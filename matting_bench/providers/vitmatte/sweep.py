from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
INFER_SCRIPT = PROVIDER_DIR / "infer.py"
EVALUATE_SCRIPT = REPO_ROOT / "matting_bench" / "evaluate.py"
DATASET_ROOT = (
    REPO_ROOT / "matting_bench" / "data" / "pet_20260710_121221_5ce7716e"
)
DEFAULT_SMOKE_DIR = DATASET_ROOT / "smoke"
DEFAULT_TEMPORAL_DIR = DATASET_ROOT / "temporal_fast_walk_24_640"
DEFAULT_RUN_ROOT = PROVIDER_DIR / "runs" / "tuning"
RESULTS_PATH = PROVIDER_DIR / "tuning_results.json"
MAX_MEASUREMENT_ATTEMPTS = 2

RADII = (2, 4, 6, 8, 12)
THRESHOLD_PRESETS = (
    {
        "name": "tight",
        "background_threshold": 0.02,
        "foreground_threshold": 0.98,
    },
    {
        "name": "relaxed",
        "background_threshold": 0.04,
        "foreground_threshold": 0.96,
    },
)
FUSION_PRESETS = (
    {"name": "conservative", "fusion_weight": 0.35, "fusion_max_delta": 0.25},
    {"name": "moderate", "fusion_weight": 0.70, "fusion_max_delta": 0.50},
)
QUALITY_KEYS = (
    "pseudo_mae",
    "background_alpha_mean",
    "foreground_loss_mean",
    "green_fringe",
    "fragment_pct",
    "soft_alpha_pct",
)
RANK_WEIGHTS = {
    "pseudo_mae": 5.0,
    "background_alpha_mean": 2.0,
    "foreground_loss_mean": 2.0,
    "green_fringe": 3.0,
    "fragment_pct": 1.0,
}
OFFICIAL_DOCS = [
    {
        "url": "https://huggingface.co/docs/transformers/v4.47.1/model_doc/vitmatte",
        "parameters": {
            "inputs": "RGB image and same-size single-channel trimap",
            "model_input": "image and trimap concatenated by VitMatteImageProcessor",
            "documented_example_output": "1x1x640x960 alpha",
        },
    },
    {
        "url": (
            "https://github.com/huggingface/transformers/blob/v4.47.1/"
            "src/transformers/models/vitmatte/image_processing_vitmatte.py"
        ),
        "parameters": {
            "rescale_factor": "1/255 for image and trimap",
            "image_normalization": "checkpoint mean/std; trimap is not normalized",
            "size_divisibility": 32,
            "padding": "right and bottom to the next divisible size; no fixed resize",
        },
    },
    {
        "url": "https://github.com/hustvl/ViTMatte/blob/main/run_one_image.py",
        "parameters": {
            "image_mode": "RGB",
            "trimap_mode": "L",
            "tensor_conversion": "torchvision to_tensor, preserving 0..1 trimap semantics",
        },
    },
    {
        "url": (
            "https://huggingface.co/hustvl/vitmatte-small-composition-1k/"
            "blob/6a58ad7646403c1df626fbd746900aec7361ea1d/"
            "preprocessor_config.json"
        ),
        "parameters": {
            "resize": "not performed by the Transformers v4.47.1 processor",
            "do_pad": True,
            "size_divisibility": 32,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ViTMatte-S trimap/fusion sweep, evaluate every output with "
            "matting_bench/evaluate.py, and retest the best two configurations."
        )
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--temporal-dir", type=Path, default=DEFAULT_TEMPORAL_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--gpu-lock-held",
        action="store_true",
        help=(
            "Required for CUDA results. Pass only when this script is itself run "
            "through matting_bench/run_with_gpu_lock.py."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def require_provider_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == PROVIDER_DIR or PROVIDER_DIR not in resolved.parents:
        raise ValueError(f"Output must stay below {PROVIDER_DIR}: {resolved}")
    return resolved


def reset_owned_directory(path: Path) -> None:
    resolved = require_provider_output(path)
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def png_summary(path: Path) -> tuple[int, tuple[int, int]]:
    from PIL import Image

    paths = sorted(path.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG frames found in {path}")
    with Image.open(paths[0]) as image:
        size = image.size
    return len(paths), size


def build_configs() -> list[dict[str, Any]]:
    configs = []
    for radius, thresholds, fusion in product(
        RADII, THRESHOLD_PRESETS, FUSION_PRESETS
    ):
        config_id = (
            f"r{radius:02d}_{thresholds['name']}_"
            f"w{int(fusion['fusion_weight'] * 100):02d}_"
            f"d{int(fusion['fusion_max_delta'] * 100):02d}"
        )
        configs.append(
            {
                "id": config_id,
                "parameters": {
                    "background_threshold": thresholds["background_threshold"],
                    "foreground_threshold": thresholds["foreground_threshold"],
                    "unknown_radius_px": radius,
                    "fusion_weight": fusion["fusion_weight"],
                    "fusion_max_delta": fusion["fusion_max_delta"],
                    "threshold_preset": thresholds["name"],
                    "fusion_preset": fusion["name"],
                    "trimap_values": {
                        "known_background": 0,
                        "unknown": 128,
                        "known_foreground": 255,
                    },
                },
            }
        )
    return configs


def empty_quality() -> dict[str, None]:
    return {key: None for key in QUALITY_KEYS}


def empty_runtime() -> dict[str, None]:
    return {
        "mean_inference_ms": None,
        "end_to_end_ms": None,
        "peak_vram_mb": None,
    }


def blank_result(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "id": config["id"],
        "parameters": config["parameters"],
        "status": "pending",
        "output_dir": relative_repo_path(output_dir),
        "quality": empty_quality(),
        "runtime": empty_runtime(),
        "unknown_region_pct": None,
        "temporal_alpha_mae": None,
        "notes": [],
    }


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see "
            f"{relative_repo_path(log_path)}"
        )


def infer_command(
    config: dict[str, Any], source_dir: Path, output_dir: Path, device: str
) -> list[str]:
    parameters = config["parameters"]
    return [
        sys.executable,
        str(INFER_SCRIPT),
        "--input-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--background-threshold",
        str(parameters["background_threshold"]),
        "--foreground-threshold",
        str(parameters["foreground_threshold"]),
        "--unknown-radius",
        str(parameters["unknown_radius_px"]),
        "--fusion-weight",
        str(parameters["fusion_weight"]),
        "--fusion-max-delta",
        str(parameters["fusion_max_delta"]),
    ]


def evaluate_command(
    config_id: str, source_dir: Path, output_dir: Path, evaluation_path: Path
) -> list[str]:
    return [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--source-dir",
        str(source_dir),
        "--provider",
        f"{config_id}={output_dir}",
        "--output",
        str(evaluation_path),
    ]


def extract_measurement(
    config: dict[str, Any], output_dir: Path, evaluation_path: Path
) -> dict[str, Any]:
    result = blank_result(config, output_dir)
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    measured = report["providers"][config["id"]]
    if measured.get("missing"):
        raise RuntimeError(f"Missing output frames: {measured['missing']}")
    mean = measured.get("mean", {})
    runtime_payload = measured.get("runtime", {})
    result["quality"] = {key: mean.get(key) for key in QUALITY_KEYS}
    peak_bytes = runtime_payload.get("cuda_memory", {}).get("peak_allocated_bytes")
    result["runtime"] = {
        "mean_inference_ms": seconds_to_ms(
            runtime_payload.get("inference_mean_seconds")
        ),
        "end_to_end_ms": seconds_to_ms(
            runtime_payload.get("end_to_end_mean_seconds")
        ),
        "peak_vram_mb": (
            float(peak_bytes) / (1024.0 * 1024.0) if peak_bytes is not None else None
        ),
        "mean_inference_excluding_first_ms": seconds_to_ms(
            runtime_payload.get("inference_mean_excluding_first_seconds")
        ),
    }
    result["unknown_region_pct"] = runtime_payload.get("pipeline", {}).get(
        "mean_unknown_pct"
    )
    result["temporal_alpha_mae"] = measured.get("temporal_alpha_mae")
    result["status"] = "ok"
    result["notes"] = [
        "Quality and temporal metrics computed by matting_bench/evaluate.py.",
        "Runtime mean includes the first inference; the warm mean is retained as an extra field.",
    ]
    return result


def seconds_to_ms(value: Any) -> float | None:
    return float(value) * 1000.0 if value is not None else None


def run_measurement(
    config: dict[str, Any],
    source_dir: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    evaluation_path = output_dir / "central_evaluation.json"
    failures = []
    for attempt in range(1, MAX_MEASUREMENT_ATTEMPTS + 1):
        reset_owned_directory(output_dir)
        try:
            run_logged(
                infer_command(config, source_dir, output_dir, device),
                output_dir / "infer.log",
            )
            run_logged(
                evaluate_command(config["id"], source_dir, output_dir, evaluation_path),
                output_dir / "evaluate.log",
            )
            result = extract_measurement(config, output_dir, evaluation_path)
            result["attempts"] = attempt
            if failures:
                result["notes"].append(
                    "Recovered after transient failure(s): " + " | ".join(failures)
                )
            return result
        except Exception as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")

    result = blank_result(config, output_dir)
    result["status"] = "error"
    result["attempts"] = MAX_MEASUREMENT_ATTEMPTS
    result["notes"] = failures
    return result


def add_selection_scores(configs: list[dict[str, Any]]) -> None:
    valid = [item for item in configs if item["status"] == "ok"]
    if not valid:
        return
    denominator = max(1, len(valid) - 1)
    totals = {item["id"]: 0.0 for item in valid}
    for key, weight in RANK_WEIGHTS.items():
        ordered = sorted(valid, key=lambda item: float(item["quality"][key]))
        for rank, item in enumerate(ordered):
            totals[item["id"]] += weight * rank / denominator
    for item in valid:
        item["selection_score"] = totals[item["id"]]
        item["selection_score_rule"] = (
            "lower is better; weighted normalized ranks over pseudo_mae=5, "
            "background_alpha_mean=2, foreground_loss_mean=2, "
            "green_fringe=3, fragment_pct=1"
        )


def write_results(payload: dict[str, Any]) -> None:
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_payload(
    smoke_dir: Path,
    temporal_dir: Path,
    device: str,
    gpu_lock_held: bool,
) -> dict[str, Any]:
    smoke_count, smoke_size = png_summary(smoke_dir)
    temporal_count, temporal_size = png_summary(temporal_dir)
    return {
        "provider": "vitmatte_adaptive_green_hybrid",
        "official_docs": OFFICIAL_DOCS,
        "dataset": {
            "id": "pet_20260710_121221_5ce7716e",
            "smoke": {
                "path": relative_repo_path(smoke_dir),
                "frames": smoke_count,
                "width": smoke_size[0],
                "height": smoke_size[1],
            },
            "temporal": {
                "path": relative_repo_path(temporal_dir),
                "frames": temporal_count,
                "width": temporal_size[0],
                "height": temporal_size[1],
            },
            "central_evaluator": "matting_bench/evaluate.py",
        },
        "benchmark": {
            "started_at_utc": utc_now(),
            "device": device,
            "gpu_lock_held": gpu_lock_held,
            "gpu_lock_wrapper": "matting_bench/run_with_gpu_lock.py",
            "timing_policy": (
                "All reported CUDA screening and final retest measurements are run "
                "inside one exclusive GPU-lock invocation."
            ),
        },
        "configs": [],
        "recommendation": {},
    }


def main() -> int:
    args = parse_args()
    smoke_dir = args.smoke_dir.resolve()
    temporal_dir = args.temporal_dir.resolve()
    run_root = require_provider_output(args.run_root)
    if args.device == "cuda" and not args.gpu_lock_held:
        raise SystemExit(
            "CUDA timing requires --gpu-lock-held and an outer "
            "matting_bench/run_with_gpu_lock.py invocation"
        )

    payload = build_payload(
        smoke_dir, temporal_dir, args.device, args.gpu_lock_held
    )
    definitions = build_configs()
    for index, definition in enumerate(definitions, start=1):
        print(f"[{index:02d}/{len(definitions):02d}] {definition['id']}", flush=True)
        output_dir = run_root / "screening" / definition["id"]
        measurement = run_measurement(
            definition, smoke_dir, output_dir, args.device
        )
        payload["configs"].append(measurement)
        write_results(payload)

    add_selection_scores(payload["configs"])
    valid = sorted(
        (item for item in payload["configs"] if item["status"] == "ok"),
        key=lambda item: (item["selection_score"], item["id"]),
    )
    finalists = valid[:2]
    definitions_by_id = {item["id"]: item for item in definitions}
    for finalist in finalists:
        config_id = finalist["id"]
        definition = definitions_by_id[config_id]
        screening_snapshot = {
            "output_dir": finalist["output_dir"],
            "quality": finalist["quality"],
            "runtime": finalist["runtime"],
            "unknown_region_pct": finalist["unknown_region_pct"],
        }
        print(f"[final smoke retest] {config_id}", flush=True)
        final_smoke = run_measurement(
            definition,
            smoke_dir,
            run_root / "final" / config_id / "smoke",
            args.device,
        )
        if final_smoke["status"] == "ok":
            finalist.update(final_smoke)
            finalist["screening"] = screening_snapshot
            finalist["notes"].append(
                "Required final smoke speed/VRAM retest completed while the GPU lock was held."
            )
        else:
            finalist["notes"].extend(final_smoke["notes"])
            finalist["status"] = "final-smoke-error"
            continue

        print(f"[final temporal retest] {config_id}", flush=True)
        temporal = run_measurement(
            definition,
            temporal_dir,
            run_root / "final" / config_id / "temporal_fast_walk_24_640",
            args.device,
        )
        finalist["temporal_output_dir"] = temporal["output_dir"]
        finalist["temporal_alpha_mae"] = temporal["temporal_alpha_mae"]
        finalist["temporal_runtime"] = temporal["runtime"]
        finalist["temporal_status"] = temporal["status"]
        finalist["notes"].append(
            "Temporal metric comes from the locked 24-frame final retest."
        )
        if temporal["status"] != "ok":
            finalist["notes"].extend(temporal["notes"])

    finalists = [item for item in finalists if item["status"] == "ok"]
    selected = finalists[0] if finalists else None
    payload["recommendation"] = {
        "selected_config_id": selected["id"] if selected else None,
        "finalists": [item["id"] for item in finalists],
        "parameters": selected["parameters"] if selected else None,
        "quality": selected["quality"] if selected else None,
        "runtime": selected["runtime"] if selected else None,
        "unknown_region_pct": selected["unknown_region_pct"] if selected else None,
        "temporal_alpha_mae": selected["temporal_alpha_mae"] if selected else None,
        "basis": (
            "Lowest weighted smoke quality rank; the two best smoke configurations "
            "were independently rerun on smoke and temporal_fast_walk_24_640 while "
            "holding the exclusive GPU benchmark lock."
        ),
    }
    payload["benchmark"]["completed_at_utc"] = utc_now()
    payload["benchmark"]["successful_screening_configs"] = sum(
        item["status"] in {"ok", "final-smoke-error"} for item in payload["configs"]
    )
    write_results(payload)
    print(RESULTS_PATH, flush=True)
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run and collect the reproducible rembg parameter sweep.

Inference artifacts stay below this provider directory. Quality is always computed by
the repository's central matting_bench/evaluate.py rather than duplicated here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from model_catalog import MODEL_SPECS


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
INFER_SCRIPT = PROVIDER_DIR / "infer.py"
EVALUATE_SCRIPT = REPO_ROOT / "matting_bench" / "evaluate.py"
GPU_LOCK_SCRIPT = REPO_ROOT / "matting_bench" / "run_with_gpu_lock.py"
DATASET_ROOT = (
    REPO_ROOT / "matting_bench" / "data" / "pet_20260710_121221_5ce7716e"
)
SMOKE_DIR = DATASET_ROOT / "smoke"
TEMPORAL_DIR = DATASET_ROOT / "temporal_fast_walk_24_640"
TUNING_OUTPUT_DIR = PROVIDER_DIR / "outputs" / "tuning"
RESULTS_PATH = PROVIDER_DIR / "tuning_results.json"


MODEL_DETAILS: dict[str, dict[str, Any]] = {
    "u2net": {
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "mask_activation": "first ONNX output, per-image min-max normalization",
        "resize": "RGB LANCZOS stretch to 320x320; mask LANCZOS to source size",
    },
    "isnet-general-use": {
        "normalization_mean": [0.5, 0.5, 0.5],
        "normalization_std": [1.0, 1.0, 1.0],
        "mask_activation": "first ONNX output, per-image min-max normalization",
        "resize": "RGB LANCZOS stretch to 1024x1024; mask LANCZOS to source size",
    },
    "birefnet-general-lite": {
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "mask_activation": "sigmoid on first ONNX output, then per-image min-max normalization",
        "resize": "RGB LANCZOS stretch to 1024x1024; mask LANCZOS to source size",
    },
}


OFFICIAL_DOCS: list[dict[str, Any]] = [
    {
        "url": "https://github.com/danielgatis/rembg/blob/main/README.md",
        "parameters": {
            "models": ["u2net", "isnet-general-use", "birefnet-general-lite"],
            "batching": "reuse one new_session() across images",
            "environment": {
                "U2NET_HOME": "model directory",
                "OMP_NUM_THREADS": "ONNX Runtime thread count",
            },
        },
    },
    {
        "url": "https://github.com/danielgatis/rembg/blob/main/USAGE.md",
        "parameters": {
            "alpha_matting": "optional post-processing refinement",
            "documented_example": {
                "foreground_threshold": 270,
                "background_threshold": 20,
                "erode_size": 11,
            },
            "post_process_mask": "optional mask post-processing",
        },
    },
    {
        "url": "https://github.com/danielgatis/rembg/blob/main/rembg/bg.py",
        "parameters": {
            "remove_defaults": {
                "alpha_matting": False,
                "foreground_threshold": 240,
                "background_threshold": 10,
                "erode_size": 10,
                "post_process_mask": False,
            },
            "trimap": "mask > foreground is known FG; mask < background is known BG; both regions are eroded",
            "post_process_mask": "disk-radius-1 opening, Gaussian sigma=2, threshold at 127 to binary",
        },
    },
    {
        "url": "https://github.com/danielgatis/rembg/blob/main/rembg/session_factory.py",
        "parameters": {
            "session_options": "default onnxruntime.SessionOptions",
            "OMP_NUM_THREADS": "sets both inter_op_num_threads and intra_op_num_threads",
            "providers": "passed through new_session() to ONNX Runtime",
        },
    },
    {
        "url": "https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2net.py",
        "parameters": {
            "input_size": [320, 320],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    {
        "url": "https://github.com/danielgatis/rembg/blob/main/rembg/sessions/dis_general_use.py",
        "parameters": {
            "input_size": [1024, 1024],
            "mean": [0.5, 0.5, 0.5],
            "std": [1.0, 1.0, 1.0],
        },
    },
    {
        "url": "https://github.com/danielgatis/rembg/blob/main/rembg/sessions/birefnet_general.py",
        "parameters": {
            "input_size": [1024, 1024],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "activation": "sigmoid before per-image min-max normalization",
        },
    },
    {
        "url": "https://github.com/xuebinqin/U-2-Net/blob/master/u2net_test.py",
        "parameters": {
            "upstream_input_size": [320, 320],
            "output": "first side output with per-image min-max normalization",
        },
    },
    {
        "url": "https://github.com/xuebinqin/DIS/blob/main/IS-Net/Inference.py",
        "parameters": {
            "upstream_input_size": [1024, 1024],
            "mean": [0.5, 0.5, 0.5],
            "std": [1.0, 1.0, 1.0],
        },
    },
    {
        "url": "https://github.com/ZhengPeng7/BiRefNet",
        "parameters": {
            "default_input_size": [1024, 1024],
            "onnx_note": "upstream reports extra ONNX latency and slight output differences",
        },
    },
]


@dataclass(frozen=True)
class TuningConfig:
    id: str
    model: str
    alpha_matting: bool
    foreground_threshold: int
    background_threshold: int
    erode_size: int
    post_process_mask: bool
    hypothesis: str


VARIANTS = (
    {
        "name": "default",
        "alpha_matting": False,
        "foreground_threshold": 240,
        "background_threshold": 10,
        "erode_size": 10,
        "post_process_mask": False,
        "hypothesis": "Unmodified rembg output; preserves the model's continuous mask.",
    },
    {
        "name": "alpha_default",
        "alpha_matting": True,
        "foreground_threshold": 240,
        "background_threshold": 10,
        "erode_size": 10,
        "post_process_mask": False,
        "hypothesis": "Official remove() alpha defaults; broad unknown trimap for edge refinement.",
    },
    {
        "name": "alpha_fur_safe",
        "alpha_matting": True,
        "foreground_threshold": 225,
        "background_threshold": 5,
        "erode_size": 3,
        "post_process_mask": False,
        "hypothesis": "Protect low-confidence fur by shrinking known BG and limiting trimap erosion.",
    },
    {
        "name": "postprocess_binary",
        "alpha_matting": False,
        "foreground_threshold": 240,
        "background_threshold": 10,
        "erode_size": 10,
        "post_process_mask": True,
        "hypothesis": "Test fragment cleanup separately; binary output is a fine-hair risk.",
    },
)


CONFIGS = tuple(
    TuningConfig(
        id=f"{model}__{variant['name']}",
        model=model,
        alpha_matting=bool(variant["alpha_matting"]),
        foreground_threshold=int(variant["foreground_threshold"]),
        background_threshold=int(variant["background_threshold"]),
        erode_size=int(variant["erode_size"]),
        post_process_mask=bool(variant["post_process_mask"]),
        hypothesis=str(variant["hypothesis"]),
    )
    for model in MODEL_SPECS
    for variant in VARIANTS
)
CONFIG_BY_ID = {config.id: config for config in CONFIGS}


# Kept explicit so `--phase all` reproduces both selected temporal runs.
RECOMMENDED_CONFIG_IDS = (
    "u2net__alpha_default",
    "isnet-general-use__alpha_default",
)
RECOMMENDATION_RATIONALE = (
    "U2Net with official alpha defaults is the primary fine-hair-safe balance: it "
    "kept more soft detail outside the binary core, matched ISNet temporal stability, "
    "and was faster. ISNet with the same alpha defaults is the tighter secondary "
    "choice with marginally lower pseudo error and fragment ratio."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("smoke", "temporal", "collect", "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(CONFIG_BY_ID),
        help="Limit the selected phase to one or more config ids.",
    )
    parser.add_argument(
        "--evaluation-python",
        default="python",
        help="Python executable with cv2 available for central evaluate.py.",
    )
    parser.add_argument(
        "--gpu-lock-python",
        default="python",
        help="Python executable used to launch central run_with_gpu_lock.py.",
    )
    parser.add_argument("--omp-num-threads", type=int)
    parser.add_argument(
        "--gpu-lock-retries",
        type=int,
        default=120,
        help="Retry count when the Windows lock helper reports temporary contention.",
    )
    parser.add_argument("--gpu-lock-retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def selected_configs(args: argparse.Namespace, phase: str) -> list[TuningConfig]:
    if args.config:
        return [CONFIG_BY_ID[config_id] for config_id in args.config]
    if phase == "temporal":
        if not RECOMMENDED_CONFIG_IDS:
            raise ValueError("No temporal configs selected; pass --config explicitly")
        return [CONFIG_BY_ID[config_id] for config_id in RECOMMENDED_CONFIG_IDS]
    return list(CONFIGS)


def output_dir(phase: str, config: TuningConfig) -> Path:
    return TUNING_OUTPUT_DIR / phase / config.id


def run_config(
    config: TuningConfig,
    phase: str,
    args: argparse.Namespace,
) -> None:
    source_dir = SMOKE_DIR if phase == "smoke" else TEMPORAL_DIR
    destination = output_dir(phase, config)
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(INFER_SCRIPT),
        "--model",
        config.model,
        "--device",
        args.device,
        "--input-dir",
        str(source_dir),
        "--output-dir",
        str(destination),
        "--metrics-json",
        str(destination / "metrics.json"),
        "--alpha-matting-foreground-threshold",
        str(config.foreground_threshold),
        "--alpha-matting-background-threshold",
        str(config.background_threshold),
        "--alpha-matting-erode-size",
        str(config.erode_size),
    ]
    if config.alpha_matting:
        command.append("--alpha-matting")
    if config.post_process_mask:
        command.append("--post-process-mask")
    if args.omp_num_threads is not None:
        command.extend(("--omp-num-threads", str(args.omp_num_threads)))
    if args.overwrite:
        command.append("--overwrite")

    if args.device == "cuda":
        command = [
            args.gpu_lock_python,
            str(GPU_LOCK_SCRIPT),
            "--",
            *command,
        ]

    print(f"[{phase}] {config.id}", flush=True)
    log_path = destination / "run.log"
    attempts = args.gpu_lock_retries if args.device == "cuda" else 1
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with log_path.open("a" if attempt > 1 else "w", encoding="utf-8") as log:
            if attempt > 1:
                log.write(f"\n--- GPU lock attempt {attempt} ---\n")
            log.write(completed.stdout)
        if completed.returncode == 0:
            break
        lock_contended = "Resource deadlock avoided" in completed.stdout
        if not lock_contended or attempt == attempts:
            raise subprocess.CalledProcessError(completed.returncode, command)
        time.sleep(args.gpu_lock_retry_delay)
    evaluate_command = [
        args.evaluation_python,
        str(EVALUATE_SCRIPT),
        "--source-dir",
        str(source_dir),
        "--provider",
        f"{config.id}={destination}",
        "--output",
        str(destination / "evaluation.json"),
    ]
    subprocess.run(evaluate_command, cwd=REPO_ROOT, check=True)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def evaluated_config(phase: str, config: TuningConfig) -> dict[str, Any] | None:
    payload = read_json(output_dir(phase, config) / "evaluation.json")
    if payload is None:
        return None
    providers = payload.get("providers", {})
    result = providers.get(config.id)
    return result if isinstance(result, dict) else None


def dataset_details(path: Path) -> dict[str, Any]:
    images = sorted(path.glob("*.png"), key=lambda item: item.name.casefold())
    size = None
    if images:
        with Image.open(images[0]) as image:
            size = [image.width, image.height]
    return {
        "path": relative(path),
        "frames": len(images),
        "frame_size": size,
        "ordered_basenames": [image.name for image in images],
    }


def quality_payload(evaluation: dict[str, Any] | None) -> dict[str, float | None]:
    mean = evaluation.get("mean", {}) if evaluation else {}
    keys = (
        "pseudo_mae",
        "background_alpha_mean",
        "foreground_loss_mean",
        "green_fringe",
        "fragment_pct",
        "soft_alpha_pct",
    )
    return {
        key: float(mean[key]) if isinstance(mean.get(key), (int, float)) else None
        for key in keys
    }


def runtime_payload(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    if metrics is None:
        return {
            "mean_inference_ms": None,
            "end_to_end_ms": None,
            "peak_vram_mb": None,
        }
    inference = metrics.get("remove_mean_excluding_first_seconds")
    end_to_end = metrics.get("end_to_end_mean_excluding_first_seconds")
    return {
        "mean_inference_ms": (
            float(inference) * 1000 if isinstance(inference, (int, float)) else None
        ),
        "end_to_end_ms": (
            float(end_to_end) * 1000
            if isinstance(end_to_end, (int, float))
            else None
        ),
        "peak_vram_mb": None,
    }


def soft_detail_comparison(model: str) -> dict[str, float | None]:
    default_dir = output_dir(
        "smoke", CONFIG_BY_ID[f"{model}__alpha_default"]
    )
    fur_safe_dir = output_dir(
        "smoke", CONFIG_BY_ID[f"{model}__alpha_fur_safe"]
    )
    binary_dir = output_dir(
        "smoke", CONFIG_BY_ID[f"{model}__postprocess_binary"]
    )
    values = []
    for default_path in sorted(default_dir.glob("*.png")):
        fur_safe_path = fur_safe_dir / default_path.name
        binary_path = binary_dir / default_path.name
        if not fur_safe_path.is_file() or not binary_path.is_file():
            continue
        default_alpha = (
            np.asarray(Image.open(default_path).convert("RGBA"), dtype=np.float32)[
                :, :, 3
            ]
            / 255.0
        )
        fur_safe_alpha = (
            np.asarray(Image.open(fur_safe_path).convert("RGBA"), dtype=np.float32)[
                :, :, 3
            ]
            / 255.0
        )
        binary_alpha = (
            np.asarray(Image.open(binary_path).convert("RGBA"), dtype=np.float32)[
                :, :, 3
            ]
            / 255.0
        )
        outside_binary = binary_alpha < 0.5
        default_detail = (default_alpha > 0.08) & outside_binary
        fur_safe_detail = (fur_safe_alpha > 0.08) & outside_binary
        values.append(
            (
                float(default_detail.sum()),
                float(fur_safe_detail.sum()),
                float(
                    (
                        default_detail
                        & (fur_safe_alpha <= 0.02)
                    ).sum()
                ),
                float(
                    (
                        fur_safe_detail
                        & (default_alpha <= 0.02)
                    ).sum()
                ),
            )
        )
    if not values:
        return {
            "alpha_default_detail_pixels_per_frame": None,
            "fur_safe_detail_pixels_per_frame": None,
            "fur_safe_lost_default_pixels_per_frame": None,
            "fur_safe_only_pixels_per_frame": None,
        }
    means = np.asarray(values, dtype=np.float64).mean(axis=0)
    return {
        "alpha_default_detail_pixels_per_frame": float(means[0]),
        "fur_safe_detail_pixels_per_frame": float(means[1]),
        "fur_safe_lost_default_pixels_per_frame": float(means[2]),
        "fur_safe_only_pixels_per_frame": float(means[3]),
    }


def config_parameters(
    config: TuningConfig,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = MODEL_SPECS[config.model]
    active_providers = metrics.get("requested_providers") if metrics else None
    environment = metrics.get("environment", {}) if metrics else {}
    return {
        "model": config.model,
        "model_input": {
            "size": list(spec.input_size),
            **MODEL_DETAILS[config.model],
        },
        "alpha_matting": config.alpha_matting,
        "alpha_matting_foreground_threshold": config.foreground_threshold,
        "alpha_matting_background_threshold": config.background_threshold,
        "alpha_matting_erode_size": config.erode_size,
        "post_process_mask": config.post_process_mask,
        "session": {
            "device": metrics.get("device") if metrics else None,
            "requested_execution_providers": active_providers,
            "active_execution_providers": environment.get("active_providers"),
            "provider_options": environment.get("provider_options"),
            "omp_num_threads": environment.get("omp_num_threads"),
            "reused_across_frames": True,
        },
    }


def collect() -> dict[str, Any]:
    entries = []
    detail_comparisons = {
        model: soft_detail_comparison(model) for model in MODEL_SPECS
    }
    for config in CONFIGS:
        smoke_dir = output_dir("smoke", config)
        smoke_metrics = read_json(smoke_dir / "metrics.json")
        smoke_evaluation = evaluated_config("smoke", config)
        temporal_evaluation = evaluated_config("temporal", config)
        if smoke_metrics is None or smoke_evaluation is None:
            status = "not_run"
        elif (
            smoke_metrics.get("status") == "ok"
            and smoke_metrics.get("image_count") == 9
            and smoke_evaluation.get("frames") == 9
            and not smoke_evaluation.get("missing")
        ):
            status = "ok"
        else:
            status = "invalid"
        temporal_value = (
            temporal_evaluation.get("temporal_alpha_mae")
            if temporal_evaluation
            else None
        )
        notes = [
            config.hypothesis,
            "Runtime means exclude the first frame; peak VRAM was not instrumented.",
        ]
        if smoke_metrics and smoke_metrics.get("device") == "cuda":
            notes.append(
                "Final smoke timing ran serially under matting_bench/run_with_gpu_lock.py."
            )
        if config.post_process_mask:
            notes.append(
                "Rejected for final use: soft_alpha_pct is zero and visual QA shows hard-cut whiskers/fur."
            )
        if config.id.endswith("__alpha_fur_safe"):
            comparison = detail_comparisons[config.model]
            lost = comparison["fur_safe_lost_default_pixels_per_frame"]
            gained = comparison["fur_safe_only_pixels_per_frame"]
            if lost is not None and gained is not None:
                notes.append(
                    f"Versus alpha_default, lost {lost:.1f} soft-detail pixels/frame "
                    f"and uniquely added only {gained:.1f}; not fine-hair safe."
                )
        if temporal_evaluation:
            notes.append(
                "24-frame temporal run completed under the GPU lock; "
                f"temporal alpha MAE={float(temporal_value):.7f}."
            )
        entries.append(
            {
                "id": config.id,
                "parameters": config_parameters(config, smoke_metrics),
                "status": status,
                "output_dir": relative(smoke_dir),
                "quality": quality_payload(smoke_evaluation),
                "runtime": runtime_payload(smoke_metrics),
                "temporal_alpha_mae": (
                    float(temporal_value)
                    if isinstance(temporal_value, (int, float))
                    else None
                ),
                "temporal_output_dir": (
                    relative(output_dir("temporal", config))
                    if temporal_evaluation
                    else None
                ),
                "temporal_frames": (
                    temporal_evaluation.get("frames") if temporal_evaluation else None
                ),
                "notes": notes,
            }
        )

    primary = RECOMMENDED_CONFIG_IDS[0] if RECOMMENDED_CONFIG_IDS else None
    secondary = (
        RECOMMENDED_CONFIG_IDS[1] if len(RECOMMENDED_CONFIG_IDS) > 1 else None
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "rembg",
        "provider_meta": {
            "version": metadata.version("rembg"),
            "models": list(MODEL_SPECS),
            "timing_protocol": (
                "CUDA inference serialized by matting_bench/run_with_gpu_lock.py; "
                "reported means exclude the first frame"
            ),
            "peak_vram_instrumented": False,
        },
        "official_docs": OFFICIAL_DOCS,
        "dataset": {
            "smoke": dataset_details(SMOKE_DIR),
            "temporal": dataset_details(TEMPORAL_DIR),
            "evaluator": relative(EVALUATE_SCRIPT),
            "ground_truth": "controlled-green-screen pseudo metrics; no hand-painted alpha",
        },
        "configs": entries,
        "recommendation": {
            "primary": primary,
            "secondary": secondary,
            "rationale": RECOMMENDATION_RATIONALE,
            "production_parameters": {
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 240,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 10,
                "post_process_mask": False,
            },
            "soft_detail_comparison": detail_comparisons,
            "rejected_patterns": [
                "post_process_mask=True: binary alpha removed whiskers and all soft alpha",
                "alpha_fur_safe 225/5/3: narrower unknown band deleted more soft detail than it recovered",
                "birefnet-general-lite: about 4.9-5.0 seconds per hot 960px frame without a quality win",
            ],
            "visual_review_artifacts": [
                relative(PROVIDER_DIR / "visual_review" / "smoke_candidates_contact.jpg"),
                relative(PROVIDER_DIR / "visual_review" / "fast_walk_mid_full_600.jpg"),
                relative(PROVIDER_DIR / "visual_review" / "idle_mid_head_600.jpg"),
            ],
            "fine_hair_guardrail": (
                "Do not select on pseudo_mae alone; inspect whiskers, ear tufts, tail fur, "
                "soft_alpha_pct, and foreground_loss_mean."
            ),
        },
    }
    RESULTS_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(RESULTS_PATH)
    return result


def main() -> int:
    args = parse_args()
    if args.omp_num_threads is not None and args.omp_num_threads < 1:
        raise ValueError("--omp-num-threads must be positive")
    if args.gpu_lock_retries < 1:
        raise ValueError("--gpu-lock-retries must be positive")
    if args.gpu_lock_retry_delay < 0:
        raise ValueError("--gpu-lock-retry-delay must be non-negative")

    if args.phase in ("smoke", "all"):
        for config in selected_configs(args, "smoke"):
            run_config(config, "smoke", args)
    if args.phase in ("temporal", "all"):
        for config in selected_configs(args, "temporal"):
            run_config(config, "temporal", args)
    if args.phase in ("collect", "all"):
        collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

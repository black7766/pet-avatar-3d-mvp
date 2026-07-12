"""Run every recommended matting provider on one pet action sequence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = REPO_ROOT / "matting_bench"
DATASET_ROOT = BENCH_ROOT / "data" / "pet_20260710_121221_5ce7716e"
LOCK_RUNNER = BENCH_ROOT / "run_with_gpu_lock.py"
DEFAULT_FRAME_COUNT = 96

PROVIDER_NAMES = (
    "adaptive_green_baseline",
    "adaptive_green_edge_v2",
    "ZhengPeng7/BiRefNet (General)",
    "vitmatte_adaptive_green_hybrid",
    "rembg",
    "ben2",
    "ZhengPeng7/BiRefNet-matting",
    "paddle_matting",
    "official_matanyone_v1",
    "official_meta_sam2_1_small_video",
)

FAST_WALK_OUTPUTS = {
    "adaptive_green_baseline": "matting_bench/outputs/tuning/baseline/temporal/despill_1_10",
    "ZhengPeng7/BiRefNet (General)": "matting_bench/outputs/birefnet_tuning/locked/temporal_fast_walk_24_640/general_1024_auto",
    "vitmatte_adaptive_green_hybrid": "matting_bench/providers/vitmatte/runs/tuning/final/r02_tight_w35_d25/temporal_fast_walk_24_640",
    "rembg": "matting_bench/providers/rembg/outputs/tuning/temporal/u2net__alpha_default",
    "ben2": "matting_bench/providers/ben2/evidence/tuning/locked_temporal/refine_r90",
    "ZhengPeng7/BiRefNet-matting": "matting_bench/outputs/birefnet_matting_tuning/locked/temporal_fast_walk_24_640/matting_1024_auto",
    "paddle_matting": "matting_bench/providers/paddle_matting/evidence/tuning/locked_temporal/default_512",
    "official_matanyone_v1": "matting_bench/providers/video_matting/runs/tuning_warmup1/rgba",
    "official_meta_sam2_1_small_video": "matting_bench/providers/sam2_video/runs/tuning_state_cpu/rgba",
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def output_frames(path: Path, expected_count: int) -> list[Path]:
    direct = sorted(path.glob("*.png"))
    if len(direct) == expected_count:
        return direct
    rgba = sorted((path / "rgba").glob("*.png"))
    if len(rgba) == expected_count:
        return rgba
    return []


def prepare_source(action: str, output: Path, frame_count: int) -> None:
    if len(sorted(output.glob("*.png"))) == frame_count:
        return
    source = DATASET_ROOT / "full" / action
    paths = sorted(source.glob("*.png"))[:frame_count]
    if len(paths) != frame_count:
        raise FileNotFoundError(
            f"expected at least {frame_count} source frames in {source}"
        )
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
            resized.save(output / f"f_{index:04d}.png", compress_level=2)


def python_in(venv: str) -> Path:
    path = REPO_ROOT / ".venvs" / venv / "Scripts" / "python.exe"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def commands(
    input_dir: Path,
    output_root: Path,
    frame_count: int,
) -> dict[str, list[str]]:
    def out(name: str) -> str:
        return str(output_root / name)

    run_slug = f"{output_root.parent.name}_{output_root.name}"
    matanyone_output = BENCH_ROOT / "providers" / "video_matting" / "runs" / run_slug
    sam2_output = BENCH_ROOT / "providers" / "sam2_video" / "runs" / run_slug

    return {
        "adaptive_green_baseline": [
            sys.executable,
            str(BENCH_ROOT / "providers" / "baseline" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("adaptive_green_baseline"),
            "--device",
            "cpu",
            "--core-despill",
            "1.10",
        ],
        "adaptive_green_edge_v2": [
            sys.executable,
            str(BENCH_ROOT / "providers" / "baseline" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("adaptive_green_edge_v2"),
            "--device",
            "cpu",
            "--core-despill",
            "1.10",
            "--edge-refine",
        ],
        "ZhengPeng7/BiRefNet (General)": [
            str(python_in("birefnet")),
            str(BENCH_ROOT / "providers" / "birefnet" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("birefnet_general"),
            "--device",
            "cuda",
            "--input-resolution",
            "1024",
            "--foreground-refinement",
            "official-auto",
            "--metrics-json",
            str(output_root / "birefnet_general" / "metrics.json"),
        ],
        "vitmatte_adaptive_green_hybrid": [
            str(python_in("vitmatte")),
            str(BENCH_ROOT / "providers" / "vitmatte" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("vitmatte"),
            "--device",
            "cuda",
            "--background-threshold",
            "0.02",
            "--foreground-threshold",
            "0.98",
            "--unknown-radius",
            "2",
            "--fusion-weight",
            "0.35",
            "--fusion-max-delta",
            "0.25",
        ],
        "rembg": [
            str(python_in("rembg")),
            str(BENCH_ROOT / "providers" / "rembg" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("rembg"),
            "--model",
            "u2net",
            "--device",
            "cuda",
            "--alpha-matting",
            "--alpha-matting-foreground-threshold",
            "240",
            "--alpha-matting-background-threshold",
            "10",
            "--alpha-matting-erode-size",
            "10",
        ],
        "ben2": [
            str(python_in("ben2")),
            str(BENCH_ROOT / "providers" / "ben2" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("ben2"),
            "--device",
            "cuda",
            "--refine-foreground",
            "--refine-radius",
            "90",
        ],
        "ZhengPeng7/BiRefNet-matting": [
            str(python_in("birefnet")),
            str(BENCH_ROOT / "providers" / "birefnet_matting" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("birefnet_matting"),
            "--device",
            "cuda",
            "--input-resolution",
            "1024",
            "--foreground-refinement",
            "official-auto",
            "--metrics-json",
            str(output_root / "birefnet_matting" / "metrics.json"),
        ],
        "paddle_matting": [
            str(python_in("paddle_matting")),
            str(BENCH_ROOT / "providers" / "paddle_matting" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            out("paddle_matting"),
            "--device",
            "cuda",
            "--max-short",
            "512",
        ],
        "official_matanyone_v1": [
            str(python_in("video_matting")),
            str(BENCH_ROOT / "providers" / "video_matting" / "matanyone_cli.py"),
            "--input",
            str(input_dir),
            "--output-dir",
            str(matanyone_output),
            "--frames",
            str(frame_count),
            "--max-size",
            "640",
            "--max-internal-size",
            "-1",
            "--warmup",
            "1",
            "--mem-every",
            "5",
            "--max-mem-frames",
            "5",
            "--init-kind",
            "mask",
            "--rgba-rgb",
            "green-clean",
        ],
        "official_meta_sam2_1_small_video": [
            str(python_in("sam2_video")),
            str(BENCH_ROOT / "providers" / "sam2_video" / "infer.py"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(sam2_output),
            "--frames",
            str(frame_count),
            "--mask-threshold",
            "128",
            "--logit-threshold",
            "0",
            "--precision",
            "fp16",
            "--offload-state-to-cpu",
        ],
    }


def run(command: list[str], *, gpu: bool) -> None:
    actual = command
    if gpu:
        actual = [sys.executable, str(LOCK_RUNNER), "--", *command]
    subprocess.run(actual, cwd=REPO_ROOT, check=True)


def evaluate(source_dir: Path, outputs: dict[str, Path], target: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(BENCH_ROOT / "evaluate.py"),
        "--source-dir",
        str(source_dir),
    ]
    for provider, directory in outputs.items():
        command.extend(("--provider", f"{provider}={directory}"))
    command.extend(("--output", str(target)))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(target.read_text(encoding="utf-8"))


def build_fast_walk_manifest(output_root: Path) -> dict[str, Any]:
    source_dir = DATASET_ROOT / "temporal_fast_walk_24_640"
    outputs = {name: REPO_ROOT / path for name, path in FAST_WALK_OUTPUTS.items()}
    for name, directory in outputs.items():
        if len(output_frames(directory, 24)) != 24:
            raise FileNotFoundError(f"missing fast-walk output for {name}: {directory}")
    evaluation_path = output_root / "evaluation.json"
    evaluation = evaluate(source_dir, outputs, evaluation_path)
    return manifest_payload(
        "fast_walk", source_dir, outputs, evaluation, evaluation_path, 24
    )


def metrics_path(directory: Path) -> Path:
    if directory.name == "rgba":
        return directory.parent / "metrics.json"
    return directory / "metrics.json"


def manifest_payload(
    action: str,
    source_dir: Path,
    outputs: dict[str, Path],
    evaluation: dict[str, Any],
    evaluation_path: Path,
    frame_count: int,
) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    for name, directory in outputs.items():
        frames = output_frames(directory, frame_count)
        providers[name] = {
            "output_dir": relative(directory),
            "metrics_json": relative(metrics_path(directory)),
            "frames": len(frames),
            "evaluation_key": name,
        }
    return {
        "schema_version": 1,
        "action": action,
        "source_dir": relative(source_dir),
        "source_frames": len(sorted(source_dir.glob("*.png"))),
        "evaluation_json": relative(evaluation_path),
        "provider_count": len(providers),
        "providers": providers,
        "evaluation": {
            "provider_count": len(evaluation.get("providers", {})),
            "metric_scope": f"action-specific {frame_count}-frame evaluation",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("fast_walk", "sleep"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("matting_bench/outputs/action_compare_5s"),
    )
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    output_root = (REPO_ROOT / args.output_root / args.action).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.frame_count == 24 and args.action == "fast_walk":
        payload = build_fast_walk_manifest(output_root)
    else:
        source_dir = DATASET_ROOT / f"temporal_{args.action}_{args.frame_count}_640"
        prepare_source(args.action, source_dir, args.frame_count)
        provider_commands = commands(source_dir, output_root, args.frame_count)
        outputs: dict[str, Path] = {}
        for name in PROVIDER_NAMES:
            command = provider_commands[name]
            if name == "adaptive_green_baseline":
                directory = output_root / "adaptive_green_baseline"
            elif name == "adaptive_green_edge_v2":
                directory = output_root / "adaptive_green_edge_v2"
            elif name == "ZhengPeng7/BiRefNet (General)":
                directory = output_root / "birefnet_general"
            elif name == "vitmatte_adaptive_green_hybrid":
                directory = output_root / "vitmatte"
            elif name == "rembg":
                directory = output_root / "rembg"
            elif name == "ben2":
                directory = output_root / "ben2"
            elif name == "ZhengPeng7/BiRefNet-matting":
                directory = output_root / "birefnet_matting"
            elif name == "paddle_matting":
                directory = output_root / "paddle_matting"
            elif name == "official_matanyone_v1":
                directory = (
                    BENCH_ROOT
                    / "providers"
                    / "video_matting"
                    / "runs"
                    / f"{output_root.parent.name}_{output_root.name}"
                )
            else:
                directory = (
                    BENCH_ROOT
                    / "providers"
                    / "sam2_video"
                    / "runs"
                    / f"{output_root.parent.name}_{output_root.name}"
                )
            existing = output_frames(directory, args.frame_count)
            expected_metrics = metrics_path(directory)
            complete = len(existing) == args.frame_count and expected_metrics.is_file()
            if not args.reuse_existing or not complete:
                can_overwrite_in_place = name in {
                    "ZhengPeng7/BiRefNet (General)",
                    "ZhengPeng7/BiRefNet-matting",
                }
                if (
                    directory.exists()
                    and any(directory.iterdir())
                    and not can_overwrite_in_place
                ):
                    raise RuntimeError(
                        f"partial output exists for {name}: {directory}; remove it explicitly before rerun"
                    )
                run(
                    command,
                    gpu=name
                    not in {"adaptive_green_baseline", "adaptive_green_edge_v2"},
                )
            frames = output_frames(directory, args.frame_count)
            if len(frames) != args.frame_count:
                raise RuntimeError(f"provider {name} produced {len(frames)} frames")
            outputs[name] = directory / "rgba" if (directory / "rgba").is_dir() else directory
        evaluation_path = output_root / "evaluation.json"
        evaluation = evaluate(source_dir, outputs, evaluation_path)
        payload = manifest_payload(
            args.action,
            source_dir,
            outputs,
            evaluation,
            evaluation_path,
            args.frame_count,
        )

    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()

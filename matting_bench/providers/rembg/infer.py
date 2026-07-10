"""Run a pinned rembg ONNX model on a flat directory of PNG images."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from model_catalog import MODEL_SPECS
from runtime import (
    MODEL_DIR,
    configure_nvidia_dll_dirs,
    configure_runtime_dirs,
    file_md5,
)


configure_runtime_dirs()

import numpy as np
import onnxruntime as ort
from PIL import Image
from rembg import new_session, remove


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Defaults to <output-dir>/metrics.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace same-name PNGs already present in the output directory.",
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def configure_providers(device: str) -> tuple[list[str], float, list[str]]:
    preload_started = time.perf_counter()
    dll_dirs: list[str] = []
    if device == "cuda":
        dll_dirs = configure_nvidia_dll_dirs()
        if not hasattr(ort, "preload_dlls"):
            raise RuntimeError("onnxruntime does not expose preload_dlls")
        ort.preload_dlls(directory="")
    preload_seconds = time.perf_counter() - preload_started

    available = ort.get_available_providers()
    requested = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    if requested[0] not in available:
        raise RuntimeError(
            f"{requested[0]} is unavailable; installed providers are {available}"
        )
    return requested, preload_seconds, dll_dirs


def alpha_summary(image: Image.Image) -> dict[str, object]:
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    return {
        "min": int(alpha.min()),
        "max": int(alpha.max()),
        "zero_fraction": float(np.mean(alpha == 0)),
        "opaque_fraction": float(np.mean(alpha == 255)),
        "partial_fraction": float(np.mean((alpha > 0) & (alpha < 255))),
    }


def environment_details(session: object) -> dict[str, object]:
    inner_session = session.inner_session
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "onnxruntime_device": ort.get_device(),
        "available_providers": ort.get_available_providers(),
        "active_providers": inner_session.get_providers(),
        "provider_options": inner_session.get_provider_options(),
        "packages": {
            "rembg": package_version("rembg"),
            "onnxruntime-gpu": package_version("onnxruntime-gpu"),
            "numpy": package_version("numpy"),
            "pillow": package_version("pillow"),
            "pooch": package_version("pooch"),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    run_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    metrics_path = (
        args.metrics_json.resolve()
        if args.metrics_json
        else output_dir / "metrics.json"
    )

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    input_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not input_paths:
        raise ValueError(f"No PNG files found directly under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / path.name for path in input_paths]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} output PNG(s) already exist; pass --overwrite to replace them"
        )

    requested_providers, preload_seconds, nvidia_dll_dirs = configure_providers(
        args.device
    )
    session_started = time.perf_counter()
    session = new_session(args.model, providers=requested_providers)
    session_load_seconds = time.perf_counter() - session_started
    active_providers = session.inner_session.get_providers()
    if not active_providers or active_providers[0] != requested_providers[0]:
        raise RuntimeError(
            f"Requested {requested_providers[0]}, but session uses {active_providers}"
        )

    spec = MODEL_SPECS[args.model]
    model_path = (MODEL_DIR / spec.filename).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"rembg did not materialize the expected model: {model_path}")
    actual_md5 = file_md5(model_path)
    if actual_md5 != spec.expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {model_path}: {actual_md5} != {spec.expected_md5}"
        )

    image_metrics: list[dict[str, object]] = []
    batch_started = time.perf_counter()
    for input_path in input_paths:
        frame_started = time.perf_counter()
        with Image.open(input_path) as opened:
            source = opened.convert("RGB")
            source.load()
        load_seconds = time.perf_counter() - frame_started

        remove_started = time.perf_counter()
        result = remove(source, session=session)
        remove_seconds = time.perf_counter() - remove_started
        frame_active_providers = session.inner_session.get_providers()
        if (
            not frame_active_providers
            or frame_active_providers[0] != requested_providers[0]
        ):
            raise RuntimeError(
                f"Execution provider changed during inference: requested "
                f"{requested_providers[0]}, active {frame_active_providers}"
            )
        if not isinstance(result, Image.Image):
            raise TypeError(f"rembg returned {type(result)!r}, expected PIL.Image.Image")
        rgba = result.convert("RGBA")
        if rgba.size != source.size:
            raise RuntimeError(
                f"Output size changed for {input_path.name}: {source.size} -> {rgba.size}"
            )

        output_path = output_dir / input_path.name
        save_started = time.perf_counter()
        rgba.save(output_path, format="PNG")
        save_seconds = time.perf_counter() - save_started
        end_to_end_seconds = time.perf_counter() - frame_started

        with Image.open(output_path) as verification:
            verification.load()
            if verification.mode != "RGBA" or verification.size != source.size:
                raise RuntimeError(
                    f"Invalid saved output {output_path}: "
                    f"mode={verification.mode}, size={verification.size}"
                )
            alpha = alpha_summary(verification)

        image_metrics.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "width": source.width,
                "height": source.height,
                "mode": "RGBA",
                "load_seconds": load_seconds,
                "remove_seconds": remove_seconds,
                "save_seconds": save_seconds,
                "end_to_end_seconds": end_to_end_seconds,
                "alpha": alpha,
            }
        )

    batch_wall_seconds = time.perf_counter() - batch_started
    remove_times = [float(item["remove_seconds"]) for item in image_metrics]
    end_to_end_times = [float(item["end_to_end_seconds"]) for item in image_metrics]
    outputs = sorted(output_dir.glob("*.png"), key=lambda path: path.name.casefold())
    expected_names = [path.name for path in input_paths]
    produced_names = [path.name for path in outputs if path.name in expected_names]
    if produced_names != expected_names:
        raise RuntimeError(
            f"Output basename mismatch: expected {expected_names}, got {produced_names}"
        )

    result = {
        "schema_version": 1,
        "status": "ok",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "rembg",
        "model": {
            **spec.as_dict(),
            "local_path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "actual_md5": actual_md5,
        },
        "device": args.device,
        "requested_providers": requested_providers,
        "nvidia_dll_dirs": nvidia_dll_dirs,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(image_metrics),
        "rgba_output_count": len(image_metrics),
        "dll_preload_seconds": preload_seconds,
        "session_load_seconds": session_load_seconds,
        "batch_wall_seconds": batch_wall_seconds,
        "remove_total_seconds": sum(remove_times),
        "remove_mean_seconds": statistics.fmean(remove_times),
        "remove_median_seconds": statistics.median(remove_times),
        "remove_mean_excluding_first_seconds": (
            statistics.fmean(remove_times[1:])
            if len(remove_times) > 1
            else remove_times[0]
        ),
        "end_to_end_total_seconds": sum(end_to_end_times),
        "end_to_end_mean_seconds": statistics.fmean(end_to_end_times),
        "run_wall_seconds": time.perf_counter() - run_started,
        "images": image_metrics,
        "environment": environment_details(session),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "status": "error",
            "model": args.model,
            "device": args.device,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

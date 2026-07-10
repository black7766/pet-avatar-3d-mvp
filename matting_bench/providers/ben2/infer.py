from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "ben2"
MODEL_DIR = MODEL_ROOT / "model"
SOURCE_DIR = MODEL_ROOT / "source"
SOURCE_SRC = SOURCE_DIR / "src"

SOURCE_REPO = "https://github.com/PramaLLC/BEN2"
SOURCE_REVISION = "2c99a5da477b5523585bfa5c893888a6e818a8f6"
MODEL_REPO = "PramaLLC/BEN2"
MODEL_REVISION = "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f"
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "ea8b7907176a09667c86343dc7d00de6a6d871076cb90bb5f753618fd6fb3ebb"
MODEL_INPUT_SIZE = (1024, 1024)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# Keep runtime caches inside this provider's permitted model directory and make
# inference independent of network availability after installation.
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf-home"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf-home" / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TORCH_HOME", str(MODEL_ROOT / "torch-home"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "xdg-cache"))

if SOURCE_SRC.is_dir():
    sys.path.insert(0, str(SOURCE_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ben2 import BEN_Base  # noqa: E402
from ben2.modeling_ben2 import (  # noqa: E402
    img_transform,
    img_transform32,
    postprocess_image,
)
from PIL import Image, ImageOps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned official BEN2 Base model on an image directory."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        required=True,
        help="PyTorch device: cuda, cuda:N, cpu, or auto.",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Full pipeline warm-up runs on the first image (default: 1).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Metrics path (default: <output-dir>/metrics.json).",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mib(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip()


def image_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not paths:
        raise RuntimeError(f"No supported images found directly under: {input_dir}")

    output_names = [path.with_suffix(".png").name.lower() for path in paths]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError("Input filenames collide after conversion to PNG names.")
    return paths


def resolve_device(value: str) -> torch.device:
    requested = value.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid --device value: {value}") from error

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)."
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("BEN2 provider supports only CUDA and CPU devices.")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_install(model_dir: Path) -> dict[str, Any]:
    source_head = git_revision(SOURCE_DIR)
    if source_head != SOURCE_REVISION:
        raise RuntimeError(
            f"Official source revision mismatch: expected {SOURCE_REVISION}, got {source_head}"
        )

    weight_path = model_dir / MODEL_FILENAME
    if not weight_path.is_file():
        raise FileNotFoundError(f"Pinned Base weight is missing: {weight_path}")

    started = time.perf_counter()
    actual_hash = sha256_file(weight_path)
    hash_seconds = time.perf_counter() - started
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(
            f"Base weight SHA-256 mismatch: expected {MODEL_SHA256}, got {actual_hash}"
        )
    return {
        "source_revision_actual": source_head,
        "weight_path": str(weight_path),
        "weight_bytes": weight_path.stat().st_size,
        "weight_mib": mib(weight_path.stat().st_size),
        "weight_sha256": actual_hash,
        "weight_hash_seconds": round(hash_seconds, 6),
    }


def cuda_snapshot(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "total_memory_mib": mib(properties.total_memory),
        "allocated_mib": mib(torch.cuda.memory_allocated(device)),
        "reserved_mib": mib(torch.cuda.memory_reserved(device)),
    }


def run_frame(
    model: BEN_Base,
    input_path: Path,
    output_path: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    frame_started = time.perf_counter()
    load_started = frame_started
    with Image.open(input_path) as opened:
        source_image = ImageOps.exif_transpose(opened).convert("RGB")
    loaded_at = time.perf_counter()

    width, height = source_image.size
    resized = source_image.resize(MODEL_INPUT_SIZE, resample=Image.Resampling.LANCZOS)
    transform = img_transform if device.type == "cuda" else img_transform32
    input_tensor = transform(resized).unsqueeze(0).to(device)
    synchronize(device)
    preprocessed_at = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(input_tensor)
    synchronize(device)
    inferred_at = time.perf_counter()
    forward_peak_allocated = (
        mib(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    alpha_array = postprocess_image(prediction, im_size=[height, width])
    alpha = Image.fromarray(alpha_array)
    rgba = source_image.copy()
    rgba.putalpha(alpha)

    output_hash = None
    output_mode = rgba.mode
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(output_path, format="PNG")
        with Image.open(output_path) as checked:
            checked.load()
            if checked.mode != "RGBA":
                raise RuntimeError(f"Output is not 8-bit RGBA: {output_path} ({checked.mode})")
            if checked.size != source_image.size:
                raise RuntimeError(
                    f"Output size mismatch for {output_path}: {checked.size} != {source_image.size}"
                )
        output_hash = sha256_file(output_path)
    synchronize(device)
    completed_at = time.perf_counter()

    alpha_float = alpha_array.astype(np.float32) / 255.0
    record: dict[str, Any] = {
        "input": input_path.name,
        "output": output_path.name if output_path is not None else None,
        "width": width,
        "height": height,
        "output_mode": output_mode,
        "load_ms": round((loaded_at - load_started) * 1000, 3),
        "preprocess_ms": round((preprocessed_at - loaded_at) * 1000, 3),
        "inference_ms": round((inferred_at - inference_started) * 1000, 3),
        "postprocess_validate_save_ms": round((completed_at - inferred_at) * 1000, 3),
        "total_ms": round((completed_at - frame_started) * 1000, 3),
        "alpha_min": round(float(alpha_float.min()), 6),
        "alpha_max": round(float(alpha_float.max()), 6),
        "alpha_mean": round(float(alpha_float.mean()), 6),
        "transparent_fraction": round(float(np.mean(alpha_float <= 0.05)), 6),
        "opaque_fraction": round(float(np.mean(alpha_float >= 0.95)), 6),
        "soft_alpha_fraction": round(
            float(np.mean((alpha_float > 0.05) & (alpha_float < 0.95))), 6
        ),
        "sha256": output_hash,
        "cuda_forward_peak_allocated_mib": forward_peak_allocated,
    }
    if device.type == "cuda":
        record["cuda_pipeline_peak_allocated_mib"] = mib(
            torch.cuda.max_memory_allocated(device)
        )
        record["cuda_pipeline_peak_reserved_mib"] = mib(
            torch.cuda.max_memory_reserved(device)
        )

    del prediction, input_tensor, resized, alpha, rgba
    return record


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def build_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "ben2": package_version("ben2"),
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "timm": package_version("timm"),
            "einops": package_version("einops"),
            "huggingface-hub": package_version("huggingface-hub"),
            "safetensors": package_version("safetensors"),
            "opencv-python-headless": package_version("opencv-python-headless"),
            "numpy": np.__version__,
            "pillow": package_version("pillow"),
        },
    }
    if device.type == "cuda":
        environment["cuda"] = {
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            **(cuda_snapshot(device) or {}),
        }
    return environment


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")

    paths = image_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    install = validate_install(model_dir)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    model = BEN_Base.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device).eval()
    synchronize(device)
    model_load_ms = (time.perf_counter() - load_started) * 1000
    model_load_peak_allocated_mib = (
        mib(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_memory_after_load = cuda_snapshot(device)

    warmup_records: list[dict[str, Any]] = []
    for _ in range(args.warmup_runs):
        warmup_records.append(run_frame(model, paths[0], None, device))

    records: list[dict[str, Any]] = []
    measured_started = time.perf_counter()
    for index, input_path in enumerate(paths, start=1):
        output_path = output_dir / input_path.with_suffix(".png").name
        record = run_frame(model, input_path, output_path, device)
        records.append(record)
        print(
            f"[{index}/{len(paths)}] {input_path.name}: "
            f"{record['inference_ms']:.3f} ms",
            file=sys.stderr,
        )
    measured_wall_ms = (time.perf_counter() - measured_started) * 1000

    inference_values = [float(record["inference_ms"]) for record in records]
    total_values = [float(record["total_ms"]) for record in records]
    warm_values = inference_values[1:] if len(inference_values) > 1 else inference_values

    summary: dict[str, Any] = {
        "image_count": len(records),
        "validated_rgba_outputs": len(records),
        "model_load_ms": round(model_load_ms, 3),
        "warmup_runs": args.warmup_runs,
        "warmup_total_ms": round(
            sum(float(record["total_ms"]) for record in warmup_records), 3
        ),
        "measured_wall_ms": round(measured_wall_ms, 3),
        "inference_total_ms": round(sum(inference_values), 3),
        "inference_mean_ms": round(statistics.fmean(inference_values), 3),
        "inference_mean_excluding_first_ms": round(statistics.fmean(warm_values), 3),
        "inference_median_ms": round(statistics.median(inference_values), 3),
        "inference_p95_ms": round(percentile(inference_values, 95), 3),
        "mean_total_ms": round(statistics.fmean(total_values), 3),
        "steady_inference_fps": round(1000.0 / statistics.fmean(warm_values), 3),
    }
    if device.type == "cuda":
        summary.update(
            {
                "model_load_peak_allocated_mib": model_load_peak_allocated_mib,
                "max_cuda_pipeline_peak_allocated_mib": max(
                    float(record["cuda_pipeline_peak_allocated_mib"])
                    for record in records
                ),
                "max_cuda_pipeline_peak_reserved_mib": max(
                    float(record["cuda_pipeline_peak_reserved_mib"])
                    for record in records
                ),
                "cuda_peak_measurement": (
                    "PyTorch allocator peak per measured frame; includes resident model, "
                    "input, forward, and official postprocess allocations"
                ),
            }
        )

    return {
        "schema_version": 1,
        "status": "ok",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "provider": "ben2",
        "model": {
            "name": "BEN2 Base",
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_dir": str(model_dir),
            "parameter_count": model_parameter_count,
            "input_size": list(MODEL_INPUT_SIZE),
            "weight_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "official_forward_autocast": "cuda/float16",
            "refine_foreground": False,
            "variant_scope": "locally self-hosted open-source Base only; no Full/API model",
            "license": "MIT",
            "license_evidence": [
                "GitHub LICENSE at source revision",
                "Hugging Face model-card metadata at model revision",
            ],
            **install,
        },
        "runtime": {
            "device_requested": args.device,
            "device": str(device),
            "model_memory_after_load": model_memory_after_load,
            "environment": build_environment(device),
        },
        "paths": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
        "summary": summary,
        "images": records,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    metrics_path = (
        args.metrics_json.resolve()
        if args.metrics_json is not None
        else output_dir / "metrics.json"
    )
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "error",
            "completed_at_utc": utc_now(),
            "provider": "ben2",
            "device_requested": args.device,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(metrics_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    write_json(metrics_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

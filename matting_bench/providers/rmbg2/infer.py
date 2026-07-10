#!/usr/bin/env python3
"""Local BRIA RMBG-2.0 directory inference.

The implementation follows the preprocessing and alpha-matte composition shown in
the official BRIA model card. The gated model snapshot is pinned and stored under
the provider-specific model directory.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "rmbg2"
MODEL_DIR = MODEL_ROOT / "model"
HF_HOME = MODEL_ROOT / "hf_home"

MODEL_REPO = "briaai/RMBG-2.0"
MODEL_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
MODEL_FILES = (
    "BiRefNet_config.py",
    "birefnet.py",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def configure_local_caches() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    cache_values = {
        "HF_HOME": HF_HOME,
        "HF_HUB_CACHE": HF_HOME / "hub",
        "HF_MODULES_CACHE": HF_HOME / "modules",
        "TRANSFORMERS_CACHE": HF_HOME / "transformers",
        "TORCH_HOME": MODEL_ROOT / "torch_home",
        "TORCHINDUCTOR_CACHE_DIR": MODEL_ROOT / "torchinductor",
    }
    for name, path in cache_values.items():
        os.environ.setdefault(name, str(path))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


configure_local_caches()

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from transformers import AutoModelForImageSegmentation  # noqa: E402


TRANSFORM_IMAGE = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official BRIA RMBG-2.0 model over an image directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="PyTorch device, for example cuda, cuda:0, cpu, or auto.",
    )
    return parser.parse_args()


def required_model_files_present() -> bool:
    return all((MODEL_DIR / name).is_file() for name in MODEL_FILES)


def ensure_model_snapshot() -> None:
    if required_model_files_present():
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=MODEL_DIR,
            allow_patterns=list(MODEL_FILES),
            token=token,
        )
    except (GatedRepoError, HfHubHTTPError) as exc:
        raise RuntimeError(
            "Official briaai/RMBG-2.0 weights are gated. Accept the model's "
            "non-commercial license on Hugging Face, then set HF_TOKEN to a "
            "read token and rerun this command."
        ) from exc

    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete model snapshot; missing: {', '.join(missing)}")


def resolve_device(value: str) -> torch.device:
    requested = "cuda" if value.lower() == "auto" and torch.cuda.is_available() else value
    if value.lower() == "auto" and not torch.cuda.is_available():
        requested = "cpu"

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device value: {value}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)."
            )
        torch.cuda.set_device(device)
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mib(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def image_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not files:
        raise RuntimeError(f"No supported images found in: {input_dir}")

    output_names = [path.with_suffix(".png").name.lower() for path in files]
    if len(output_names) != len(set(output_names)):
        raise RuntimeError("Input filenames collide after conversion to .png.")
    return files


def cuda_memory(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_mib": mib(properties.total_memory),
        "allocated_mib": mib(torch.cuda.memory_allocated(device)),
        "reserved_mib": mib(torch.cuda.memory_reserved(device)),
    }


def load_model(device: torch.device) -> tuple[Any, float]:
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    model = AutoModelForImageSegmentation.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval().to(device)
    synchronize(device)
    return model, (time.perf_counter() - started) * 1000


def run_image(
    model: Any,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    with Image.open(input_path) as opened:
        image = opened.convert("RGB")

    preprocess_started = time.perf_counter()
    input_tensor = TRANSFORM_IMAGE(image).unsqueeze(0).to(device)
    synchronize(device)
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(input_tensor)[-1].sigmoid()
    synchronize(device)
    inference_ms = (time.perf_counter() - inference_started) * 1000

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = mib(torch.cuda.max_memory_allocated(device))
        peak_reserved = mib(torch.cuda.max_memory_reserved(device))

    postprocess_started = time.perf_counter()
    matte = transforms.ToPILImage()(prediction[0].squeeze().float().cpu())
    matte = matte.resize(image.size)
    rgba = image.copy()
    rgba.putalpha(matte)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output_path, format="PNG")
    postprocess_ms = (time.perf_counter() - postprocess_started) * 1000

    return {
        "input": input_path.name,
        "output": output_path.name,
        "width": image.width,
        "height": image.height,
        "output_mode": rgba.mode,
        "preprocess_ms": round(preprocess_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "postprocess_save_ms": round(postprocess_ms, 2),
        "total_ms": round((time.perf_counter() - total_started) * 1000, 2),
        "cuda_peak_allocated_mib": peak_allocated,
        "cuda_peak_reserved_mib": peak_reserved,
    }


def summarize(
    args: argparse.Namespace,
    device: torch.device,
    model_load_ms: float,
    memory_after_load: dict[str, Any] | None,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    inference_times = [item["inference_ms"] for item in results]
    warm_times = inference_times[1:] if len(inference_times) > 1 else inference_times
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "local_dir": str(MODEL_DIR),
            "input_size": [1024, 1024],
            "dtype": "float32",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device": str(device),
            "model_load_ms": round(model_load_ms, 2),
            "cuda_after_model_load": memory_after_load,
        },
        "paths": {
            "input_dir": str(args.input_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        },
        "summary": {
            "image_count": len(results),
            "mean_inference_ms_all": round(statistics.fmean(inference_times), 2),
            "mean_inference_ms_excluding_first": round(statistics.fmean(warm_times), 2),
            "median_inference_ms_all": round(statistics.median(inference_times), 2),
            "max_cuda_peak_allocated_mib": max(
                (item["cuda_peak_allocated_mib"] or 0 for item in results), default=0
            ),
            "max_cuda_peak_reserved_mib": max(
                (item["cuda_peak_reserved_mib"] or 0 for item in results), default=0
            ),
        },
        "images": results,
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("--input-dir and --output-dir must be different directories.")

    files = image_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_model_snapshot()
    device = resolve_device(args.device)
    model, model_load_ms = load_model(device)
    memory_after_load = cuda_memory(device)

    results = []
    for input_path in files:
        output_path = output_dir / input_path.with_suffix(".png").name
        results.append(run_image(model, input_path, output_path, device))

    print(
        json.dumps(
            summarize(args, device, model_load_ms, memory_after_load, results),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

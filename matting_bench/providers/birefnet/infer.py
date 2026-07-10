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


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_REPO = "ZhengPeng7/BiRefNet"
MODEL_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
DEFAULT_MODEL_DIR = REPO_ROOT / ".models" / "birefnet" / "ZhengPeng7--BiRefNet"
MODEL_ROOT = REPO_ROOT / ".models" / "birefnet"
INPUT_SIZE = (1024, 1024)

# Keep all runtime caches inside the provider's permitted model directory.
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf-cache" / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(MODEL_ROOT / "hf-cache" / "modules"))
os.environ.setdefault("HF_TOKEN_PATH", str(MODEL_ROOT / "hf-cache" / "token"))
os.environ.setdefault("TORCH_HOME", str(MODEL_ROOT / "torch-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "xdg-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import cv2
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from transformers import AutoModelForImageSegmentation


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned official BiRefNet model on a directory of PNG files."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda", "cpu"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional path for machine-readable timing and memory metrics.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def preprocess(image: Image.Image, dtype: torch.dtype) -> torch.Tensor:
    resized = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0).to(dtype=dtype)


def alpha_from_prediction(
    prediction: torch.Tensor, output_size: tuple[int, int]
) -> Image.Image:
    resized = torch_functional.interpolate(
        prediction,
        size=(output_size[1], output_size[0]),
        mode="bilinear",
        align_corners=True,
    )
    alpha = (
        resized[0, 0]
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(alpha, mode="L")


def foreground_estimate(
    image: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    blurred_alpha = cv2.blur(alpha, (radius, radius))[:, :, None]
    blurred_foreground_alpha = cv2.blur(foreground * alpha, (radius, radius))
    blurred_foreground = blurred_foreground_alpha / (blurred_alpha + 1e-5)
    blurred_background_alpha = cv2.blur(background * (1 - alpha), (radius, radius))
    blurred_background = blurred_background_alpha / ((1 - blurred_alpha) + 1e-5)
    foreground = blurred_foreground + alpha * (
        image - alpha * blurred_foreground - (1 - alpha) * blurred_background
    )
    return np.clip(foreground, 0, 1), blurred_background


def refine_foreground(image: Image.Image, alpha: Image.Image) -> Image.Image:
    """Mirror the foreground refinement used by the official HF handler."""
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float64) / 255.0
    alpha_array = (np.asarray(alpha, dtype=np.float64) / 255.0)[:, :, None]
    foreground, blurred_background = foreground_estimate(
        image_array, image_array, image_array, alpha_array, radius=90
    )
    foreground, _ = foreground_estimate(
        image_array, foreground, blurred_background, alpha_array, radius=6
    )
    return Image.fromarray((foreground * 255.0).astype(np.uint8), mode="RGB")


def build_environment(device: torch.device) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "packages": {
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "transformers": package_version("transformers"),
            "timm": package_version("timm"),
            "kornia": package_version("kornia"),
            "numpy": np.__version__,
            "pillow": package_version("pillow"),
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment["cuda"] = {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return environment


def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = utc_now()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model snapshot does not exist: {model_dir}. Run download_model.py first."
        )

    input_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not input_paths:
        raise ValueError(f"No PNG files found directly under: {input_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    model = AutoModelForImageSegmentation.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    image_metrics: list[dict[str, object]] = []
    batch_started = time.perf_counter()
    for input_path in input_paths:
        end_to_end_started = time.perf_counter()
        with Image.open(input_path) as opened:
            image = opened.convert("RGB")

        input_tensor = preprocess(image, dtype).to(device, non_blocking=False)
        synchronize(device)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(input_tensor)[-1].sigmoid().to(torch.float32)
        synchronize(device)
        inference_seconds = time.perf_counter() - inference_started

        alpha = alpha_from_prediction(prediction, image.size)
        rgba = refine_foreground(image, alpha)
        rgba.putalpha(alpha)
        output_path = output_dir / input_path.name
        rgba.save(output_path, format="PNG")
        end_to_end_seconds = time.perf_counter() - end_to_end_started

        image_metrics.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "width": image.width,
                "height": image.height,
                "inference_seconds": inference_seconds,
                "end_to_end_seconds": end_to_end_seconds,
            }
        )

    batch_seconds = time.perf_counter() - batch_started
    inference_times = [float(item["inference_seconds"]) for item in image_metrics]
    end_to_end_times = [float(item["end_to_end_seconds"]) for item in image_metrics]

    result: dict[str, object] = {
        "status": "ok",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "model": {
            "repo_id": args.model_repo,
            "revision": args.model_revision,
            "local_dir": str(model_dir),
            "input_size": list(INPUT_SIZE),
            "dtype": str(dtype).removeprefix("torch."),
            "foreground_refinement": "official HF handler two-pass blur fusion (r=90, r=6)",
        },
        "device": args.device,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(image_metrics),
        "model_load_seconds": model_load_seconds,
        "batch_wall_seconds": batch_seconds,
        "inference_total_seconds": sum(inference_times),
        "inference_mean_seconds": statistics.fmean(inference_times),
        "inference_median_seconds": statistics.median(inference_times),
        "inference_mean_excluding_first_seconds": (
            statistics.fmean(inference_times[1:])
            if len(inference_times) > 1
            else inference_times[0]
        ),
        "end_to_end_total_seconds": sum(end_to_end_times),
        "images": image_metrics,
        "environment": build_environment(device),
    }
    if device.type == "cuda":
        result["cuda_peak_memory"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
            "measurement": "torch allocator peak from before model load through final output",
        }
    return result


def write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "status": "error",
            "completed_at_utc": utc_now(),
            "device": args.device,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if args.metrics_json:
            write_json(args.metrics_json, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1

    if args.metrics_json:
        write_json(args.metrics_json, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

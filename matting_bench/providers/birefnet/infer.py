from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "birefnet"
SUPPORTED_INPUT_RESOLUTIONS = (512, 768, 1024)
FOREGROUND_REFINEMENT_MODES = (
    "official-auto",
    "official-cpu",
    "official-gpu",
    "none",
)
OFFICIAL_FOREGROUND_RADIUS = 90
OFFICIAL_SECOND_PASS_RADIUS = 6

# Keep runtime caches in the provider's existing model area.
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf-cache" / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(MODEL_ROOT / "hf-cache" / "modules"))
os.environ.setdefault("HF_TOKEN_PATH", str(MODEL_ROOT / "hf-cache" / "token"))
os.environ.setdefault("TORCH_HOME", str(MODEL_ROOT / "torch-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "xdg-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from transformers import AutoModelForImageSegmentation


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    model_dir: Path
    variant: str


GENERAL_MODEL = ModelSpec(
    repo_id="ZhengPeng7/BiRefNet",
    revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
    model_dir=MODEL_ROOT / "ZhengPeng7--BiRefNet",
    variant="general",
)


def parse_args(model_spec: ModelSpec = GENERAL_MODEL) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run the pinned official {model_spec.repo_id} checkpoint."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda", "cpu"))
    parser.add_argument(
        "--input-resolution",
        type=int,
        choices=SUPPORTED_INPUT_RESOLUTIONS,
        default=1024,
        help="Square model input resolution. Official standard-checkpoint default: 1024.",
    )
    parser.add_argument(
        "--foreground-refinement",
        choices=FOREGROUND_REFINEMENT_MODES,
        default="official-auto",
        help=(
            "Official two-pass foreground estimator on CPU/GPU, automatic selection, "
            "or none for an alpha-only ablation."
        ),
    )
    parser.add_argument("--model-dir", type=Path, default=model_spec.model_dir)
    parser.add_argument("--model-repo", default=model_spec.repo_id)
    parser.add_argument("--model-revision", default=model_spec.revision)
    parser.add_argument("--model-variant", default=model_spec.variant)
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


def preprocess(
    image: Image.Image, input_resolution: int, dtype: torch.dtype
) -> torch.Tensor:
    resized = image.resize(
        (input_resolution, input_resolution), Image.Resampling.BILINEAR
    )
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


def foreground_estimate_cpu(
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


def refine_foreground_cpu(image: Image.Image, alpha: Image.Image) -> Image.Image:
    """Pinned HF handler's two-pass OpenCV foreground estimator."""
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float64) / 255.0
    alpha_array = (np.asarray(alpha, dtype=np.float64) / 255.0)[:, :, None]
    foreground, blurred_background = foreground_estimate_cpu(
        image_array,
        image_array,
        image_array,
        alpha_array,
        radius=OFFICIAL_FOREGROUND_RADIUS,
    )
    foreground, _ = foreground_estimate_cpu(
        image_array,
        foreground,
        blurred_background,
        alpha_array,
        radius=OFFICIAL_SECOND_PASS_RADIUS,
    )
    return Image.fromarray((foreground * 255.0).astype(np.uint8), mode="RGB")


def mean_blur(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Match the current official Demo's torch implementation of cv2.blur."""
    if kernel_size % 2 == 0:
        pad_left = kernel_size // 2 - 1
        pad_right = kernel_size // 2
        pad_top = kernel_size // 2 - 1
        pad_bottom = kernel_size // 2
    else:
        pad_left = pad_right = pad_top = pad_bottom = kernel_size // 2
    padded = torch_functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="replicate",
    )
    return torch_functional.avg_pool2d(
        padded,
        kernel_size=(kernel_size, kernel_size),
        stride=1,
        count_include_pad=False,
    )


def foreground_estimate_gpu(
    image: torch.Tensor,
    foreground: torch.Tensor,
    background: torch.Tensor,
    alpha: torch.Tensor,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_dtype = image.dtype
    image = image.float()
    foreground = foreground.float()
    background = background.float()
    alpha = alpha.float()
    blurred_alpha = mean_blur(alpha, radius)
    blurred_foreground = mean_blur(foreground * alpha, radius) / (
        blurred_alpha + 1e-5
    )
    blurred_background = mean_blur(background * (1 - alpha), radius) / (
        (1 - blurred_alpha) + 1e-5
    )
    foreground_output = blurred_foreground + alpha * (
        image - alpha * blurred_foreground - (1 - alpha) * blurred_background
    )
    return (
        foreground_output.clamp(0, 1).to(input_dtype),
        blurred_background.to(input_dtype),
    )


def refine_foreground_gpu(
    image: Image.Image, alpha: Image.Image, device: torch.device
) -> Image.Image:
    """Current official Demo's GPU two-pass foreground estimator."""
    if device.type != "cuda":
        raise ValueError("official-gpu foreground refinement requires --device cuda")
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device)
    )
    alpha_tensor = (
        torch.from_numpy(alpha_array)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device)
    )
    foreground, blurred_background = foreground_estimate_gpu(
        image_tensor,
        image_tensor,
        image_tensor,
        alpha_tensor,
        radius=OFFICIAL_FOREGROUND_RADIUS,
    )
    foreground, _ = foreground_estimate_gpu(
        image_tensor,
        foreground,
        blurred_background,
        alpha_tensor,
        radius=OFFICIAL_SECOND_PASS_RADIUS,
    )
    packed = (
        foreground[0]
        .mul(255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return Image.fromarray(packed, mode="RGB")


def resolve_refinement_mode(requested: str, device: torch.device) -> str:
    if requested == "official-auto":
        return "official-gpu" if device.type == "cuda" else "official-cpu"
    if requested == "official-gpu" and device.type != "cuda":
        raise ValueError("official-gpu foreground refinement requires --device cuda")
    return requested


def refine_foreground(
    image: Image.Image,
    alpha: Image.Image,
    mode: str,
    device: torch.device,
) -> Image.Image:
    if mode == "none":
        return image.copy()
    if mode == "official-cpu":
        return refine_foreground_cpu(image, alpha)
    if mode == "official-gpu":
        return refine_foreground_gpu(image, alpha, device)
    raise ValueError(f"Unsupported foreground refinement mode: {mode}")


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
            "opencv": cv2.__version__,
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
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not input_paths:
        raise ValueError(f"No PNG files found directly under: {input_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    refinement_mode = resolve_refinement_mode(args.foreground_refinement, device)
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

        preprocess_started = time.perf_counter()
        input_tensor = preprocess(image, args.input_resolution, dtype).to(
            device, non_blocking=False
        )
        synchronize(device)
        preprocess_seconds = time.perf_counter() - preprocess_started

        inference_started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(input_tensor)[-1].sigmoid().to(torch.float32)
        synchronize(device)
        inference_seconds = time.perf_counter() - inference_started

        alpha = alpha_from_prediction(prediction, image.size)
        synchronize(device)
        refinement_started = time.perf_counter()
        with torch.inference_mode():
            foreground = refine_foreground(image, alpha, refinement_mode, device)
        synchronize(device)
        refinement_seconds = time.perf_counter() - refinement_started
        foreground.putalpha(alpha)
        output_path = output_dir / input_path.name
        foreground.save(output_path, format="PNG")
        end_to_end_seconds = time.perf_counter() - end_to_end_started

        image_metrics.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "width": image.width,
                "height": image.height,
                "preprocess_seconds": preprocess_seconds,
                "inference_seconds": inference_seconds,
                "foreground_refinement_seconds": refinement_seconds,
                "end_to_end_seconds": end_to_end_seconds,
            }
        )

    batch_seconds = time.perf_counter() - batch_started
    inference_times = [float(item["inference_seconds"]) for item in image_metrics]
    refinement_times = [
        float(item["foreground_refinement_seconds"]) for item in image_metrics
    ]
    end_to_end_times = [float(item["end_to_end_seconds"]) for item in image_metrics]

    result: dict[str, object] = {
        "status": "ok",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "model": {
            "repo_id": args.model_repo,
            "revision": args.model_revision,
            "variant": args.model_variant,
            "local_dir": str(model_dir),
            "input_size": [args.input_resolution, args.input_resolution],
            "dtype": str(dtype).removeprefix("torch."),
            "foreground_refinement_requested": args.foreground_refinement,
            "foreground_refinement_resolved": refinement_mode,
            "foreground_refinement_radii": (
                [OFFICIAL_FOREGROUND_RADIUS, OFFICIAL_SECOND_PASS_RADIUS]
                if refinement_mode != "none"
                else None
            ),
        },
        "parameters": {
            "input_resolution": args.input_resolution,
            "foreground_refinement": args.foreground_refinement,
            "foreground_refinement_resolved": refinement_mode,
            "device": args.device,
            "dtype": str(dtype).removeprefix("torch."),
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
        "foreground_refinement_mean_seconds": statistics.fmean(refinement_times),
        "end_to_end_total_seconds": sum(end_to_end_times),
        "end_to_end_mean_seconds": statistics.fmean(end_to_end_times),
        "end_to_end_mean_excluding_first_seconds": (
            statistics.fmean(end_to_end_times[1:])
            if len(end_to_end_times) > 1
            else end_to_end_times[0]
        ),
        "images": image_metrics,
        "environment": build_environment(device),
    }
    if device.type == "cuda":
        result["cuda_peak_memory"] = {
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
            "measurement": "torch allocator peak from model load through final output",
        }
    return result


def write_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(model_spec: ModelSpec = GENERAL_MODEL) -> int:
    args = parse_args(model_spec)
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "status": "error",
            "completed_at_utc": utc_now(),
            "device": args.device,
            "parameters": {
                "input_resolution": args.input_resolution,
                "foreground_refinement": args.foreground_refinement,
            },
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

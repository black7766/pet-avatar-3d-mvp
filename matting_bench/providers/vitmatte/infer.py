from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


sys.dont_write_bytecode = True


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "vitmatte"
DEFAULT_MODEL_DIR = MODEL_ROOT / "hustvl--vitmatte-small-composition-1k"
MODEL_REPO = "hustvl/vitmatte-small-composition-1k"
MODEL_REVISION = "6a58ad7646403c1df626fbd746900aec7361ea1d"
WEIGHT_SHA256 = "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
DEFAULT_BG_THRESHOLD = 0.02
DEFAULT_FG_THRESHOLD = 0.98
DEFAULT_UNKNOWN_RADIUS = 2
DEFAULT_FUSION_WEIGHT = 0.35
DEFAULT_FUSION_MAX_DELTA = 0.25

# Inference is deliberately offline. download_model.py is the only networked step.
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf-cache" / "hub"))
os.environ.setdefault("HF_TOKEN_PATH", str(MODEL_ROOT / "hf-cache" / "token"))
os.environ.setdefault("TORCH_HOME", str(MODEL_ROOT / "torch-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "xdg-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VitMatteForImageMatting, VitMatteImageProcessor


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc import (  # noqa: E402
    adaptive_green_matte_frame,
    profile_green_screen,
    refine_reframed_halo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine the repository's adaptive green-screen alpha with the pinned "
            "official hustvl ViTMatte checkpoint and emit same-name RGBA PNGs."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda", "cpu"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--background-threshold", type=float, default=DEFAULT_BG_THRESHOLD
    )
    parser.add_argument(
        "--foreground-threshold", type=float, default=DEFAULT_FG_THRESHOLD
    )
    parser.add_argument(
        "--unknown-radius",
        type=int,
        default=DEFAULT_UNKNOWN_RADIUS,
        help="Inward foreground-erosion radius in source pixels.",
    )
    parser.add_argument(
        "--fusion-weight",
        type=float,
        default=DEFAULT_FUSION_WEIGHT,
        help=(
            "ViTMatte contribution in the unknown region: 0 keeps adaptive alpha, "
            "1 uses the model prediction."
        ),
    )
    parser.add_argument(
        "--fusion-max-delta",
        type=float,
        default=DEFAULT_FUSION_MAX_DELTA,
        help=(
            "Maximum absolute per-pixel alpha correction before fusion weighting; "
            "1 disables practical clipping."
        ),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="Optional directory for baseline alpha, trimap, model alpha, and final alpha PNGs.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, list[Path]]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = args.model_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model snapshot does not exist: {model_dir}. Run download_model.py first."
        )
    weight_path = model_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Pinned safetensors weight is missing: {weight_path}")
    if not 0.0 <= args.background_threshold < args.foreground_threshold <= 1.0:
        raise ValueError(
            "Expected 0 <= background-threshold < foreground-threshold <= 1"
        )
    if not 1 <= args.unknown_radius <= 64:
        raise ValueError("unknown-radius must be between 1 and 64 source pixels")
    if not 0.0 <= args.fusion_weight <= 1.0:
        raise ValueError("fusion-weight must be between 0 and 1")
    if not 0.0 < args.fusion_max_delta <= 1.0:
        raise ValueError("fusion-max-delta must be greater than 0 and at most 1")
    input_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not input_paths:
        raise ValueError(f"No PNG files found directly under: {input_dir}")
    return input_dir, output_dir, model_dir, input_paths


def build_trimap(
    adaptive_alpha: np.ndarray,
    background_threshold: float,
    foreground_threshold: float,
    unknown_radius: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    support = adaptive_alpha > background_threshold
    core = adaptive_alpha >= foreground_threshold
    diameter = unknown_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    contracted_core = cv2.erode(core.astype(np.uint8), kernel) > 0

    trimap = np.full(adaptive_alpha.shape, 128, dtype=np.uint8)
    # The chroma key's hard background is reliable and must not be expanded into.
    trimap[~support] = 0
    trimap[contracted_core] = 255
    pixel_count = trimap.size
    stats: dict[str, float | int] = {
        "width": int(trimap.shape[1]),
        "height": int(trimap.shape[0]),
        "known_background_pct": float((trimap == 0).sum() * 100.0 / pixel_count),
        "unknown_pct": float((trimap == 128).sum() * 100.0 / pixel_count),
        "known_foreground_pct": float((trimap == 255).sum() * 100.0 / pixel_count),
    }
    return trimap, stats


def clamp_to_trimap(model_alpha: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    alpha = np.clip(model_alpha, 0.0, 1.0).astype(np.float32, copy=True)
    alpha[trimap == 0] = 0.0
    alpha[trimap == 255] = 1.0
    return alpha


def fuse_unknown_alpha(
    adaptive_alpha: np.ndarray,
    model_alpha: np.ndarray,
    trimap: np.ndarray,
    fusion_weight: float,
    fusion_max_delta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    unknown = trimap == 128
    raw_delta = model_alpha - adaptive_alpha
    limited_delta = np.clip(raw_delta, -fusion_max_delta, fusion_max_delta)
    fused = adaptive_alpha.astype(np.float32, copy=True)
    fused[unknown] += fusion_weight * limited_delta[unknown]
    fused[trimap == 0] = 0.0
    fused[trimap == 255] = 1.0
    np.clip(fused, 0.0, 1.0, out=fused)

    if unknown.any():
        stats = {
            "mean_abs_model_delta_unknown": float(np.abs(raw_delta[unknown]).mean()),
            "mean_abs_fused_delta_unknown": float(
                np.abs(fused[unknown] - adaptive_alpha[unknown]).mean()
            ),
            "clipped_delta_pct_unknown": float(
                (np.abs(raw_delta[unknown]) > fusion_max_delta).mean() * 100.0
            ),
        }
    else:
        stats = {
            "mean_abs_model_delta_unknown": 0.0,
            "mean_abs_fused_delta_unknown": 0.0,
            "clipped_delta_pct_unknown": 0.0,
        }
    return fused, stats


def save_gray(path: Path, values: np.ndarray) -> None:
    if values.dtype != np.uint8:
        values = (np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(values, mode="L").save(path, format="PNG")


def build_environment(device: torch.device) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "transformers": package_version("transformers"),
            "huggingface-hub": package_version("huggingface-hub"),
            "safetensors": package_version("safetensors"),
            "numpy": np.__version__,
            "pillow": package_version("pillow"),
            "opencv-python-headless": package_version("opencv-python-headless"),
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        environment["cuda"] = {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": properties.name,
            "total_memory_bytes": total_bytes,
            "free_memory_before_run_bytes": free_bytes,
            "capability": list(torch.cuda.get_device_capability(device)),
        }
    return environment


def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = utc_now()
    input_dir, output_dir, model_dir, input_paths = validate_args(args)
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
    diagnostics_dir = args.diagnostics_dir.resolve() if args.diagnostics_dir else None
    if diagnostics_dir:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    environment = build_environment(device)
    load_started = time.perf_counter()
    processor = VitMatteImageProcessor.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model = VitMatteForImageMatting.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    profile_started = time.perf_counter()
    profile = profile_green_screen(input_paths)
    profile_seconds = time.perf_counter() - profile_started

    image_metrics: list[dict[str, object]] = []
    batch_started = time.perf_counter()
    for input_path in input_paths:
        frame_started = time.perf_counter()
        with Image.open(input_path) as opened:
            image = opened.convert("RGB")
        adaptive_started = time.perf_counter()
        adaptive_rgba = np.asarray(
            adaptive_green_matte_frame(image, profile), dtype=np.float32
        ) / 255.0
        adaptive_alpha = adaptive_rgba[:, :, 3]
        adaptive_seconds = time.perf_counter() - adaptive_started

        trimap_started = time.perf_counter()
        trimap, trimap_stats = build_trimap(
            adaptive_alpha,
            args.background_threshold,
            args.foreground_threshold,
            args.unknown_radius,
        )
        trimap_seconds = time.perf_counter() - trimap_started

        preprocess_started = time.perf_counter()
        inputs = processor(
            images=image,
            trimaps=Image.fromarray(trimap, mode="L"),
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(
            device=device, dtype=dtype, non_blocking=False
        )
        model_input_height = int(pixel_values.shape[-2])
        model_input_width = int(pixel_values.shape[-1])
        synchronize(device)
        preprocess_seconds = time.perf_counter() - preprocess_started

        inference_started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(pixel_values=pixel_values).alphas
        synchronize(device)
        inference_seconds = time.perf_counter() - inference_started

        raw_model_alpha = prediction[0, 0].to(torch.float32).cpu().numpy()
        height, width = adaptive_alpha.shape
        if raw_model_alpha.shape[0] < height or raw_model_alpha.shape[1] < width:
            raise RuntimeError(
                f"Model alpha {raw_model_alpha.shape} is smaller than source {(height, width)}"
            )
        raw_model_alpha = raw_model_alpha[:height, :width]
        model_alpha = clamp_to_trimap(raw_model_alpha, trimap)
        fused_alpha, fusion_stats = fuse_unknown_alpha(
            adaptive_alpha,
            model_alpha,
            trimap,
            args.fusion_weight,
            args.fusion_max_delta,
        )

        postprocess_started = time.perf_counter()
        clean_rgb, final_alpha = refine_reframed_halo(
            adaptive_rgba[:, :, :3], fused_alpha, profile="real"
        )
        clean_rgb[final_alpha == 0.0] = 0.0
        packed = np.dstack((np.clip(clean_rgb, 0.0, 1.0), final_alpha))
        output_path = output_dir / input_path.name
        Image.fromarray(
            (packed * 255.0 + 0.5).astype(np.uint8), mode="RGBA"
        ).save(output_path, format="PNG")
        postprocess_seconds = time.perf_counter() - postprocess_started

        if diagnostics_dir:
            save_gray(diagnostics_dir / f"{input_path.stem}__baseline_alpha.png", adaptive_alpha)
            save_gray(diagnostics_dir / f"{input_path.stem}__trimap.png", trimap)
            save_gray(diagnostics_dir / f"{input_path.stem}__model_alpha.png", model_alpha)
            save_gray(diagnostics_dir / f"{input_path.stem}__fused_alpha.png", fused_alpha)
            save_gray(diagnostics_dir / f"{input_path.stem}__final_alpha.png", final_alpha)

        unknown = trimap == 128
        alpha_abs_delta = np.abs(model_alpha - adaptive_alpha)
        frame_seconds = time.perf_counter() - frame_started
        image_metrics.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "width": image.width,
                "height": image.height,
                "model_input_width": model_input_width,
                "model_input_height": model_input_height,
                "padding_right": model_input_width - image.width,
                "padding_bottom": model_input_height - image.height,
                "adaptive_seconds": adaptive_seconds,
                "trimap_seconds": trimap_seconds,
                "preprocess_seconds": preprocess_seconds,
                "inference_seconds": inference_seconds,
                "postprocess_and_save_seconds": postprocess_seconds,
                "end_to_end_seconds": frame_seconds,
                "trimap": trimap_stats,
                "alpha": {
                    "adaptive_soft_pct": float(
                        ((adaptive_alpha > 0.02) & (adaptive_alpha < 0.98)).mean()
                        * 100.0
                    ),
                    "final_soft_pct": float(
                        ((final_alpha > 0.02) & (final_alpha < 0.98)).mean() * 100.0
                    ),
                    "mean_abs_delta_unknown": (
                        float(alpha_abs_delta[unknown].mean()) if unknown.any() else 0.0
                    ),
                    "max_abs_delta_unknown": (
                        float(alpha_abs_delta[unknown].max()) if unknown.any() else 0.0
                    ),
                    "fusion": fusion_stats,
                },
            }
        )
        del pixel_values, prediction

    synchronize(device)
    batch_seconds = time.perf_counter() - batch_started
    inference_times = [float(item["inference_seconds"]) for item in image_metrics]
    end_to_end_times = [float(item["end_to_end_seconds"]) for item in image_metrics]
    unknown_percentages = [
        float(item["trimap"]["unknown_pct"]) for item in image_metrics  # type: ignore[index]
    ]

    result: dict[str, object] = {
        "status": "ok",
        "provider": "vitmatte_adaptive_green_hybrid",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "local_dir": str(model_dir),
            "implementation": "transformers.VitMatteForImageMatting",
            "checkpoint": "ViTMatte-S Composition-1k",
            "weight_file": "model.safetensors",
            "weight_sha256": file_sha256(model_dir / "model.safetensors"),
            "expected_weight_sha256": WEIGHT_SHA256,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "licenses": {
            "upstream_hustvl_code": "MIT",
            "hugging_face_model_card": "apache-2.0",
            "transformers": "Apache-2.0",
        },
        "pipeline": {
            "trimap_source": (
                "repository poc.profile_green_screen + adaptive_green_matte_frame"
            ),
            "trimap_values": {"known_background": 0, "unknown": 128, "known_foreground": 255},
            "background_threshold": args.background_threshold,
            "foreground_threshold": args.foreground_threshold,
            "unknown_radius_source_pixels_inward": args.unknown_radius,
            "morphology_kernel": "ellipse; foreground erosion only",
            "fusion_weight": args.fusion_weight,
            "fusion_max_delta": args.fusion_max_delta,
            "fusion_policy": (
                "adaptive + weight * clip(model - adaptive, +/-max_delta) in "
                "unknown pixels only"
            ),
            "background_policy": (
                "adaptive alpha <= background threshold is locked to known background"
            ),
            "known_region_policy": (
                "force background=0 and foreground=1 after model fusion; the "
                "downstream real-profile halo refinement may feather halo pixels"
            ),
            "rgb_policy": (
                "adaptive_green_matte_frame cleaned RGB + poc.refine_reframed_halo"
            ),
            "mean_unknown_pct": statistics.fmean(unknown_percentages),
            "processor": {
                "image_and_trimap_same_source_size": True,
                "trimap_channel_values_uint8": [0, 128, 255],
                "size_divisibility": int(processor.size_divisibility),
                "padding": "right and bottom only; output cropped to source size",
            },
            "green_profile": {
                "bg_floor": float(profile["bg_floor"]),
                "key_rgb_0_1": [float(value) for value in profile["key_rgb"]],
                "sampled_frames": int(profile["sampled_frames"]),
            },
        },
        "device": args.device,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "diagnostics_dir": str(diagnostics_dir) if diagnostics_dir else None,
        "image_count": len(image_metrics),
        "model_load_seconds": model_load_seconds,
        "green_profile_seconds": profile_seconds,
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
        "end_to_end_mean_seconds": statistics.fmean(end_to_end_times),
        "images": image_metrics,
        "environment": environment,
    }
    if device.type == "cuda":
        free_after_bytes, total_bytes = torch.cuda.mem_get_info(device)
        result["cuda_memory"] = {
            "total_bytes": total_bytes,
            "free_after_run_bytes": free_after_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "measurement": (
                "PyTorch allocator peak reset immediately before model load; "
                "includes model and all nine sequential inferences"
            ),
        }
    return result


def write_metrics(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


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
        try:
            write_metrics(args.output_dir.resolve(), failure)
        except OSError:
            pass
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1

    write_metrics(args.output_dir.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run official MatAnyone on a bounded frame sequence and emit RGBA PNGs.

The target assignment is derived from the repository's existing green-screen
matte on frame zero. The central harness is imported read-only and is not
modified by this provider.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_BASE = REPO_ROOT / ".models" / "video_matting"
MODEL_REPO = MODEL_BASE / "MatAnyone"
CHECKPOINT = MODEL_BASE / "checkpoints" / "matanyone.pth"
OFFICIAL_COMMIT = "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f"
CHECKPOINT_SHA256 = "dd26b991d020ed5eb4be50996f97354c45cfdfc0f59958e8983ac6a198f4809d"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
QUALITY_FIELDS = (
    "pseudo_mae",
    "background_alpha_mean",
    "foreground_loss_mean",
    "green_fringe",
    "fragment_pct",
    "soft_alpha_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official MatAnyone arbitrary-object video matting wrapper."
    )
    parser.add_argument("--input", required=True, type=Path, help="Video file or frame directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Run directory under this provider's runs/ directory",
    )
    parser.add_argument("--frames", type=int, default=24, help="Consecutive frames to process")
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument(
        "--max-size",
        type=int,
        default=640,
        help="Downscale when the shortest side exceeds this value; -1 keeps source size",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Official recurrent refinement passes")
    parser.add_argument(
        "--init-kind",
        choices=("alpha", "mask"),
        default="mask",
        help="Feed the soft green-screen alpha or its binary target mask",
    )
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument(
        "--max-internal-size",
        type=int,
        default=-1,
        help="Official InferenceCore short-side limit; output is restored to decoded size",
    )
    parser.add_argument("--mem-every", type=int, default=5)
    parser.add_argument("--max-mem-frames", type=int, default=5)
    parser.add_argument(
        "--use-long-term",
        action="store_true",
        help="Enable the official long-term memory tier",
    )
    parser.add_argument(
        "--rgba-rgb",
        choices=("green-clean", "source"),
        default="green-clean",
        help="Use existing green cleanup for RGB, or preserve source RGB",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA float16 autocast")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_within(path: Path, parent: Path) -> None:
    if path == parent:
        raise ValueError(f"write path must be a child of {parent}, not the directory itself")
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"write path must stay under {parent}: {path}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def resize_rgb(frame: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return frame
    height, width = frame.shape[:2]
    short_side = min(height, width)
    if short_side <= max_size:
        return frame
    scale = max_size / short_side
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def decode_frames(
    input_path: Path, frame_offset: int, frame_count: int, max_size: int
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if frame_count < 1:
        raise ValueError("--frames must be positive")
    if frame_offset < 0:
        raise ValueError("--frame-offset cannot be negative")

    frames: list[np.ndarray] = []
    metadata: dict[str, Any] = {}
    if input_path.is_dir():
        paths = sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        selected = paths[frame_offset : frame_offset + frame_count]
        for path in selected:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"OpenCV failed to decode {path}")
            frames.append(resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_size))
        metadata.update(
            {
                "kind": "frame_directory",
                "source_frame_count": len(paths),
                "source_fps": None,
                "selected_files": [path.name for path in selected],
            }
        )
    else:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV failed to open {input_path}")
        source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
        try:
            while len(frames) < frame_count:
                ok, bgr = capture.read()
                if not ok:
                    break
                frames.append(resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_size))
        finally:
            capture.release()
        metadata.update(
            {
                "kind": "video",
                "source_frame_count": source_count,
                "source_fps": round(source_fps, 6),
                "source_size": [source_width, source_height],
            }
        )

    if len(frames) != frame_count:
        raise RuntimeError(
            f"requested {frame_count} frames at offset {frame_offset}, decoded {len(frames)}"
        )
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise RuntimeError("all input frames must have the same dimensions")
    metadata["processed_size"] = [shape[1], shape[0]]
    metadata["decoded_frames"] = len(frames)
    metadata["frame_offset"] = frame_offset
    return frames, metadata


def load_green_module() -> Any:
    poc_path = REPO_ROOT / "poc.py"
    spec = importlib.util.spec_from_file_location("petavatar_green_initializer", poc_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import existing green-screen implementation from {poc_path}")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def load_green_initializer(
    module: Any, first_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    poc_path = REPO_ROOT / "poc.py"

    profile = module.profile_green_arrays([first_rgb])
    green_rgba = np.asarray(
        module.adaptive_green_matte_frame(Image.fromarray(first_rgb, "RGB"), profile),
        dtype=np.uint8,
    )
    alpha = green_rgba[:, :, 3]
    if not 0.01 < float((alpha > 8).mean()) < 0.95:
        raise RuntimeError("green-screen initializer produced implausible target coverage")
    profile_meta = {
        "source_file": str(poc_path),
        "profile_function": "profile_green_arrays([frame0])",
        "matte_function": "adaptive_green_matte_frame(frame0, profile)",
        "bg_floor": round(float(profile["bg_floor"]), 6),
        "key_rgb": [round(float(value) * 255) for value in profile["key_rgb"]],
        "sampled_frames": int(profile["sampled_frames"]),
        "alpha_min": int(alpha.min()),
        "alpha_max": int(alpha.max()),
        "alpha_mean": round(float(alpha.mean()), 6),
        "visible_fraction_gt_8": round(float((alpha > 8).mean()), 6),
        "opaque_fraction_ge_250": round(float((alpha >= 250).mean()), 6),
    }
    return green_rgba, alpha, profile_meta


def build_green_reference(
    module: Any, frames: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Run the existing chroma key for RGB cleanup and green-footage diagnostics."""
    profile = module.profile_green_arrays(frames)
    clean_rgbs: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for frame in frames:
        rgba = np.asarray(
            module.adaptive_green_matte_frame(Image.fromarray(frame, "RGB"), profile),
            dtype=np.uint8,
        )
        clean_rgbs.append(rgba[:, :, :3])
        alphas.append(rgba[:, :, 3])
    return clean_rgbs, alphas, {
        "role": "RGB cleanup and diagnostic reference only; never substituted for model alpha",
        "profile_function": "profile_green_arrays(all selected frames)",
        "matte_function": "adaptive_green_matte_frame(frame, clip_profile)",
        "bg_floor": round(float(profile["bg_floor"]), 6),
        "key_rgb": [round(float(value) * 255) for value in profile["key_rgb"]],
        "sampled_frames": int(profile["sampled_frames"]),
    }


def verify_model_assets() -> dict[str, Any]:
    if not (MODEL_REPO / ".git").exists():
        raise FileNotFoundError(f"missing official source checkout: {MODEL_REPO}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"missing checkpoint: {CHECKPOINT}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(MODEL_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"unexpected MatAnyone commit {actual_commit}; expected {OFFICIAL_COMMIT}")
    source_changes = subprocess.check_output(
        ["git", "-C", str(MODEL_REPO), "status", "--porcelain"], text=True
    ).strip()
    if source_changes:
        raise RuntimeError("official MatAnyone source checkout has local changes")
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch {checkpoint_hash}; expected {CHECKPOINT_SHA256}"
        )
    return {
        "name": "MatAnyone v1",
        "repository": "https://github.com/pq-yang/MatAnyone",
        "commit": actual_commit,
        "checkpoint_url": (
            "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
        ),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "license": "S-Lab License 1.0 (non-commercial; commercial use requires permission)",
    }


def choose_device(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("this verified provider requires a CUDA-capable PyTorch runtime")
    return torch.device("cuda")


def load_official_model(
    device: Any,
    *,
    max_internal_size: int,
    mem_every: int,
    max_mem_frames: int,
    use_long_term: bool,
) -> tuple[Any, dict[str, Any]]:
    if str(MODEL_REPO) not in sys.path:
        sys.path.insert(0, str(MODEL_REPO))

    import_started = time.perf_counter()
    import torch
    import torchvision
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from matanyone.inference.inference_core import InferenceCore
    from matanyone.model.matanyone import MatAnyone
    module_import_seconds = time.perf_counter() - import_started

    started = time.perf_counter()
    config_dir = (MODEL_REPO / "matanyone" / "config").resolve()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(config_dir)):
        config = compose(config_name="eval_matanyone_config")
    with open_dict(config):
        config.weights = str(CHECKPOINT.resolve())
        # The release checkpoint includes both encoders. This avoids two redundant
        # ImageNet downloads performed by the upstream constructor default.
        config.model.pretrained_resnet = False
        config.max_internal_size = max_internal_size
        config.mem_every = mem_every
        config.max_mem_frames = max_mem_frames
        config.use_long_term = use_long_term

    network = MatAnyone(config, single_object=True).to(device).eval()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    network.load_weights(state)
    processor = InferenceCore(network, cfg=config, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metadata = {
        "module_import_seconds": round(module_import_seconds, 6),
        "load_seconds": round(time.perf_counter() - started, 6),
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_driver": query_gpu_driver() if device.type == "cuda" else None,
        "pretrained_resnet_download_disabled": True,
        "inference_config": {
            "max_internal_size": max_internal_size,
            "mem_every": mem_every,
            "max_mem_frames": max_mem_frames,
            "use_long_term": use_long_term,
            "stagger_updates": int(config.stagger_updates),
            "top_k": int(config.top_k),
        },
    }
    return processor, metadata


def query_gpu_driver() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        return None


def cuda_memory(torch: Any, device: Any) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 3),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 3),
        "driver_free_mib": round(free_bytes / 2**20, 3),
        "device_total_mib": round(total_bytes / 2**20, 3),
    }


def run_inference(
    processor: Any,
    frames: list[np.ndarray],
    init: np.ndarray,
    warmup: int,
    device: Any,
    amp_enabled: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    import torch

    if warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    loaded_memory = cuda_memory(torch, device)

    init_tensor = torch.from_numpy(np.ascontiguousarray(init)).float().to(device)
    sequence = [frames[0]] * warmup + frames
    alphas: list[np.ndarray] = []
    warmup_times: list[float] = []
    output_times: list[float] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for time_index, rgb in enumerate(sequence):
            image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            image = image.float().div_(255.0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            frame_started = time.perf_counter()
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled and device.type == "cuda",
            ):
                if time_index == 0:
                    processor.step(image, init_tensor, objects=[1])
                    output_probability = processor.step(image, first_frame_pred=True)
                elif time_index <= warmup:
                    output_probability = processor.step(image, first_frame_pred=True)
                else:
                    output_probability = processor.step(
                        image, end=time_index == len(sequence) - 1
                    )
                alpha_tensor = processor.output_prob_to_mask(output_probability)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - frame_started
            if time_index < warmup:
                warmup_times.append(elapsed)
            else:
                output_times.append(elapsed)
                alpha = alpha_tensor.detach().float().clamp_(0.0, 1.0).cpu().numpy()
                alphas.append(np.round(alpha * 255.0).astype(np.uint8))

    if len(alphas) != len(frames):
        raise RuntimeError(f"expected {len(frames)} alpha frames, got {len(alphas)}")
    total_seconds = time.perf_counter() - started
    memory = cuda_memory(torch, device)
    return alphas, {
        "total_with_warmup_seconds": round(total_seconds, 6),
        "warmup_passes": warmup,
        "warmup_seconds": round(sum(warmup_times), 6),
        "output_inference_seconds": round(sum(output_times), 6),
        "output_frames_per_second": round(len(frames) / max(sum(output_times), 1e-9), 6),
        "output_first_frame_seconds": round(output_times[0], 6),
        "steady_after_first_frames_per_second": round(
            max(0, len(output_times) - 1) / max(sum(output_times[1:]), 1e-9), 6
        ),
        "output_frame_seconds_p50": round(percentile(output_times, 0.50), 6),
        "output_frame_seconds_p95": round(percentile(output_times, 0.95), 6),
        "cuda_after_model_load": loaded_memory,
        "cuda_peak_during_inference": memory,
    }


def alpha_diagnostics(alphas: list[np.ndarray]) -> dict[str, Any]:
    coverages: list[float] = []
    soft_coverages: list[float] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    components: list[int] = []
    consecutive_diffs: list[float] = []
    edge_touch_frames = 0
    alpha_values: set[int] = set()

    for index, alpha in enumerate(alphas):
        alpha_values.update(int(value) for value in np.unique(alpha))
        visible = alpha >= 16
        coverages.append(float(visible.mean()))
        soft_coverages.append(float(((alpha > 0) & (alpha < 250)).mean()))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            visible.astype(np.uint8), 8
        )
        components.append(max(0, count - 1))
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            centroid_x.append(float(centroids[largest, 0]))
            centroid_y.append(float(centroids[largest, 1]))
        if visible[0].any() or visible[-1].any() or visible[:, 0].any() or visible[:, -1].any():
            edge_touch_frames += 1
        if index:
            consecutive_diffs.append(
                float(np.mean(np.abs(alpha.astype(np.float32) - alphas[index - 1].astype(np.float32))))
                / 255.0
            )

    coverage_mean = statistics.fmean(coverages)
    coverage_cv = (
        statistics.pstdev(coverages) / coverage_mean if coverage_mean > 0 and len(coverages) > 1 else 0.0
    )
    return {
        "alpha_type": "fractional probability quantized to uint8",
        "alpha_unique_value_count": len(alpha_values),
        "alpha_value_min": min(alpha_values, default=0),
        "alpha_value_max": max(alpha_values, default=0),
        "visible_coverage_mean": round(coverage_mean, 6),
        "visible_coverage_cv": round(coverage_cv, 6),
        "visible_coverage_min": round(min(coverages), 6),
        "visible_coverage_max": round(max(coverages), 6),
        "soft_alpha_coverage_mean": round(statistics.fmean(soft_coverages), 6),
        "consecutive_alpha_mae_mean": round(statistics.fmean(consecutive_diffs), 6),
        "consecutive_alpha_mae_p95": round(percentile(consecutive_diffs, 0.95), 6),
        "largest_component_centroid_span_px": [
            round(max(centroid_x) - min(centroid_x), 3) if centroid_x else 0.0,
            round(max(centroid_y) - min(centroid_y), 3) if centroid_y else 0.0,
        ],
        "component_count_max": max(components, default=0),
        "edge_touch_frames": edge_touch_frames,
        "empty_frames": sum(coverage == 0 for coverage in coverages),
    }


def compare_to_green_reference(
    predictions: list[np.ndarray], references: list[np.ndarray]
) -> dict[str, Any]:
    frame_rows: list[dict[str, float]] = []
    for prediction, reference in zip(predictions, references, strict=True):
        interior = reference >= 250
        background = reference <= 5
        frame_rows.append(
            {
                "alpha_mae": float(
                    np.abs(reference.astype(np.float32) - prediction.astype(np.float32)).mean()
                    / 255.0
                ),
                "reference_interior_soft_fraction": float(
                    ((prediction < 250) & interior).sum() / max(1, int(interior.sum()))
                ),
                "reference_interior_hole_fraction": float(
                    ((prediction < 128) & interior).sum() / max(1, int(interior.sum()))
                ),
                "reference_background_leak_fraction": float(
                    ((prediction > 16) & background).sum() / max(1, int(background.sum()))
                ),
            }
        )
    return {
        key: round(statistics.fmean(row[key] for row in frame_rows), 6)
        for key in frame_rows[0]
    }


def standardized_benchmark_metrics(
    frames: list[np.ndarray], output_rgbs: list[np.ndarray], alphas: list[np.ndarray]
) -> dict[str, Any]:
    """Use the repository benchmark metric implementation without writing centrally."""
    evaluator_path = REPO_ROOT / "matting_bench" / "evaluate.py"
    spec = importlib.util.spec_from_file_location(
        "petavatar_matanyone_benchmark_evaluator", evaluator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting

    sources = [frame.astype(np.float32) / 255.0 for frame in frames]
    alpha_float = [alpha.astype(np.float32) / 255.0 for alpha in alphas]
    outputs = [
        np.dstack([rgb.astype(np.float32) / 255.0, alpha])
        for rgb, alpha in zip(output_rgbs, alpha_float, strict=True)
    ]
    frame_values = [
        module.frame_metrics(source, output)
        for source, output in zip(sources, outputs, strict=True)
    ]
    mean = module.summarize(frame_values)
    return {
        "definition_file": str(evaluator_path),
        "quality": {field: float(mean[field]) for field in QUALITY_FIELDS},
        "temporal_alpha_mae": module.temporal_error(sources, alpha_float),
        "frames": len(frame_values),
        "reference_kind": "controlled-green-screen confident-region proxy",
    }


def checkerboard(width: int, height: int, tile: int = 16) -> Image.Image:
    y, x = np.indices((height, width))
    pattern = ((x // tile + y // tile) % 2).astype(bool)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[pattern] = (218, 222, 218)
    rgb[~pattern] = (242, 244, 242)
    return Image.fromarray(rgb, "RGB")


def make_contact_sheet(
    rgba_frames: list[Image.Image], alpha_frames: list[Image.Image], output_dir: Path
) -> None:
    sample_count = min(8, len(rgba_frames))
    indices = sorted(
        set(round(index * (len(rgba_frames) - 1) / max(1, sample_count - 1)) for index in range(sample_count))
    )
    thumb_size = 240
    label_height = 24
    rgba_sheet = Image.new("RGB", (thumb_size * 4, (thumb_size + label_height) * 2), "white")
    alpha_sheet = Image.new("L", rgba_sheet.size, 255)
    draw = ImageDraw.Draw(rgba_sheet)
    alpha_draw = ImageDraw.Draw(alpha_sheet)
    for slot, frame_index in enumerate(indices):
        x = (slot % 4) * thumb_size
        y = (slot // 4) * (thumb_size + label_height)
        background = checkerboard(thumb_size, thumb_size)
        rgba = rgba_frames[frame_index].copy()
        rgba.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        paste_x = x + (thumb_size - rgba.width) // 2
        paste_y = y + (thumb_size - rgba.height) // 2
        background.paste(rgba, (paste_x - x, paste_y - y), rgba)
        rgba_sheet.paste(background, (x, y))
        alpha = alpha_frames[frame_index].resize(
            (thumb_size, thumb_size), Image.Resampling.LANCZOS
        )
        alpha_sheet.paste(alpha, (x, y))
        label = f"frame {frame_index:04d}"
        draw.text((x + 6, y + thumb_size + 4), label, fill=(24, 32, 28))
        alpha_draw.text((x + 6, y + thumb_size + 4), label, fill=0)
    rgba_sheet.save(output_dir / "contact_sheet_rgba.png", compress_level=4)
    alpha_sheet.save(output_dir / "contact_sheet_alpha.png", compress_level=4)


def write_outputs(
    output_dir: Path,
    rgb_frames: list[np.ndarray],
    alphas: list[np.ndarray],
    rgb_policy: str,
) -> tuple[dict[str, Any], list[Image.Image], list[Image.Image]]:
    rgba_dir = output_dir / "rgba"
    alpha_dir = output_dir / "alpha"
    rgba_dir.mkdir(parents=True)
    alpha_dir.mkdir(parents=True)
    rgba_images: list[Image.Image] = []
    alpha_images: list[Image.Image] = []
    digest = hashlib.sha256()
    started = time.perf_counter()
    for index, (rgb, alpha) in enumerate(zip(rgb_frames, alphas, strict=True)):
        clean_rgb = rgb.copy()
        clean_rgb[alpha == 0] = 0
        rgba = np.dstack([clean_rgb, alpha])
        rgba_image = Image.fromarray(rgba, "RGBA")
        alpha_image = Image.fromarray(alpha, "L")
        rgba_path = rgba_dir / f"frame_{index:04d}.png"
        alpha_path = alpha_dir / f"frame_{index:04d}.png"
        rgba_image.save(rgba_path, compress_level=4)
        alpha_image.save(alpha_path, compress_level=4)
        digest.update(rgba_path.read_bytes())
        rgba_images.append(rgba_image)
        alpha_images.append(alpha_image)
    make_contact_sheet(rgba_images, alpha_images, output_dir)
    return (
        {
            "rgba_directory": str(rgba_dir),
            "alpha_directory": str(alpha_dir),
            "rgba_frames": len(rgba_images),
            "rgba_sequence_sha256": digest.hexdigest(),
            "write_seconds": round(time.perf_counter() - started, 6),
            "rgb_policy": rgb_policy,
        },
        rgba_images,
        alpha_images,
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = resolved(args.input)
    output_dir = resolved(args.output_dir)
    provider_root = resolved(PROVIDER_DIR)
    runs_root = provider_root / "runs"
    ensure_within(output_dir, runs_root)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {output_dir}")
        ensure_within(output_dir, runs_root)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    run_started = time.perf_counter()
    try:
        if not 0 <= args.mask_threshold <= 255:
            raise ValueError("--mask-threshold must be in [0, 255]")
        if args.mem_every < 1:
            raise ValueError("--mem-every must be positive")
        if args.max_mem_frames < 1:
            raise ValueError("--max-mem-frames must be positive")
        if args.max_internal_size == 0 or args.max_internal_size < -1:
            raise ValueError("--max-internal-size must be -1 or a positive integer")
        asset_meta = verify_model_assets()
        decode_started = time.perf_counter()
        frames, input_meta = decode_frames(
            input_path, args.frame_offset, args.frames, args.max_size
        )
        decode_seconds = time.perf_counter() - decode_started

        init_started = time.perf_counter()
        green_module = load_green_module()
        green_rgba, init_alpha, initializer_meta = load_green_initializer(
            green_module, frames[0]
        )
        init_mask = np.where(init_alpha >= args.mask_threshold, 255, 0).astype(np.uint8)
        initializer_meta.update(
            {
                "init_kind_passed_to_model": args.init_kind,
                "mask_threshold": args.mask_threshold,
                "mask_coverage": round(float((init_mask > 0).mean()), 6),
            }
        )
        init_dir = output_dir / "init"
        init_dir.mkdir()
        Image.fromarray(frames[0], "RGB").save(init_dir / "first_frame_rgb.png")
        Image.fromarray(green_rgba, "RGBA").save(init_dir / "first_frame_green_rgba.png")
        Image.fromarray(init_alpha, "L").save(init_dir / "first_frame_alpha.png")
        Image.fromarray(init_mask, "L").save(init_dir / "first_frame_mask.png")
        write_json(init_dir / "initializer.json", initializer_meta)
        init_seconds = time.perf_counter() - init_started

        reference_started = time.perf_counter()
        green_clean_rgbs, green_reference_alphas, green_reference_meta = build_green_reference(
            green_module, frames
        )
        green_reference_seconds = time.perf_counter() - reference_started

        torch_import_started = time.perf_counter()
        import torch
        torch_import_seconds = time.perf_counter() - torch_import_started

        device = choose_device(torch)
        processor, runtime_meta = load_official_model(
            device,
            max_internal_size=args.max_internal_size,
            mem_every=args.mem_every,
            max_mem_frames=args.max_mem_frames,
            use_long_term=args.use_long_term,
        )
        selected_init = init_alpha if args.init_kind == "alpha" else init_mask
        alphas, inference_meta = run_inference(
            processor=processor,
            frames=frames,
            init=selected_init,
            warmup=args.warmup,
            device=device,
            amp_enabled=not args.no_amp,
        )
        if args.rgba_rgb == "green-clean":
            output_rgbs = green_clean_rgbs
            rgb_policy = (
                "RGB from existing adaptive_green_matte_frame using a clip-level profile; "
                "alpha is unmodified MatAnyone output"
            )
        else:
            output_rgbs = frames
            rgb_policy = "source RGB; RGB is zeroed only where MatAnyone alpha equals zero"
        output_meta, _, _ = write_outputs(
            output_dir, output_rgbs, alphas, rgb_policy=rgb_policy
        )
        benchmark_meta = standardized_benchmark_metrics(frames, output_rgbs, alphas)
        diagnostics = alpha_diagnostics(alphas)
        green_reference_meta["timing_seconds"] = round(green_reference_seconds, 6)
        green_reference_meta["alpha_diagnostics"] = alpha_diagnostics(green_reference_alphas)
        green_reference_meta["model_comparison"] = compare_to_green_reference(
            alphas, green_reference_alphas
        )

        end_to_end_seconds = time.perf_counter() - run_started
        metrics = {
            "status": "succeeded",
            "provider": "official_matanyone_v1",
            "model": asset_meta,
            "input": {
                "path": str(input_path),
                "bytes": input_path.stat().st_size if input_path.is_file() else None,
                **input_meta,
            },
            "initializer": initializer_meta,
            "green_reference": green_reference_meta,
            "runtime": {
                **runtime_meta,
                "torch_import_seconds": round(torch_import_seconds, 6),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "amp": not args.no_amp and device.type == "cuda",
            },
            "timing": {
                "decode_seconds": round(decode_seconds, 6),
                "initializer_seconds": round(init_seconds, 6),
                "green_reference_seconds": round(green_reference_seconds, 6),
                **inference_meta,
                "output_write_seconds": output_meta["write_seconds"],
                "end_to_end_seconds": round(end_to_end_seconds, 6),
            },
            "diagnostics": diagnostics,
            "benchmark": benchmark_meta,
            "output": output_meta,
            "tuning_parameters": {
                "decoded_max_size": args.max_size,
                "max_internal_size": args.max_internal_size,
                "warmup": args.warmup,
                "init_kind": args.init_kind,
                "mask_threshold": args.mask_threshold,
                "precision": "fp16_autocast" if not args.no_amp else "fp32",
                "mem_every": args.mem_every,
                "max_mem_frames": args.max_mem_frames,
                "use_long_term": args.use_long_term,
            },
            "notes": [
                "The central poc.py is imported read-only for frame-0 target initialization.",
                "MatAnyone predicts alpha only; existing green cleanup supplies RGB without changing predicted alpha.",
                "Green-reference comparison is diagnostic for this controlled footage, not arbitrary-object ground truth.",
                "Consecutive alpha difference includes real object motion and is not a ground-truth flicker metric.",
            ],
        }
        write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "input": str(input_path),
            "output_dir": str(output_dir),
            "elapsed_seconds": round(time.perf_counter() - run_started, 6),
        }
        write_json(output_dir / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=True, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

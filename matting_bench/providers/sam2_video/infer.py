#!/usr/bin/env python3
"""Isolated SAM 2.1 video-mask propagation experiment for pet footage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "sam2_video"
MODEL_REPO = MODEL_ROOT / "repo"
CHECKPOINT = MODEL_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"

OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/sam2"
OFFICIAL_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_small.pt"
)
CHECKPOINT_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
MODEL_NAME = "SAM 2.1 Hiera Small"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    default_input = (
        REPO_ROOT
        / "matting_bench"
        / "data"
        / "pet_20260710_121221_5ce7716e"
        / "full"
        / "fast_walk"
    )
    default_output = PROVIDER_DIR / "runs" / "fast_walk_24_sam2_1_small"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"write path must stay within {root}: {path}") from exc


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    output_dir = resolved(path)
    provider_root = resolved(PROVIDER_DIR)
    ensure_within(output_dir, provider_root)
    if output_dir == provider_root:
        raise ValueError("the provider root itself cannot be used as an output directory")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_dir}")
        ensure_within(output_dir, provider_root)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sort_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else sys.maxsize, path.name.lower())


def select_frames(input_dir: Path, offset: int, count: int) -> list[Path]:
    if offset < 0:
        raise ValueError("frame offset must be non-negative")
    if count < 1:
        raise ValueError("frame count must be positive")
    input_dir = resolved(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    available = sorted(
        (path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=frame_sort_key,
    )
    selected = available[offset : offset + count]
    if len(selected) != count:
        raise RuntimeError(
            f"requested {count} frames at offset {offset}, found only {len(selected)}"
        )
    output_names = [path.with_suffix(".png").name for path in selected]
    if len(set(output_names)) != len(output_names):
        raise RuntimeError("selected frames do not have unique output names")
    return selected


def load_rgb_frames(paths: list[Path]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    expected_size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        size = (rgb.shape[1], rgb.shape[0])
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise RuntimeError(f"mixed frame sizes: expected {expected_size}, got {size} at {path}")
        frames.append(rgb)
    return frames


def stage_frames(paths: list[Path], staging_dir: Path) -> dict[str, Any]:
    """Expose lossless source bytes under the numeric JPEG names SAM 2 expects."""
    staging_dir.mkdir()
    methods: list[str] = []
    for index, source in enumerate(paths):
        destination = staging_dir / f"{index:05d}.jpg"
        try:
            os.link(source, destination)
            methods.append("hardlink")
        except OSError:
            shutil.copy2(source, destination)
            methods.append("copy")
    return {
        "directory": str(staging_dir),
        "format_note": "PNG bytes are exposed under numeric .jpg names; PIL detects by content.",
        "methods": sorted(set(methods)),
        "frames": len(paths),
    }


def verify_model_assets() -> dict[str, Any]:
    if not (MODEL_REPO / ".git").is_dir():
        raise FileNotFoundError(f"missing official source checkout: {MODEL_REPO}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing model checkpoint: {CHECKPOINT}")

    commit = subprocess.check_output(
        ["git", "-C", str(MODEL_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"unexpected SAM 2 commit {commit}; expected {OFFICIAL_COMMIT}")
    origin = subprocess.check_output(
        ["git", "-C", str(MODEL_REPO), "remote", "get-url", "origin"], text=True
    ).strip()
    normalized_origin = origin.removesuffix(".git").rstrip("/")
    if normalized_origin != OFFICIAL_REPOSITORY:
        raise RuntimeError(f"unexpected SAM 2 origin: {origin}")

    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch {checkpoint_hash}; expected {CHECKPOINT_SHA256}"
        )

    license_path = MODEL_REPO / "LICENSE"
    readme_text = (MODEL_REPO / "README.md").read_text(encoding="utf-8")
    license_text = license_path.read_text(encoding="utf-8")
    license_verified = (
        "Apache License" in license_text
        and "Version 2.0" in license_text
        and "model checkpoints" in readme_text
        and "licensed under [Apache 2.0]" in readme_text
    )
    if not license_verified:
        raise RuntimeError("official Apache-2.0 license statements were not found")

    dirty_lines = subprocess.check_output(
        ["git", "-C", str(MODEL_REPO), "status", "--porcelain"], text=True
    ).splitlines()
    return {
        "name": MODEL_NAME,
        "architecture": "sam2.1_hiera_small",
        "config": MODEL_CONFIG,
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "checkout_dirty": bool(dirty_lines),
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": checkpoint_hash,
        "license": "Apache-2.0",
        "license_file": str(license_path),
        "license_file_sha256": sha256_file(license_path),
        "readme_checkpoint_license_statement_verified": license_verified,
        "commercial_use_assessment": (
            "Acceptable for commercial evaluation under Apache-2.0, subject to preserving "
            "the license/notice and separately reviewing third-party dependencies."
        ),
    }


def import_green_screen_module() -> Any:
    poc_path = REPO_ROOT / "poc.py"
    spec = importlib.util.spec_from_file_location("pet_avatar_poc_read_only", poc_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import existing green-screen implementation: {poc_path}")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def green_screen_masks(
    frames: list[np.ndarray], threshold: int
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, Any]]:
    module = import_green_screen_module()
    profile = module.profile_green_arrays([frames[0]])
    alphas: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    first_rgba: np.ndarray | None = None
    for frame in frames:
        rgba = np.asarray(
            module.adaptive_green_matte_frame(Image.fromarray(frame), profile),
            dtype=np.uint8,
        )
        if first_rgba is None:
            first_rgba = rgba
        alpha = rgba[:, :, 3]
        alphas.append(alpha)
        masks.append(alpha >= threshold)
    assert first_rgba is not None
    first_alpha = alphas[0]
    prompt_coverage = float(masks[0].mean())
    if not 0.01 < prompt_coverage < 0.95:
        raise RuntimeError(f"green-screen prompt coverage is implausible: {prompt_coverage}")
    metadata = {
        "source_file": str(REPO_ROOT / "poc.py"),
        "profile_function": "profile_green_arrays([frame0])",
        "matte_function": "adaptive_green_matte_frame(frame, frame0_profile)",
        "only_frame_zero_passed_to_sam2": True,
        "threshold": threshold,
        "bg_floor": round(float(profile["bg_floor"]), 6),
        "key_rgb": [round(float(value) * 255) for value in profile["key_rgb"]],
        "sampled_frames_for_profile": int(profile["sampled_frames"]),
        "prompt_coverage": round(prompt_coverage, 6),
        "prompt_alpha_mean": round(float(first_alpha.mean()), 6),
        "prompt_soft_alpha_fraction": round(
            float(((first_alpha > 0) & (first_alpha < 255)).mean()), 6
        ),
    }
    return alphas, masks, first_rgba, metadata


def save_initializer(
    output_dir: Path,
    first_rgb: np.ndarray,
    first_green_rgba: np.ndarray,
    first_alpha: np.ndarray,
    first_mask: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    init_dir = output_dir / "init"
    init_dir.mkdir()
    Image.fromarray(first_rgb).save(init_dir / "first_frame_rgb.png")
    Image.fromarray(first_green_rgba).save(init_dir / "first_frame_green_rgba.png")
    Image.fromarray(first_alpha).save(init_dir / "first_frame_alpha.png")
    Image.fromarray(first_mask.astype(np.uint8) * 255).save(
        init_dir / "first_frame_mask.png"
    )
    write_json(init_dir / "initializer.json", metadata)


def cuda_snapshot(torch: Any) -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    divisor = 1024**2
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / divisor, 3),
        "reserved_mib": round(torch.cuda.memory_reserved() / divisor, 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / divisor, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / divisor, 3),
        "driver_free_mib": round(free_bytes / divisor, 3),
        "device_total_mib": round(total_bytes / divisor, 3),
    }


def timed_cuda(torch: Any, operation: Callable[[], Any]) -> tuple[Any, float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    result = operation()
    end_event.record()
    end_event.synchronize()
    return result, time.perf_counter() - started, start_event.elapsed_time(end_event)


def autocast_context(torch: Any, precision: str) -> Any:
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cuda", enabled=False)


def load_predictor(torch: Any, precision: str) -> tuple[Any, dict[str, Any]]:
    if str(MODEL_REPO) not in sys.path:
        sys.path.insert(0, str(MODEL_REPO))
    from sam2.build_sam import build_sam2_video_predictor

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def build() -> Any:
        with autocast_context(torch, precision):
            return build_sam2_video_predictor(
                MODEL_CONFIG,
                str(CHECKPOINT),
                device="cuda",
                mode="eval",
                apply_postprocessing=True,
                vos_optimized=False,
            )

    predictor, wall_seconds, cuda_ms = timed_cuda(torch, build)
    parameters = sum(parameter.numel() for parameter in predictor.parameters())
    return predictor, {
        "wall_seconds": round(wall_seconds, 6),
        "cuda_elapsed_ms": round(cuda_ms, 3),
        "parameters": int(parameters),
        "memory": cuda_snapshot(torch),
    }


def run_propagation(
    torch: Any,
    predictor: Any,
    staging_dir: Path,
    prompt_mask: np.ndarray,
    frame_count: int,
    object_id: int,
    precision: str,
    offload_state_to_cpu: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats()

    def initialize() -> Any:
        with autocast_context(torch, precision):
            return predictor.init_state(
                video_path=str(staging_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=offload_state_to_cpu,
                async_loading_frames=False,
            )

    state, init_wall, init_cuda_ms = timed_cuda(torch, initialize)

    def add_prompt() -> Any:
        with autocast_context(torch, precision):
            return predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=object_id,
                mask=prompt_mask,
            )

    _, prompt_wall, prompt_cuda_ms = timed_cuda(torch, add_prompt)

    propagated: dict[int, np.ndarray] = {}
    per_frame_cuda_ms: list[float] = []
    per_frame_wall_ms: list[float] = []
    propagation_started = time.perf_counter()
    with autocast_context(torch, precision):
        iterator = iter(
            predictor.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=frame_count,
                reverse=False,
            )
        )
        while True:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            frame_started = time.perf_counter()
            try:
                frame_idx, object_ids, mask_logits = next(iterator)
            except StopIteration:
                break
            object_index = list(object_ids).index(object_id)
            mask = (
                mask_logits[object_index, 0]
                .gt(0.0)
                .detach()
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
                .astype(bool)
            )
            end_event.record()
            end_event.synchronize()
            per_frame_cuda_ms.append(float(start_event.elapsed_time(end_event)))
            per_frame_wall_ms.append((time.perf_counter() - frame_started) * 1000.0)
            propagated[int(frame_idx)] = mask

    missing = sorted(set(range(frame_count)) - set(propagated))
    if missing:
        raise RuntimeError(f"SAM 2 did not emit all requested frames: {missing}")
    masks = [propagated[index] for index in range(frame_count)]
    propagation_wall = time.perf_counter() - propagation_started
    return masks, {
        "init_state_wall_seconds": round(init_wall, 6),
        "init_state_cuda_ms": round(init_cuda_ms, 3),
        "prompt_wall_seconds": round(prompt_wall, 6),
        "prompt_cuda_ms": round(prompt_cuda_ms, 3),
        "propagation_wall_seconds": round(propagation_wall, 6),
        "propagation_cuda_ms_sum": round(float(sum(per_frame_cuda_ms)), 3),
        "propagation_fps_wall": round(frame_count / propagation_wall, 6),
        "frame_cuda_ms_p50": round(float(np.percentile(per_frame_cuda_ms, 50)), 3),
        "frame_cuda_ms_p95": round(float(np.percentile(per_frame_cuda_ms, 95)), 3),
        "frame_wall_ms_p50": round(float(np.percentile(per_frame_wall_ms, 50)), 3),
        "frame_wall_ms_p95": round(float(np.percentile(per_frame_wall_ms, 95)), 3),
        "per_frame_cuda_ms": [round(value, 3) for value in per_frame_cuda_ms],
        "peak_memory": cuda_snapshot(torch),
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": offload_state_to_cpu,
    }


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return float(x.mean()), float(y.mean())


def diagnostics(
    masks: list[np.ndarray],
    green_masks: list[np.ndarray],
    green_alphas: list[np.ndarray],
) -> dict[str, Any]:
    areas = np.asarray([float(mask.mean()) for mask in masks], dtype=np.float64)
    reference_areas = np.asarray(
        [float(mask.mean()) for mask in green_masks], dtype=np.float64
    )
    consecutive_iou = np.asarray(
        [binary_iou(masks[index - 1], masks[index]) for index in range(1, len(masks))]
    )
    reference_iou = np.asarray(
        [binary_iou(mask, reference) for mask, reference in zip(masks, green_masks, strict=True)]
    )
    soft_fractions: list[float] = []
    soft_band_containment: list[float] = []
    component_counts: list[int] = []
    centroids: list[tuple[float, float] | None] = []
    for mask, alpha in zip(masks, green_alphas, strict=True):
        soft_band = (alpha > 8) & (alpha < 247)
        soft_fractions.append(float(soft_band.mean()))
        soft_band_containment.append(float(mask[soft_band].mean()) if soft_band.any() else 1.0)
        count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        component_counts.append(max(0, int(count) - 1))
        centroids.append(mask_centroid(mask))

    centroid_steps = []
    for previous, current in zip(centroids[:-1], centroids[1:], strict=True):
        if previous is not None and current is not None:
            centroid_steps.append(float(np.hypot(current[0] - previous[0], current[1] - previous[1])))
    area_changes = np.abs(np.diff(areas))
    trend = float(np.polyfit(np.arange(len(reference_iou)), reference_iou, 1)[0])
    return {
        "mask_coverage_mean": round(float(areas.mean()), 6),
        "mask_coverage_min": round(float(areas.min()), 6),
        "mask_coverage_max": round(float(areas.max()), 6),
        "mask_coverage_cv": round(float(areas.std() / max(areas.mean(), 1e-12)), 6),
        "absolute_area_change_mean": round(float(area_changes.mean()), 6),
        "absolute_area_change_p95": round(float(np.percentile(area_changes, 95)), 6),
        "consecutive_mask_iou_mean": round(float(consecutive_iou.mean()), 6),
        "consecutive_mask_iou_p05": round(float(np.percentile(consecutive_iou, 5)), 6),
        "green_reference_iou_mean": round(float(reference_iou.mean()), 6),
        "green_reference_iou_p05": round(float(np.percentile(reference_iou, 5)), 6),
        "green_reference_iou_min": round(float(reference_iou.min()), 6),
        "green_reference_iou_trend_per_frame": round(trend, 8),
        "green_reference_area_mae": round(float(np.abs(areas - reference_areas).mean()), 6),
        "centroid_step_px_mean": round(float(np.mean(centroid_steps)), 3),
        "centroid_step_px_p95": round(float(np.percentile(centroid_steps, 95)), 3),
        "component_count_max": max(component_counts),
        "empty_frames": int(sum(not mask.any() for mask in masks)),
        "full_frames": int(sum(mask.all() for mask in masks)),
        "green_soft_band_fraction_mean": round(float(np.mean(soft_fractions)), 6),
        "green_soft_band_containment_mean": round(
            float(np.mean(soft_band_containment)), 6
        ),
        "alpha_values": [0, 255],
        "temporal_metric_caveat": (
            "Raw consecutive IoU includes real pet motion. Green-screen comparisons use the "
            "existing chroma-key result as a proxy, not hand-labeled ground truth."
        ),
    }


def checkerboard(width: int, height: int, tile: int = 16) -> Image.Image:
    y, x = np.indices((height, width))
    pattern = ((x // tile + y // tile) % 2).astype(bool)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[pattern] = (218, 222, 218)
    rgb[~pattern] = (242, 244, 242)
    return Image.fromarray(rgb)


def make_contact_sheets(
    rgba_images: list[Image.Image], mask_images: list[Image.Image], output_dir: Path
) -> None:
    sample_count = min(8, len(rgba_images))
    indices = sorted(
        {
            round(index * (len(rgba_images) - 1) / max(1, sample_count - 1))
            for index in range(sample_count)
        }
    )
    thumb = 240
    label_height = 24
    rgba_sheet = Image.new("RGB", (thumb * 4, (thumb + label_height) * 2), "white")
    mask_sheet = Image.new("L", rgba_sheet.size, 255)
    rgba_draw = ImageDraw.Draw(rgba_sheet)
    mask_draw = ImageDraw.Draw(mask_sheet)
    for slot, frame_index in enumerate(indices):
        x = (slot % 4) * thumb
        y = (slot // 4) * (thumb + label_height)
        background = checkerboard(thumb, thumb)
        rgba = rgba_images[frame_index].resize((thumb, thumb), Image.Resampling.LANCZOS)
        background.paste(rgba, (0, 0), rgba)
        rgba_sheet.paste(background, (x, y))
        mask = mask_images[frame_index].resize((thumb, thumb), Image.Resampling.NEAREST)
        mask_sheet.paste(mask, (x, y))
        label = f"frame {frame_index:04d}"
        rgba_draw.text((x + 6, y + thumb + 4), label, fill=(24, 32, 28))
        mask_draw.text((x + 6, y + thumb + 4), label, fill=0)
    rgba_sheet.save(output_dir / "contact_sheet_rgba.png", compress_level=4)
    mask_sheet.save(output_dir / "contact_sheet_mask.png", compress_level=4)


def write_outputs(
    output_dir: Path,
    input_paths: list[Path],
    frames: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, Any]:
    rgba_dir = output_dir / "rgba"
    mask_dir = output_dir / "mask"
    rgba_dir.mkdir()
    mask_dir.mkdir()
    rgba_images: list[Image.Image] = []
    mask_images: list[Image.Image] = []
    digest = hashlib.sha256()
    rgb_mismatches: list[str] = []
    output_names: list[str] = []
    started = time.perf_counter()
    for input_path, rgb, mask in zip(input_paths, frames, masks, strict=True):
        output_name = input_path.with_suffix(".png").name
        alpha = mask.astype(np.uint8) * 255
        rgba_array = np.dstack([rgb, alpha])
        rgba_image = Image.fromarray(rgba_array)
        mask_image = Image.fromarray(alpha)
        rgba_path = rgba_dir / output_name
        mask_path = mask_dir / output_name
        rgba_image.save(rgba_path, compress_level=4)
        mask_image.save(mask_path, compress_level=4)
        with Image.open(rgba_path) as check:
            written = np.asarray(check.convert("RGBA"), dtype=np.uint8)
        if not np.array_equal(written[:, :, :3], rgb):
            rgb_mismatches.append(output_name)
        if not np.array_equal(written[:, :, 3], alpha):
            raise RuntimeError(f"written alpha does not match propagated mask: {rgba_path}")
        digest.update(output_name.encode("utf-8"))
        digest.update(rgba_path.read_bytes())
        rgba_images.append(rgba_image)
        mask_images.append(mask_image)
        output_names.append(output_name)
    make_contact_sheets(rgba_images, mask_images, output_dir)
    expected_names = [path.with_suffix(".png").name for path in input_paths]
    return {
        "rgba_directory": str(rgba_dir),
        "mask_directory": str(mask_dir),
        "frame_count": len(output_names),
        "output_names": output_names,
        "same_names_as_input": output_names == expected_names,
        "sequence_sha256": digest.hexdigest(),
        "rgb_policy": "Source RGB is retained byte-for-byte at every pixel.",
        "rgb_mismatch_count": len(rgb_mismatches),
        "rgb_mismatch_names": rgb_mismatches,
        "alpha_policy": "Binary SAM 2 logit > 0 mask mapped to alpha values 0 and 255.",
        "write_and_validate_seconds": round(time.perf_counter() - started, 6),
    }


def package_versions() -> dict[str, str]:
    names = (
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "opencv-python-headless",
        "hydra-core",
        "iopath",
        "tqdm",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    model = metrics["model"]
    timing = metrics["timing"]
    memory = timing["inference"]["peak_memory"]
    diagnostics_data = metrics["diagnostics"]
    output = metrics["output"]
    lines = [
        "# SAM 2.1 Small Pet Video Mask Propagation",
        "",
        f"Status: **{metrics['status']}**",
        "",
        "## Experiment",
        "",
        f"- Input: `{metrics['input']['directory']}`",
        f"- Frames: {metrics['input']['frame_count']} consecutive frames, "
        f"{metrics['input']['width']}x{metrics['input']['height']}",
        "- Prompt: existing green-screen subject mask from frame 0 only",
        "- Output: source RGB plus binary propagated SAM 2 mask as alpha",
        "",
        "## Model and license",
        "",
        f"- Model: {model['name']} (`{model['architecture']}`)",
        f"- Official source: {model['repository']} at `{model['commit']}`",
        f"- Checkpoint: `{Path(model['checkpoint_path']).name}` "
        f"({model['checkpoint_bytes']:,} bytes)",
        f"- Checkpoint SHA-256: `{model['checkpoint_sha256']}`",
        f"- License: {model['license']} for official code and checkpoints, verified "
        "against the vendored official README and LICENSE",
        "- Commercial note: retain the Apache-2.0 license/notice and review dependency "
        "licenses for the final distribution; this is an engineering record, not legal advice.",
        "",
        "## CUDA performance",
        "",
        f"- GPU: {metrics['runtime']['gpu']}",
        f"- Precision: {metrics['runtime']['precision']} (BF16 is not used on Turing)",
        f"- Model load: {timing['model_load']['wall_seconds']:.3f} s wall, "
        f"{timing['model_load']['cuda_elapsed_ms']:.1f} ms CUDA",
        f"- State init: {timing['inference']['init_state_wall_seconds']:.3f} s wall, "
        f"{timing['inference']['init_state_cuda_ms']:.1f} ms CUDA",
        f"- Frame-0 prompt: {timing['inference']['prompt_wall_seconds']:.3f} s wall, "
        f"{timing['inference']['prompt_cuda_ms']:.1f} ms CUDA",
        f"- Propagation: {timing['inference']['propagation_wall_seconds']:.3f} s wall, "
        f"{timing['inference']['propagation_cuda_ms_sum']:.1f} ms summed CUDA, "
        f"{timing['inference']['propagation_fps_wall']:.2f} fps",
        f"- Peak process VRAM: {memory['peak_allocated_mib']:.1f} MiB allocated, "
        f"{memory['peak_reserved_mib']:.1f} MiB reserved",
        "- CUDA values are elapsed between synchronized CUDA events; they include stream "
        "idle time caused by host work between those events.",
        "- The optional SAM 2 connected-component CUDA extension was not built in this "
        "Windows environment, so the official fallback skipped tiny-hole filling.",
        "",
        "## Temporal stability",
        "",
        f"- Consecutive mask IoU: mean {diagnostics_data['consecutive_mask_iou_mean']:.4f}, "
        f"p05 {diagnostics_data['consecutive_mask_iou_p05']:.4f}",
        f"- Mask-area CV: {diagnostics_data['mask_coverage_cv']:.4f}",
        f"- Green-reference IoU: mean {diagnostics_data['green_reference_iou_mean']:.4f}, "
        f"p05 {diagnostics_data['green_reference_iou_p05']:.4f}, "
        f"trend/frame {diagnostics_data['green_reference_iou_trend_per_frame']:.6f}",
        f"- Empty/full frames: {diagnostics_data['empty_frames']}/"
        f"{diagnostics_data['full_frames']}",
        "- Caveat: consecutive IoU includes actual motion; the per-frame green result is a "
        "proxy reference rather than hand-labeled ground truth.",
        "",
        "## Fur and alpha limitations",
        "",
        "- SAM 2 is a binary segmentation/tracking model, not an alpha-matting model. It "
        "cannot represent fractional transparency at individual fur strands.",
        f"- The green reference labels {diagnostics_data['green_soft_band_fraction_mean']:.4%} "
        "of pixels as a soft transition band; the binary SAM mask can only include or exclude "
        "those pixels opaquely.",
        "- The frame-0 alpha is thresholded before prompting, so low-opacity fur below the "
        "threshold is absent from the prompt and may not be recovered during propagation.",
        "- Source RGB is deliberately preserved. No green-spill removal or foreground color "
        "decontamination is performed, so retained edge pixels can still contain green color.",
        "- Fast or blurred fur motion can produce edge snapping even when the whole-pet mask "
        "remains temporally coherent.",
        "",
        "## Output validation",
        "",
        f"- RGBA frames: {output['frame_count']}",
        f"- Same names as inputs: {output['same_names_as_input']}",
        f"- RGB mismatch count: {output['rgb_mismatch_count']}",
        f"- Sequence SHA-256: `{output['sequence_sha256']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_started = time.perf_counter()
    output_dir: Path | None = None
    staging_dir: Path | None = None
    try:
        if not 0 <= args.mask_threshold <= 255:
            raise ValueError("mask threshold must be in [0, 255]")
        model_meta = verify_model_assets()
        input_paths = select_frames(args.input_dir, args.frame_offset, args.frames)
        frames = load_rgb_frames(input_paths)
        output_dir = prepare_output_dir(args.output_dir, args.overwrite)
        staging_dir = output_dir / ".sam2_numeric_frames"
        staging_meta = stage_frames(input_paths, staging_dir)

        green_started = time.perf_counter()
        green_alphas, green_masks, first_green_rgba, initializer_meta = green_screen_masks(
            frames, args.mask_threshold
        )
        initializer_seconds = time.perf_counter() - green_started
        save_initializer(
            output_dir,
            frames[0],
            first_green_rgba,
            green_alphas[0],
            green_masks[0],
            initializer_meta,
        )

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this experiment")
        capability = torch.cuda.get_device_capability(0)
        if capability != (7, 5):
            warnings.warn(f"experiment was requested for RTX 2080 Ti; found capability {capability}")

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            predictor, model_load_meta = load_predictor(torch, args.precision)
            masks, inference_meta = run_propagation(
                torch=torch,
                predictor=predictor,
                staging_dir=staging_dir,
                prompt_mask=green_masks[0],
                frame_count=args.frames,
                object_id=args.object_id,
                precision=args.precision,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
        warning_messages = list(dict.fromkeys(str(item.message) for item in caught_warnings))

        output_meta = write_outputs(output_dir, input_paths, frames, masks)
        diagnostics_meta = diagnostics(masks, green_masks, green_alphas)
        if not args.keep_staging:
            shutil.rmtree(staging_dir)
            staging_meta["removed_after_run"] = True
        else:
            staging_meta["removed_after_run"] = False

        height, width = frames[0].shape[:2]
        input_digest = hashlib.sha256()
        for path in input_paths:
            input_digest.update(path.name.encode("utf-8"))
            input_digest.update(path.read_bytes())
        runtime = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(capability),
            "precision": args.precision,
            "autocast_dtype": "float16" if args.precision == "fp16" else "float32",
            "bf16_used": False,
            "optional_sam2_cuda_extension_built": False,
            "warning_messages": warning_messages,
        }
        metrics = {
            "status": "succeeded",
            "provider": "official_meta_sam2_1_video",
            "model": model_meta,
            "input": {
                "directory": str(resolved(args.input_dir)),
                "frame_offset": args.frame_offset,
                "frame_count": len(input_paths),
                "frame_names": [path.name for path in input_paths],
                "width": width,
                "height": height,
                "sequence_sha256": input_digest.hexdigest(),
                "staging": staging_meta,
            },
            "initializer": initializer_meta,
            "runtime": runtime,
            "timing": {
                "cuda_timing_method": (
                    "Synchronized CUDA-event elapsed time; includes stream idle time from "
                    "host work between events. Propagation is timed per yielded frame."
                ),
                "initializer_seconds": round(initializer_seconds, 6),
                "model_load": model_load_meta,
                "inference": inference_meta,
                "end_to_end_seconds": round(time.perf_counter() - run_started, 6),
            },
            "diagnostics": diagnostics_meta,
            "output": output_meta,
            "limitations": [
                "SAM 2.1 emits a binary semantic mask, not a fractional alpha matte.",
                "Fine or semi-transparent fur is either opaque or absent after thresholding.",
                "Original RGB is retained without green-spill removal or color decontamination.",
                "Temporal metrics use chroma keying as a proxy reference, not manual ground truth.",
            ],
        }
        write_json(output_dir / "metrics.json", metrics)
        write_report(output_dir / "REPORT.md", metrics)
        print(json.dumps(metrics, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.perf_counter() - run_started, 6),
        }
        if output_dir is not None and output_dir.exists():
            if staging_dir is not None and staging_dir.exists():
                shutil.rmtree(staging_dir)
            write_json(output_dir / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=True, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

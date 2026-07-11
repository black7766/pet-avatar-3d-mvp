"""PaddleSeg PP-MattingV2 provider for the local matting benchmark.

The CLI follows the repository provider contract and writes 8-bit RGBA PNGs
with the same basenames as the input PNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
DEFAULT_MODEL_DIR = (
    REPO_ROOT
    / ".models"
    / "paddle_matting"
    / "ppmattingv2-stdc1-human_512"
)
MODEL_NAME = "PP-MattingV2-STDC1-human-512"
MODEL_SOURCE_URL = (
    "https://paddleseg.bj.bcebos.com/matting/models/deploy/"
    "ppmattingv2-stdc1-human_512.zip"
)
SUPPORTED_TRANSFORMS = ("LoadImages", "LimitShort", "ResizeToIntMult", "Normalize")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official PaddleSeg PP-MattingV2 and emit RGBA PNGs."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "gpu", "cpu"),
        default="auto",
        help="'cuda' and 'gpu' are aliases; explicit CUDA requests never silently fall back.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--gpu-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--max-short",
        type=int,
        help=(
            "Override LimitShort.max_short from deploy.yaml. "
            "Omit to retain the official model default."
        ),
    )
    parser.add_argument(
        "--min-short",
        type=int,
        help="Override LimitShort.min_short; deploy.yaml leaves it unset by default.",
    )
    parser.add_argument(
        "--resize-mult",
        type=int,
        help=(
            "Override ResizeToIntMult.mult_int from deploy.yaml. "
            "Omit to retain the official model default."
        ),
    )
    return parser.parse_args()


def collect_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not paths:
        raise FileNotFoundError(f"no PNG inputs found in: {input_dir}")
    return paths


def load_deploy_spec(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "deploy.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing PaddleSeg deploy config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle)
    deploy = root.get("Deploy") if isinstance(root, dict) else None
    if not isinstance(deploy, dict):
        raise ValueError(f"invalid Deploy section in: {config_path}")

    transform_types = tuple(item.get("type") for item in deploy.get("transforms", []))
    if transform_types != SUPPORTED_TRANSFORMS:
        raise ValueError(
            "unsupported deploy transform pipeline: "
            f"{transform_types}; expected {SUPPORTED_TRANSFORMS}"
        )

    model_path = model_dir / str(deploy.get("model", ""))
    params_path = model_dir / str(deploy.get("params", ""))
    for path in (model_path, params_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing inference model file: {path}")

    limit_short = deploy["transforms"][1]
    resize_mult = deploy["transforms"][2]
    normalize = deploy["transforms"][3]
    return {
        "config_path": config_path,
        "model_path": model_path,
        "params_path": params_path,
        "max_short": int(limit_short.get("max_short", 512)),
        "min_short": (
            int(limit_short["min_short"])
            if limit_short.get("min_short") is not None
            else None
        ),
        "mult_int": int(resize_mult.get("mult_int", 32)),
        "mean": tuple(float(value) for value in normalize.get("mean", (0.5, 0.5, 0.5))),
        "std": tuple(float(value) for value in normalize.get("std", (0.5, 0.5, 0.5))),
        "transforms": list(transform_types),
    }


def apply_transform_overrides(
    spec: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    effective = dict(spec)
    if args.max_short is not None:
        effective["max_short"] = args.max_short
    if args.min_short is not None:
        effective["min_short"] = args.min_short
    if args.resize_mult is not None:
        effective["mult_int"] = args.resize_mult

    for name in ("max_short", "min_short", "mult_int"):
        value = effective[name]
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        effective["max_short"] is not None
        and effective["min_short"] is not None
        and effective["min_short"] > effective["max_short"]
    ):
        raise ValueError("--min-short must not exceed --max-short")
    return effective


def preprocess_image(
    path: Path, spec: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = original_rgb
    restore_shapes: list[tuple[int, int]] = []

    # Match PaddleSeg Matting's LimitShort transform exactly.
    height, width = image.shape[:2]
    restore_shapes.append((height, width))
    short_edge = min(height, width)
    max_short = spec["max_short"]
    min_short = spec["min_short"]
    target_short = short_edge
    if max_short is not None and short_edge > max_short:
        target_short = max_short
    elif min_short is not None and short_edge < min_short:
        target_short = min_short
    if target_short != short_edge:
        scale = target_short / float(short_edge)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        image = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )

    # Match ResizeToIntMult: floor both dimensions to a multiple of 32.
    height, width = image.shape[:2]
    restore_shapes.append((height, width))
    mult_int = spec["mult_int"]
    resized_width = width - width % mult_int
    resized_height = height - height % mult_int
    if resized_width <= 0 or resized_height <= 0:
        raise ValueError(
            f"input {path} is too small for ResizeToIntMult({mult_int}): "
            f"{width}x{height}"
        )
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )

    image = image.astype(np.float32, copy=False) / 255.0
    mean = np.asarray(spec["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(spec["std"], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[np.newaxis, ...])
    return tensor, original_rgb, restore_shapes, (resized_height, resized_width)


def restore_alpha(alpha: np.ndarray, restore_shapes: list[tuple[int, int]]) -> np.ndarray:
    restored = np.asarray(alpha, dtype=np.float32).squeeze()
    if restored.ndim != 2:
        raise ValueError(f"expected a 2D alpha output, got shape {restored.shape}")
    for height, width in reversed(restore_shapes):
        restored = cv2.resize(restored, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(restored, 0.0, 1.0)


class PaddleMattingPredictor:
    def __init__(self, args: argparse.Namespace, spec: dict[str, Any]) -> None:
        import paddle
        from paddle.inference import Config, create_predictor

        requested = args.device
        has_cuda = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        if requested == "auto":
            use_cuda = has_cuda
        elif requested in ("cuda", "gpu"):
            if not has_cuda:
                raise RuntimeError(
                    "CUDA was explicitly requested, but PaddlePaddle cannot see a CUDA GPU. "
                    "Use --device cpu for an explicit fallback."
                )
            use_cuda = True
        else:
            use_cuda = False

        config = Config(str(spec["model_path"]), str(spec["params_path"]))
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_ir_optim(True)
        if use_cuda:
            config.enable_use_gpu(args.gpu_memory_mb, args.gpu_id)
            self.device = "cuda"
            self.memory_device = f"gpu:{args.gpu_id}"
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(args.cpu_threads)
            if args.enable_mkldnn:
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            self.device = "cpu"
            self.memory_device = None

        self.predictor = create_predictor(config)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        if self.input_names != ["img"]:
            raise ValueError(f"unexpected model inputs: {self.input_names}")
        if not self.output_names:
            raise ValueError("model has no outputs")
        self.input_handle = self.predictor.get_input_handle(self.input_names[0])
        self.output_handle = self.predictor.get_output_handle(self.output_names[0])
        self.paddle = paddle

    def reset_peak_memory_stats(self) -> None:
        if self.memory_device is not None:
            self.paddle.device.reset_peak_memory_stats(self.memory_device)

    def peak_memory(self) -> dict[str, float | None]:
        if self.memory_device is None:
            return {"peak_allocated_mb": None, "peak_reserved_mb": None}
        return {
            "peak_allocated_mb": round(
                self.paddle.device.max_memory_allocated(self.memory_device)
                / (1024 * 1024),
                3,
            ),
            "peak_reserved_mb": round(
                self.paddle.device.max_memory_reserved(self.memory_device)
                / (1024 * 1024),
                3,
            ),
        }

    def run(self, tensor: np.ndarray) -> np.ndarray:
        self.input_handle.reshape(tensor.shape)
        self.input_handle.copy_from_cpu(tensor)
        # Paddle 3.3 returns None on success and raises on inference failure.
        self.predictor.run()
        return self.output_handle.copy_to_cpu()

    def environment(self) -> dict[str, Any]:
        paddle = self.paddle
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "paddle": paddle.__version__,
            "compiled_with_cuda": bool(paddle.is_compiled_with_cuda()),
            "cuda_device_count": int(paddle.device.cuda.device_count()),
            "paddle_cuda": paddle.version.cuda(),
            "paddle_cudnn": paddle.version.cudnn(),
        }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise SystemExit("input and output directories must be different")

    paths = collect_inputs(input_dir)
    model_dir = args.model_dir.resolve()
    spec = apply_transform_overrides(load_deploy_spec(model_dir), args)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_started = time.perf_counter()
    predictor_started = time.perf_counter()
    predictor = PaddleMattingPredictor(args, spec)
    predictor_load_seconds = time.perf_counter() - predictor_started

    warmup_seconds = 0.0
    if args.warmup_runs:
        warmup_tensor, _, _, _ = preprocess_image(paths[0], spec)
        warmup_started = time.perf_counter()
        for _ in range(args.warmup_runs):
            predictor.run(warmup_tensor)
        warmup_seconds = time.perf_counter() - warmup_started

    predictor.reset_peak_memory_stats()

    records: list[dict[str, Any]] = []
    measured_started = time.perf_counter()
    for path in paths:
        frame_started = time.perf_counter()
        tensor, original_rgb, restore_shapes, inference_size = preprocess_image(path, spec)
        preprocessed_at = time.perf_counter()
        raw_alpha = predictor.run(tensor)
        inferred_at = time.perf_counter()
        alpha = restore_alpha(raw_alpha, restore_shapes)
        alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
        rgba = np.dstack((original_rgb, alpha_u8))
        rgba[alpha_u8 == 0, :3] = 0
        output_path = output_dir / path.name
        Image.fromarray(rgba, mode="RGBA").save(output_path)
        saved_at = time.perf_counter()

        with Image.open(output_path) as check:
            if check.mode != "RGBA" or check.size != (original_rgb.shape[1], original_rgb.shape[0]):
                raise RuntimeError(
                    f"invalid output {output_path}: mode={check.mode}, size={check.size}"
                )

        records.append(
            {
                "input": str(path),
                "output": str(output_path),
                "width": int(original_rgb.shape[1]),
                "height": int(original_rgb.shape[0]),
                "inference_width": int(inference_size[1]),
                "inference_height": int(inference_size[0]),
                "preprocess_ms": round((preprocessed_at - frame_started) * 1000.0, 3),
                "inference_ms": round((inferred_at - preprocessed_at) * 1000.0, 3),
                "postprocess_and_save_ms": round((saved_at - inferred_at) * 1000.0, 3),
                "total_ms": round((saved_at - frame_started) * 1000.0, 3),
                "alpha_min": round(float(alpha.min()), 6),
                "alpha_max": round(float(alpha.max()), 6),
                "alpha_mean": round(float(alpha.mean()), 6),
                "transparent_fraction": round(float(np.mean(alpha <= 0.05)), 6),
                "opaque_fraction": round(float(np.mean(alpha >= 0.95)), 6),
                "sha256": sha256_file(output_path),
            }
        )

    measured_seconds = time.perf_counter() - measured_started
    peak_memory = predictor.peak_memory()
    inference_ms = [float(record["inference_ms"]) for record in records]
    total_ms = [float(record["total_ms"]) for record in records]
    metrics = {
        "provider": "paddle_matting",
        "model": MODEL_NAME,
        "official_training_domain": "human portrait matting",
        "test_domain": "pet video frames (cross-domain, unsupported by official model card)",
        "device_requested": args.device,
        "device": predictor.device,
        "frames": len(records),
        "validated_rgba_outputs": len(records),
        "predictor_load_seconds": round(predictor_load_seconds, 4),
        "warmup_runs": args.warmup_runs,
        "warmup_seconds": round(warmup_seconds, 4),
        "measured_seconds": round(measured_seconds, 4),
        "mean_inference_ms": round(float(np.mean(inference_ms)), 3),
        "median_inference_ms": round(float(np.median(inference_ms)), 3),
        "p95_inference_ms": round(percentile(inference_ms, 95), 3),
        "mean_total_ms": round(float(np.mean(total_ms)), 3),
        "peak_vram_mb": peak_memory["peak_allocated_mb"],
        "peak_reserved_vram_mb": peak_memory["peak_reserved_mb"],
        "process_seconds_excluding_cli_import": round(time.perf_counter() - process_started, 4),
        "model_source": MODEL_SOURCE_URL,
        "model_license_note": (
            "The model archive contains no separate license file. It is distributed by the "
            "official PaddleSeg model zoo; PaddleSeg source is Apache-2.0."
        ),
        "model_files": {
            spec["model_path"].name: sha256_file(spec["model_path"]),
            spec["params_path"].name: sha256_file(spec["params_path"]),
            spec["config_path"].name: sha256_file(spec["config_path"]),
        },
        "deploy_transforms": spec["transforms"],
        "effective_transform_parameters": {
            "LimitShort": {
                "max_short": spec["max_short"],
                "min_short": spec["min_short"],
            },
            "ResizeToIntMult": {"mult_int": spec["mult_int"]},
            "Normalize": {"mean": spec["mean"], "std": spec["std"]},
        },
        "environment": predictor.environment(),
        "per_frame": records,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "provider": metrics["provider"],
                "model": metrics["model"],
                "device": metrics["device"],
                "frames": metrics["frames"],
                "mean_inference_ms": metrics["mean_inference_ms"],
                "mean_total_ms": metrics["mean_total_ms"],
                "metrics": str(metrics_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

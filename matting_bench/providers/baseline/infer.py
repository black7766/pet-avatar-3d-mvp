"""Adapter for the project's adaptive green-screen matte implementation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import poc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--foreground-score", type=float, default=poc.GREEN_MATTE_FOREGROUND_SCORE
    )
    parser.add_argument(
        "--border-quantile", type=float, default=poc.GREEN_MATTE_BORDER_QUANTILE
    )
    parser.add_argument("--alpha-gamma", type=float, default=poc.GREEN_MATTE_ALPHA_GAMMA)
    parser.add_argument(
        "--core-despill", type=float, default=poc.GREEN_CORE_DESPILL_STRENGTH
    )
    parser.add_argument(
        "--core-radius-ratio", type=float, default=poc.GREEN_CORE_DESPILL_RADIUS_RATIO
    )
    parser.add_argument(
        "--halo-strength", type=float, default=poc.GREEN_OPAQUE_HALO_STRENGTH
    )
    parser.add_argument(
        "--halo-profile", choices=("real", "cartoon", "none"), default="real"
    )
    parser.add_argument(
        "--edge-refine",
        action="store_true",
        help="anti-alias a narrow silhouette band and rebuild contaminated edge color",
    )
    parser.add_argument(
        "--temporal-refine",
        action="store_true",
        help="flow-gated clip-level alpha stabilization plus single-frame fragment removal",
    )
    parser.add_argument(
        "--temporal-flow-size",
        type=int,
        default=poc.GREEN_TEMPORAL_FLOW_SIZE,
        help="square optical-flow working resolution for temporal refinement",
    )
    args = parser.parse_args()

    if not 0.001 <= args.foreground_score <= 0.10:
        raise SystemExit("--foreground-score must be in [0.001, 0.10]")
    if not 0.0 <= args.border_quantile <= 0.10:
        raise SystemExit("--border-quantile must be in [0, 0.10]")
    if not 0.5 <= args.alpha_gamma <= 2.5:
        raise SystemExit("--alpha-gamma must be in [0.5, 2.5]")
    if not 0.0 <= args.core_despill <= 1.5:
        raise SystemExit("--core-despill must be in [0, 1.5]")
    if not 0.02 <= args.core_radius_ratio <= 0.40:
        raise SystemExit("--core-radius-ratio must be in [0.02, 0.40]")
    if not 0.0 <= args.halo_strength <= 1.5:
        raise SystemExit("--halo-strength must be in [0, 1.5]")
    if not 192 <= args.temporal_flow_size <= 640:
        raise SystemExit("--temporal-flow-size must be in [192, 640]")

    poc.GREEN_MATTE_FOREGROUND_SCORE = args.foreground_score
    poc.GREEN_MATTE_BORDER_QUANTILE = args.border_quantile
    poc.GREEN_MATTE_ALPHA_GAMMA = args.alpha_gamma
    poc.GREEN_CORE_DESPILL_STRENGTH = args.core_despill
    poc.GREEN_CORE_DESPILL_RADIUS_RATIO = args.core_radius_ratio
    poc.GREEN_OPAQUE_HALO_STRENGTH = args.halo_strength

    paths = sorted(args.input_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNGs in {args.input_dir}")
    frames = [Image.open(path).convert("RGB") for path in paths]
    started = time.perf_counter()
    profile = poc.profile_green_screen(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    times = []
    rgba_stack = []
    source_stack = []
    temporal_stats = None
    for path, frame in zip(paths, frames):
        frame_started = time.perf_counter()
        source_rgb = np.asarray(frame, dtype=np.float32) / 255.0
        result = poc.adaptive_green_matte_frame(frame, profile)
        rgba = np.asarray(result, dtype=np.float32) / 255.0
        if args.halo_profile == "none":
            clean_rgb, clean_alpha = rgba[:, :, :3], rgba[:, :, 3]
        else:
            clean_rgb, clean_alpha = poc.refine_reframed_halo(
                rgba[:, :, :3], rgba[:, :, 3], profile=args.halo_profile
            )
        if args.edge_refine:
            clean_rgb, clean_alpha = poc.refine_adaptive_edge(
                clean_rgb, clean_alpha, profile=args.halo_profile
            )
        clean_rgb[clean_alpha == 0.0] = 0.0
        packed = np.dstack((np.clip(clean_rgb, 0.0, 1.0), clean_alpha)).astype(
            np.float32
        )
        times.append(time.perf_counter() - frame_started)
        if args.temporal_refine:
            rgba_stack.append(packed)
            source_stack.append(source_rgb)
        else:
            result = Image.fromarray((packed * 255.0 + 0.5).astype(np.uint8), "RGBA")
            result.save(args.output_dir / path.name)
    if args.temporal_refine:
        temporal_started = time.perf_counter()
        rgba_stack, temporal_stats = poc.stabilize_alpha_temporal(
            rgba_stack,
            source_rgbs=source_stack,
            flow_size=args.temporal_flow_size,
        )
        temporal_stats["temporal_seconds"] = round(
            time.perf_counter() - temporal_started, 4
        )
        for path, packed in zip(paths, rgba_stack):
            result = Image.fromarray((packed * 255.0 + 0.5).astype(np.uint8), "RGBA")
            result.save(args.output_dir / path.name)
    elapsed = time.perf_counter() - started
    if args.temporal_refine:
        provider_name = "adaptive_green_temporal_v3"
    elif args.edge_refine:
        provider_name = "adaptive_green_edge_v2"
    else:
        provider_name = "adaptive_green_baseline"
    metrics = {
        "provider": provider_name,
        "device": "cpu",
        "frames": len(paths),
        "total_seconds": round(elapsed, 4),
        "mean_ms_per_frame": round(sum(times) * 1000 / len(times), 3),
        "parameters": {
            "foreground_score": args.foreground_score,
            "border_quantile": args.border_quantile,
            "alpha_gamma": args.alpha_gamma,
            "core_despill": args.core_despill,
            "core_radius_ratio": args.core_radius_ratio,
            "halo_strength": args.halo_strength,
            "halo_profile": args.halo_profile,
            "edge_refine": args.edge_refine,
            "temporal_refine": args.temporal_refine,
            "temporal_flow_size": args.temporal_flow_size,
        },
        "temporal": temporal_stats,
        "profile": {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in profile.items()
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

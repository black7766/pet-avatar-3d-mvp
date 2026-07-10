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

from poc import (  # noqa: E402
    adaptive_green_matte_frame,
    profile_green_screen,
    refine_reframed_halo,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNGs in {args.input_dir}")
    frames = [Image.open(path).convert("RGB") for path in paths]
    started = time.perf_counter()
    profile = profile_green_screen(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    times = []
    for path, frame in zip(paths, frames):
        frame_started = time.perf_counter()
        result = adaptive_green_matte_frame(frame, profile)
        rgba = np.asarray(result, dtype=np.float32) / 255.0
        clean_rgb, clean_alpha = refine_reframed_halo(
            rgba[:, :, :3], rgba[:, :, 3], profile="real"
        )
        clean_rgb[clean_alpha == 0.0] = 0.0
        packed = np.dstack((np.clip(clean_rgb, 0.0, 1.0), clean_alpha))
        result = Image.fromarray((packed * 255.0 + 0.5).astype(np.uint8), "RGBA")
        result.save(args.output_dir / path.name)
        times.append(time.perf_counter() - frame_started)
    elapsed = time.perf_counter() - started
    metrics = {
        "provider": "adaptive_green_baseline",
        "device": "cpu",
        "frames": len(paths),
        "total_seconds": round(elapsed, 4),
        "mean_ms_per_frame": round(sum(times) * 1000 / len(times), 3),
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

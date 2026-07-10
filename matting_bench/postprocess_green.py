"""Apply the same green-screen color decontamination to any model-provided alpha."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poc import refine_reframed_halo  # noqa: E402


def green_score(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (green - np.maximum(red, blue)) / np.maximum(green, 1.0 / 255.0)


def decontaminate(
    rgb: np.ndarray, alpha: np.ndarray, refine_alpha: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    height, width = alpha.shape
    score = green_score(rgb)
    band = max(8, min(height, width) // 50)
    border = np.zeros_like(alpha, dtype=bool)
    border[:band] = True
    border[-band:] = True
    border[:, :band] = True
    border[:, -band:] = True
    key_pixels = border & (score > 0.30) & (rgb[:, :, 1] > 0.14)
    key_rgb = np.median(rgb[key_pixels], axis=0) if key_pixels.any() else np.array((0, 1, 0))

    background_candidate = alpha < 0.995
    count, labels = cv2.connectedComponents(background_candidate.astype(np.uint8), 8)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    reachable = np.isin(labels, border_labels)

    safe_alpha = np.maximum(alpha[:, :, None], 0.055)
    recovered = np.clip(
        (rgb - (1.0 - alpha[:, :, None]) * key_rgb[None, None, :]) / safe_alpha,
        0.0,
        1.0,
    )
    recover_weight = (
        np.clip((1.0 - alpha) * 1.8, 0.0, 1.0) * reachable
    )[:, :, None]
    clean = rgb * (1.0 - recover_weight) + recovered * recover_weight

    edge = reachable & (alpha > 0.018) & (alpha < 0.995) & (score > 0.02)
    core = alpha > 0.985
    if edge.any() and core.any():
        _, labels = cv2.distanceTransformWithLabels(
            (~core).astype(np.uint8),
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.float32)
        lut[labels[core]] = rgb[core]
        nearest = lut[labels]
        weight = np.clip((0.985 - alpha) / 0.72, 0.0, 1.0)[:, :, None]
        clean[edge] = clean[edge] * (1.0 - weight[edge]) + nearest[edge] * weight[edge]

    max_rb = np.maximum(clean[:, :, 0], clean[:, :, 2])
    clean[:, :, 1][edge] = np.minimum(clean[:, :, 1][edge], max_rb[edge] + 0.006)
    refined_rgb, refined_alpha = refine_reframed_halo(clean, alpha, profile="real")
    silhouette_zone = cv2.dilate(
        reachable.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1
    ).astype(bool)
    refined_rgb[~silhouette_zone] = clean[~silhouette_zone]
    refined_alpha[~silhouette_zone] = alpha[~silhouette_zone]
    if not refine_alpha:
        refined_alpha = alpha.copy()
    clean = refined_rgb
    clean[refined_alpha == 0.0] = 0.0
    return np.clip(clean, 0.0, 1.0), refined_alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--rgba-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--refine-alpha",
        action="store_true",
        help="allow the green-edge pass to feather alpha; keep model alpha by default",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed = 0
    for source_path in sorted(args.source_dir.glob("*.png")):
        rgba_path = args.rgba_dir / source_path.name
        if not rgba_path.exists():
            continue
        rgb = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32) / 255.0
        model_rgba = np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.float32) / 255.0
        clean, alpha = decontaminate(
            rgb, model_rgba[:, :, 3], refine_alpha=args.refine_alpha
        )
        packed = np.dstack((clean, alpha))
        Image.fromarray((packed * 255.0 + 0.5).astype(np.uint8), "RGBA").save(
            args.output_dir / source_path.name
        )
        processed += 1
    elapsed = time.perf_counter() - started
    metrics = {
        "provider": "green_decontamination_postprocess",
        "device": "cpu",
        "frames": processed,
        "total_seconds": round(elapsed, 4),
        "mean_ms_per_frame": round(elapsed * 1000 / max(1, processed), 3),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

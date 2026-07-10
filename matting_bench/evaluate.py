"""Evaluate RGBA matting outputs without requiring a hand-painted alpha ground truth.

The benchmark source uses a controlled green screen. Metrics therefore use only
high-confidence green background and non-green foreground pixels, plus temporal
consistency after optical-flow compensation. Boundary quality still requires visual QA.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_provider(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("provider must be NAME=OUTPUT_DIR")
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path).resolve()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def green_score(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (green - np.maximum(red, blue)) / np.maximum(green, 1.0 / 255.0)


def confident_regions(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = green_score(rgb)
    green = rgb[:, :, 1]
    background_candidate = (score > 0.32) & (green > 0.18)

    count, labels = cv2.connectedComponents(background_candidate.astype(np.uint8), 8)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border_labels = border_labels[border_labels != 0]
    confident_bg = np.isin(labels, border_labels) & (score > 0.42)
    confident_fg = (score < 0.025) | (green < 0.10)
    return confident_bg, confident_fg


def component_metrics(alpha: np.ndarray) -> tuple[float, int]:
    mask = alpha > 0.08
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(areas.max())
    fragments = int(areas.sum() - main)
    return fragments / max(1, int(areas.sum())), int((areas >= 4).sum() - 1)


def frame_metrics(source: np.ndarray, output: np.ndarray) -> dict[str, float]:
    if source.shape[:2] != output.shape[:2]:
        raise ValueError(f"shape mismatch: source={source.shape} output={output.shape}")
    alpha = output[:, :, 3]
    confident_bg, confident_fg = confident_regions(source)
    bg_leak = float(alpha[confident_bg].mean()) if confident_bg.any() else 1.0
    fg_loss = float((1.0 - alpha[confident_fg]).mean()) if confident_fg.any() else 1.0
    score = green_score(source)
    soft_edge = (alpha > 0.02) & (alpha < 0.98)
    output_rgb = output[:, :, :3]
    edge_green = np.maximum(
        0.0,
        output_rgb[:, :, 1] - (output_rgb[:, :, 0] + output_rgb[:, :, 2]) * 0.5,
    )
    if soft_edge.any():
        green_fringe = float((edge_green[soft_edge] * alpha[soft_edge]).mean())
    else:
        green_fringe = 0.0
    opaque_green = float(((alpha > 0.50) & (score > 0.25)).mean())
    fragment_ratio, fragments = component_metrics(alpha)
    return {
        "pseudo_mae": (bg_leak + fg_loss) * 0.5,
        "background_alpha_mean": bg_leak,
        "foreground_loss_mean": fg_loss,
        "green_fringe": green_fringe,
        "opaque_green_leak_pct": opaque_green * 100.0,
        "soft_alpha_pct": float(soft_edge.mean() * 100.0),
        "coverage_pct": float((alpha > 0.02).mean() * 100.0),
        "fragment_pct": fragment_ratio * 100.0,
        "fragment_count": float(fragments),
    }


def temporal_error(sources: list[np.ndarray], alphas: list[np.ndarray]) -> float | None:
    if len(sources) < 2:
        return None
    errors = []
    for previous_rgb, current_rgb, previous_alpha, current_alpha in zip(
        sources, sources[1:], alphas, alphas[1:]
    ):
        size = (480, 480)
        previous_gray = cv2.cvtColor(
            cv2.resize((previous_rgb * 255).astype(np.uint8), size), cv2.COLOR_RGB2GRAY
        )
        current_gray = cv2.cvtColor(
            cv2.resize((current_rgb * 255).astype(np.uint8), size), cv2.COLOR_RGB2GRAY
        )
        previous_small = cv2.resize(previous_alpha, size, interpolation=cv2.INTER_AREA)
        current_small = cv2.resize(current_alpha, size, interpolation=cv2.INTER_AREA)
        backward_flow = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            0.5,
            3,
            21,
            3,
            5,
            1.2,
            0,
        )
        grid_x, grid_y = np.meshgrid(
            np.arange(size[0], dtype=np.float32),
            np.arange(size[1], dtype=np.float32),
        )
        warped = cv2.remap(
            previous_small,
            grid_x + backward_flow[:, :, 0],
            grid_y + backward_flow[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        union = (warped > 0.02) | (current_small > 0.02)
        if union.any():
            errors.append(float(np.abs(warped - current_small)[union].mean()))
    return float(np.mean(errors)) if errors else None


def summarize(values: list[dict[str, float]]) -> dict[str, float]:
    keys = values[0].keys()
    return {key: float(np.mean([item[key] for item in values])) for key in keys}


def evaluate_provider(source_dir: Path, output_dir: Path) -> dict:
    paths = sorted(source_dir.glob("*.png"))
    frame_values = []
    sources = []
    alphas = []
    missing = []
    for source_path in paths:
        output_path = output_dir / source_path.name
        if not output_path.exists():
            missing.append(source_path.name)
            continue
        source = load_rgb(source_path)
        output = load_rgba(output_path)
        frame_values.append(frame_metrics(source, output))
        sources.append(source)
        alphas.append(output[:, :, 3])
    consecutive = all(re.fullmatch(r"f_\d+", path.stem) for path in paths)
    result = {
        "frames": len(frame_values),
        "missing": missing,
        "mean": summarize(frame_values) if frame_values else {},
        "temporal_alpha_mae": temporal_error(sources, alphas) if consecutive else None,
    }
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        try:
            result["runtime"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            result["runtime"] = {"error": "invalid metrics.json"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--provider", type=parse_provider, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {"source_dir": str(args.source_dir.resolve()), "providers": {}}
    for name, output_dir in args.provider:
        report["providers"][name] = evaluate_provider(args.source_dir.resolve(), output_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

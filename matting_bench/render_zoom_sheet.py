"""Render synchronized edge crops on checker, white, black, and alpha views."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_provider(value: str) -> tuple[str, Path]:
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path)


def checker(size: tuple[int, int], block: int = 20) -> Image.Image:
    image = Image.new("RGBA", size, (240, 242, 240, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], block):
        for x in range(0, size[0], block):
            if (x // block + y // block) % 2:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill=(205, 211, 206, 255))
    return image


def alpha_bbox(path: Path) -> tuple[int, int, int, int]:
    alpha = np.asarray(Image.open(path).convert("RGBA"))[:, :, 3]
    ys, xs = np.where(alpha > 5)
    if not len(xs):
        raise ValueError(f"empty alpha: {path}")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def select_region(bbox: tuple[int, int, int, int], region: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    if region == "head":
        return x0 + int(width * 0.12), y0, x0 + int(width * 0.88), y0 + int(height * 0.48)
    if region == "lower":
        return x0, y0 + int(height * 0.46), x1, y1
    if region == "left":
        return x0, y0 + int(height * 0.08), x0 + int(width * 0.48), y0 + int(height * 0.92)
    if region == "right":
        return x0 + int(width * 0.52), y0 + int(height * 0.08), x1, y0 + int(height * 0.92)
    return bbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--provider", type=parse_provider, action="append", required=True)
    parser.add_argument("--region", choices=("full", "head", "lower", "left", "right"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", type=int, default=360)
    args = parser.parse_args()

    bbox = alpha_bbox(args.reference_dir / args.frame)
    crop = select_region(bbox, args.region)
    padding = max(4, int(max(crop[2] - crop[0], crop[3] - crop[1]) * 0.04))
    sample = Image.open(args.reference_dir / args.frame)
    crop = (
        max(0, crop[0] - padding),
        max(0, crop[1] - padding),
        min(sample.width, crop[2] + padding),
        min(sample.height, crop[3] + padding),
    )
    header = 30
    columns = ("checker", "white", "black", "alpha")
    sheet = Image.new("RGB", (args.cell * 4, (args.cell + header) * len(args.provider)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (name, output_dir) in enumerate(args.provider):
        rgba = Image.open(output_dir / args.frame).convert("RGBA").crop(crop)
        rgba.thumbnail((args.cell, args.cell), Image.Resampling.LANCZOS)
        alpha = rgba.getchannel("A").convert("RGB")
        for col, background in enumerate(columns):
            x = col * args.cell
            y = row * (args.cell + header)
            draw.rectangle((x, y, x + args.cell, y + header), fill=(29, 37, 33))
            draw.text((x + 7, y + 7), name if col == 0 else background, fill="white")
            if background == "alpha":
                canvas = Image.new("RGB", (args.cell, args.cell), "black")
                canvas.paste(alpha, ((args.cell - alpha.width) // 2, (args.cell - alpha.height) // 2))
            else:
                if background == "checker":
                    canvas_rgba = checker((args.cell, args.cell))
                else:
                    color = (255, 255, 255, 255) if background == "white" else (12, 14, 13, 255)
                    canvas_rgba = Image.new("RGBA", (args.cell, args.cell), color)
                canvas_rgba.alpha_composite(
                    rgba,
                    ((args.cell - rgba.width) // 2, (args.cell - rgba.height) // 2),
                )
                canvas = canvas_rgba.convert("RGB")
            sheet.paste(canvas, (x, y + header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()

"""Render all provider outputs on the same checkerboard for visual review."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_provider(value: str) -> tuple[str, Path]:
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path)


def checker(size: tuple[int, int], block: int = 18) -> Image.Image:
    image = Image.new("RGB", size, (238, 240, 238))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], block):
        for x in range(0, size[0], block):
            if (x // block + y // block) % 2:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill=(214, 219, 215))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--provider", type=parse_provider, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", type=int, default=220)
    args = parser.parse_args()

    sources = sorted(args.source_dir.glob("*.png"))
    providers = args.provider
    header = 32
    sheet = Image.new(
        "RGB",
        (args.cell * len(sources), (args.cell + header) * len(providers)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, (name, output_dir) in enumerate(providers):
        for col, source_path in enumerate(sources):
            x = col * args.cell
            y = row * (args.cell + header)
            draw.rectangle((x, y, x + args.cell, y + header), fill=(31, 40, 35))
            label = name if col == 0 else source_path.stem
            draw.text((x + 7, y + 8), label, fill="white")
            rgba_path = output_dir / source_path.name
            canvas = checker((args.cell, args.cell)).convert("RGBA")
            if rgba_path.exists():
                rgba = Image.open(rgba_path).convert("RGBA")
                rgba.thumbnail((args.cell, args.cell), Image.Resampling.LANCZOS)
                px = (args.cell - rgba.width) // 2
                py = (args.cell - rgba.height) // 2
                canvas.alpha_composite(rgba, (px, py))
            else:
                missing = ImageDraw.Draw(canvas)
                missing.text((12, 12), "missing", fill=(160, 30, 30, 255))
            sheet.paste(canvas.convert("RGB"), (x, y + header))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    print(args.output)


if __name__ == "__main__":
    main()

"""Create a fixed-size benchmark view without modifying decoded source frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(args.input_dir.glob("*.png")):
        image = Image.open(path).convert("RGB")
        image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
        image.save(args.output_dir / path.name, compress_level=2)
        count += 1
    print(f"{args.output_dir} ({count} frames)")


if __name__ == "__main__":
    main()

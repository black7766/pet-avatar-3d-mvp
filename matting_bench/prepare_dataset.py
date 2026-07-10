"""Decode a fixed pet matting benchmark from generated green-screen videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image


CLIPS = ("idle", "fast_walk", "sleep")


def decode_video(video: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if not cv2.imwrite(str(output / f"f_{count:04d}.png"), frame):
            raise RuntimeError(f"failed to write frame {count} from {video}")
        count += 1
    capture.release()
    if count < 2:
        raise RuntimeError(f"decoded only {count} frames from {video}")
    return {"frames": count, "fps": fps, "width": width, "height": height}


def link_or_copy(source: Path, target: Path) -> None:
    import shutil

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--natural",
        action="append",
        default=[],
        metavar="NAME=IMAGE",
        help="optional natural-background still image",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    manifest = {
        "source": str(source),
        "clips": {},
        "smoke": [],
        "temporal": [],
        "natural": [],
    }
    for clip in CLIPS:
        video = source / f"raw_{clip}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        clip_dir = output / "full" / clip
        info = decode_video(video, clip_dir)
        manifest["clips"][clip] = {"video": str(video), **info}
        indices = sorted({0, info["frames"] // 2, info["frames"] - 1})
        for index in indices:
            name = f"{clip}__f_{index:04d}.png"
            link_or_copy(clip_dir / f"f_{index:04d}.png", output / "smoke" / name)
            manifest["smoke"].append({"clip": clip, "frame": index, "file": name})
        if clip == "fast_walk":
            temporal_count = min(24, info["frames"])
            for index in range(temporal_count):
                name = f"f_{index:04d}.png"
                link_or_copy(clip_dir / name, output / "temporal_fast_walk_24" / name)
                manifest["temporal"].append(
                    {"clip": clip, "frame": index, "file": name}
                )

    for item in args.natural:
        if "=" not in item:
            raise ValueError(f"natural input must be NAME=IMAGE: {item}")
        name, raw_path = item.split("=", 1)
        natural_path = Path(raw_path).resolve()
        target = output / "natural" / f"{name.strip()}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.open(natural_path).convert("RGB").save(target, compress_level=2)
        manifest["natural"].append(
            {"name": name.strip(), "source": str(natural_path), "file": target.name}
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output / "manifest.json")


if __name__ == "__main__":
    main()

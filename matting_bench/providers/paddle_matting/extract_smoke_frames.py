"""Extract first, middle, and last frames from the three raw benchmark videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2


CLIPS = ("idle", "fast_walk", "sleep")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing raw_idle.mp4, raw_fast_walk.mp4, and raw_sleep.mp4.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def extract_clip(video_path: Path, output_dir: Path, clip: str) -> dict[str, object]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if declared_frames <= 0:
        capture.release()
        raise RuntimeError(f"video reports no frames: {video_path}")
    selected = {0, (declared_frames - 1) // 2, declared_frames - 1}
    outputs: list[dict[str, object]] = []
    decoded_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = decoded_frames
        decoded_frames += 1
        if frame_index not in selected:
            continue
        output_path = output_dir / f"{clip}__f_{frame_index:04d}.png"
        if not cv2.imwrite(str(output_path), frame):
            capture.release()
            raise RuntimeError(f"failed to write extracted frame: {output_path}")
        outputs.append(
            {
                "frame": frame_index,
                "file": output_path.name,
                "sha256": sha256_file(output_path),
            }
        )
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if decoded_frames != declared_frames:
        raise RuntimeError(
            f"decoded {decoded_frames} frames but container declared {declared_frames}: {video_path}"
        )
    if len(outputs) != 3:
        raise RuntimeError(f"expected 3 extracted frames, got {len(outputs)}: {video_path}")
    return {
        "clip": clip,
        "video": str(video_path.resolve()),
        "video_sha256": sha256_file(video_path),
        "frames": decoded_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "selected": outputs,
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    clips = [
        extract_clip(source_dir / f"raw_{clip}.mp4", output_dir, clip)
        for clip in CLIPS
    ]
    manifest = {
        "source_dir": str(source_dir),
        "selection": "first, middle, last",
        "clips": clips,
        "total_extracted_frames": sum(len(item["selected"]) for item in clips),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    manifest_path = output_dir / "extraction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

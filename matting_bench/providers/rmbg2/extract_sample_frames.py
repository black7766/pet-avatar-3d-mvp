#!/usr/bin/env python3
"""Extract three deterministic frames from each real_after raw video."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


VIDEO_NAMES = ("raw_fast_walk.mp4", "raw_idle.mp4", "raw_sleep.mp4")
SAMPLE_RATIOS = (0.2, 0.5, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_indices(frame_count: int) -> list[int]:
    if frame_count < 3:
        raise RuntimeError(f"Video has fewer than three frames: {frame_count}")
    indices = [round((frame_count - 1) * ratio) for ratio in SAMPLE_RATIOS]
    if len(set(indices)) != 3:
        indices = [0, frame_count // 2, frame_count - 1]
    return indices


def extract_video(source: Path, output_dir: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wanted = frame_indices(frame_count)
    wanted_set = set(wanted)
    outputs: list[dict[str, Any]] = []
    segment = source.stem.removeprefix("raw_")

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted_set:
            output = output_dir / f"{segment}_f{index:04d}.png"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Could not write frame: {output}")
            outputs.append(
                {
                    "frame_index": index,
                    "timestamp_s": round(index / fps, 4) if fps > 0 else None,
                    "file": output.name,
                    "sha256": sha256_file(output),
                }
            )
        index += 1

    capture.release()
    if [item["frame_index"] for item in outputs] != wanted:
        raise RuntimeError(
            f"Expected frames {wanted} from {source.name}, got "
            f"{[item['frame_index'] for item in outputs]}"
        )

    return {
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "frame_count": frame_count,
        "fps": round(fps, 4),
        "width": width,
        "height": height,
        "duration_s": round(frame_count / fps, 4) if fps > 0 else None,
        "samples": outputs,
    }


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    for name in VIDEO_NAMES:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing source video: {source}")
        videos.append(extract_video(source, output_dir))

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_ratios": list(SAMPLE_RATIOS),
        "videos": videos,
    }
    manifest_path = output_dir / "frame_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

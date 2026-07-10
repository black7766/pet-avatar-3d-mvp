from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


VIDEO_NAMES = ("raw_idle.mp4", "raw_sleep.mp4", "raw_fast_walk.mp4")
FRACTIONS = (0.25, 0.5, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract three deterministic interior frames from each benchmark video."
    )
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []

    for video_name in VIDEO_NAMES:
        video_path = video_dir / video_name
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < 3 or fps <= 0:
            capture.release()
            raise RuntimeError(
                f"Invalid video metadata for {video_path}: frames={frame_count}, fps={fps}"
            )

        for fraction in FRACTIONS:
            frame_index = round((frame_count - 1) * fraction)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(
                    f"Could not read frame {frame_index} from {video_path}"
                )

            output_name = f"{video_path.stem.removeprefix('raw_')}_f{frame_index:04d}.png"
            output_path = output_dir / output_name
            if not cv2.imwrite(str(output_path), frame):
                capture.release()
                raise RuntimeError(f"Could not write PNG: {output_path}")

            samples.append(
                {
                    "source_video": str(video_path),
                    "source_frame_index_zero_based": frame_index,
                    "source_time_seconds": frame_index / fps,
                    "fraction": fraction,
                    "video_frame_count": frame_count,
                    "video_fps": fps,
                    "width": width,
                    "height": height,
                    "output": str(output_path),
                }
            )
        capture.release()

    manifest = {
        "sampling": "round((frame_count - 1) * fraction)",
        "fractions": list(FRACTIONS),
        "sample_count": len(samples),
        "samples": samples,
    }
    manifest_path = output_dir / "samples.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

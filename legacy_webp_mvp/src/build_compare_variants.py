from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
KEYFRAMES = ASSETS / "keyframes"
COMPARE = ASSETS / "compare"
SIZE = 520

ACTION_CONFIG = {
    "happy": {
        "label": "开心",
        "sequence": "pingpong",
        "target_duration_ms": 1550,
    },
    "eat": {
        "label": "进食",
        "sequence": "straight",
        "target_duration_ms": 880,
    },
}

VARIANTS = [
    {
        "tier": "boost",
        "label": "增强帧",
        "inbetweens": 7,
        "note": "大幅补帧，动作时长保持不变，目标是先接近 45fps 体感。",
    },
    {
        "tier": "max60",
        "label": "接近 60fps",
        "inbetweens": 9,
        "note": "把动作压到接近浏览器 60Hz 上限，适合判断本地插帧的实用极限。",
    },
    {
        "tier": "ultra",
        "label": "超采样 60fps",
        "inbetweens": 15,
        "playback_fps": 60,
        "note": "生成更多中间帧，但按 60fps 播放全部帧；动作会更慢，主要看细腻度上限。",
    },
]


def load_rgba(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


def pil_from_rgba(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGBA")


def rgba_to_flow_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = frame[:, :, :3].astype(np.float32)
    alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full_like(rgb, 240.0)
    comp = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(comp, 0, 255).astype(np.uint8)


def remap_frame(frame: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[:, :, 0] * scale
    map_y = grid_y - flow[:, :, 1] * scale
    channels = []
    for channel in range(4):
        warped = cv2.remap(
            frame[:, :, channel],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        channels.append(warped)
    return np.stack(channels, axis=2)


def interpolate_pair(a: np.ndarray, b: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps <= 0:
        return []

    rgb_a = rgba_to_flow_rgb(a)
    rgb_b = rgba_to_flow_rgb(b)
    gray_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
    flow_ab = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 4, 31, 4, 7, 1.5, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, 0.5, 4, 31, 4, 7, 1.5, 0)

    frames: list[np.ndarray] = []
    for step in range(1, steps + 1):
        t = step / (steps + 1)
        wa = remap_frame(a, flow_ab, t).astype(np.float32)
        wb = remap_frame(b, flow_ba, 1.0 - t).astype(np.float32)
        blended = wa * (1.0 - t) + wb * t
        frames.append(np.clip(blended, 0, 255).astype(np.uint8))
    return frames


def load_keyframes(action: str) -> list[np.ndarray]:
    paths = [KEYFRAMES / action / f"frame_{action}_{index:02d}.png" for index in range(1, 7)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing keyframes for {action}: {missing}")
    return [load_rgba(path) for path in paths]


def action_sequence(action: str, keyframes: list[np.ndarray]) -> list[np.ndarray]:
    mode = ACTION_CONFIG[action]["sequence"]
    if mode == "pingpong":
        return keyframes + keyframes[-2:0:-1]
    if mode == "straight":
        return keyframes
    raise ValueError(f"Unsupported sequence mode: {mode}")


def build_frames(action: str, inbetweens: int) -> list[np.ndarray]:
    sequence = action_sequence(action, load_keyframes(action))
    output: list[np.ndarray] = []
    for index in range(len(sequence) - 1):
        current_frame = sequence[index]
        next_frame = sequence[index + 1]
        output.append(current_frame)
        output.extend(interpolate_pair(current_frame, next_frame, inbetweens))
    output.append(sequence[-1])
    return output


def write_frames(out_dir: Path, frames: list[np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    for index, frame in enumerate(frames, 1):
        pil_from_rgba(frame).save(out_dir / f"frame_{index:02d}.png")


def write_webp(path: Path, frames: list[np.ndarray], frame_ms: int) -> int:
    images = [pil_from_rgba(frame).copy() for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
        lossless=False,
        quality=72,
        method=0,
        minimize_size=False,
    )
    return path.stat().st_size


def make_contact_sheet(manifest: list[dict[str, object]]) -> None:
    thumb = 132
    cols = 6
    rows = len(manifest)
    label_h = 34
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "#f3eee7")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row, item in enumerate(manifest):
        frames_dir = COMPARE / str(item["id"])
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        if not frame_paths:
            continue
        picks = np.linspace(0, len(frame_paths) - 1, cols).round().astype(int)
        y0 = row * (thumb + label_h)
        draw.text((8, y0 + 8), f'{item["id"]}  {item["frame_count"]}f / {item["fps"]}fps', fill="#2b211c", font=font)
        for col, frame_index in enumerate(picks):
            image = Image.open(frame_paths[int(frame_index)]).convert("RGBA")
            tile = Image.new("RGBA", (thumb, thumb), (255, 255, 255, 0))
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            tile.alpha_composite(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
            white = Image.new("RGB", (thumb, thumb), "#fbfaf7")
            white.paste(tile, mask=tile.split()[3])
            sheet.paste(white, (col * thumb, y0 + label_h))

    sheet.save(COMPARE / "compare_contact_sheet.jpg", quality=88)


def clear_compare_dir() -> None:
    COMPARE.mkdir(parents=True, exist_ok=True)
    resolved = COMPARE.resolve()
    if ASSETS.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
    for child in COMPARE.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    clear_compare_dir()
    manifest: list[dict[str, object]] = []

    for action, config in ACTION_CONFIG.items():
        for variant in VARIANTS:
            started = time.perf_counter()
            item_id = f'{action}_{variant["tier"]}'
            frames = build_frames(action, int(variant["inbetweens"]))
            if "playback_fps" in variant:
                frame_ms = round(1000 / float(variant["playback_fps"]))
            else:
                duration_ms = int(config["target_duration_ms"])
                frame_ms = max(10, round(duration_ms / len(frames)))
            fps = round(1000 / frame_ms, 1)
            out_dir = COMPARE / item_id
            write_frames(out_dir, frames)
            webp_size = write_webp(COMPARE / f"{item_id}.webp", frames, frame_ms)
            elapsed = time.perf_counter() - started

            manifest.append(
                {
                    "id": item_id,
                    "action": action,
                    "action_label": config["label"],
                    "tier": variant["tier"],
                    "tier_label": variant["label"],
                    "inbetweens": variant["inbetweens"],
                    "frame_count": len(frames),
                    "frame_ms": frame_ms,
                    "fps": fps,
                    "duration_ms": frame_ms * len(frames),
                    "webp": f"assets/compare/{item_id}.webp",
                    "webp_size_bytes": webp_size,
                    "frame_path": f"assets/compare/{item_id}/frame_{{index}}.png",
                    "local_build_seconds": round(elapsed, 2),
                    "runtime_cost": "0 API cost; local Canvas/WebP playback",
                    "generation_cost": "0 API cost; generated from existing 6 keyframes by local optical-flow interpolation",
                    "note": variant["note"],
                }
            )
            print(f"{item_id}: {len(frames)} frames, {fps} fps, {webp_size / 1024:.1f} KB, {elapsed:.2f}s")

    (COMPARE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(manifest)
    print(f"Wrote {COMPARE / 'manifest.json'}")


if __name__ == "__main__":
    main()

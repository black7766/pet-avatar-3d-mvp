from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ASSETS / "native"
CANVAS_SIZE = 520
COLS = 6
ROWS = 4
FRAME_COUNT = COLS * ROWS

SHEETS = {
    "happy": {
        "label": "开心",
        "sheet": ASSETS / "native_happy_24_sheet.png",
        "duration_ms": 1200,
        "note": "24 张 AI 原生开心姿态，猫的爪子、尾巴、表情和身体都有真实姿态变化。",
    },
    "eat": {
        "label": "进食",
        "sheet": ASSETS / "native_eat_24_sheet.png",
        "duration_ms": 1320,
        "note": "24 张 AI 原生进食姿态，包含低头、咬食、抬头、眨眼等完整动作变化。",
    },
}


def green_background_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    green_like = (g > 105) & (g - np.maximum(r, b) > 26) & (g > r * 1.12) & (g > b * 1.12)
    green_u8 = green_like.astype(np.uint8)

    labels_count, labels = cv2.connectedComponents(green_u8, connectivity=8)
    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    border_labels.discard(0)

    mask = np.zeros(green_like.shape, dtype=bool)
    for label in border_labels:
        if label < labels_count:
            mask |= labels == label
    return mask


def remove_green(cell: Image.Image) -> Image.Image:
    rgba = cell.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    rgb = arr[:, :, :3].copy()
    bg = green_background_mask(rgb)

    # Feather only around the connected background. This keeps green eyes intact.
    bg_float = bg.astype(np.float32)
    blurred = cv2.GaussianBlur(bg_float, (0, 0), 1.2)
    alpha = np.clip((1.0 - blurred) * 255, 0, 255).astype(np.uint8)
    alpha[bg] = 0

    # Despill green fringe without changing opaque subject colors too much.
    fringe = (alpha > 0) & (alpha < 255)
    rgb_float = rgb.astype(np.float32)
    max_rb = np.maximum(rgb_float[:, :, 0], rgb_float[:, :, 2])
    rgb_float[:, :, 1] = np.where(fringe, np.minimum(rgb_float[:, :, 1], max_rb * 1.05), rgb_float[:, :, 1])

    out = np.dstack([np.clip(rgb_float, 0, 255).astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA")


def keep_main_alpha_components(frame: Image.Image, keep_count: int, min_area: int = 600) -> Image.Image:
    arr = np.array(frame.convert("RGBA"), dtype=np.uint8)
    alpha = arr[:, :, 3]
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 8).astype(np.uint8), connectivity=8)
    components: list[tuple[int, int]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    keep = np.zeros(alpha.shape, dtype=bool)
    for _, label in components[:keep_count]:
        keep |= labels == label
    arr[:, :, 3] = np.where(keep, alpha, 0).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def slice_sheet(action: str, config: dict[str, object]) -> list[Image.Image]:
    sheet_path = Path(config["sheet"])
    if not sheet_path.exists():
        raise FileNotFoundError(sheet_path)
    sheet = Image.open(sheet_path).convert("RGB")
    cell_w = sheet.width // COLS
    cell_h = sheet.height // ROWS
    frames: list[Image.Image] = []

    for row in range(ROWS):
        for col in range(COLS):
            box = (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
            cell = sheet.crop(box)
            alpha_cell = remove_green(cell)
            alpha_cell = alpha_cell.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)
            alpha_cell = keep_main_alpha_components(alpha_cell, keep_count=2 if action == "eat" else 1)
            frames.append(alpha_cell)
    return frames


def write_frames(action: str, frames: list[Image.Image]) -> Path:
    action_dir = OUT / f"{action}_24"
    action_dir.mkdir(parents=True, exist_ok=True)
    for old in action_dir.glob("frame_*.png"):
        old.unlink()
    for index, frame in enumerate(frames, 1):
        frame.save(action_dir / f"frame_{index:02d}.png")
    return action_dir


def write_webp(action: str, frames: list[Image.Image], frame_ms: int) -> Path:
    path = OUT / f"{action}_native24.webp"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        lossless=False,
        quality=78,
        method=0,
        minimize_size=False,
    )
    return path


def write_contact_sheet(items: list[dict[str, object]]) -> None:
    thumb = 150
    label_h = 38
    sheet = Image.new("RGB", (COLS * thumb, len(items) * (ROWS * thumb + label_h)), "#f4f0eb")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for block, item in enumerate(items):
        action_dir = OUT / str(item["folder"])
        y_base = block * (ROWS * thumb + label_h)
        draw.text((8, y_base + 10), f'{item["label"]} native 24 frames', fill="#211b17", font=font)
        for index, path in enumerate(sorted(action_dir.glob("frame_*.png"))):
            image = Image.open(path).convert("RGBA")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            tile = Image.new("RGBA", (thumb, thumb), (255, 255, 255, 0))
            tile.alpha_composite(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
            white = Image.new("RGB", (thumb, thumb), "#fffaf3")
            white.paste(tile, mask=tile.split()[3])
            x = (index % COLS) * thumb
            y = y_base + label_h + (index // COLS) * thumb
            sheet.paste(white, (x, y))
    sheet.save(OUT / "native24_contact_sheet.jpg", quality=88)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for action, config in SHEETS.items():
        started = time.perf_counter()
        frames = slice_sheet(action, config)
        frame_ms = round(int(config["duration_ms"]) / len(frames))
        action_dir = write_frames(action, frames)
        webp = write_webp(action, frames, frame_ms)
        elapsed = time.perf_counter() - started

        item = {
            "id": f"{action}_native24",
            "action": action,
            "label": config["label"],
            "folder": f"{action}_24",
            "frame_count": len(frames),
            "frame_ms": frame_ms,
            "fps": round(1000 / frame_ms, 1),
            "duration_ms": frame_ms * len(frames),
            "webp": f"assets/native/{action}_native24.webp",
            "webp_size_bytes": webp.stat().st_size,
            "frame_path": f"assets/native/{action}_24/frame_{{index}}.png",
            "source_sheet": f"assets/native_{action}_24_sheet.png",
            "local_build_seconds": round(elapsed, 2),
            "generation_route": "image_gen generated 24 native poses in one 6x4 sprite sheet, then local chroma-key extraction",
            "note": config["note"],
        }
        manifest.append(item)
        print(
            f"{item['id']}: {item['frame_count']} frames, {item['fps']} fps, "
            f"{item['webp_size_bytes'] / 1024:.1f} KB, {item['local_build_seconds']}s"
        )

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_contact_sheet(manifest)
    print(f"Wrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()

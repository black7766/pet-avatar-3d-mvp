from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SIZE = 520

MOTION_SHEETS = {
    "expressive": {
        "source": Path(
            r"C:\Users\20408\.codex\generated_images\019eb4cb-3c61-7560-8c39-d75b23c377bb\ig_0d091699738e396e016a2aaf2dacec8199abe53edc3c43311a.png"
        ),
        "rows": ["talk", "wave", "happy"],
        "file": "pet_motion_sheet_expressive.png",
    },
    "daily": {
        "source": Path(
            r"C:\Users\20408\.codex\generated_images\019eb4cb-3c61-7560-8c39-d75b23c377bb\ig_0d091699738e396e016a2aaf8f9bf4819998027506b3f29e38.png"
        ),
        "rows": ["idle", "eat", "sleep"],
        "file": "pet_motion_sheet_daily.png",
    },
}

DURATIONS = {
    "idle": 120,
    "talk": 92,
    "wave": 92,
    "happy": 86,
    "eat": 110,
    "sleep": 130,
}


def remove_magenta(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    data = np.array(rgba)
    r = data[:, :, 0].astype(np.int16)
    g = data[:, :, 1].astype(np.int16)
    b = data[:, :, 2].astype(np.int16)
    a = data[:, :, 3].astype(np.int16)
    key = (r > 185) & (b > 175) & (g < 130) & ((r - g) > 60) & ((b - g) > 55)
    near = (r > 145) & (b > 135) & (g < 155) & ((r - g) > 38) & ((b - g) > 30)
    a[key] = 0
    a[near & ~key] = np.minimum(a[near & ~key], np.maximum(0, (g[near & ~key] - 88) * 2))
    data[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    out = Image.fromarray(data.astype(np.uint8), "RGBA")
    alpha = out.getchannel("A").filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.18))
    out.putalpha(alpha)
    return out


def remove_small_components(image: Image.Image, min_area: int = 900) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    foreground = alpha > 16
    h, w = foreground.shape
    seen = np.zeros_like(foreground, dtype=bool)
    keep = np.zeros_like(foreground, dtype=bool)
    for y in range(h):
        for x in range(w):
            if seen[y, x] or not foreground[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                px, py = stack.pop()
                component.append((px, py))
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < w and 0 <= ny < h and foreground[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(component) >= min_area:
                for px, py in component:
                    keep[py, px] = True
    rgba.putalpha(Image.fromarray(np.where(keep, alpha, 0).astype("uint8"), "L"))
    return rgba


def keep_largest_component(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    foreground = alpha > 16
    h, w = foreground.shape
    seen = np.zeros_like(foreground, dtype=bool)
    best: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or not foreground[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                px, py = stack.pop()
                component.append((px, py))
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < w and 0 <= ny < h and foreground[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(component) > len(best):
                best = component
    keep = np.zeros_like(foreground, dtype=bool)
    for px, py in best:
        keep[py, px] = True
    rgba.putalpha(Image.fromarray(np.where(keep, alpha, 0).astype("uint8"), "L"))
    return rgba


def despill(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    data = np.array(rgba).astype(np.int16)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    fringe = (a > 0) & (r > 140) & (b > 125) & (g < 155) & ((r - g) > 32) & ((b - g) > 24)
    weak = fringe & (a < 118)
    a[weak] = 0
    strong = fringe & ~weak
    neutral = ((r + g + b) // 3)
    r[strong] = np.minimum(neutral[strong] + 42, 232)
    g[strong] = np.maximum(neutral[strong] - 12, 112)
    b[strong] = np.minimum(neutral[strong] - 8, 152)
    data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3] = r, g, b, a
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), "RGBA")


def fit_cell(cell: Image.Image, action: str) -> Image.Image:
    cutout = despill(remove_small_components(remove_magenta(cell)))
    bbox = cutout.getbbox()
    if not bbox:
        return Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    pad = 10
    crop = cutout.crop((
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(cutout.width, bbox[2] + pad),
        min(cutout.height, bbox[3] + pad),
    ))
    scale_h = 0.88
    scale_w = 0.88
    y_bias = 16
    if action == "happy":
        scale_h = 0.92
        y_bias = 6
    elif action == "eat":
        scale_h = 0.75
        scale_w = 0.94
        y_bias = 70
    elif action == "sleep":
        scale_h = 0.62
        scale_w = 0.92
        y_bias = 110
    scale = min(SIZE * scale_w / crop.width, SIZE * scale_h / crop.height)
    resized = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(resized, ((SIZE - resized.width) // 2, (SIZE - resized.height) // 2 + y_bias))
    canvas = keep_largest_component(canvas)
    rgb = ImageEnhance.Sharpness(canvas.convert("RGB")).enhance(1.06)
    result = rgb.convert("RGBA")
    result.putalpha(canvas.getchannel("A"))
    return result


def extract_rows() -> dict[str, list[Image.Image]]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    actions: dict[str, list[Image.Image]] = {}
    for sheet in MOTION_SHEETS.values():
      source = sheet["source"]
      target = ASSETS / sheet["file"]
      shutil.copy2(source, target)
      image = Image.open(target).convert("RGBA")
      cell_w = image.width // 6
      cell_h = image.height // 3
      for row, action in enumerate(sheet["rows"]):
          frames = []
          for col in range(6):
              cell = image.crop((col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h))
              frame = fit_cell(cell, action)
              frame.save(ASSETS / f"frame_{action}_{col + 1:02d}.png")
              frames.append(frame)
          actions[action] = frames
    return actions


def loopify(action: str, frames: list[Image.Image]) -> list[Image.Image]:
    if action in {"talk", "wave", "eat"}:
        return frames
    if action == "sleep":
        return frames + frames[-2:0:-1]
    if action == "idle":
        return frames + frames[-2:0:-1]
    if action == "happy":
        return frames + frames[-2:0:-1]
    return frames


def save_webp(action: str, frames: list[Image.Image]) -> None:
    frames = loopify(action, frames)
    frames[0].save(
        ASSETS / f"{action}_momo.webp",
        save_all=True,
        append_images=frames[1:],
        duration=[DURATIONS[action]] * len(frames),
        loop=0,
        lossless=True,
        quality=92,
        method=4,
        minimize_size=False,
        kmin=1,
        kmax=1,
    )
    frames[0].save(ASSETS / f"{action}_momo_poster.png")


def main() -> None:
    actions = extract_rows()
    for action, frames in actions.items():
        save_webp(action, frames)
    print(f"sequence-sheet WebP actions written to {ASSETS}")


if __name__ == "__main__":
    main()

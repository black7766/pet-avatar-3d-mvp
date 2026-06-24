from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SOURCE = Path(
    r"C:\Users\20408\.codex\generated_images\019eb4cb-3c61-7560-8c39-d75b23c377bb\ig_0d091699738e396e016a2aab4a2cb88199b6669fc753a0cc32.png"
)
SHEET = ASSETS / "pet_pose_sheet_imagegen.png"
SIZE = 520

POSE_NAMES = ["idle", "talk", "wave", "happy", "eat", "sleep"]


def remove_magenta(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    key_pixels: set[tuple[int, int]] = set()
    near_pixels: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            is_key = r > 195 and b > 185 and g < 110
            near_key = r > 165 and b > 150 and g < 140 and (r - g) > 65 and (b - g) > 55
            if is_key:
                pixels[x, y] = (r, g, b, 0)
                key_pixels.add((x, y))
            elif near_key:
                alpha = max(0, min(255, int((g - 60) * 3.0)))
                pixels[x, y] = (r, g, b, min(a, alpha))
                near_pixels.add((x, y))
    alpha = rgba.getchannel("A")
    alpha = alpha.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.18))
    alpha_px = alpha.load()
    for x, y in key_pixels:
        alpha_px[x, y] = 0
    for x, y in near_pixels:
        alpha_px[x, y] = min(alpha_px[x, y], 42)
    rgba.putalpha(alpha)
    return rgba


def remove_small_alpha_components(image: Image.Image, min_area: int = 1100) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    foreground = alpha > 18
    height, width = foreground.shape
    seen = np.zeros_like(foreground, dtype=bool)
    keep = np.zeros_like(foreground, dtype=bool)
    for y in range(height):
        for x in range(width):
            if not foreground[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                px, py = stack.pop()
                component.append((px, py))
                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if 0 <= nx < width and 0 <= ny < height and foreground[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(component) >= min_area:
                for px, py in component:
                    keep[py, px] = True
    cleaned_alpha = np.where(keep, alpha, 0).astype("uint8")
    rgba.putalpha(Image.fromarray(cleaned_alpha, "L"))
    return rgba


def fit_pose(image: Image.Image, action: str) -> Image.Image:
    image = image.convert("RGBA")
    bbox = image.getbbox()
    if not bbox:
        return Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    pad = 18
    crop = image.crop((
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(image.width, bbox[2] + pad),
        min(image.height, bbox[3] + pad),
    ))
    scale = min(SIZE * 0.86 / crop.width, SIZE * 0.88 / crop.height)
    if action == "sleep":
        scale = min(SIZE * 0.90 / crop.width, SIZE * 0.78 / crop.height)
    elif action == "eat":
        scale = min(SIZE * 0.88 / crop.width, SIZE * 0.82 / crop.height)
    resized = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    y_bias = 18 if action in {"idle", "talk", "wave", "happy"} else 44
    if action == "sleep":
        y_bias = 74
    canvas.alpha_composite(resized, ((SIZE - resized.width) // 2, (SIZE - resized.height) // 2 + y_bias))
    return canvas


def extract_poses() -> dict[str, Image.Image]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, SHEET)
    source = Image.open(SHEET).convert("RGBA")
    cell_w = source.width // 3
    cell_h = source.height // 2
    poses: dict[str, Image.Image] = {}
    for index, name in enumerate(POSE_NAMES):
        col = index % 3
        row = index // 3
        cell = source.crop((col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h))
        pose = fit_pose(remove_small_alpha_components(remove_magenta(cell)), name)
        pose = despill(pose)
        pose.save(ASSETS / f"pose_{name}_imagegen.png")
        poses[name] = pose
    return poses


def despill(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a and r > 145 and b > 125 and g < 135 and (r - g) > 45 and (b - g) > 35:
                if a < 96:
                    pixels[x, y] = (r, g, b, 0)
                else:
                    neutral = int((r + g + b) / 3)
                    pixels[x, y] = (min(neutral + 36, 230), max(neutral - 18, 105), min(neutral - 10, 150), a)
    return rgba


def transform(image: Image.Image, scale_x: float = 1.0, scale_y: float = 1.0, rotate: float = 0.0, dx: int = 0, dy: int = 0) -> Image.Image:
    width = max(1, int(image.width * scale_x))
    height = max(1, int(image.height * scale_y))
    frame = image.resize((width, height), Image.Resampling.BICUBIC)
    frame = frame.rotate(rotate, resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(frame, ((SIZE - frame.width) // 2 + dx, (SIZE - frame.height) // 2 + dy))
    return canvas


def blend_pose(a: Image.Image, b: Image.Image, amount: float) -> Image.Image:
    amount = max(0.0, min(1.0, amount))
    return Image.blend(a, b, amount)


def sharpen_frames(frames: list[Image.Image]) -> list[Image.Image]:
    out = []
    for frame in frames:
        rgb = frame.convert("RGB")
        rgb = ImageEnhance.Sharpness(rgb).enhance(1.08)
        fixed = rgb.convert("RGBA")
        fixed.putalpha(frame.getchannel("A"))
        out.append(fixed)
    return out


def make_idle(poses: dict[str, Image.Image]) -> list[Image.Image]:
    base = poses["idle"]
    frames = []
    total = 30
    for i in range(total):
        t = i / total
        breath = math.sin(t * math.tau)
        frames.append(transform(base, 1 - breath * 0.006, 1 + breath * 0.012, math.cos(t * math.tau) * 0.8, 0, int(-breath * 4)))
    return frames


def make_pose_loop(poses: dict[str, Image.Image], action: str, total: int) -> list[Image.Image]:
    idle = poses["idle"]
    target = poses[action]
    frames = []
    for i in range(total):
        t = i / (total - 1)
        if t < 0.18:
            amount = t / 0.18
        elif t > 0.78:
            amount = (1 - t) / 0.22
        else:
            amount = 1.0
        pose = blend_pose(idle, target, amount) if action in {"talk", "wave"} else target
        bounce = abs(math.sin(t * math.tau))
        if action == "talk":
            frames.append(transform(pose, 1.0, 1.0 + bounce * 0.006, math.sin(t * math.tau) * 0.8, 0, int(-bounce * 4)))
        elif action == "wave":
            frames.append(transform(pose, 1.0, 1.0, math.sin(t * math.tau) * 1.2, 0, int(-bounce * 4)))
        elif action == "happy":
            frames.append(transform(pose, 1 + bounce * 0.018, 1 - bounce * 0.012, math.sin(t * math.tau) * 2.8, 0, int(-bounce * 24)))
        elif action == "eat":
            frames.append(transform(pose, 1.0 + bounce * 0.006, 1.0 - bounce * 0.004, math.sin(t * math.tau) * 0.45, 0, int(bounce * 5)))
        elif action == "sleep":
            frames.append(transform(pose, 1.0 + bounce * 0.005, 1.0 - bounce * 0.004, math.sin(t * math.tau) * 0.24, 0, int(bounce * 3)))
    return frames


def save_webp(name: str, frames: list[Image.Image], duration: int) -> None:
    frames = sharpen_frames(frames)
    frames[0].save(
        ASSETS / f"{name}_momo.webp",
        save_all=True,
        append_images=frames[1:],
        duration=[duration] * len(frames),
        loop=0,
        lossless=True,
        quality=92,
        method=4,
        minimize_size=False,
        kmin=1,
        kmax=1,
    )
    frames[0].save(ASSETS / f"{name}_momo_poster.png")


def main() -> None:
    poses = extract_poses()
    save_webp("idle", make_idle(poses), 70)
    save_webp("talk", make_pose_loop(poses, "talk", 28), 54)
    save_webp("wave", make_pose_loop(poses, "wave", 30), 54)
    save_webp("happy", make_pose_loop(poses, "happy", 28), 56)
    save_webp("eat", make_pose_loop(poses, "eat", 30), 68)
    save_webp("sleep", make_pose_loop(poses, "sleep", 32), 88)
    print(f"pose-sheet WebP actions written to {ASSETS}")


if __name__ == "__main__":
    main()

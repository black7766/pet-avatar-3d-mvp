from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
INPUT = Path(r"C:\Users\20408\Pictures\u=2398577323,3273266194&fm=253&app=138&f=JPEG.jpg")


def fit_canvas(image: Image.Image, size: int = 520) -> Image.Image:
    image = image.convert("RGBA")
    scale = min(size * 0.82 / image.width, size * 0.86 / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(resized, ((size - resized.width) // 2, int(size * 0.09)))
    return canvas


def foreground_mask(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    sat = rgb.convert("HSV").split()[1]
    bright = gray.point(lambda p: 255 if p > 68 else 0)
    mid = gray.point(lambda p: 255 if p > 46 else 0)
    colorful = sat.point(lambda p: 255 if p > 36 else 0)
    mask = ImageChops.lighter(bright, ImageChops.multiply(mid, colorful))
    mask = mask.filter(ImageFilter.MedianFilter(5))
    mask = mask.filter(ImageFilter.MaxFilter(21))
    mask = mask.filter(ImageFilter.GaussianBlur(2.4))
    mask = mask.point(lambda p: 255 if p > 54 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.4))
    return mask


def cartoonize(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    smooth = rgb.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.SMOOTH_MORE)
    smooth = ImageEnhance.Color(smooth).enhance(1.28)
    smooth = ImageEnhance.Contrast(smooth).enhance(1.10)
    poster = ImageOps.posterize(smooth, 5)
    edges = ImageOps.grayscale(rgb).filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(0.8)).point(lambda p: 130 if p > 32 else 0)
    dark_edges = Image.new("RGBA", image.size, (80, 52, 42, 0))
    dark_edges.putalpha(edges)
    out = poster.convert("RGBA")
    out.alpha_composite(dark_edges)
    out.putalpha(mask)
    return out


def crop_subject(image: Image.Image) -> Image.Image:
    bbox = image.getbbox()
    if not bbox:
        return image
    pad = 34
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(image.width, bbox[2] + pad)
    bottom = min(image.height, bbox[3] + pad)
    return image.crop((left, top, right, bottom))


def add_outline(image: Image.Image, color=(89, 58, 46, 255), width: int = 10) -> Image.Image:
    alpha = image.getchannel("A")
    outline = alpha.filter(ImageFilter.MaxFilter(width * 2 + 1))
    outline = ImageChops.subtract(outline, alpha)
    base = Image.new("RGBA", image.size, (0, 0, 0, 0))
    stroke = Image.new("RGBA", image.size, color)
    stroke.putalpha(outline.filter(ImageFilter.GaussianBlur(0.5)))
    base.alpha_composite(stroke)
    base.alpha_composite(image)
    return base


def make_avatar() -> Image.Image:
    original = Image.open(INPUT)
    mask = foreground_mask(original)
    cutout = original.convert("RGBA")
    cutout.putalpha(mask)
    cutout = crop_subject(cutout)
    mask = cutout.getchannel("A")
    cartoon = cartoonize(cutout, mask)
    cartoon = add_outline(cartoon, width=8)
    avatar = fit_canvas(cartoon, 520)
    return avatar


def transform_avatar(avatar: Image.Image, frame: int, total: int, mode: str) -> Image.Image:
    mode = mode.split("_", 1)[0]
    t = frame / total
    phase = math.sin(t * math.tau)
    phase2 = math.cos(t * math.tau)

    scale_x = 1.0
    scale_y = 1.0
    rot = 0.0
    y = 0
    x = 0
    extra = []

    if mode == "idle":
        scale_x = 1.0 - phase * 0.010
        scale_y = 1.0 + phase * 0.018
        y = int(-phase * 5)
        rot = phase2 * 1.4
    elif mode == "happy":
        scale_x = 1.0 + abs(phase) * 0.035
        scale_y = 1.0 - abs(phase) * 0.022
        y = int(-abs(phase) * 34)
        rot = phase * 5.0
        for i in range(4):
            extra.append(("heart", 80 + i * 62, 94 + int(math.sin(t * math.tau + i) * 12), i))
    elif mode == "eat":
        scale_x = 1.0 + max(0, phase) * 0.018
        scale_y = 1.0 - max(0, phase) * 0.012
        y = int(max(0, phase) * 18)
        rot = phase * 1.6
        extra.append(("bowl", 260, 438, 0))
    elif mode == "sleep":
        scale_x = 1.04
        scale_y = 0.88 + phase * 0.010
        y = 54 + int(phase * 4)
        rot = -6
        for i in range(3):
            extra.append(("z", 332 + i * 30, 120 - i * 28 + int(phase2 * 4), i))
    elif mode == "blink":
        scale_x = 1.0
        scale_y = 1.0
        y = int(-phase * 4)
        if 0.42 < t < 0.58:
            extra.append(("blink", 260, 216, 0))

    w, h = avatar.size
    resized = avatar.resize((int(w * scale_x), int(h * scale_y)), Image.Resampling.BICUBIC)
    rotated = resized.rotate(rot, resample=Image.Resampling.BICUBIC, expand=True)
    canvas_size = 520
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    canvas.alpha_composite(rotated, ((canvas_size - rotated.width) // 2 + x, (canvas_size - rotated.height) // 2 + y))
    draw = ImageDraw.Draw(canvas)

    for kind, px, py, idx in extra:
        if kind == "heart":
            fill = (222, 96, 120, max(80, 210 - idx * 24))
            draw.ellipse((px - 11, py - 14, px + 5, py + 2), fill=fill)
            draw.ellipse((px - 1, py - 14, px + 15, py + 2), fill=fill)
            draw.polygon([(px - 14, py - 4), (px + 14, py - 4), (px, py + 18)], fill=fill)
        elif kind == "bowl":
            draw.ellipse((px - 118, py + 14, px + 118, py + 64), fill=(120, 78, 58, 235))
            draw.rounded_rectangle((px - 100, py - 18, px + 100, py + 48), radius=32, fill=(190, 122, 82, 255))
            draw.ellipse((px - 82, py - 22, px + 82, py + 26), fill=(245, 218, 186, 255))
            for k in range(18):
                ax = px - 62 + (k % 9) * 15
                ay = py - 4 + (k // 9) * 12
                draw.ellipse((ax, ay, ax + 8, ay + 8), fill=(118, 75, 48, 255))
        elif kind == "z":
            alpha = 230 - idx * 52
            draw.text((px, py), "Z", fill=(116, 94, 130, alpha), anchor="mm")
        elif kind == "blink":
            draw.rounded_rectangle((px - 78, py - 10, px - 18, py + 1), radius=6, fill=(70, 45, 38, 210))
            draw.rounded_rectangle((px + 22, py - 10, px + 82, py + 1), radius=6, fill=(70, 45, 38, 210))

    return canvas


def save_webp(name: str, avatar: Image.Image, frames: int, duration: int) -> None:
    generated = [transform_avatar(avatar, i, frames, name) for i in range(frames)]
    generated[0].save(
        ASSETS / f"{name}.webp",
        save_all=True,
        append_images=generated[1:],
        duration=[duration] * len(generated),
        loop=0,
        lossless=True,
        quality=90,
        method=4,
        minimize_size=False,
        kmin=1,
        kmax=1,
    )
    generated[0].save(ASSETS / f"{name}_poster.png")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    avatar = make_avatar()
    avatar.save(ASSETS / "pet_avatar.png")
    avatar.resize((256, 256), Image.Resampling.LANCZOS).save(ASSETS / "pet_avatar_thumb.png")
    save_webp("idle", avatar, 16, 80)
    save_webp("happy", avatar, 16, 64)
    save_webp("eat", avatar, 18, 78)
    save_webp("sleep", avatar, 18, 96)
    save_webp("blink", avatar, 16, 80)
    print(f"assets written to {ASSETS}")


if __name__ == "__main__":
    main()

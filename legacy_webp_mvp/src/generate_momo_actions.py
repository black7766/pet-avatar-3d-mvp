from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
AVATAR = ASSETS / "pet_avatar_imagegen.png"
SIZE = 520


def eased(value: float) -> float:
    return 0.5 - math.cos(value * math.pi) * 0.5


def fit_layer(layer: Image.Image, scale_x: float, scale_y: float) -> Image.Image:
    width = max(1, int(layer.width * scale_x))
    height = max(1, int(layer.height * scale_y))
    return layer.resize((width, height), Image.Resampling.BICUBIC)


def transform(layer: Image.Image, scale_x: float = 1.0, scale_y: float = 1.0, rotate: float = 0.0, dx: int = 0, dy: int = 0) -> Image.Image:
    scaled = fit_layer(layer, scale_x, scale_y)
    rotated = scaled.rotate(rotate, resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(rotated, ((SIZE - rotated.width) // 2 + dx, (SIZE - rotated.height) // 2 + dy))
    return canvas


def draw_closed_eyes(draw: ImageDraw.ImageDraw, alpha: int = 230, wink: bool = False) -> None:
    color = (71, 46, 34, alpha)
    draw.arc((160, 120, 222, 166), 18, 162, fill=color, width=7)
    if wink:
        draw.arc((244, 130, 306, 174), 8, 172, fill=color, width=8)
    else:
        draw.arc((244, 130, 306, 174), 18, 162, fill=color, width=7)


def draw_talk_mouth(draw: ImageDraw.ImageDraw, amount: float) -> None:
    amount = max(0.0, min(1.0, amount))
    if amount < 0.12:
        draw.arc((210, 184, 262, 212), 18, 162, fill=(78, 42, 32, 185), width=4)
        return
    h = 18 + int(34 * amount)
    top = 184 - int(5 * amount)
    draw.ellipse((210, top, 262, top + h), fill=(64, 30, 31, 238))
    draw.ellipse((220, top + h * 0.48, 252, top + h - 4), fill=(221, 101, 116, 230))
    draw.polygon([(220, top + 5), (228, top + 5), (224, top + 19)], fill=(255, 246, 230, 238))
    draw.polygon([(244, top + 5), (252, top + 5), (248, top + 19)], fill=(255, 246, 230, 238))


def draw_blush(draw: ImageDraw.ImageDraw, level: float) -> None:
    alpha = int(68 * level)
    draw.ellipse((140, 176, 194, 200), fill=(235, 131, 124, alpha))
    draw.ellipse((276, 186, 330, 210), fill=(235, 131, 124, alpha))


def make_paw(lift: float) -> Image.Image:
    paw = Image.new("RGBA", (122, 170), (0, 0, 0, 0))
    draw = ImageDraw.Draw(paw)
    fur = (224, 171, 104, 246)
    shade = (108, 70, 45, 122)
    draw.rounded_rectangle((36, 20, 92, 134), radius=28, fill=fur)
    draw.ellipse((23, 4, 54, 42), fill=fur)
    draw.ellipse((51, 0, 82, 38), fill=fur)
    draw.ellipse((77, 8, 106, 48), fill=fur)
    draw.arc((38, 28, 94, 132), 266, 90, fill=shade, width=4)
    for y in (54, 78, 102):
        draw.arc((35, y, 92, y + 28), 184, 246, fill=(74, 45, 33, 128), width=3)
    draw.ellipse((58, 108, 78, 128), fill=(123, 76, 63, 160))
    draw.ellipse((39, 95, 54, 110), fill=(123, 76, 63, 142))
    draw.ellipse((80, 94, 96, 110), fill=(123, 76, 63, 142))
    return paw.rotate(-25 + 44 * math.sin(lift * math.pi), resample=Image.Resampling.BICUBIC, expand=True)


def draw_heart(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float, alpha: int) -> None:
    r = int(11 * scale)
    fill = (220, 91, 116, alpha)
    draw.ellipse((cx - r, cy - r, cx + 2, cy + 2), fill=fill)
    draw.ellipse((cx - 2, cy - r, cx + r, cy + 2), fill=fill)
    draw.polygon([(cx - r - 1, cy - 1), (cx + r + 1, cy - 1), (cx, cy + int(1.35 * r))], fill=fill)


def draw_bowl(draw: ImageDraw.ImageDraw, chew: float) -> None:
    x, y = 272, 438
    draw.ellipse((x - 118, y + 16, x + 118, y + 66), fill=(91, 56, 43, 150))
    draw.rounded_rectangle((x - 106, y - 22, x + 106, y + 52), radius=34, fill=(132, 79, 58, 255))
    draw.ellipse((x - 92, y - 28, x + 92, y + 22), fill=(248, 219, 178, 255))
    draw.ellipse((x - 76, y - 18, x + 76, y + 14), fill=(229, 175, 116, 255))
    for i in range(22):
        px = x - 62 + (i % 11) * 12
        py = y - 10 + (i // 11) * 12 + int(math.sin(i + chew * math.tau) * 1.5)
        draw.ellipse((px, py, px + 7, py + 7), fill=(104, 65, 42, 255))


def pose(base: Image.Image, *, mouth: float = 0.0, blink: bool = False, wink: bool = False, blush: float = 0.0, paw: float = 0.0, bowl: float = 0.0, zed: float = 0.0, hearts: float = 0.0) -> Image.Image:
    frame = base.copy()
    draw = ImageDraw.Draw(frame, "RGBA")
    if mouth:
        draw_talk_mouth(draw, mouth)
    if blink:
        draw_closed_eyes(draw, wink=wink)
    if blush:
        draw_blush(draw, blush)
    if paw:
        raised = make_paw(paw)
        px = 276 + int(12 * math.sin(paw * math.tau))
        py = 278 - int(126 * eased(min(1.0, paw)))
        frame.alpha_composite(raised, (px, py))
    if bowl:
        draw_bowl(draw, bowl)
    if zed:
        for i in range(3):
            x = 318 + i * 34
            y = 128 - i * 28 + int(math.sin(zed * math.tau + i) * 5)
            draw.text((x, y), "Z", fill=(92, 79, 123, 210 - i * 48), anchor="mm")
    if hearts:
        for i in range(5):
            phase = (hearts + i * 0.17) % 1
            alpha = int(225 * (1 - phase))
            draw_heart(draw, 112 + i * 56, 120 - int(58 * phase), 0.82 + i * 0.08, alpha)
    return frame


def make_frames(action: str, base: Image.Image) -> list[Image.Image]:
    frames: list[Image.Image] = []
    if action == "idle":
        total = 28
        for i in range(total):
            t = i / total
            breath = math.sin(t * math.tau)
            blink = 0.43 < t < 0.50
            layer = pose(base, blink=blink)
            frames.append(transform(layer, 1 - breath * 0.006, 1 + breath * 0.014, math.cos(t * math.tau) * 0.9, 0, int(-breath * 4)))
    elif action == "talk":
        total = 26
        for i in range(total):
            t = i / total
            open_amount = max(0.0, math.sin(t * math.tau * 3))
            layer = pose(base, mouth=open_amount, blush=0.32)
            frames.append(transform(layer, 1.0, 1.0 + open_amount * 0.01, math.sin(t * math.tau) * 0.8, 0, int(-open_amount * 5)))
    elif action == "wave":
        total = 30
        for i in range(total):
            t = i / (total - 1)
            lift = math.sin(t * math.pi)
            layer = pose(base, blink=0.38 < t < 0.58, wink=True, paw=lift, blush=0.28)
            frames.append(transform(layer, 1.0, 1.0, math.sin(t * math.tau) * 1.2, 0, int(-lift * 6)))
    elif action == "happy":
        total = 24
        for i in range(total):
            t = i / total
            bounce = abs(math.sin(t * math.tau))
            layer = pose(base, mouth=0.75 * bounce, blush=0.58, hearts=t)
            frames.append(transform(layer, 1 + bounce * 0.02, 1 - bounce * 0.014, math.sin(t * math.tau) * 3.6, 0, int(-bounce * 24)))
    elif action == "eat":
        total = 28
        for i in range(total):
            t = i / total
            chew = abs(math.sin(t * math.tau * 2))
            layer = pose(base, mouth=0.24 * chew, bowl=t)
            frames.append(transform(layer, 1.0 + chew * 0.01, 1.0 - chew * 0.006, math.sin(t * math.tau) * 0.8, 0, int(chew * 9)))
    elif action == "sleep":
        total = 32
        for i in range(total):
            t = i / total
            breath = math.sin(t * math.tau)
            layer = pose(base, blink=True, zed=t)
            frames.append(transform(layer, 1.04, 0.90 + breath * 0.006, -6.5, -6, 46 + int(breath * 4)))
    else:
        raise ValueError(f"unknown action: {action}")
    return frames


def save_action(action: str, base: Image.Image, duration: int) -> None:
    frames = make_frames(action, base)
    target = ASSETS / f"{action}_momo.webp"
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=[duration] * len(frames),
        loop=0,
        lossless=True,
        quality=90,
        method=4,
        minimize_size=False,
        kmin=1,
        kmax=1,
    )
    frames[0].save(ASSETS / f"{action}_momo_poster.png")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    base = Image.open(AVATAR).convert("RGBA")
    # Strengthen edge alpha once so the WebP keeps the transparent pet layer crisp.
    alpha = base.getchannel("A").filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.25))
    body = base.copy()
    body.putalpha(ImageChops.lighter(base.getchannel("A"), alpha))
    for action, duration in {
        "idle": 70,
        "talk": 58,
        "wave": 54,
        "happy": 56,
        "eat": 68,
        "sleep": 86,
    }.items():
        save_action(action, body, duration)
    print(f"momo action assets written to {ASSETS}")


if __name__ == "__main__":
    main()

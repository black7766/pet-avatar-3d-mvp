from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from generate_pet_assets import ASSETS, save_webp


INPUT = Path(r"C:\Users\20408\AppData\Local\Temp\codex-clipboard-f325f820-e526-432f-975a-32b981491af9.png")


def remove_checkerboard(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    rgb = rgba.convert("RGB")
    pixels = rgb.load()
    w, h = rgb.size

    mask = Image.new("L", rgb.size, 255)
    mask_px = mask.load()
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            near_neutral = abs(r - g) < 4 and abs(g - b) < 4
            checker_light = 230 <= r <= 255
            if near_neutral and checker_light:
                mask_px[x, y] = 0

    # Recover the cat body from internal bright fur while keeping background transparent.
    bg = Image.new("L", mask.size, 0)
    bg_px = bg.load()
    src_px = mask.load()
    stack = []
    for x in range(w):
        stack.append((x, 0))
        stack.append((x, h - 1))
    for y in range(h):
        stack.append((0, y))
        stack.append((w - 1, y))
    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if bg_px[x, y] or src_px[x, y] > 0:
            continue
        bg_px[x, y] = 255
        stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    alpha = ImageChops.invert(bg)
    alpha = alpha.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(0.8))
    rgba.putalpha(alpha)
    return rgba


def crop_and_fit(image: Image.Image, size: int = 520) -> Image.Image:
    bbox = image.getbbox()
    if bbox:
        pad = 30
        crop = image.crop((
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(image.width, bbox[2] + pad),
            min(image.height, bbox[3] + pad),
        ))
    else:
        crop = image
    scale = min(size * 0.88 / crop.width, size * 0.90 / crop.height)
    resized = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    avatar = crop_and_fit(remove_checkerboard(Image.open(INPUT)))
    avatar.save(ASSETS / "pet_avatar_imagegen.png")
    avatar.resize((256, 256), Image.Resampling.LANCZOS).save(ASSETS / "pet_avatar_imagegen_thumb.png")
    save_webp("idle_imagegen", avatar, 16, 80)
    save_webp("happy_imagegen", avatar, 16, 64)
    save_webp("eat_imagegen", avatar, 18, 78)
    save_webp("sleep_imagegen", avatar, 18, 96)
    save_webp("blink_imagegen", avatar, 16, 80)
    print(f"image_gen assets written to {ASSETS}")


if __name__ == "__main__":
    main()

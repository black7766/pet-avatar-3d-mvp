from __future__ import annotations

import argparse
import base64
import json
import os
import mimetypes
import sys
from pathlib import Path
from urllib import request
from urllib.error import HTTPError

from generate_pet_assets import ASSETS, save_webp
from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = """
把参考图中的宠物转换成一个高质量移动 App 宠物管家 mascot：
Q版卡通、全身坐姿、三分之二正面、保留原宠物毛色和眼睛特征、
头部略大、眼睛有神、毛发柔软但不过度写实、轮廓干净、
透明背景、无文字、无道具、无边框、适合做手机首页宠物形象。
输出必须是单只宠物，主体居中，完整身体，不要裁切耳朵和尾巴。
""".strip()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Copy .env.example to .env and fill it.")
    return value


def download_image(url: str, output: Path) -> None:
    with request.urlopen(url, timeout=120) as response:
        payload = response.read()
    output.write_bytes(payload)


def save_response_image(data: dict, output: Path) -> None:
    # OpenAI-compatible image APIs commonly return either data[0].url or data[0].b64_json.
    items = data.get("data") or data.get("images") or []
    if not items:
        raise RuntimeError(f"No image found in response: {json.dumps(data, ensure_ascii=False)[:500]}")
    first = items[0]
    if isinstance(first, str):
        download_image(first, output)
        return
    if "url" in first:
        download_image(first["url"], output)
        return
    if "b64_json" in first:
        output.write_bytes(base64.b64decode(first["b64_json"]))
        return
    raise RuntimeError(f"Unsupported image response item: {json.dumps(first, ensure_ascii=False)[:500]}")


def image_file_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        mime = "image/jpeg"

    image = Image.open(path).convert("RGB")
    if image.width * image.height < 3_686_400:
        canvas_size = 2048
        scale = min(canvas_size * 0.86 / image.width, canvas_size * 0.86 / image.height)
        resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (canvas_size, canvas_size), (245, 242, 237))
        canvas.paste(resized, ((canvas_size - resized.width) // 2, (canvas_size - resized.height) // 2))
        upscaled = ASSETS / "seedream_reference_2048.jpg"
        canvas.save(upscaled, quality=94, optimize=True)
        path = upscaled
        mime = "image/jpeg"

    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def call_seedream(prompt: str, image_input: str, output: Path) -> None:
    api_key = require_env("ARK_API_KEY")
    model = os.environ.get("ARK_IMAGE_MODEL", "doubao-seedream-4-5-251128")
    base_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
    endpoint = f"{base_url}/images/generations"

    # This follows the common Ark/OpenAI-compatible image generation shape.
    # Some Seedream endpoints name the reference field differently. If your
    # console model requires another field, adjust only this payload block.
    payload = {
        "model": model,
        "prompt": prompt,
        "size": os.environ.get("ARK_IMAGE_SIZE", "3K"),
        "response_format": "url",
        "images": [image_input],
        "watermark": False,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Seedream HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(
            "Seedream request failed. Check ARK_IMAGE_MODEL, image reference URL, and account permissions."
        ) from exc
    save_response_image(data, output)


def clean_alpha(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    # If the model already returns transparent PNG, this keeps it. If it returns
    # a near-white/near-solid background, this makes a lightweight cleanup pass.
    alpha = image.getchannel("A")
    if alpha.getextrema()[0] < 240:
        return image
    rgb = image.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    inv = gray.point(lambda p: 0 if p > 242 else 255)
    mask = inv.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(1.2))
    image.putalpha(mask)
    return image


def generate_motion_from_avatar(avatar: Image.Image) -> None:
    avatar.save(ASSETS / "pet_avatar_seedream.png")
    avatar.resize((256, 256), Image.Resampling.LANCZOS).save(ASSETS / "pet_avatar_seedream_thumb.png")
    save_webp("idle_seedream", avatar, 16, 80)
    save_webp("happy_seedream", avatar, 16, 64)
    save_webp("eat_seedream", avatar, 18, 78)
    save_webp("sleep_seedream", avatar, 18, 96)
    save_webp("blink_seedream", avatar, 16, 80)


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-url", default=os.environ.get("PET_REFERENCE_IMAGE_URL"))
    parser.add_argument("--image-file", default=os.environ.get("PET_REFERENCE_IMAGE_FILE"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-output", default=str(ASSETS / "seedream_raw.png"))
    args = parser.parse_args()

    image_input = args.image_url
    if args.image_file:
        image_path = Path(args.image_file)
        if not image_path.exists():
            print(f"Image file not found: {image_path}", file=sys.stderr)
            return 2
        image_input = image_file_to_data_url(image_path)

    if not image_input:
        print("Missing PET_REFERENCE_IMAGE_URL/--image-url or PET_REFERENCE_IMAGE_FILE/--image-file.", file=sys.stderr)
        return 2

    ASSETS.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw_output)
    call_seedream(args.prompt, image_input, raw)
    avatar = clean_alpha(raw)
    generate_motion_from_avatar(avatar)
    print(json.dumps({
        "ok": True,
        "raw": str(raw),
        "avatar": str(ASSETS / "pet_avatar_seedream.png"),
        "actions": [
            str(ASSETS / "idle_seedream.webp"),
            str(ASSETS / "happy_seedream.webp"),
            str(ASSETS / "eat_seedream.webp"),
            str(ASSETS / "sleep_seedream.webp"),
            str(ASSETS / "blink_seedream.webp"),
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# PaiMomo-Style Pet MVP

## What This Demo Implements

This is a local prototype for the likely PaiMomo-style implementation:

1. User pet photo input.
2. Local subject cutout.
3. Lightweight cartoon/posterized styling.
4. Transparent pet avatar PNG.
5. Single-action animated WebP loops.
6. Mobile app UI with pet health, food, vaccine, spending, and action state switching.

## Local Files

- Demo page: `index.html`
- Source image processor: `src/generate_pet_assets.py`
- Static avatar: `assets/pet_avatar.png`
- Thumbnail: `assets/pet_avatar_thumb.png`
- Action loops:
  - `assets/idle.webp`
  - `assets/happy.webp`
  - `assets/eat.webp`
  - `assets/sleep.webp`
  - `assets/blink.webp`

## Current Technical Route

The current version avoids cloud model dependencies and uses local image processing:

- Pillow threshold mask
- local transparent cutout
- posterized color treatment
- outline stroke
- frame-by-frame transform
- animated WebP export

This is intentionally close to the implementation hinted in the Douyin comments: image generation for the avatar, traditional image processing for cutout/refinement, and WebP single-action loops for animation.

## Production Upgrade Path

Replace the local cartoon filter with:

- Doubao / Seedream image-to-image for pet Q-avatar generation.
- RMBG / MODNet / rembg / SAM for stronger matting.
- Seedance / Kling / Runway / ComfyUI AnimateDiff for richer action frame generation.
- WebP/APNG/Lottie/Rive for runtime delivery.

For a first production MVP, animated WebP is the most practical runtime format because it supports transparency and animation in modern browsers and mobile webviews.

## Seedream / Doubao Image-to-Image Integration

The scaffold is in:

```text
src/seedream_pet_pipeline.py
```

You need to provide:

1. `ARK_API_KEY`: Volcengine Ark / Doubao API key.
2. `ARK_IMAGE_MODEL`: the image model or endpoint enabled in your console.
3. `PET_REFERENCE_IMAGE_URL`: a publicly accessible URL of the pet photo.
4. Optional: confirm whether your enabled Seedream endpoint accepts the reference image field as `image`, `image_url`, or another provider-specific field.

Create:

```powershell
Copy-Item .env.example .env
notepad .env
```

Then run:

```powershell
python src/seedream_pet_pipeline.py
```

Expected outputs:

```text
assets/pet_avatar_seedream.png
assets/pet_avatar_seedream_thumb.png
assets/idle_seedream.webp
assets/happy_seedream.webp
assets/eat_seedream.webp
assets/sleep_seedream.webp
assets/blink_seedream.webp
```

The current local demo uses Pillow-generated assets. Once Seedream outputs are generated, update `index.html` to use the `_seedream` file names.

## Image Generation Prompt

```text
把参考图中的宠物转换成一个高质量移动 App 宠物管家 mascot：
Q版卡通、全身坐姿、三分之二正面、保留原宠物毛色和眼睛特征、
头部略大、眼睛有神、毛发柔软但不过度写实、轮廓干净、
透明背景、无文字、无道具、无边框、适合做手机首页宠物形象。
输出必须是单只宠物，主体居中，完整身体，不要裁切耳朵和尾巴。
```

## Run

```powershell
cd D:\work\ai_tools\paimomo_pet_mvp
python -m http.server 8788 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8788/index.html
```

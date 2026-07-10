# PaddlePaddle / PaddleSeg PP-MattingV2 provider

This provider runs Baidu PaddlePaddle's official
`PP-MattingV2-STDC1-human-512` PaddleSeg inference model and follows the central
benchmark CLI contract. It emits one same-basename, 8-bit RGBA PNG per input PNG.

## Domain warning

The official PaddleSeg Matting model zoo calls PP-MattingV2 a real-time **human
matting** model. The released checkpoint is trained for human portraits, not pets
and not arbitrary foreground objects. The included cat video test is a deliberate
cross-domain probe; a successful command does not imply suitable pet-matting
quality.

## Environment

Pinned runtime location: `D:\work\pet-avatar-3d-mvp\.venvs\paddle_matting`

```powershell
python -m venv .venvs\paddle_matting
.venvs\paddle_matting\Scripts\python.exe -m pip install `
  paddlepaddle-gpu==3.3.0 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
.venvs\paddle_matting\Scripts\python.exe -m pip install `
  opencv-python-headless==4.13.0.92 PyYAML==6.0.3
```

CPU alternative, in a clean environment:

```powershell
.venvs\paddle_matting\Scripts\python.exe -m pip install `
  paddlepaddle==3.3.0 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Do not install CPU and GPU Paddle wheels in the same environment.

## Model

The official archive and pinned PaddleSeg `v2.10.0` sparse checkout live under
`.models\paddle_matting\`. See `MODEL_CARD.md` for hashes, domain, transforms, and
license notes.

## Inference CLI

```powershell
.venvs\paddle_matting\Scripts\python.exe `
  matting_bench\providers\paddle_matting\infer.py `
  --input-dir matting_bench\data\pet_20260710_121221_5ce7716e\smoke `
  --output-dir matting_bench\providers\paddle_matting\evidence\rgba_cuda `
  --device cuda
```

`--device cuda` and `--device gpu` are aliases. Explicit CUDA requests fail if no
GPU is available; use `--device cpu` for an intentional fallback. `--device auto`
prefers CUDA and otherwise uses CPU.

## Reproduce the three-video smoke run

The script independently decodes frame 0, 48, and 96 from each 97-frame raw
video, then runs all nine images in one warmed-up inference process:

```powershell
powershell -ExecutionPolicy Bypass -File `
  matting_bench\providers\paddle_matting\run_smoke.ps1
```

Artifacts, extraction hashes, and per-frame metrics are written only below this
provider's `evidence\` directory.

## Verified result

The 2026-07-10 run succeeded on CUDA and CPU. CUDA mean model inference was
49.087 ms/frame (P95 73.605 ms); CPU mean was 1600.599 ms/frame. All 18 generated
CUDA/CPU files passed mode, size, and bit-depth validation. See `REPORT.md` and
`evidence/contact_sheet_cuda.jpg` for the complete record and visual review.

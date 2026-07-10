# ViTMatte adaptive-green provider

This provider uses the official `hustvl/vitmatte-small-composition-1k` checkpoint
through Hugging Face Transformers' native `VitMatteForImageMatting` implementation.
Inference is offline after the pinned model snapshot has been downloaded and verified.

## Install

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venvs/vitmatte
.venvs/vitmatte/Scripts/python.exe -m pip install --upgrade pip
.venvs/vitmatte/Scripts/python.exe -m pip install `
  -r matting_bench/providers/vitmatte/requirements.lock.txt
.venvs/vitmatte/Scripts/python.exe `
  matting_bench/providers/vitmatte/download_model.py
```

## Run

```powershell
.venvs/vitmatte/Scripts/python.exe `
  matting_bench/providers/vitmatte/infer.py `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/providers/vitmatte/evidence/smoke_rgba `
  --device cuda
```

The provider emits one 8-bit RGBA PNG with the same basename as every input PNG,
plus `metrics.json`. Optional `--diagnostics-dir` saves baseline alpha, generated
trimap, clamped model alpha, and final alpha images.

## Hybrid pipeline

1. The repository's existing adaptive green-screen implementation builds a shared
   profile and produces the initial alpha.
2. Initial alpha values at or below `0.02` seed known background; values at or above
   `0.98` seed known foreground.
3. Adaptive alpha at or below the background threshold stays locked as background.
   A 6-pixel-radius elliptical erosion of certain foreground creates a narrow inward
   unknown band without allowing the model to grow into pure green. These parameters
   are configurable with `--background-threshold`,
   `--foreground-threshold`, and `--unknown-radius`.
4. ViTMatte predicts alpha, but predictions are used only in the unknown band. Known
   background and foreground are forced back to 0 and 1.
5. RGB comes from the existing adaptive green-screen cleanup, while the repository's
   `refine_reframed_halo` applies its opaque green-halo refinement before RGBA packing.

## Provenance and licenses

- Upstream implementation: <https://github.com/hustvl/ViTMatte>, MIT license.
- Checkpoint: <https://huggingface.co/hustvl/vitmatte-small-composition-1k>, model
  card marked Apache-2.0.
- Pinned revision: `6a58ad7646403c1df626fbd746900aec7361ea1d`.
- Weight: `model.safetensors`, 103,294,572 bytes, SHA-256
  `bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb`.
- Transformers library: Apache-2.0.

The upstream code and hosted model card use different license labels. Distribution
should preserve both notices and should be reviewed against the intended product use.

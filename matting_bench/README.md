# Pet Matting Benchmark

This lab compares the current adaptive green-screen pipeline with local matting models
on exactly the same frames. Heavy model weights, virtual environments, decoded frames,
and output PNGs are intentionally ignored by Git.

## Dataset

```powershell
python matting_bench/prepare_dataset.py `
  --source poc_output/pet_20260710_121221_5ce7716e_real_after `
  --output matting_bench/data/pet_20260710_121221_5ce7716e
```

The default dataset contains all frames from `idle`, `fast_walk`, and `sleep`, plus a
fixed nine-frame subset (`0`, middle, last frame from every clip) for installation
smoke tests.

## Provider contract

Every provider exposes an `infer.py` command with this interface:

```powershell
python infer.py --input-dir <png_dir> --output-dir <rgba_dir> --device cuda
```

Output files must use the same basename as each input and must be 8-bit RGBA PNGs.

## Baseline

```powershell
python matting_bench/providers/baseline/infer.py `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/outputs/baseline/smoke
```

## Tested providers

- Current adaptive chroma-key baseline
- BiRefNet and BiRefNet-matting
- ViTMatte-S
- PaddleSeg PP-MattingV2 human checkpoint
- BEN2 Base
- rembg U2Net, ISNet General, and BiRefNet General Lite
- MatAnyone v1 and SAM 2.1 Small for consecutive video propagation
- RMBG-2.0 adapter only; its weight is gated and its self-hosted license is not
  suitable for this commercial production evaluation without separate acceptance

The fixed conclusions and model/license links are in `benchmark_catalog.json`.
The full benchmark report is in `MATTING_MODEL_BENCHMARK_20260710.md`.

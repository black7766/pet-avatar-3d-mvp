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

## Parameter tuning

Provider-specific tuning results follow `TUNING_SCHEMA.md`. The final sweep contains
69 configurations across nine providers. It uses a tuned adaptive-green baseline,
quality guardrails, commercial/fractional-alpha constraints, and a Pareto frontier;
it deliberately does not collapse matting quality into one weighted score.

```powershell
python matting_bench/aggregate_tuning.py `
  --output matting_bench/outputs/tuning/aggregate_final.json `
  --strict-outputs

python matting_bench/render_tuning_html.py `
  --aggregate matting_bench/outputs/tuning/aggregate_final.json `
  --output poc_output/matting_tuning_report_20260711.html
```

The renderer copies each provider's recommended output into a page-local asset
directory so the existing `poc_output` HTTP server can display it without exposing
the model/weight directories.

- Final tuning report: `MATTING_PARAMETER_TUNING_20260711.md`
- Local dashboard: `http://127.0.0.1:8792/matting_tuning_report_20260711.html`
- GPU serialization helper: `run_with_gpu_lock.py`

## Entity-only animated comparison

Build a page that converts every provider's recommended full fast-walk and
sleep outputs to synchronized transparent WebPs. It also reads the real-version generation
metrics and shows API generation time, local matting time, token usage, per-provider
runtime, VRAM, static quality metrics, and temporal error.

```powershell
python matting_bench/run_action_compare.py --action fast_walk
python matting_bench/run_action_compare.py --action sleep
python matting_bench/render_animated_compare.py
```

- Local page: `http://127.0.0.1:8792/matting_animated_compare_real_20260711.html`
- Scope: entity/real version only; no PaiMomo/cute-version assets
- Actions: entity-version `fast_walk` and `sleep`, switchable on the same page
- Playback: 640x640, 96 real consecutive frames, 19.2 FPS, 5-second silent loop
- Metrics: runtime and quality values switch with the selected action
- Controls: production/research filter, checker/white/black background, synchronized replay
- Loading: production candidates load first; hidden research animations load on demand

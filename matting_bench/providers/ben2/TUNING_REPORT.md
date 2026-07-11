# BEN2 Base official-parameter tuning report

Date: 2026-07-11

## Scope and outcome

This run tuned the pinned open-source BEN2 Base implementation without changing
weights or the central evaluator. Three postprocess configurations were tested
on the same nine frames, including the unchanged official default.

The quality recommendation is `refine_foreground=true` with the official
`r=90` radius. It lowers the central green-fringe metric from `0.036927` to
`0.004709` and slightly improves the alpha pseudo metrics. The tradeoff is
substantial CPU postprocessing: the independent locked recheck measured
809.414 ms/frame end to end, versus 397.093 ms/frame without refinement.

## Official parameter sources

Pinned BEN2 source revision:
`2c99a5da477b5523585bfa5c893888a6e818a8f6`.

| Source | Tunable or fixed behavior established by the source |
|---|---|
| [Official quick start](https://github.com/PramaLLC/BEN2/blob/2c99a5da477b5523585bfa5c893888a6e818a8f6/README.md#quick-start-code-inside-cloned-repo) | `BEN_Base.inference(image, refine_foreground=False)`; refinement is optional, slower, and intended to improve edges. |
| [Official batch example](https://github.com/PramaLLC/BEN2/blob/2c99a5da477b5523585bfa5c893888a6e818a8f6/README.md#batch-image-processing) | Input may be one PIL image or an image list; README recommends no more than three images on consumer GPUs. |
| [Inference implementation](https://github.com/PramaLLC/BEN2/blob/2c99a5da477b5523585bfa5c893888a6e818a8f6/src/ben2/modeling_ben2.py#L1154-L1225) | No-refine uses `postprocess_image`; refine uses raw sigmoid alpha plus `refine_foreground_process`. |
| [Input transforms](https://github.com/PramaLLC/BEN2/blob/2c99a5da477b5523585bfa5c893888a6e818a8f6/src/ben2/modeling_ben2.py#L1357-L1386) | RGB conversion, fixed 1024x1024 LANCZOS resize, ImageNet normalization, float16 on CUDA and float32 on CPU. |
| [Postprocess helpers](https://github.com/PramaLLC/BEN2/blob/2c99a5da477b5523585bfa5c893888a6e818a8f6/src/ben2/modeling_ben2.py#L1520-L1559) | Foreground refinement exposes initial blur radius `r=90`; default alpha restore is bilinear plus per-image min-max normalization. |
| [Official Base model repository](https://huggingface.co/PramaLLC/BEN2) | Pinned `model.safetensors` and model configuration. |

The official Base implementation does not expose a tunable network resolution:
all inputs are resized to 1024x1024. This benchmark therefore kept that fixed
instead of introducing unsupported 512/768 variants. The provider now exposes
`--refine-foreground` and `--refine-radius`; omitting them preserves the
official no-refine default.

Important semantic detail: `refine_foreground` is not only an RGB cleanup
switch in the pinned code. It also changes alpha postprocessing from bilinear
restore plus min-max normalization to quantized raw sigmoid alpha resized by
PIL. The adapter follows both official branches exactly.

## Method

- Smoke input: `matting_bench/data/pet_20260710_121221_5ce7716e/smoke`
- Frames: 9 fixed 960x960 PNGs, three each from fast walk, idle, and sleep
- Temporal input: `temporal_fast_walk_24_640`, 24 consecutive 640x640 PNGs
- Device: NVIDIA GeForce RTX 2080 Ti 11 GB, driver 591.86
- Runtime: Python 3.11.9, PyTorch 2.5.1+cu121
- Warm-up: one full pipeline call before measurement
- Evaluator: unchanged `matting_bench/evaluate.py`
- Serialization: every reported CUDA command ran through
  `python matting_bench/run_with_gpu_lock.py -- ...`

`mean_inference_ms` is synchronized CUDA model-forward time.
`end_to_end_ms` includes image load, fixed-square preprocessing, model forward,
official alpha/RGB postprocess, output validation, SHA-256, and PNG save.
`peak_vram_mb` is the PyTorch allocator peak and includes the resident model.
The central quality metrics are green-screen pseudo metrics, not alpha ground
truth.

## Locked smoke results

| Config | Refine | Radius | Pseudo MAE | BG alpha | FG loss | Green fringe | Fragment % | Soft alpha % | Inference ms | End-to-end ms | Peak VRAM MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default_no_refine` | no | n/a | 0.005452 | 0.000565 | 0.010339 | 0.036927 | 0.012798 | 1.3107 | 266.655 | 397.093 | 2587.360 |
| `refine_r90` | yes | 90 | 0.005304 | 0.000506 | 0.010101 | 0.004709 | 0.012795 | 1.1823 | 264.574 | 809.414 | 2587.360 |
| `refine_r45` | yes | 45 | 0.005304 | 0.000506 | 0.010101 | 0.005444 | 0.012795 | 1.1823 | 255.907 | 701.505 | 2587.360 |

The `refine_r90` speed and VRAM values are from the required independent locked
recheck. Its first locked matrix run measured 259.699 ms inference and
711.169 ms end to end. The forward and VRAM values stay nearly unchanged across
configs because refinement is a CPU RGB estimator after the CUDA model.

`r=90` and `r=45` have identical alpha metrics because radius affects only the
estimated foreground RGB. The official 90 radius is retained because its
`green_fringe` is lower. Use the official no-refine default when latency matters
more than delivering cleaned foreground RGB, or when a downstream stage already
performs foreground-color estimation.

Visual evidence:

- `evidence/tuning/locked_smoke_contact_sheet.jpg`
- `evidence/tuning/locked_smoke_fast_walk_0048_zoom.jpg`
- `evidence/tuning/locked_smoke_evaluation.json`
- `evidence/tuning/locked_recheck_evaluation.json`

## Recommended temporal run

Configuration: fixed 1024x1024 input, float16 CUDA transform,
`refine_foreground=true`, `r=90`.

| Frames | Pseudo MAE | BG alpha | FG loss | Green fringe | Fragment % | Soft alpha % | Temporal alpha MAE | Inference ms | End-to-end ms | Peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.007830 | 0.000562 | 0.015097 | 0.002902 | 0.000828 | 1.4482 | 0.01470284 | 263.269 | 513.361 | 2587.360 |

Output and central evaluation:

- `evidence/tuning/locked_temporal/refine_r90`
- `evidence/tuning/locked_temporal_evaluation.json`

## Failures and lock handling

No BEN2 model configuration failed. The Windows lock helper intermittently
returned `OSError [Errno 36] Resource deadlock avoided` before launching the
model. Those attempts are preserved in `evidence/tuning` logs and excluded from
all results. Every accepted run log contains both `acquired GPU benchmark lock`
and `released GPU benchmark lock`.

## Reproduction

```powershell
python matting_bench/run_with_gpu_lock.py -- `
  .venvs/ben2/Scripts/python.exe `
  matting_bench/providers/ben2/infer.py `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/providers/ben2/evidence/tuning/repro/refine_r90 `
  --device cuda --warmup-runs 1 `
  --refine-foreground --refine-radius 90

python matting_bench/evaluate.py `
  --source-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --provider refine_r90=matting_bench/providers/ben2/evidence/tuning/repro/refine_r90 `
  --output matting_bench/providers/ben2/evidence/tuning/repro/evaluation.json
```

Machine-readable results are in `tuning_results.json`.

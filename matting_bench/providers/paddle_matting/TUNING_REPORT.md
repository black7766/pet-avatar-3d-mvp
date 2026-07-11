# PP-MattingV2 official-parameter tuning report

Date: 2026-07-11

## Scope and outcome

This run tuned the pinned PaddleSeg `PP-MattingV2-STDC1-human-512` deployment
without changing weights or the central evaluator. The official 512 short-edge
configuration remains the recommendation for this provider.

The 640 run has the lowest green-screen pseudo MAE, but visual review found a
detached green fragment above `idle__f_0048`; its mean `fragment_pct` is
`0.166049`, versus `0.008115` at 512. The default 512 setting is therefore the
better quality/stability tradeoff. This conclusion is only about settings for
this checkpoint. PaddleSeg labels it as a human-matting model, so the pet test
is cross-domain and does not make it the production pet-matting recommendation.

## Official parameter sources

Pinned PaddleSeg revision: `d459390adcec7fa6dd010c21b71aeb73f2afded9`
(`v2.10.0`).

| Source | Tunable or fixed behavior established by the source |
|---|---|
| [ResizeToIntMult source](https://github.com/PaddlePaddle/PaddleSeg/blob/d459390adcec7fa6dd010c21b71aeb73f2afded9/Matting/ppmatting/transforms/transforms.py#L231-L253) | `ResizeToIntMult.mult_int=32` floors both dimensions to an integer multiple. |
| [LimitShort source](https://github.com/PaddlePaddle/PaddleSeg/blob/d459390adcec7fa6dd010c21b71aeb73f2afded9/Matting/ppmatting/transforms/transforms.py#L442-L493) | Optional `max_short` downsizes a larger short edge; optional `min_short` enlarges a smaller short edge; aspect ratio is preserved. |
| [PP-MattingV2 human-512 config](https://github.com/PaddlePaddle/PaddleSeg/blob/d459390adcec7fa6dd010c21b71aeb73f2afded9/Matting/configs/ppmattingv2/ppmattingv2-stdc1-human_512.yml) | Released validation pipeline: `LimitShort(max_short=512)`, `ResizeToIntMult(32)`, `Normalize`. |
| [Official deploy CLI](https://github.com/PaddlePaddle/PaddleSeg/blob/d459390adcec7fa6dd010c21b71aeb73f2afded9/Matting/deploy/python/infer.py#L39-L146) | Other deploy controls include batch size, CPU/GPU, foreground estimation, CPU threads, MKLDNN, TensorRT, and TensorRT precision. |
| [Paddle peak-memory API](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/max_memory_allocated_cn.html) | `max_memory_allocated` returns peak tensor memory in bytes for the selected device. |

The checked-in exported `deploy.yaml` confirms `input_shape: [1,3,512,512]`,
`max_short: 512`, and `mult_int: 32`. The provider now exposes `--max-short`,
`--min-short`, and `--resize-mult`; omitting all three preserves that file.

## Method

- Smoke input: `matting_bench/data/pet_20260710_121221_5ce7716e/smoke`
- Frames: 9 fixed 960x960 PNGs, three each from fast walk, idle, and sleep
- Temporal input: `temporal_fast_walk_24_640`, 24 consecutive 640x640 PNGs
- Device: NVIDIA GeForce RTX 2080 Ti 11 GB, driver 591.86
- Runtime: Python 3.11.9, PaddlePaddle 3.3.0
- Warm-up: one full inference before measurement
- Evaluator: unchanged `matting_bench/evaluate.py`
- Serialization: every reported CUDA command ran through
  `python matting_bench/run_with_gpu_lock.py -- ...`

`mean_inference_ms` measures the predictor call and output copy.
`end_to_end_ms` is the per-frame mean including decode, preprocessing,
restoration, validation, SHA-256, and PNG save. `peak_vram_mb` is the Paddle
allocator peak after warm-up and includes resident inference allocations. The
central quality metrics are green-screen pseudo metrics, not alpha ground truth.

## Locked smoke results

| Config | Network size on smoke | Pseudo MAE | BG alpha | FG loss | Green fringe | Fragment % | Soft alpha % | Inference ms | End-to-end ms | Peak VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short_384` | 384x384 | 0.003111 | 0.004045 | 0.002178 | 0.100696 | 0.037520 | 2.2464 | 15.226 | 110.022 | 175.990 |
| `default_512` | 512x512 | 0.002345 | 0.003455 | 0.001235 | 0.107512 | 0.008115 | 1.7642 | 18.532 | 120.048 | 186.709 |
| `short_640` | 640x640 | 0.002169 | 0.003492 | 0.000846 | 0.114897 | 0.166049 | 1.6246 | 20.026 | 126.196 | 200.490 |

The `default_512` speed and VRAM values above come from the required independent
locked recheck. Its first locked matrix run measured 24.626 ms inference and
124.250 ms end to end; the central quality metrics reproduced exactly. Timing
variance is expected on this shared Windows desktop, so the recheck is retained
as the final runtime datum.

Visual evidence:

- `evidence/tuning/locked_smoke_contact_sheet.jpg`
- `evidence/tuning/locked_smoke_fast_walk_0048_zoom.jpg`
- `evidence/tuning/locked_smoke_evaluation.json`
- `evidence/tuning/locked_recheck_evaluation.json`

## Recommended temporal run

Configuration: official `max_short=512`, no `min_short`, `mult_int=32`.

| Frames | Pseudo MAE | BG alpha | FG loss | Green fringe | Fragment % | Soft alpha % | Temporal alpha MAE | Inference ms | End-to-end ms | Peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.002822 | 0.003556 | 0.002088 | 0.093759 | 0.041644 | 1.9872 | 0.01207656 | 18.334 | 65.325 | 186.709 |

Output and central evaluation:

- `evidence/tuning/locked_temporal/default_512`
- `evidence/tuning/locked_temporal_evaluation.json`

## Failures and lock handling

No model configuration failed. The Windows lock helper intermittently returned
`OSError [Errno 36] Resource deadlock avoided` before launching the model. The
640 smoke run succeeded on retry; the recommended recheck succeeded after 21
such acquisition failures. Failed attempts are preserved in `evidence/tuning`
logs and are excluded from all runtime numbers. Every accepted log contains
both `acquired GPU benchmark lock` and `released GPU benchmark lock`.

## Reproduction

```powershell
python matting_bench/run_with_gpu_lock.py -- `
  .venvs/paddle_matting/Scripts/python.exe `
  matting_bench/providers/paddle_matting/infer.py `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/providers/paddle_matting/evidence/tuning/repro/default_512 `
  --device cuda --warmup-runs 1

python matting_bench/evaluate.py `
  --source-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --provider default_512=matting_bench/providers/paddle_matting/evidence/tuning/repro/default_512 `
  --output matting_bench/providers/paddle_matting/evidence/tuning/repro/evaluation.json
```

Machine-readable results are in `tuning_results.json`.

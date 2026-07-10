# PaddleSeg PP-MattingV2 local deployment report

Date: 2026-07-10

## Outcome

Official PaddleSeg `PP-MattingV2-STDC1-human-512` is deployed locally and works
on both CUDA and CPU under Windows. The unified `infer.py` contract produced and
independently validated 9/9 same-basename, 960x960, 8-bit RGBA PNGs.

The latest stable PaddleSeg release remains `v2.10.0`; its Matting model zoo still
identifies PP-MattingV2 as the lightweight real-time choice. No official pet
matting checkpoint is listed there.

PP-MattingV2 was selected over PP-Matting-512 because the official V100 table
reports 98.89 FPS and 8.95M parameters for V2 versus 28.9 FPS and 24.5M parameters
for PP-Matting-512. Both downloadable `human_512` checkpoints are human-focused;
the higher-accuracy, slower PP-Matting checkpoint was not duplicated in this
real-time provider test.

This checkpoint is **human-portrait-specific**. PaddleSeg describes the released
PP-MattingV2 checkpoint as a real-time human matting model. The cat results below
are cross-domain observations, not an officially supported pet use case.

## Fixed inputs

Source directory:
`poc_output/pet_20260710_121221_5ce7716e_real_after`

| Clip | Video SHA-256 | Frames / FPS / size | Selected frames |
|---|---|---|---|
| `idle` | `5eee8e51caf4341fb60bddaf76381cf878365a59e82015aa160f0601bf168357` | 97 / 24 / 960x960 | 0, 48, 96 |
| `fast_walk` | `4a854c4fec7d0c3e0f60ac34f3e0b9521a79906a664dc01baeac7901ec9c90f5` | 97 / 24 / 960x960 | 0, 48, 96 |
| `sleep` | `e80995b53873e36ff70b8e4fa59195217e84924cf02757ef0104e33ffc62bd3f` | 97 / 24 / 960x960 | 0, 48, 96 |

Frames were sequentially decoded with OpenCV rather than random-seeked. Full
frame hashes are in `evidence/frames/extraction_manifest.json`. All nine hashes
also match the central benchmark's independently prepared smoke frames.

## Runtime

- OS: Windows 10 build `26200`, x86-64
- Python: `3.11.9`
- GPU: NVIDIA GeForce RTX 2080 Ti, compute capability 7.5
- Driver API reported by Paddle: CUDA `13.1`
- PaddlePaddle GPU wheel: `3.3.0`, bundled CUDA `11.8`, cuDNN `8.9.6`
- OpenCV: `4.13.0.92`
- Model inference input: 512x512 for all fixed 960x960 frames
- GPU environment footprint: 4.157 GiB / 9,875 files
- Model/source footprint: 0.075 GiB / 480 files

Paddle's GPU self-check succeeded: `PaddlePaddle works well on 1 GPU`. The
self-check took 79.746 seconds after import on the first cold run.
Installing the official GPU wheel and its CUDA dependencies took 629.3 seconds;
downloading and extracting the 33.38 MB model archive took 7.3 seconds.

## Timing results

One warm-up pass was excluded from measured per-frame timing.

| Device | Predictor load | Warm-up | Mean inference | Median | P95 | Mean total incl. PNG save |
|---|---:|---:|---:|---:|---:|---:|
| CUDA | 9.917 s | 1.013 s | 49.087 ms | 47.741 ms | 73.605 ms | 398.521 ms |
| CPU, 8 threads | 7.673 s | 9.494 s | 1600.599 ms | 1556.587 ms | 2279.828 ms | 1813.848 ms |

CUDA inference was 32.6x faster than CPU on this sample. The first GPU frame's
PNG save was an outlier, so `mean_total_ms` should not be treated as model speed.
Use `mean_inference_ms` for provider comparison.

CPU and CUDA alpha outputs were numerically close after 8-bit packing: mean
absolute alpha difference was `0.0453 / 255`; the maximum single-pixel difference
was 60/255 in a small boundary region.

Independent post-run validation confirmed both device output sets contain the
same nine basenames and every file decodes as `uint8[960, 960, 4]` in RGBA mode.

## Quality observations

Visual result: `evidence/contact_sheet_cuda.jpg`

Against the repository's adaptive green-screen baseline on the same nine frames:

| Metric | Adaptive baseline | PP-MattingV2 | Interpretation |
|---|---:|---:|---|
| Pseudo MAE | 0.001791 | 0.002345 | PP-MattingV2 is 30.9% higher |
| Background alpha mean | 0.000831 | 0.003455 | more background leakage |
| Foreground loss mean | 0.002752 | 0.001235 | better foreground retention |
| Green fringe | 0.014523 | 0.107512 | 7.4x more green spill at soft edges |
| Opaque green leak | 0.1037% | 0.2991% | 2.9x higher |
| Fragment ratio | 0.1345% | 0.0081% | much cleaner connected silhouette |

The model extracts the cat body, ears, legs, and tail surprisingly well despite
the human-only training domain. It is not a drop-in replacement for the current
pipeline because source RGB at soft alpha edges retains visible green. A later
integration experiment should compare model alpha plus the existing despill/edge
postprocess, but this provider does not modify the central harness.

## Reproduction

Environment and model setup:

```powershell
python -m venv .venvs\paddle_matting
.venvs\paddle_matting\Scripts\python.exe -m pip install `
  paddlepaddle-gpu==3.3.0 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
.venvs\paddle_matting\Scripts\python.exe -m pip install `
  opencv-python-headless==4.13.0.92 PyYAML==6.0.3

curl.exe -L --fail `
  -o .models\paddle_matting\ppmattingv2-stdc1-human_512.zip `
  https://paddleseg.bj.bcebos.com/matting/models/deploy/ppmattingv2-stdc1-human_512.zip
Expand-Archive `
  .models\paddle_matting\ppmattingv2-stdc1-human_512.zip `
  .models\paddle_matting -Force
```

Nine-frame CUDA run:

```powershell
powershell -ExecutionPolicy Bypass -File `
  matting_bench\providers\paddle_matting\run_smoke.ps1 `
  -Device cuda
```

CPU fallback:

```powershell
.venvs\paddle_matting\Scripts\python.exe `
  matting_bench\providers\paddle_matting\infer.py `
  --input-dir matting_bench\providers\paddle_matting\evidence\frames `
  --output-dir matting_bench\providers\paddle_matting\evidence\rgba_cpu `
  --device cpu --warmup-runs 1
```

## Failures and warnings retained

1. Initial smoke launch stopped before inference because Windows PowerShell 5
   converted Paddle's harmless native stderr (`ccache` not found) into a
   terminating error under `$ErrorActionPreference = "Stop"`. The runner now
   checks native exit codes explicitly while allowing stderr. Log:
   `evidence/run_smoke_cuda.log`.
2. The first corrected launch reached model inference but the adapter treated
   Paddle 3.3's successful `Predictor.run()` return value (`None`) as false. The
   adapter now follows the official example and relies on raised exceptions.
   Log: `evidence/run_smoke_cuda_retry.log`.
3. Paddle emits a generic warning that compiled cuDNN 8.9 and machine cuDNN 8.9
   "may cause serious incompatible bug", even though the reported versions match.
   GPU self-check, static-model smoke, warm-up, and all nine outputs succeeded;
   no incompatibility was observed. Final log: `evidence/run_smoke_cuda_final.log`.

## Model and license

- PaddleSeg tag: `v2.10.0`
- Commit: `d459390adcec7fa6dd010c21b71aeb73f2afded9`
- Model archive SHA-256:
  `daff48b08c61958b9a21093791f6aed8eb3939b34b7418e40c18b2348136893d`
- PaddleSeg code license: Apache-2.0, copied to
  `.models/paddle_matting/LICENSE.PaddleSeg.Apache-2.0.txt`
- The official weight archive includes no separate license file. It is hosted by
  the official PaddleSeg model zoo. Confirm a separate weight-distribution grant
  with Baidu/PaddlePaddle before commercial redistribution if required.

Exact model-file hashes and preprocessing are in `MODEL_CARD.md`; the complete
installed environment is in `.models/paddle_matting/environment.freeze.txt`.

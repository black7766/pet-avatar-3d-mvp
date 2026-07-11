# MatAnyone v1 Tuning Report

## Recommendation

For this 24-frame 640x640 pet clip, use `warmup=1`, binary frame-0 mask
threshold `128`, FP16 autocast, `mem_every=5`, `max_mem_frames=5`, and no
long-term memory. This configuration had the lowest proxy/reference error,
foreground loss, flow-compensated temporal alpha MAE, and internal-hole rate.
Visual review confirmed that it retained the ears, tail, legs, and body silhouette.

MatAnyone emits fractional alpha but not foreground color. Keep the existing
green-screen RGB cleanup; do not treat this model as a complete spill-removal path.
The pinned S-Lab License 1.0 is also a commercial-use blocker until permission is
obtained.

## Controlled Test

- Input: `matting_bench/data/pet_20260710_121221_5ce7716e/temporal_fast_walk_24_640`
- Sequence: `f_0000.png` through `f_0023.png`, 24 consecutive frames
- Size: 640x640 for every configuration
- Sequence SHA-256: `550eec0273bc719d663aa8c455683f8deeecb4cb397c989478304674fb484e1e`
- GPU: NVIDIA GeForce RTX 2080 Ti 11GB, driver 591.86, Torch 2.5.1+cu124
- Model: official MatAnyone v1 checkpoint at commit
  `e5ddc534c1fff9bb9e54cf476095d29071b7cb4f`
- Serialization: every reported CUDA run was launched through
  `python matting_bench/run_with_gpu_lock.py -- ...`

Quality uses the repository's controlled-green-screen confident-region proxy.
`temporal_alpha_mae` is optical-flow compensated at 480x480. The green reference
is diagnostic, not hand-labeled alpha ground truth.

## Results

| Config | Pseudo MAE | BG alpha | FG loss | Green fringe | Fragments | Soft alpha | Temporal MAE | Ref alpha MAE | Internal holes | Inference ms/frame | Peak VRAM MiB | Process E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.006150 | 0.003440 | 0.008861 | 0.008507 | 0.000125% | 1.996958% | 0.014170 | 0.004751 | 0.3665% | 82.733 | 697.736 | 58841.258 |
| threshold64 | 0.003411 | 0.003444 | 0.003378 | 0.010744 | 0.000077% | 1.513255% | 0.013164 | 0.003396 | 0.0888% | 67.075 | 697.736 | 17823.006 |
| **warmup1** | **0.002175** | **0.003434** | **0.000916** | 0.012534 | 0.000080% | 1.321828% | **0.012900** | **0.002823** | **0.0272%** | 83.467 | 697.736 | 20532.177 |
| mem_every1 | 0.007227 | 0.003441 | 0.011013 | **0.008150** | 0.000170% | 2.087585% | 0.014224 | 0.005261 | 0.6133% | 64.721 | 697.736 | 17759.893 |

All model outputs used 256 uint8 alpha values. Lower is better for every metric in
the table except soft-alpha percentage, which is descriptive rather than a quality
score. `warmup1` increases the soft-edge green-fringe score versus default, so edge
color should be rechecked on more fur colors even though its alpha is materially
cleaner.

Process end-to-end time is a fresh Python-process measurement, but the first run paid
one-time Windows module and CUDA kernel-cache costs. OS/CUDA caches were not flushed
between configurations, so use per-frame inference for within-session throughput and
do not rank configurations by the E2E column alone.

## Official Tunables

| Area | Official control | Finding for this run |
|---|---|---|
| First-frame assignment | The official API consumes a first-frame segmentation mask; its CLI also exposes erosion and dilation. The project wrapper adds a green-alpha threshold before that API. | Threshold 64 reduced holes, but one warmup pass with threshold 128 was better overall. |
| Warmup | Upstream defaults to 10 passes, uses 1 for its 512x288 evaluation recipe, and 10 for 1920x1080. | One pass was best at 640 and removed most pet-interior gray holes. |
| Resolution | `max_size` downsizes by shortest side; `max_internal_size` can downsize inside `InferenceCore` and restore output size. | Locked to 640 for fair comparison; internal resizing remained disabled (`-1`). |
| Precision | Autocast is supported; this wrapper exposes FP16 and `--no-amp` FP32. | FP16 was held constant. No FP32 timing is claimed because the optional run never acquired the shared lock. |
| Working memory | `mem_every`, `max_mem_frames`, `stagger_updates`, and `top_k`. | `mem_every=1` was worse; retain official `5/5`, stagger 5, top-k 30. |
| Long-term memory | `use_long_term` plus prototype/token limits. | Not useful for only 24 frames; disabled. Test separately on long clips. |
| Object batching | `chunk_size`; optional `flip_aug`. | Single object, unlimited chunking, no flip augmentation. |
| Offload | No official inference-state/video CPU-offload switch is present in MatAnyone v1. | Report as unsupported rather than inventing an offload result. |

Official sources are pinned in `tuning_results.json`. Key files are the upstream
README, `evaluation/inference_matanyone_yt.py`,
`matanyone/config/eval_matanyone_config.yaml`, and
`matanyone/inference/inference_core.py`.

## Propagation and Alpha Limits

- The model is presented upstream as human video matting with target assignment; pet
  behavior is out of the stated primary domain.
- `mem_every=1` increased recurrent interior corruption, showing that denser memory
  updates are not automatically safer.
- The proxy foreground includes highly non-green pixels. Eyes and dark markings can
  therefore dominate foreground-loss changes; the separate opaque-interior hole metric
  and contact sheets were checked alongside it.
- The model predicts alpha only. `green_fringe` here also depends on the existing RGB
  cleanup policy, not solely on MatAnyone.
- A 24-frame clip cannot validate long-term memory eviction, reappearance after
  occlusion, or scene-cut recovery.

## Reproduce Recommended Run

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python .\matting_bench\run_with_gpu_lock.py -- `
  .\.venvs\video_matting\Scripts\python.exe `
  .\matting_bench\providers\video_matting\matanyone_cli.py `
  --input .\matting_bench\data\pet_20260710_121221_5ce7716e\temporal_fast_walk_24_640 `
  --output-dir .\matting_bench\providers\video_matting\runs\tuning_warmup1 `
  --frames 24 --max-size 640 --warmup 1 --init-kind mask `
  --mask-threshold 128 --mem-every 5 --max-mem-frames 5 --overwrite
```

# SAM 2.1 Small Video Tuning Report

## Recommendation

For this bounded 24-frame, one-object workload, use frame-0 threshold `128`,
propagated-logit threshold `0`, FP16, video CPU offload, state CPU offload,
synchronous loading, official postprocessing enabled, and no VOS compilation.
The state-offload result was bit-identical to the quality-best default, reduced peak
allocated VRAM by 16.514 MiB, and had no measurable propagation penalty here.

For long or multi-object videos, keep state on GPU unless memory pressure requires
offload. The official predictor explicitly warns that state CPU offload lowers tracking
fps. The 24-frame result is not evidence that the cost disappears at larger scale.

Most importantly, SAM 2.1 is a segmentation/tracking model. Every delivered alpha
pixel is exactly 0 or 255. It cannot represent fractional fur opacity and is not a
replacement for a matting model.

## Controlled Test

- Input: `matting_bench/data/pet_20260710_121221_5ce7716e/temporal_fast_walk_24_640`
- Sequence: `f_0000.png` through `f_0023.png`, 24 consecutive frames
- Input/output size: 640x640; official model internal size: 1024x1024
- Sequence SHA-256: `550eec0273bc719d663aa8c455683f8deeecb4cb397c989478304674fb484e1e`
- GPU: NVIDIA GeForce RTX 2080 Ti 11GB, driver 591.86, Torch 2.5.1+cu124
- Model: official SAM 2.1 Hiera Small at commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Checkpoint: `sam2.1_hiera_small.pt`, SHA-256
  `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`
- Serialization: every reported CUDA run was launched through
  `python matting_bench/run_with_gpu_lock.py -- ...`

Quality uses the repository's controlled-green-screen confident-region proxy.
`temporal_alpha_mae` is optical-flow compensated at 480x480. The green reference
is diagnostic, not hand-labeled alpha ground truth.

## Results

| Config | Pseudo MAE | BG alpha | FG loss | Green fringe | Fragments | Soft alpha | Temporal MAE | Ref IoU | Ref alpha MAE | Internal holes | Inference ms/frame | Peak VRAM MiB | Process E2E ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | **0.000935** | **0.000707** | 0.001162 | 0.000000 | 0.000000% | 0.000000% | 0.014994 | 0.993051 | **0.002707** | 0.0741% | 79.804 | 712.615 | 14203.008 |
| threshold64 | 0.001155 | 0.001704 | 0.000606 | 0.000000 | 0.000000% | 0.000000% | **0.014688** | 0.992891 | 0.002833 | 0.0286% | **79.023** | 712.615 | 13872.960 |
| **state_cpu** | **0.000935** | **0.000707** | 0.001162 | 0.000000 | 0.000000% | 0.000000% | 0.014994 | 0.993051 | **0.002707** | 0.0741% | 79.797 | **696.101** | **13795.255** |
| fp32 | 0.000935 | **0.000707** | 0.001164 | 0.000000 | 0.000000% | 0.000000% | 0.014995 | **0.993053** | **0.002707** | 0.0741% | 176.012 | 762.135 | 16592.627 |
| logit-0.5 | 0.000943 | 0.001372 | **0.000515** | 0.000000 | 0.000000% | 0.000000% | 0.014746 | 0.992431 | 0.002764 | **0.0265%** | 81.344 | 712.615 | 14931.010 |

`green_fringe=0` and `soft_alpha_pct=0` are structural consequences of binary alpha.
The benchmark's green-fringe metric is evaluated only on fractional-alpha pixels, so
zero here does **not** establish clean fur color or spill-free edges. Source RGB is
preserved and can contain green contamination at opaque boundary pixels.

The final default timing is a dedicated GPU-locked rerun after functional screening.
Process E2E is a fresh Python-process measurement, but OS and CUDA caches were not
flushed. Per-frame propagation is the more useful comparison; model load, state init,
prompt encoding, output PNG writing, and report generation are included only in E2E.

## Official Tunables

| Area | Official control | Finding for this run |
|---|---|---|
| First-frame prompt | `add_new_mask` accepts a binary prompt. The project wrapper thresholds its green alpha first. | Threshold 64 reduced holes but raised background leakage; keep 128. |
| Propagation | `start_frame_idx`, `max_frame_num_to_track`, and `reverse`; output is raw mask logits. | Forward frame 0 through 23. Wrapper logit `-0.5` expanded edges but reduced overall reference agreement. |
| Resolution | Small config fixes `image_size: 1024` and resizes outputs to source dimensions. | Same 640 input/output for all runs; internal 1024 was not altered. |
| Precision | Official examples use autocast BF16. | RTX 2080 Ti is Turing (7.5), so BF16 was not used. FP32 was 2.21x slower than final FP16 with no material quality gain. |
| Video offload | `offload_video_to_cpu`; official comment says small overhead. | Enabled for all runs to bound frame-tensor VRAM. |
| State offload | `offload_state_to_cpu`; official comment warns of lower fps. | Saved 16.514 MiB and was output-identical with no short-clip penalty. Re-test long/multi-object video. |
| Frame loading | `async_loading_frames`. | Disabled for deterministic bounded testing and to avoid a persistent loader thread. |
| Postprocessing | Builder enables dynamic multimask behavior, prompt-mask binarization for memory, and `fill_hole_area=8`. | Enabled, but the optional `_C` CUDA extension is unavailable on this Windows runtime, so tiny-hole filling was skipped. |
| Compilation | `vos_optimized=True` compiles the full VOS model; config also exposes `compile_image_encoder`. | Not used. Compilation warmup would dominate a 24-frame cold run and needs a separate resident-service benchmark. |
| Warmup | No recurrent warmup parameter exists in the standard SAM 2 video predictor. | Reported as unsupported, not mapped from MatAnyone. |

Official sources are pinned in `tuning_results.json`: the README, `build_sam.py`,
`sam2_video_predictor.py`, the Small config, and `INSTALL.md`.

## Binary Alpha Limit

- All five configurations produced only alpha values `{0, 255}`.
- The green reference contains a mean 0.9561% fractional transition band. SAM can
  only include those pixels fully or discard them.
- Lower prompt/output thresholds exchange internal retention for opaque background
  leakage; they do not recover real fur transparency.
- `fragment_pct=0` means one connected binary subject, not accurate hair detail.
- A downstream true matting/refinement stage is required when fractional fur edges are
  part of the product contract.

## Reproduce Recommended Run

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python .\matting_bench\run_with_gpu_lock.py -- `
  .\.venvs\sam2_video\Scripts\python.exe `
  .\matting_bench\providers\sam2_video\infer.py `
  --input-dir .\matting_bench\data\pet_20260710_121221_5ce7716e\temporal_fast_walk_24_640 `
  --output-dir .\matting_bench\providers\sam2_video\runs\tuning_state_cpu `
  --frames 24 --mask-threshold 128 --logit-threshold 0 --precision fp16 `
  --offload-video-to-cpu --offload-state-to-cpu --apply-postprocessing --overwrite
```

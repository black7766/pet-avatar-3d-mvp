# Official MatAnyone video-matting provider

This directory is an isolated Windows/CUDA deployment of the official
[pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone) v1 code. It targets
arbitrary assigned objects from a frame-zero mask and does not substitute a
human-only RVM model for pet footage.

The original deployment result and evidence index are in
[`BENCHMARK.md`](BENCHMARK.md). The controlled default-plus-three sweep is in
[`TUNING_REPORT.md`](TUNING_REPORT.md), with machine-readable results in
[`tuning_results.json`](tuning_results.json). For the fixed 24-frame 640 input,
the newer tuning report supersedes the historical warmup recommendation.

## Reproduce

From the repository root in PowerShell:

```powershell
& .\matting_bench\providers\video_matting\setup.ps1

python .\matting_bench\run_with_gpu_lock.py -- `
  powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\matting_bench\providers\video_matting\run.ps1 `
  -InputPath .\matting_bench\data\pet_20260710_121221_5ce7716e\temporal_fast_walk_24_640 `
  -OutputDir .\matting_bench\providers\video_matting\runs\tuning_warmup1 `
  -Frames 24 `
  -MaxSize 640 `
  -Warmup 1 `
  -InitKind mask `
  -Overwrite
```

The CLI refuses to write outside this provider's `runs/` directory. The run emits:

- `init/`: frame-zero RGB, existing green-screen alpha, binary mask, and profile metadata.
- `alpha/frame_*.png`: MatAnyone alpha sequence.
- `rgba/frame_*.png`: unassociated RGBA sequence; source RGB is zeroed only at alpha zero.
- `contact_sheet_*.png`: visual checks across the sequence.
- `metrics.json`: source/checkpoint identity, timings, Torch allocator VRAM, and temporal diagnostics.

## Integration boundary

`matanyone_cli.py` loads `poc.py` read-only and calls
`profile_green_arrays([frame0])` followed by
`adaptive_green_matte_frame(frame0, profile)`. The default target assignment is
the thresholded binary mask, which matches the upstream training/inference
contract and performed better on the pet clip. Use `-InitKind alpha` to test the
soft alpha directly.

MatAnyone predicts alpha, not decontaminated foreground color. By default the
RGBA sequence therefore takes cleaned RGB from the existing green-screen
algorithm while retaining MatAnyone alpha unchanged. `-RgbaRgb source` disables
that color cleanup. The existing per-frame green alpha is retained only as a
diagnostic reference in `metrics.json`; it is never substituted for MatAnyone
alpha.

Video decoding uses OpenCV because the stock entrypoint relies on
`torchvision.io.read_video`/PyAV and its MP4 writers require an FFmpeg runtime.
The model, checkpoint loading, recurrent refinement, memory manager, and
`InferenceCore.step` are the official implementation at the pinned commit.

The upstream constructor defaults to downloading ImageNet ResNet-18 and
ResNet-50 weights before immediately replacing them with the release
checkpoint. The wrapper sets `cfg.model.pretrained_resnet=False`; this changes
no model layers or checkpoint values and removes two redundant downloads.

## Pinned assets

- Source: `https://github.com/pq-yang/MatAnyone`
- Commit: `e5ddc534c1fff9bb9e54cf476095d29071b7cb4f`
- Checkpoint: official GitHub release `v1.0.0/matanyone.pth`
- Checkpoint size: `141429992` bytes
- SHA-256: `dd26b991d020ed5eb4be50996f97354c45cfdfc0f59958e8983ac6a198f4809d`
- Environment: Python 3.11, Torch 2.5.1+cu124, TorchVision 0.20.1+cu124

## License and production constraint

MatAnyone uses **S-Lab License 1.0**. Source and binary redistribution/use are
permitted for non-commercial purposes under its conditions. Commercial use
requires permission from the contributors. This is a production licensing
blocker until written permission or a commercially compatible replacement is
secured.

## Evidence

Generated installation and CUDA/model-load logs are retained in `evidence/`.
Run-specific success or failure evidence is retained under `runs/` as
`metrics.json` or `failure.json`.

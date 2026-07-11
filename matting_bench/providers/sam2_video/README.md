# Official SAM 2.1 pet video-mask provider

This is an isolated Windows/CUDA experiment using Meta's official Apache-2.0
`facebookresearch/sam2` implementation. It propagates the existing green-screen
subject mask from frame 0 over 24 consecutive `fast_walk` frames with
`sam2.1_hiera_small`, selected for an RTX 2080 Ti 11 GB.

The controlled default-plus-four sweep is in
[`TUNING_REPORT.md`](TUNING_REPORT.md), with machine-readable results in
[`tuning_results.json`](tuning_results.json).

The provider is intentionally not registered in the central benchmark harness. It only
reads `poc.py` and the prepared input frames. All generated files remain under this
provider; source and environment assets remain under `.models/sam2_video` and
`.venvs/sam2_video`.

## Reproduce

From the repository root in PowerShell:

```powershell
& .\matting_bench\providers\sam2_video\setup.ps1
python .\matting_bench\run_with_gpu_lock.py -- `
  powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\matting_bench\providers\sam2_video\run.ps1 `
  -OffloadStateToCpu -Overwrite
```

The exact direct invocation is:

```powershell
& .\.venvs\sam2_video\Scripts\python.exe `
  .\matting_bench\providers\sam2_video\infer.py `
  --input-dir .\matting_bench\data\pet_20260710_121221_5ce7716e\temporal_fast_walk_24_640 `
  --output-dir .\matting_bench\providers\sam2_video\runs\tuning_state_cpu `
  --frames 24 `
  --mask-threshold 128 `
  --logit-threshold 0 `
  --precision fp16 `
  --offload-video-to-cpu `
  --offload-state-to-cpu `
  --overwrite
```

BF16 is deliberately not used because the RTX 2080 Ti is a Turing device (compute
capability 7.5). Video tensors are offloaded to CPU; recurrent state stays on GPU by
default.

## Output contract

- `init/`: original frame-0 RGB, existing green-screen RGBA/alpha, binary prompt mask,
  and initializer metadata.
- `mask/f_*.png`: propagated binary mask, preserving each source basename.
- `rgba/f_*.png`: byte-identical source RGB plus propagated mask as alpha, preserving
  each source basename.
- `metrics.json`: checkpoint provenance, CUDA timing, peak VRAM, temporal diagnostics,
  output validation, and limitations.
- `REPORT.md`: concise experiment result and commercial-license record.
- `contact_sheet_rgba.png` and `contact_sheet_mask.png`: visual QA samples.

SAM 2 is a segmentation/tracking model, not an alpha-matting model. Its alpha output is
binary. It cannot recover fractional fur transparency, and this experiment deliberately
does not perform green-spill removal because the output contract requires original RGB.

See `LICENSE_NOTES.md` for provenance and distribution obligations.

# rembg ONNX provider

This provider pins the official rembg package and compares three models from its
supported model registry:

- `u2net`
- `isnet-general-use`
- `birefnet-general-lite`

All dependencies, model files, caches, and benchmark outputs stay in the provider's
assigned directories. The adapter does not depend on or modify the central harness.

## Environment

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venvs/rembg
.venvs/rembg/Scripts/python.exe -m pip install -r matting_bench/providers/rembg/requirements.lock.txt
```

`onnxruntime-gpu[cuda,cudnn]` includes isolated CUDA 12 and cuDNN 9 runtime DLLs.
The GPU package also exposes `CPUExecutionProvider`, so the same venv supports both
`--device cuda` and `--device cpu`. `requirements.txt` records the two direct pins;
`requirements.lock.txt` freezes the complete tested environment.

## Models

Download all pinned weights through rembg's model registry and verify their official
MD5 checksums:

```powershell
.venvs/rembg/Scripts/python.exe matting_bench/providers/rembg/download_models.py
```

Weights and the generated download manifest are stored in `.models/rembg/`.

## Inference

The command reads PNG files directly under the input directory and writes same-name,
same-size, 8-bit RGBA PNGs to the output directory. A single rembg session is reused
for the full directory.

```powershell
.venvs/rembg/Scripts/python.exe matting_bench/providers/rembg/infer.py `
  --model birefnet-general-lite `
  --device cuda `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/providers/rembg/outputs/smoke/birefnet-general-lite
```

Each output directory includes `metrics.json` with model provenance, checksum, weight
size, active ONNX Runtime providers, session-load time, per-frame `remove()` time,
end-to-end time, and alpha-channel validation.

## Smoke summary

After running all three models:

```powershell
.venvs/rembg/Scripts/python.exe matting_bench/providers/rembg/summarize_smoke.py
```

This writes `smoke_results.json` and `SMOKE_RESULTS.md` in this provider directory.

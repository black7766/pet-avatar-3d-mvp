# BRIA RMBG-2.0 local provider

Standalone local inference for the official gated
[`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) model. The
implementation follows BRIA's official [GitHub repository](https://github.com/Bria-AI/RMBG-2.0)
and model-card preprocessing. It is intentionally not wired into the central
matting harness.

## License restriction

The Hugging Face weights are released under CC BY-NC 4.0 for non-commercial
use. Self-hosted commercial use is not granted by the model download and
requires a separate commercial agreement with BRIA. Do not use this provider
in a commercial product or production service without that agreement.

## Environment

The checked environment lives at `.venvs/rmbg2`. Recreate it from the repository
root with:

```powershell
py -3.11 -m venv .venvs\rmbg2
.venvs\rmbg2\Scripts\python.exe -m pip install -r matting_bench\providers\rmbg2\requirements.txt
```

The CUDA build was validated on an NVIDIA GeForce RTX 2080 Ti with the installed
driver. CPU inference is supported by the CLI but is expected to be much slower.

## Gated model access

1. Sign in at the official model page and accept its non-commercial terms.
2. Create a Hugging Face read token.
3. Set the token for the current shell. Do not store it in this repository.

```powershell
$env:HF_TOKEN = "hf_..."
```

On first use, `infer.py` downloads only the required files from the official
repository at pinned revision `5df4c9c76d8170882c34f6986e848ee07fd0ba43`.
All Hugging Face and Torch caches are redirected under `.models/rmbg2`.

## Inference

```powershell
.venvs\rmbg2\Scripts\python.exe matting_bench\providers\rmbg2\infer.py `
  --input-dir matting_bench\providers\rmbg2\samples\input `
  --output-dir matting_bench\providers\rmbg2\samples\output `
  --device cuda
```

Every supported input image produces a same-stem `.png` in RGBA mode. The CLI
prints JSON containing model-load time, per-image timing, and CUDA peak allocated
and reserved memory.

The official safetensors file is `884,878,856` bytes (about 844 MiB / 885 MB),
as reported by the official Hugging Face repository metadata.

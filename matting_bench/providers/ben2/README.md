# BEN2 Base local provider

This provider runs only the locally self-hosted, open-source **BEN2 Base**
variant from PramaLLC. It does not use the Full model, hosted API, or web demo.

## Pinned official assets

- Source: https://github.com/PramaLLC/BEN2
- Source revision: `2c99a5da477b5523585bfa5c893888a6e818a8f6`
- Weight repository: https://huggingface.co/PramaLLC/BEN2
- Weight revision: `e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f`
- Weight: `model.safetensors`, 380,577,976 bytes (362.947 MiB)
- Weight SHA-256:
  `ea8b7907176a09667c86343dc7d00de6a6d871076cb90bb5f753618fd6fb3ebb`

The source repository includes an MIT license (Copyright 2025 Prama LLC), and
the pinned Hugging Face model card declares `license: mit`. The GitHub README
states that Base is open source and directs users to the separate commercial
channel for the Full model.

## Unified inference

~~~powershell
.venvs\ben2\Scripts\python.exe `
  matting_bench\providers\ben2\infer.py `
  --input-dir matting_bench\data\pet_20260710_121221_5ce7716e\smoke `
  --output-dir matting_bench\providers\ben2\evidence\rgba_cuda `
  --device cuda
~~~

The command loads only local files, performs one default warm-up, and writes an
8-bit RGBA PNG for every supported input image. PNG inputs retain the exact same
filename. Runtime and per-frame validation data are written to
`<output-dir>/metrics.json`.

The adapter follows the official 1024x1024 preprocessing and Base forward path,
uses the official per-image alpha postprocessing, and keeps
`refine_foreground=False`. It invokes these official primitives directly so
core CUDA forward time can be separated from image load, preprocessing, alpha
postprocessing, validation, and PNG save time.

## Local layout

- `.models/ben2/source`: pinned official GitHub checkout
- `.models/ben2/model`: pinned safetensors weight, config, and model card
- `.models/ben2/LICENSE.PramaLLC-BEN2.MIT.txt`: copied official license
- `.models/ben2/huggingface-model-info.json`: pinned model metadata
- `.venvs/ben2`: isolated Python environment

See `REPORT.md` for the fixed nine-frame CUDA result and pet-fur risk review.

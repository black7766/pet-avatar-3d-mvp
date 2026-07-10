# BEN2 Base local deployment and smoke report

Date: 2026-07-10

## Outcome

The official PramaLLC BEN2 Base model is deployed locally with a pinned source
checkout, pinned safetensors weight, isolated CUDA environment, and offline-only
runtime caches. The unified provider command produced and validated 9/9
same-name, 960x960, 8-bit RGBA PNGs.

Only the open-source Base variant was tested. The Full model, hosted API, web
demo, ONNX export, and legacy `BEN2_Base.pth` were not used. The run used the
official 1024x1024 Base path with `refine_foreground=False`.

## Official source and license

| Asset | Pinned source |
|---|---|
| Code | https://github.com/PramaLLC/BEN2 |
| Code revision | `2c99a5da477b5523585bfa5c893888a6e818a8f6` |
| Weight/model card | https://huggingface.co/PramaLLC/BEN2 |
| Weight revision | `e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f` |
| Weight file | `model.safetensors` |
| Weight size | 380,577,976 bytes / 362.947 MiB / 0.354 GiB |
| Weight SHA-256 | `ea8b7907176a09667c86343dc7d00de6a6d871076cb90bb5f753618fd6fb3ebb` |
| Parameters | 94,134,658 |

The pinned GitHub repository contains the MIT license, Copyright 2025 Prama
LLC. The pinned Hugging Face model card independently declares `license: mit`.
The GitHub README identifies Base as open source and routes the separate Full
model to Prama's commercial channel. The MIT notice is copied to
`.models/ben2/LICENSE.PramaLLC-BEN2.MIT.txt`, and pinned Hugging Face metadata
is retained in `.models/ben2/huggingface-model-info.json`.

On that published evidence, the tested code and Base weight are MIT-licensed,
including commercial-use permission subject to retaining the notice. This is a
technical license check, not legal advice; it does not extend to the Full/API
model.

## Fixed test

Input:
`matting_bench/data/pet_20260710_121221_5ce7716e/smoke`

The sample contains 9 fixed 960x960 PNGs: frames 0, 48, and 96 from each of
`fast_walk`, `idle`, and `sleep`.

Primary output:
`matting_bench/providers/ben2/evidence/rgba_cuda`

Repeat output:
`matting_bench/providers/ben2/evidence/rgba_cuda_repeat`

All nine basenames match their inputs, all files decode as 960x960 RGBA, and
all nine SHA-256 values match between the two independent processes.

## Runtime

- OS: Windows 10 build `26200`, x86-64
- Python: `3.11.9`
- GPU: NVIDIA GeForce RTX 2080 Ti, compute capability 7.5, 11,263.688 MiB
- NVIDIA driver: `591.86`
- PyTorch: `2.5.1+cu121`; torchvision `0.20.1+cu121`
- cuDNN reported by PyTorch: `9.1.0`
- Official weight dtype: float32; official CUDA forward autocast: float16

This was a shared desktop GPU, not an isolated benchmark host. Immediately
before the primary run, `nvidia-smi` reported 2,675 MiB used and 9% GPU
utilization. A later idle sample remained at 2,715-2,726 MiB and 16%-19%.
Consequently, timing includes contention variance. PyTorch allocator peaks below
measure only this BEN2 process, not other processes or WDDM graphics memory.

## CUDA results

One full pipeline warm-up was excluded from measured frame timing.

| Metric | Primary | Independent repeat |
|---|---:|---:|
| Model load | 4.135 s | 3.068 s |
| Warm-up | 14.367 s | 1.543 s |
| 9-frame measured wall | 6.125 s | 6.561 s |
| Mean CUDA forward | 361.697 ms | 369.517 ms |
| Median CUDA forward | 321.183 ms | 294.745 ms |
| P95 CUDA forward | 555.746 ms | 626.820 ms |
| Mean total incl. I/O, validation, SHA, PNG | 668.109 ms | 714.130 ms |
| Peak allocated VRAM | 2,587.360 MiB | 2,587.360 MiB |
| Peak reserved VRAM | 3,238.000 MiB | 3,238.000 MiB |

The two forward means differ by 2.162%. Their simple mean is 365.607 ms, about
2.74 frames/s, but the primary per-run value should remain the comparison datum.
The first-ever 14.367 s warm-up created CUDA/kernel caches; the next fresh
process warmed in 1.543 s. Model weights alone occupy about 368.4 MiB allocated
VRAM after load.

`inference_ms` is synchronized wall time around the official model forward.
The peak is the PyTorch allocator maximum per measured frame and includes the
resident model, input, forward intermediates, and official alpha postprocess.

## Green-screen evaluation

The repository evaluator uses confident green background and non-green
foreground regions; it is a proxy because no hand-painted pet alpha ground truth
exists.

| Mean metric | Adaptive baseline | BEN2 Base | Reading |
|---|---:|---:|---|
| Pseudo MAE | 0.001791 | 0.005452 | BEN2 is 3.04x higher |
| Background alpha mean | 0.000831 | 0.000565 | BEN2 leaks 32.0% less background |
| Foreground loss mean | 0.002752 | 0.010339 | BEN2 loses 3.76x more confident foreground |
| Green fringe | 0.014523 | 0.036927 | BEN2 has 2.54x more soft-edge green spill |
| Opaque green leak | 0.1037% | 0.0931% | BEN2 is 10.2% lower |
| Soft alpha coverage | 0.9979% | 1.3107% | BEN2 has a wider soft transition |
| Fragment ratio | 0.1345% | 0.0128% | BEN2 is much more contiguous |
| Mean fragment count | 27.56 | 0.44 | far fewer detached islands |

## Pet fur and output risk

Visual evidence:
`matting_bench/providers/ben2/evidence/contact_sheet_detail.jpg`

- Body, ears, paws, and tail remain connected in all nine frames. BEN2 removes
  small mask islands and produces a cleaner gross silhouette than the adaptive
  green baseline.
- Ear tufts, back fur, and flank fur retain a soft band, but some fine guard
  hairs and low-contrast whiskers are thinned or omitted. The 3.76x foreground
  loss is consistent with this risk.
- Some visible whiskers survive in the idle and sleep samples, but the sparse
  nine-frame set cannot establish reliable subpixel whisker retention.
- The official non-refined path preserves source RGB beneath alpha. Green-screen
  contamination therefore remains in semi-transparent fur pixels. On neutral or
  light backgrounds this can appear as green or bright edge halo; the measured
  2.54x green-fringe increase makes this the main production blocker.
- The sample is three sparse frames per clip, so no valid temporal flicker metric
  is available. BEN2 is frame-independent here; fine-hair alpha can shimmer in a
  full animation even when isolated frames look acceptable.

Conclusion: BEN2 Base is a strong local candidate for contiguous pet silhouette
extraction, but this sample does not support a drop-in production decision for
fur-quality RGBA. A production integration would need controlled-background
despill/foreground-color estimation and a full consecutive-frame temporal test.
Those changes were intentionally not made to the central harness.

## Reproduction

~~~powershell
.venvs\ben2\Scripts\python.exe `
  matting_bench\providers\ben2\infer.py `
  --input-dir matting_bench\data\pet_20260710_121221_5ce7716e\smoke `
  --output-dir matting_bench\providers\ben2\evidence\rgba_cuda `
  --device cuda
~~~

Complete per-frame timing, alpha statistics, hashes, package versions, and CUDA
memory values are in `evidence/rgba_cuda/metrics.json`. Proxy metrics are in
`evidence/evaluation.json`. The full environment is frozen in
`.models/ben2/environment.freeze.txt`.

## Retained warnings

1. The first Hugging Face transfer disconnected after 361,436,411 bytes. The
   official client resumed the partial file; final size and SHA-256 match the
   pinned asset.
2. The initial CLI launch exposed a Windows PyTorch device-index requirement for
   bare `cuda`. The adapter now normalizes it to `cuda:0`; both complete runs
   succeeded using the required three-argument interface.
3. Official BEN2 source triggers PyTorch's future `torch.meshgrid` indexing
   warning. It did not affect either run or output determinism.

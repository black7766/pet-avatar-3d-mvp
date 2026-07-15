# Pet Matting Benchmark

This lab compares the current adaptive green-screen pipeline with local matting models
on exactly the same frames. Heavy model weights, virtual environments, decoded frames,
and output PNGs are intentionally ignored by Git.

## Dataset

```powershell
python matting_bench/prepare_dataset.py `
  --source poc_output/pet_20260710_121221_5ce7716e_real_after `
  --output matting_bench/data/pet_20260710_121221_5ce7716e
```

The default dataset contains all frames from `idle`, `fast_walk`, and `sleep`, plus a
fixed nine-frame subset (`0`, middle, last frame from every clip) for installation
smoke tests.

## Provider contract

Every provider exposes an `infer.py` command with this interface:

```powershell
python infer.py --input-dir <png_dir> --output-dir <rgba_dir> --device cuda
```

Output files must use the same basename as each input and must be 8-bit RGBA PNGs.

## Baseline

```powershell
python matting_bench/providers/baseline/infer.py `
  --input-dir matting_bench/data/pet_20260710_121221_5ce7716e/smoke `
  --output-dir matting_bench/outputs/baseline/smoke
```

## Tested providers

- Current adaptive chroma-key baseline
- BiRefNet and BiRefNet-matting
- ViTMatte-S
- PaddleSeg PP-MattingV2 human checkpoint
- BEN2 Base
- rembg U2Net, ISNet General, and BiRefNet General Lite
- MatAnyone v1 and SAM 2.1 Small for consecutive video propagation
- RMBG-2.0 adapter only; its weight is gated and its self-hosted license is not
  suitable for this commercial production evaluation without separate acceptance

The fixed conclusions and model/license links are in `benchmark_catalog.json`.
The full benchmark report is in `MATTING_MODEL_BENCHMARK_20260710.md`.

## Parameter tuning

Provider-specific tuning results follow `TUNING_SCHEMA.md`. The final sweep contains
69 configurations across nine providers. It uses a tuned adaptive-green baseline,
quality guardrails, commercial/fractional-alpha constraints, and a Pareto frontier;
it deliberately does not collapse matting quality into one weighted score.

```powershell
python matting_bench/aggregate_tuning.py `
  --output matting_bench/outputs/tuning/aggregate_final.json `
  --strict-outputs

python matting_bench/render_tuning_html.py `
  --aggregate matting_bench/outputs/tuning/aggregate_final.json `
  --output poc_output/matting_tuning_report_20260711.html
```

The renderer copies each provider's recommended output into a page-local asset
directory so the existing `poc_output` HTTP server can display it without exposing
the model/weight directories.

- Final tuning report: `MATTING_PARAMETER_TUNING_20260711.md`
- Local dashboard: `http://127.0.0.1:8792/matting_tuning_report_20260711.html`
- GPU serialization helper: `run_with_gpu_lock.py`

## Entity-only animated comparison

Build a page that converts every provider's recommended full fast-walk and
sleep outputs to synchronized transparent WebPs. It also reads the real-version generation
metrics and shows API generation time, local matting time, token usage, per-provider
runtime, VRAM, static quality metrics, and temporal error.

```powershell
python matting_bench/run_action_compare.py --action fast_walk
python matting_bench/run_action_compare.py --action sleep
python matting_bench/render_animated_compare.py
```

- Local page: `http://127.0.0.1:8792/matting_animated_compare_real_20260711.html`
- Scope: entity/real version only; no PaiMomo/cute-version assets
- Actions: entity-version `fast_walk` and `sleep`, switchable on the same page
- Playback: 640x640, 96 real consecutive frames, 19.2 FPS, 5-second silent loop
- Metrics: runtime and quality values switch with the selected action
- Self-developed variants: original adaptive green baseline, `edge_v2`, and `temporal_v3`
- `edge_v2`: 2-3px contour anti-aliasing plus nearest-core color reconstruction
- `temporal_v3`: `edge_v2` plus bidirectional flow-gated temporal alpha fusion
  (photometrically verified Farneback warp, static/soft-band weights 0.65/0.50)
  and flow-aligned adjacent-frame fragment voting. Large detached body components
  are protected from fragment removal.
- Production default: `edge_v2` and `temporal_v3` are enabled for entity/real assets
  by `petavatar_server.py`; temporal flow runs at 384 px. Set
  `PETAVATAR_GREEN_TEMPORAL_REFINE=0` to roll back to per-frame `edge_v2`, or set
  `PETAVATAR_TEMPORAL_FLOW_SIZE=320|384|480` for an explicit quality/speed tradeoff.
- Controls: production/research filter, checker/white/black background, synchronized replay
- Loading: production candidates load first; hidden research animations load on demand

## 2026-07-14 production A/B

```powershell
python matting_bench/run_production_temporal_ab.py --flow-size 384
python matting_bench/run_real_lighting_prompt_ab.py
```

- Temporal page: `http://127.0.0.1:8792/temporal_v3_production_ab_20260714.html`
- Lighting-prompt page: `http://127.0.0.1:8792/real_lighting_prompt_ab_20260714.html`
- Fixed report: `PRODUCTION_OPTIMIZATION_AB_20260714.md`

## 2026-07-15 structural support for moving limbs

The production `fast_walk` pipeline now combines the self-developed soft chroma
alpha with SAM 2.1 Hiera Small as an interior-only structural prior. SAM2 does
not replace the outside contour, so fractional fur, whiskers, and the tail tip
remain controlled by the adaptive-green matte. It only raises alpha inside a
tracked subject mask and reconstructs color for severe internal holes.

- Optional enhancement switch: `PETAVATAR_SAM2_STRUCTURAL_REFINE=1`
- Default scope: `PETAVATAR_SAM2_STRUCTURAL_CLIPS=fast_walk`
- Production default is off; V4 uses motion-aware adaptive green plus temporal_v3
- Failure behavior: log the provider error and continue with temporal_v3
- A/B page: `http://127.0.0.1:8792/structural_hybrid_ab_20260715.html`
- Test clip result: 163,278 reinforced pixels, 48,014 severe holes repaired,
  temporal fragment count 343 -> 320, WebP 4.62 MB -> 4.56 MB
- Local cost: zero API tokens; cold SAM2 job 90.23 seconds for 120 frames,
  including model/process initialization, while propagation itself is 9.23 seconds

The next speed optimization is a persistent SAM2 worker that keeps the model in
GPU memory. The current subprocess implementation is intentionally isolated and
safe for the existing demo server, but pays cold-start overhead on every job.

## 2026-07-15 upstream side-view locomotion

The primary fast-walk failure was traced to source generation rather than alpha
recovery: the previous three-quarter view rotated toward the camera during the
clip, causing leg occlusion, tail reconstruction, and large silhouette changes.
The production prompt now generates a fixed near-side keyframe and one compact,
repeated four-beat walk cycle with a low-amplitude tail.

- Dynamic A/B: `http://127.0.0.1:8792/upstream_sideview_production_ab_20260715.html`
- Same pet, Seedance 1.5, 720p, 5 seconds, 120 output frames
- Silhouette width CV: 25.84% -> 3.53%
- Silhouette area CV: 16.22% -> 2.66%
- Horizontal centroid range: 70.7 px -> 23.5 px
- Output fragment ratio: 0.0186% -> 0.0015%
- New candidate uses temporal_v3 without SAM2: 70.7 seconds local processing

`poc.py` accepts `PETAVATAR_ANIMATE_MODEL` for controlled model A/B. The account
lists `doubao-seedance-2-0-260128`, but task creation currently returns
`ModelNotOpen`; do not switch the server default until that model is activated.

## 2026-07-15 upstream no-ground locomotion

The remaining paw defects were traced to contact lighting already baked into the
source video. Mentioning an invisible baseline still caused the video model to
infer a physical floor, contact shadow, paw compression, and a green brightness
gradient. The production prompt now describes a suspended animation-software
walk-cycle preview over a flat 2D compositing fill. It requires a continuous
uniform green band below every complete paw and explicitly forbids any rendered
surface, contact patch, ambient occlusion, reflection, or floor gradient.

- Dynamic A/B: `http://127.0.0.1:8792/upstream_ground_airgap_ab_20260715.html`
- Source/background confidence: 0.6544 -> 0.8235
- Output loop seam: 0.004400 -> 0.001345
- Temporal fragments removed: 311 -> 247
- WebP size: 4.71 MB -> 4.19 MB
- Local temporal_v3 processing: 70.7 s -> 70.0 s
- Test configuration: Seedance 1.5, 720p, 5 seconds, temporal_v3, no SAM2

The transparent asset intentionally contains no floor or baked shadow. Product
clients should align the lowest visible paw to their desktop/app baseline and
render a separate stable ellipse shadow beneath the pet when visual grounding is
needed. This keeps contact lighting temporally stable and independent of matting.

## Production default: airgap_motion_v4

Both production stages now use the V4 path by default:

- Generation: no physical floor, suspended locomotion cycle, full paws, and a
  uniform 2D green compositing fill below the lowest paw.
- Matting: motion-aware linear alpha un-mixing restores blurred dark limbs and
  tails without reviving pure-green contact shadows.
- Temporal pass: bidirectional flow remains enabled, but the current-frame lower
  motion silhouette is protected for `fast_walk` to prevent translucent duplicate
  paws and amputated legs.
- SAM2 is opt-in rather than part of the default path. Set
  `PETAVATAR_SAM2_STRUCTURAL_REFINE=1` only for an explicit structural A/B.
- Metrics record `pipeline_version: airgap_motion_v4` for new video and matte jobs.

Rollback controls are `PETAVATAR_GREEN_MOTION_ALPHA_REFINE=0`,
`PETAVATAR_GREEN_PRESERVE_LOWER_MOTION=0`, and
`PETAVATAR_GREEN_TEMPORAL_REFINE=0`.

For real/entity assets, V4 also applies a final odds-domain alpha contrast of
`1.18`. This keeps alpha 0, 0.5, and 1 fixed while compressing an overly broad
semi-transparent transition, so the silhouette does not shrink or move. Disable
it with `PETAVATAR_REAL_ALPHA_EDGE_CONTRAST=1.0`.

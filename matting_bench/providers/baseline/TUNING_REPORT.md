# Adaptive green-screen parameter tuning

Date: 2026-07-11

## Scope

The internal adaptive chroma-key implementation was tuned on the same nine
`960x960` smoke frames as the external models. Four candidates were also tested
on the same 24 consecutive `640x640` fast-walk frames.

The implementation parameters are documented in the repository's `poc.py`:

- `foreground_score`: green-connectivity foreground threshold.
- `border_quantile`: clip-level green background floor estimator.
- `alpha_gamma`: transition-band alpha curve.
- `core_despill`: opaque yellow-green contamination correction strength.
- `core_radius_ratio`: maximum silhouette distance for opaque despill.
- `halo_profile`: baked rim-light color/alpha treatment.

## Result

| Configuration | pseudo MAE | Foreground loss | Green fringe | Fragment % | Temporal alpha MAE |
|---|---:|---:|---:|---:|---:|
| default | 0.001791 | 0.002752 | 0.014523 | 0.1345 | **0.013821** |
| despill 1.10 | 0.001788 | **0.002745** | 0.013816 | 0.1345 | 0.013823 |
| gamma 1.32 + despill 1.05 | 0.001762 | 0.002796 | 0.013595 | 0.1296 | 0.013868 |
| gamma 1.40 + despill 1.10 | **0.001746** | 0.002836 | **0.013171** | **0.1239** | 0.013904 |

Recommended production delta: `core_despill=1.10`, with all other parameters
unchanged.

- Green fringe improves by about 4.87%.
- Pseudo MAE improves slightly.
- Temporal alpha error changes by about +0.015%, which is within run-to-run
  optical-flow metric noise and expected because despill does not change alpha.
- The change does not alter foreground topology or fragment count.

## Rejected variants

`halo_profile=none` and `halo_profile=cartoon` appear best under pseudo MAE because
they do not lower alpha around bright rims. Black-background visual review shows
that they retain an obvious white baked rim around the raised leg, so they are
not acceptable despite the lower automatic score.

Increasing `alpha_gamma` to `1.32-1.40` reduces static green fringe and fragments,
but raises continuous-frame alpha error by roughly 0.34-0.60%. It is retained as
an experimental option, not the production default.

## Reproduction

```powershell
python matting_bench/providers/baseline/tuning_sweep.py
```

Machine-readable results are in `tuning_results.json`. Large RGBA outputs and
evaluation JSON remain under ignored `matting_bench/outputs/tuning/baseline/`.

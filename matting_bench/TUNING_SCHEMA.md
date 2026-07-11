# Tuning result contract

Each provider tuning task writes `providers/<provider>/tuning_results.json`.
Large outputs remain in ignored `evidence/`, `runs/`, or `matting_bench/outputs/`
directories. The committed JSON contains only parameters, metrics, paths, and
official documentation references.

Required top-level fields:

```json
{
  "provider": "provider-name",
  "provider_meta": {"version": "optional runtime metadata"},
  "official_docs": [
    {"url": "https://official.example/docs", "parameters": ["input_size"]}
  ],
  "dataset": {"smoke": "matting_bench/data/.../smoke"},
  "configs": [],
  "recommendation": {"config_id": "default", "reason": "..."}
}
```

`provider_meta` is optional. Recommendation selectors may use `config_id`,
`recommended_config_id`, `selected_config_id`, or `primary`; the aggregator
normalizes them to `recommended_config_id`.

Each configuration uses this shape:

```json
{
  "id": "input768_refine",
  "parameters": {"input_size": 768, "refine_foreground": true},
  "status": "ok",
  "output_dir": "matting_bench/outputs/provider/input768_refine",
  "quality": {
    "pseudo_mae": 0.0,
    "background_alpha_mean": 0.0,
    "foreground_loss_mean": 0.0,
    "green_fringe": 0.0,
    "fragment_pct": 0.0,
    "soft_alpha_pct": 0.0
  },
  "runtime": {
    "mean_inference_ms": 0.0,
    "end_to_end_ms": 0.0,
    "peak_vram_mb": 0.0
  },
  "temporal_alpha_mae": null,
  "notes": ""
}
```

Use `null` when a metric is not available. Do not replace missing measurements
with zero. `aggregate_tuning.py` validates the schema, calculates deltas against
the current baseline, applies transparent guardrails, and computes a Pareto
frontier without inventing a single weighted quality score.

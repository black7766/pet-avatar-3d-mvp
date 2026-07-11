"""Validate and aggregate provider-specific tuning results.

The aggregator intentionally avoids a single weighted score. Matting has competing
objectives: a configuration can remove fragments by deleting valid fur. Instead it
reports guardrails, deltas against the current baseline, and a Pareto frontier.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


QUALITY_FIELDS = (
    "pseudo_mae",
    "background_alpha_mean",
    "foreground_loss_mean",
    "green_fringe",
    "fragment_pct",
    "soft_alpha_pct",
)
RUNTIME_FIELDS = ("mean_inference_ms", "end_to_end_ms", "peak_vram_mb")
PARETO_FIELDS = (
    "pseudo_mae",
    "foreground_loss_mean",
    "background_alpha_mean",
    "green_fringe",
    "runtime_ms",
)

RECOMMENDATION_KEYS = (
    "config_id",
    "recommended_config_id",
    "selected_config_id",
    "primary",
    "id",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_number_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )


def recommendation_id(recommendation: dict[str, Any]) -> str | None:
    for key in RECOMMENDATION_KEYS:
        value = recommendation.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def selection_eligibility(provider: str, config_id: str) -> dict[str, Any]:
    """Apply product constraints that proxy metrics cannot represent.

    Binary masks can score well by deleting fur, and MatAnyone's current upstream
    license is not suitable for a commercial product. Visually rejected settings
    are retained in the report, but excluded from the production Pareto frontier.
    """

    provider_key = provider.lower()
    config_key = config_id.lower()
    fractional_alpha = True
    commercial_use = True
    visual_review = True
    domain_fit = True
    reasons: list[str] = []

    if "sam2" in provider_key:
        fractional_alpha = False
        reasons.append("SAM2 outputs a binary segmentation mask, not a fur matte")
    if "postprocess_binary" in config_key:
        fractional_alpha = False
        reasons.append("binary post-processing removes whiskers and soft fur")
    if provider_key == "official_matanyone_v1":
        commercial_use = False
        reasons.append("upstream S-Lab License 1.0 is non-commercial")
    if provider_key == "zhengpeng7/birefnet-matting":
        domain_fit = False
        reasons.append("tested checkpoint is predominantly human-matting data and underperformed the General checkpoint on pet smoke frames")
    if provider_key == "paddle_matting":
        domain_fit = False
        reasons.append("released PP-MattingV2 checkpoint is human-domain and failed the pet green-edge guardrail")
    if provider_key == "adaptive_green_baseline" and config_key in {
        "halo_none",
        "halo_cartoon",
    }:
        visual_review = False
        reasons.append("visual review found a baked white rim around raised fur/legs")
    if provider_key == "rembg" and "alpha_fur_safe" in config_key:
        visual_review = False
        reasons.append("visual review found more deleted soft detail than recovered detail")

    return {
        "fractional_alpha": fractional_alpha,
        "commercial_use": commercial_use,
        "visual_review": visual_review,
        "domain_fit": domain_fit,
        "eligible_for_final": fractional_alpha and commercial_use and visual_review and domain_fit,
        "exclusion_reasons": reasons,
    }


def normalize_config(provider: str, item: dict[str, Any], source: Path) -> dict[str, Any]:
    quality = {field: (item.get("quality") or {}).get(field) for field in QUALITY_FIELDS}
    runtime = {field: (item.get("runtime") or {}).get(field) for field in RUNTIME_FIELDS}
    runtime_ms = runtime.get("mean_inference_ms")
    if runtime_ms is None:
        runtime_ms = runtime.get("end_to_end_ms")
    eligibility = selection_eligibility(provider, item["id"])
    return {
        "key": f"{provider}:{item['id']}",
        "provider": provider,
        "id": item["id"],
        "parameters": item.get("parameters") or {},
        "status": item.get("status", "unknown"),
        "output_dir": item.get("output_dir"),
        "quality": quality,
        "runtime": runtime,
        "runtime_ms": runtime_ms,
        "temporal_alpha_mae": item.get("temporal_alpha_mae"),
        "notes": item.get("notes") or "",
        "source_file": str(source),
        **eligibility,
    }


def validate_result(path: Path, repo_root: Path, strict_outputs: bool) -> tuple[dict[str, Any], list[str]]:
    payload = read_json(path)
    errors: list[str] = []
    raw_provider = payload.get("provider")
    provider_meta = payload.get("provider_meta")
    if not isinstance(provider_meta, dict):
        provider_meta = {}
    if isinstance(raw_provider, dict):
        provider_meta = {**raw_provider, **provider_meta}
        provider = raw_provider.get("name")
    else:
        provider = raw_provider
    if not isinstance(provider, str) or not provider.strip():
        errors.append("provider must be a non-empty string")
        provider = path.parent.name
    docs = payload.get("official_docs")
    if not isinstance(docs, list) or not docs:
        errors.append("official_docs must contain at least one item")
    else:
        for index, doc in enumerate(docs):
            if not isinstance(doc, dict) or not str(doc.get("url", "")).startswith("http"):
                errors.append(f"official_docs[{index}] must contain an HTTP(S) url")
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs:
        errors.append("configs must contain at least one configuration")
        configs = []
    if len(configs) < 3:
        errors.append(f"expected at least 3 configurations, got {len(configs)}")
    ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(configs):
        prefix = f"configs[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        config_id = item.get("id")
        if not isinstance(config_id, str) or not config_id:
            errors.append(f"{prefix}.id must be a non-empty string")
            continue
        if config_id in ids:
            errors.append(f"duplicate config id: {config_id}")
        ids.add(config_id)
        quality = item.get("quality") or {}
        runtime = item.get("runtime") or {}
        for field in QUALITY_FIELDS:
            if not is_number_or_none(quality.get(field)):
                errors.append(f"{prefix}.quality.{field} must be finite number or null")
        for field in RUNTIME_FIELDS:
            if not is_number_or_none(runtime.get(field)):
                errors.append(f"{prefix}.runtime.{field} must be finite number or null")
        if not is_number_or_none(item.get("temporal_alpha_mae")):
            errors.append(f"{prefix}.temporal_alpha_mae must be finite number or null")
        output_dir = item.get("output_dir")
        if strict_outputs and item.get("status") == "ok" and output_dir:
            resolved = (repo_root / output_dir).resolve()
            if not resolved.is_dir():
                errors.append(f"{prefix}.output_dir does not exist: {resolved}")
        normalized.append(normalize_config(provider, item, path))
    recommendation = payload.get("recommendation") or {}
    recommended_id = recommendation_id(recommendation)
    if recommended_id is not None and recommended_id not in ids:
        errors.append(f"recommendation references unknown config: {recommended_id}")
    return {
        "provider": provider,
        "provider_meta": provider_meta,
        "official_docs": docs or [],
        "dataset": payload.get("dataset") or {},
        "configs": normalized,
        "recommendation": recommendation,
        "recommended_config_id": recommended_id,
        "source_file": str(path),
    }, errors


def metric_vector(config: dict[str, Any]) -> tuple[float, ...] | None:
    quality = config["quality"]
    values = [quality.get(field) for field in PARETO_FIELDS[:-1]] + [config.get("runtime_ms")]
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)


def pareto_keys(
    configs: list[dict[str, Any]],
    *,
    final_only: bool,
    recommended_only: bool = False,
    require_guardrails: bool = False,
) -> list[str]:
    candidates = [
        (item, metric_vector(item))
        for item in configs
        if item.get("status") == "ok"
        and (not final_only or item.get("eligible_for_final"))
        and (not recommended_only or item.get("provider_recommended"))
        and (not require_guardrails or item.get("passes_all_guardrails"))
    ]
    candidates = [(item, vector) for item, vector in candidates if vector is not None]
    frontier: list[str] = []
    for item, vector in candidates:
        dominated = False
        for other, other_vector in candidates:
            if other is item:
                continue
            no_worse = all(a <= b for a, b in zip(other_vector, vector))
            strictly_better = any(a < b for a, b in zip(other_vector, vector))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(item["key"])
    return sorted(frontier)


def add_baseline_deltas(config: dict[str, Any], baseline: dict[str, float]) -> None:
    deltas: dict[str, float | None] = {}
    ratios: dict[str, float | None] = {}
    for field in QUALITY_FIELDS:
        value = config["quality"].get(field)
        reference = baseline.get(field)
        deltas[field] = None if value is None or reference is None else value - reference
        ratios[field] = (
            None if value is None or not reference else value / reference
        )
    config["delta_vs_baseline"] = deltas
    config["ratio_vs_baseline"] = ratios
    config["guardrails"] = {
        "foreground_retention": (
            config["quality"].get("foreground_loss_mean") is not None
            and config["quality"]["foreground_loss_mean"] <= baseline["foreground_loss_mean"] * 1.5
        ),
        "background_control": (
            config["quality"].get("background_alpha_mean") is not None
            and config["quality"]["background_alpha_mean"] <= max(
                baseline["background_alpha_mean"] * 4.0, 0.004
            )
        ),
        "green_edge": (
            config["quality"].get("green_fringe") is not None
            and config["quality"]["green_fringe"] <= baseline["green_fringe"] * 1.5
        ),
    }
    config["passes_all_guardrails"] = all(config["guardrails"].values())


def tuned_baseline(
    providers: list[dict[str, Any]], original: dict[str, float]
) -> tuple[dict[str, float], str]:
    for provider in providers:
        if provider["provider"] != "adaptive_green_baseline":
            continue
        selected = provider.get("recommended_config_id")
        for config in provider["configs"]:
            if config["id"] != selected:
                continue
            quality = config.get("quality") or {}
            if all(isinstance(quality.get(field), (int, float)) for field in QUALITY_FIELDS):
                return {field: float(quality[field]) for field in QUALITY_FIELDS}, config["key"]
    return original, "legacy:matting_bench/outputs/final_smoke_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers-dir", type=Path, default=Path("matting_bench/providers"))
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("matting_bench/outputs/final_smoke_metrics.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-outputs", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    result_paths = sorted(args.providers_dir.glob("*/tuning_results.json"))
    if not result_paths:
        raise SystemExit(f"no tuning_results.json files found under {args.providers_dir}")
    baseline_report = read_json(args.baseline_report)
    original_baseline = baseline_report["providers"]["baseline"]["mean"]
    providers: list[dict[str, Any]] = []
    errors: list[str] = []
    configs: list[dict[str, Any]] = []
    for path in result_paths:
        provider_result, provider_errors = validate_result(path, repo_root, args.strict_outputs)
        providers.append(provider_result)
        configs.extend(provider_result["configs"])
        errors.extend(f"{path}: {message}" for message in provider_errors)
    baseline, baseline_key = tuned_baseline(providers, original_baseline)
    recommended_keys = {
        f"{provider['provider']}:{provider['recommended_config_id']}"
        for provider in providers
        if provider.get("recommended_config_id")
    }
    for config in configs:
        add_baseline_deltas(config, baseline)
        config["provider_recommended"] = config["key"] in recommended_keys
    payload = {
        "schema_version": 1,
        "baseline": baseline,
        "baseline_key": baseline_key,
        "original_baseline": original_baseline,
        "provider_count": len(providers),
        "config_count": len(configs),
        "validation_errors": errors,
        "pareto_frontier": pareto_keys(
            configs,
            final_only=True,
            recommended_only=True,
            require_guardrails=True,
        ),
        "parameter_pareto_frontier": pareto_keys(configs, final_only=True),
        "research_pareto_frontier": pareto_keys(configs, final_only=False),
        "providers": providers,
        "configs": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(f"providers={len(providers)} configs={len(configs)} errors={len(errors)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

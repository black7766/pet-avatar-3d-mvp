"""Parameter sweep for the project's adaptive green-screen baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]

CONFIGS = [
    {"id": "default", "parameters": {}},
    {"id": "gamma_1_05", "parameters": {"alpha_gamma": 1.05}},
    {"id": "gamma_1_40", "parameters": {"alpha_gamma": 1.40}},
    {"id": "fg_0_018", "parameters": {"foreground_score": 0.018}},
    {"id": "fg_0_035", "parameters": {"foreground_score": 0.035}},
    {"id": "despill_0_65", "parameters": {"core_despill": 0.65}},
    {"id": "despill_1_10", "parameters": {"core_despill": 1.10}},
    {"id": "radius_0_10", "parameters": {"core_radius_ratio": 0.10}},
    {
        "id": "balanced_gamma_1_32_despill_1_05",
        "parameters": {"alpha_gamma": 1.32, "core_despill": 1.05},
    },
    {
        "id": "clean_gamma_1_40_despill_1_10",
        "parameters": {"alpha_gamma": 1.40, "core_despill": 1.10},
    },
    {"id": "halo_none", "parameters": {"halo_profile": "none"}},
    {"id": "halo_cartoon", "parameters": {"halo_profile": "cartoon"}},
]

DEFAULTS = {
    "foreground_score": 0.025,
    "border_quantile": 0.002,
    "alpha_gamma": 1.22,
    "core_despill": 0.90,
    "core_radius_ratio": 0.16,
    "halo_strength": 0.90,
    "halo_profile": "real",
}

QUALITY_FIELDS = (
    "pseudo_mae",
    "background_alpha_mean",
    "foreground_loss_mean",
    "green_fringe",
    "fragment_pct",
    "soft_alpha_pct",
)
TEMPORAL_IDS = (
    "default",
    "despill_1_10",
    "balanced_gamma_1_32_despill_1_05",
    "clean_gamma_1_40_despill_1_10",
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("matting_bench/data/pet_20260710_121221_5ce7716e/smoke"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("matting_bench/outputs/tuning/baseline")
    )
    parser.add_argument(
        "--temporal-source-dir",
        type=Path,
        default=Path(
            "matting_bench/data/pet_20260710_121221_5ce7716e/temporal_fast_walk_24_640"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    source_dir = (REPO_ROOT / args.source_dir).resolve()
    temporal_source_dir = (REPO_ROOT / args.temporal_source_dir).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    provider_args: list[str] = []
    results: list[dict] = []
    for config in CONFIGS:
        parameters = DEFAULTS | config["parameters"]
        output_dir = output_root / config["id"]
        command = [
            args.python,
            str(PROVIDER_DIR / "infer.py"),
            "--input-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
        for key, value in parameters.items():
            command.extend((f"--{key.replace('_', '-')}", str(value)))
        if not args.reuse_existing or not (output_dir / "metrics.json").exists():
            run(command)
        provider_args.extend((f"baseline_{config['id']}={output_dir}",))

    evaluation_path = output_root / "evaluation.json"
    command = [
        args.python,
        str(REPO_ROOT / "matting_bench/evaluate.py"),
        "--source-dir",
        str(source_dir),
    ]
    for provider in provider_args:
        command.extend(("--provider", provider))
    command.extend(("--output", str(evaluation_path)))
    run(command)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    for config in CONFIGS:
        config_id = config["id"]
        output_dir = output_root / config_id
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        quality_mean = evaluation["providers"][f"baseline_{config_id}"]["mean"]
        results.append(
            {
                "id": config_id,
                "parameters": DEFAULTS | config["parameters"],
                "status": "ok",
                "output_dir": str(output_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                "quality": {field: quality_mean.get(field) for field in QUALITY_FIELDS},
                "runtime": {
                    "mean_inference_ms": None,
                    "end_to_end_ms": metrics.get("mean_ms_per_frame"),
                    "peak_vram_mb": 0.0,
                },
                "temporal_alpha_mae": None,
                "notes": "CPU end-to-end includes halo refinement and PNG output.",
            }
        )

    viable = [
        item
        for item in results
        if item["quality"]["foreground_loss_mean"] <= 0.0042
        and item["quality"]["background_alpha_mean"] <= 0.004
        and item["parameters"]["halo_profile"] == "real"
    ]
    temporal_args: list[str] = []
    by_id = {item["id"]: item for item in results}
    config_by_id = {item["id"]: item for item in CONFIGS}
    for config_id in TEMPORAL_IDS:
        parameters = DEFAULTS | config_by_id[config_id]["parameters"]
        output_dir = output_root / "temporal" / config_id
        command = [
            args.python,
            str(PROVIDER_DIR / "infer.py"),
            "--input-dir",
            str(temporal_source_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
        for key, value in parameters.items():
            command.extend((f"--{key.replace('_', '-')}", str(value)))
        if not args.reuse_existing or not (output_dir / "metrics.json").exists():
            run(command)
        temporal_args.extend((f"baseline_{config_id}={output_dir}",))
    temporal_evaluation_path = output_root / "temporal_evaluation.json"
    command = [
        args.python,
        str(REPO_ROOT / "matting_bench/evaluate.py"),
        "--source-dir",
        str(temporal_source_dir),
    ]
    for provider in temporal_args:
        command.extend(("--provider", provider))
    command.extend(("--output", str(temporal_evaluation_path)))
    run(command)
    temporal_evaluation = json.loads(temporal_evaluation_path.read_text(encoding="utf-8"))
    for config_id in TEMPORAL_IDS:
        by_id[config_id]["temporal_alpha_mae"] = temporal_evaluation["providers"][
            f"baseline_{config_id}"
        ]["temporal_alpha_mae"]
    default_temporal = by_id["default"]["temporal_alpha_mae"]
    stable_viable = [
        item
        for item in viable
        if item["temporal_alpha_mae"] is not None
        and item["temporal_alpha_mae"] <= default_temporal * 1.001
    ]
    recommended = min(
        stable_viable or viable or results,
        key=lambda item: (
            item["quality"]["green_fringe"],
            item["quality"]["pseudo_mae"],
        ),
    )
    payload = {
        "provider": "adaptive_green_baseline",
        "official_docs": [
            {
                "url": "https://github.com/black7766/pet-avatar-3d-mvp/blob/master/poc.py",
                "parameters": list(DEFAULTS),
            }
        ],
        "dataset": {
            "smoke": str(source_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            "temporal": str(temporal_source_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "configs": results,
        "recommendation": {
            "config_id": recommended["id"],
            "reason": "lowest green fringe among foreground/background and temporal-stability guardrail passes",
        },
    }
    result_path = PROVIDER_DIR / "tuning_results.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()

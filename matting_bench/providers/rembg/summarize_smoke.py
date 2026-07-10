"""Consolidate per-model rembg smoke metrics into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from model_catalog import MODEL_SPECS
from runtime import PROVIDER_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROVIDER_DIR / "outputs" / "smoke",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROVIDER_DIR / "smoke_results.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROVIDER_DIR / "SMOKE_RESULTS.md",
    )
    return parser.parse_args()


def nvidia_gpu_details() -> dict[str, object] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_gpu = completed.stdout.strip().splitlines()[0]
    name, driver, memory_mib = (part.strip() for part in first_gpu.split(",", 2))
    return {
        "name": name,
        "driver_version": driver,
        "memory_mib": int(memory_mib),
    }


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    raw_metrics: dict[str, object] = {}
    for model_name, spec in MODEL_SPECS.items():
        metrics_path = args.results_dir / model_name / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing smoke metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "ok" or metrics.get("image_count") != 9:
            raise RuntimeError(f"Incomplete smoke result in {metrics_path}")
        if metrics.get("rgba_output_count") != 9:
            raise RuntimeError(f"RGBA validation failed in {metrics_path}")

        model = metrics["model"]
        row = {
            "model": model_name,
            "source": spec.upstream_repo,
            "weight_url": spec.weight_url,
            "upstream_license": spec.upstream_license,
            "weight_size_bytes": model["size_bytes"],
            "md5": model["actual_md5"],
            "device": metrics["device"],
            "active_providers": metrics["environment"]["active_providers"],
            "frames": metrics["image_count"],
            "dll_preload_seconds": metrics["dll_preload_seconds"],
            "session_load_seconds": metrics["session_load_seconds"],
            "remove_total_seconds": metrics["remove_total_seconds"],
            "remove_mean_ms": metrics["remove_mean_seconds"] * 1000,
            "remove_mean_excluding_first_ms": (
                metrics["remove_mean_excluding_first_seconds"] * 1000
            ),
            "end_to_end_total_seconds": metrics["end_to_end_total_seconds"],
            "output_dir": metrics["output_dir"],
        }
        rows.append(row)
        raw_metrics[model_name] = metrics

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": nvidia_gpu_details(),
        "method": (
            "One cold process per model; no warm-up frame. remove_seconds measures the "
            "rembg remove() call, including model preprocessing and mask postprocessing."
        ),
        "models": rows,
        "metrics": raw_metrics,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# rembg 9-frame smoke results",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "Each model ran in a cold Python process with no warm-up frame. `remove` timing "
        "covers rembg preprocessing, ONNX inference, and mask postprocessing; end-to-end "
        "timing also includes PNG load/save.",
        "",
    ]
    if payload["gpu"]:
        gpu = payload["gpu"]
        lines.extend(
            [
                f"CUDA host: `{gpu['name']}`, driver `{gpu['driver_version']}`, "
                f"{gpu['memory_mib']} MiB VRAM.",
                "",
            ]
        )
    lines.extend(
        [
            "## Runtime",
            "",
            "| Model | Device / active EP | Weight MiB | Session load s | 9-frame remove s | Mean ms/frame | Mean excl. first ms | End-to-end s | Output |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        active = ", ".join(row["active_providers"])
        lines.append(
            f"| `{row['model']}` | `{row['device']}` / `{active}` | "
            f"{row['weight_size_bytes'] / (1024 ** 2):.2f} | "
            f"{row['session_load_seconds']:.3f} | "
            f"{row['remove_total_seconds']:.3f} | "
            f"{row['remove_mean_ms']:.1f} | "
            f"{row['remove_mean_excluding_first_ms']:.1f} | "
            f"{row['end_to_end_total_seconds']:.3f} | `{row['output_dir']}` |"
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "License values below are the licenses declared by the upstream model "
            "repositories. rembg itself is MIT-licensed.",
            "",
            "| Model | Upstream source | License | Official rembg weight | MD5 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        spec = MODEL_SPECS[row["model"]]
        lines.append(
            f"| `{row['model']}` | [repository]({row['source']}) | "
            f"[{row['upstream_license']}]({spec.upstream_license_url}) | "
            f"[ONNX]({row['weight_url']}) | `{row['md5']}` |"
        )
    lines.append("")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"models": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

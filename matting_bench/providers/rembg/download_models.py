"""Download and verify the pinned models through rembg's own model registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from model_catalog import MODEL_SPECS
from runtime import MODEL_DIR, configure_runtime_dirs, file_md5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Models to download; defaults to the complete benchmark set.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MODEL_DIR / "download_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_runtime_dirs()

    from rembg.sessions import sessions_class

    registry = {session_class.name(): session_class for session_class in sessions_class}
    downloaded: list[dict[str, object]] = []
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        session_class = registry.get(model_name)
        if session_class is None:
            raise RuntimeError(
                f"rembg {metadata.version('rembg')} does not support {model_name!r}"
            )

        model_path = Path(session_class.download_models()).resolve()
        actual_md5 = file_md5(model_path)
        if actual_md5 != spec.expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {model_name}: {actual_md5} != {spec.expected_md5}"
            )
        downloaded.append(
            {
                **spec.as_dict(),
                "local_path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "actual_md5": actual_md5,
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rembg_version": metadata.version("rembg"),
        "model_dir": str(MODEL_DIR.resolve()),
        "models": downloaded,
    }
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

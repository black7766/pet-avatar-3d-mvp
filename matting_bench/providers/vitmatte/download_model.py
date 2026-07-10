from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "vitmatte"
MODEL_DIR = MODEL_ROOT / "hustvl--vitmatte-small-composition-1k"
MODEL_REPO = "hustvl/vitmatte-small-composition-1k"
MODEL_REVISION = "6a58ad7646403c1df626fbd746900aec7361ea1d"
WEIGHT_NAME = "model.safetensors"
WEIGHT_SIZE_BYTES = 103_294_572
WEIGHT_SHA256 = "bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"

# Keep every Hugging Face cache and credential lookup under the permitted model root.
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf-cache" / "hub"))
os.environ.setdefault("HF_TOKEN_PATH", str(MODEL_ROOT / "hf-cache" / "token"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "xdg-cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub import snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "README.md",
            "config.json",
            "preprocessor_config.json",
            WEIGHT_NAME,
        ],
    )

    weight_path = MODEL_DIR / WEIGHT_NAME
    actual_size = weight_path.stat().st_size
    actual_sha256 = sha256(weight_path)
    if actual_size != WEIGHT_SIZE_BYTES:
        raise RuntimeError(
            f"Weight size mismatch: expected {WEIGHT_SIZE_BYTES}, got {actual_size}"
        )
    if actual_sha256 != WEIGHT_SHA256:
        raise RuntimeError(
            f"Weight SHA-256 mismatch: expected {WEIGHT_SHA256}, got {actual_sha256}"
        )

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "implementation": "Hugging Face Transformers VitMatteForImageMatting",
        "upstream_code": "https://github.com/hustvl/ViTMatte",
        "upstream_code_license": "MIT",
        "model_page": f"https://huggingface.co/{MODEL_REPO}",
        "model_card_license": "apache-2.0",
        "local_dir": str(MODEL_DIR.resolve()),
        "weight": {
            "file": WEIGHT_NAME,
            "size_bytes": actual_size,
            "sha256": actual_sha256,
            "format": "safetensors",
        },
        "verified": True,
    }
    (MODEL_DIR / "local_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

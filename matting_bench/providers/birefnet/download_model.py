from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "birefnet"
MODEL_DIR = MODEL_ROOT / "ZhengPeng7--BiRefNet"
MODEL_REPO = "ZhengPeng7/BiRefNet"
MODEL_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
WEIGHT_SIZE_BYTES = 444_473_596
WEIGHT_SHA256 = "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"

os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf-cache" / "hub"))
os.environ.setdefault("HF_TOKEN_PATH", str(MODEL_ROOT / "hf-cache" / "token"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "xdg-cache"))

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
            "BiRefNet_config.py",
            "README.md",
            "birefnet.py",
            "config.json",
            "handler.py",
            "model.safetensors",
            "requirements.txt",
        ],
    )

    weight_path = MODEL_DIR / "model.safetensors"
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

    print(
        json.dumps(
            {
                "repo_id": MODEL_REPO,
                "revision": MODEL_REVISION,
                "local_dir": str(MODEL_DIR),
                "weight_path": str(weight_path),
                "weight_size_bytes": actual_size,
                "weight_sha256": actual_sha256,
                "verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

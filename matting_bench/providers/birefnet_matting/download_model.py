from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_ROOT = REPO_ROOT / ".models" / "birefnet"
MODEL_DIR = MODEL_ROOT / "ZhengPeng7--BiRefNet-matting"
MODEL_REPO = "ZhengPeng7/BiRefNet-matting"
MODEL_REVISION = "57f9f68b43ba337c75762b14cf3075d659007268"

os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ROOT / "hf-cache" / "hub"))
os.environ.setdefault("HF_TOKEN_PATH", str(MODEL_ROOT / "hf-cache" / "token"))

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
        max_workers=1,
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
    result = {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "local_dir": str(MODEL_DIR),
        "weight_path": str(weight_path),
        "weight_size_bytes": weight_path.stat().st_size,
        "weight_sha256": sha256(weight_path),
        "verified_revision": True,
    }
    (PROVIDER_DIR / "model_manifest.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

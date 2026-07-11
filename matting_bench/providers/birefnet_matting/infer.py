from __future__ import annotations

import sys
from pathlib import Path


PROVIDER_DIR = Path(__file__).resolve().parent
PROVIDERS_DIR = PROVIDER_DIR.parent
REPO_ROOT = PROVIDER_DIR.parents[2]
if str(PROVIDERS_DIR) not in sys.path:
    sys.path.insert(0, str(PROVIDERS_DIR))

from birefnet.infer import ModelSpec, main  # noqa: E402


MATTING_MODEL = ModelSpec(
    repo_id="ZhengPeng7/BiRefNet-matting",
    revision="57f9f68b43ba337c75762b14cf3075d659007268",
    model_dir=REPO_ROOT / ".models" / "birefnet" / "ZhengPeng7--BiRefNet-matting",
    variant="matting",
)


if __name__ == "__main__":
    raise SystemExit(main(MATTING_MODEL))

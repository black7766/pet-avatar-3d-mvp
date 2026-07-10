"""Runtime paths and integrity helpers for the isolated rembg provider."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


PROVIDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROVIDER_DIR.parents[2]
MODEL_DIR = REPO_ROOT / ".models" / "rembg"
_DLL_DIRECTORY_HANDLES: list[object] = []


def configure_runtime_dirs() -> None:
    cache_dirs = {
        "U2NET_HOME": MODEL_DIR,
        "XDG_DATA_HOME": MODEL_DIR / "xdg-data",
        "XDG_CACHE_HOME": MODEL_DIR / "xdg-cache",
        "NUMBA_CACHE_DIR": MODEL_DIR / "numba-cache",
        "TEMP": MODEL_DIR / "tmp",
        "TMP": MODEL_DIR / "tmp",
    }
    for path in set(cache_dirs.values()):
        path.mkdir(parents=True, exist_ok=True)
    for name, path in cache_dirs.items():
        os.environ[name] = str(path)


def configure_nvidia_dll_dirs() -> list[str]:
    """Keep split NVIDIA wheel DLL directories visible to Windows LoadLibrary."""
    if os.name != "nt":
        return []

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    dll_dirs = sorted(
        path.resolve()
        for path in nvidia_root.glob("*/bin")
        if path.is_dir()
    )
    current_path = os.environ.get("PATH", "")
    prefixes = [str(path) for path in dll_dirs]
    os.environ["PATH"] = os.pathsep.join(prefixes + [current_path])
    for path in dll_dirs:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
    return prefixes


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

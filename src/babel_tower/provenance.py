"""Facts about how a run was made, for run.json."""

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PACKAGES = ("babel-tower", "torch", "sentence-transformers", "transformers", "numpy", "ebooklib")
REPO_ROOT = Path(__file__).resolve().parents[2]


def file_info(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def code_info() -> dict[str, Any] | None:
    """The commit of this code, and whether src/ or pyproject.toml has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock").strip())
    except subprocess.CalledProcessError, FileNotFoundError:
        return None
    return {"commit": commit, "dirty": dirty}


def env_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {pkg: _version(pkg) for pkg in PACKAGES},
    }
    try:
        import torch
    except ImportError:
        return info
    info["torch_cuda"] = torch.version.cuda
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _version(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None

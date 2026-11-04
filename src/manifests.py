"""Environment and provenance manifests without credentials or raw text."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

from .io_utils import sha256_file, write_json


DEFAULT_PACKAGES = (
    "torch", "transformers", "datasets", "accelerate", "sentencepiece",
    "protobuf", "huggingface_hub", "safetensors", "scikit-learn", "pandas",
    "numpy", "scipy", "matplotlib", "pyyaml", "pytest", "pyarrow", "nbformat",
)


def package_versions(packages: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def git_revision(root: str | Path) -> dict:
    root = Path(root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def environment_manifest(root: str | Path) -> dict:
    cuda = {"available": False, "torch_version": None}
    try:
        import torch
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": package_versions(),
        "cuda": cuda,
        "git": git_revision(root),
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }


def write_environment_manifest(path: str | Path, root: str | Path) -> Path:
    return write_json(path, environment_manifest(root))


def file_manifest(paths: Iterable[str | Path]) -> dict[str, dict]:
    output = {}
    for item in paths:
        path = Path(item)
        output[str(path)] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return output

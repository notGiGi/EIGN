"""Environment detection and path resolution for portable deployment."""
from __future__ import annotations

import os
from pathlib import Path


def detect_environment() -> str:
    """Detect runtime environment.

    Returns:
        "kaggle" if running on Kaggle
        "local" otherwise
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root
    """
    # Assume this file is in src/eign/env.py
    return Path(__file__).resolve().parents[2]


def get_cache_dir() -> Path:
    """Get cache directory for tokenized datasets.

    Returns:
        Path to cache directory (environment-aware)
    """
    env = detect_environment()
    if env == "kaggle":
        cache_dir = Path("/kaggle/working/cache")
    else:
        cache_dir = get_project_root() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_runs_dir() -> Path:
    """Get runs directory for training outputs.

    Returns:
        Path to runs directory (environment-aware)
    """
    env = detect_environment()
    if env == "kaggle":
        runs_dir = Path("/kaggle/working/runs")
    else:
        runs_dir = get_project_root() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_data_dir() -> Path:
    """Get data directory for datasets.

    Returns:
        Path to data directory (environment-aware)
    """
    env = detect_environment()
    if env == "kaggle":
        # Kaggle datasets are typically mounted at /kaggle/input
        data_dir = Path("/kaggle/input")
    else:
        data_dir = get_project_root() / "data"
    return data_dir


def get_artifacts_dir() -> Path:
    """Get artifacts directory for tokenizers and other artifacts.

    Returns:
        Path to artifacts directory (environment-aware)
    """
    env = detect_environment()
    if env == "kaggle":
        artifacts_dir = Path("/kaggle/working/artifacts")
    else:
        artifacts_dir = get_project_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir

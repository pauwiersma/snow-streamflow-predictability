"""Portable path resolution via DATA_ROOT / repo-relative defaults."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root for analysis data shipped with this repository.

    Set ``SSP_DATA_ROOT`` or ``DATA_ROOT`` to override. Defaults to ``./data``.
    """
    env = os.environ.get("SSP_DATA_ROOT") or os.environ.get("DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "data").resolve()


def descriptors_dir() -> Path:
    return data_root() / "descriptors"


def tables_dir() -> Path:
    return data_root() / "tables"


def figures_dir() -> Path:
    out = data_root() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def config_dir() -> Path:
    return REPO_ROOT / "config"

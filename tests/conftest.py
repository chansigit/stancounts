"""Shared pytest fixtures and data-path resolution.

Real-data tests are skipped unless the following environment variables point
to readable files/directories:

    STANCOUNTS_PBMC_DIR    - directory containing PBMC h5ad files
    STANCOUNTS_CHONDRO_DIR - directory containing chondro-atlas h5ad files

Defaults point to the maintainer's workstation, so CI / third-party clones
cleanly skip rather than fail.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


@pytest.fixture(scope="session")
def pbmc_dir() -> Path:
    return _env_path("STANCOUNTS_PBMC_DIR",
                     "/home/users/chensj16/s/projects/genofoundation/data/pbmc")


@pytest.fixture(scope="session")
def chondro_dir() -> Path:
    return _env_path("STANCOUNTS_CHONDRO_DIR",
                     "/home/users/chensj16/s/data/chondro-atlas/h5ad")

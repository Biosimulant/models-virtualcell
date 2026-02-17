from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MODEL_DIR = Path(__file__).resolve().parents[1]
_MODELS_DIR = _MODEL_DIR.parent

_DEPS = ["virtualcell-perturbation-source"]


@pytest.fixture(scope="session", autouse=True)
def _paths():
    paths = [str(_MODEL_DIR)]
    paths += [str(_MODELS_DIR / d) for d in _DEPS]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


@pytest.fixture(scope="session")
def bsim(_paths):
    import bsim as _bsim
    return _bsim

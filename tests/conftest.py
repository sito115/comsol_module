"""
Shared pytest fixtures for ComsolVtu tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from comsol_module import ComsolVtu

TEST_DATA_FOLDER = Path(__file__).resolve().parent / "data"

STATIONARY_PATH = TEST_DATA_FOLDER / "Example_Stationary_NonSweep.vtu"
TRANSIENT_PATH = TEST_DATA_FOLDER / "Example_Transient_NonSweep.vtu"
SWEEP_PATH = TEST_DATA_FOLDER / "Example_TransientSweep.vtu"
CSV_PATH = TEST_DATA_FOLDER / "Scen_1AvLineProductionWell.csv"


def _load(path: Path) -> ComsolVtu:
    if not path.exists():
        pytest.skip(f"VTU file not found: {path}")
    return ComsolVtu.from_file(path)


@pytest.fixture(scope="session")
def vtu_stationary() -> ComsolVtu:
    return _load(STATIONARY_PATH)


@pytest.fixture(scope="session")
def vtu_transient() -> ComsolVtu:
    return _load(TRANSIENT_PATH)


@pytest.fixture(scope="session")
def vtu_sweep() -> ComsolVtu:
    return _load(SWEEP_PATH)

"""
Shared pytest fixtures for ComsolVtu tests.

Three session-scoped fixtures are provided, one per study type:

  vtu_stationary  → Example_Stationary_NonSweep.vtu
  vtu_transient   → Example_Transient_NonSweep.vtu
  vtu_sweep       → Example_TransientSweep.vtu

Each is skipped automatically when its file is not present on disk.

The legacy ``vtu`` / ``vtu_path`` fixtures are kept for backwards
compatibility with any test that still references them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from comsol_module import ComsolVtu

# ---------------------------------------------------------------------------
# Project root + canonical file paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent

_STATIONARY_PATH = _PROJECT_ROOT / "Example_Stationary_NonSweep.vtu"
_TRANSIENT_PATH = _PROJECT_ROOT / "Example_Transient_NonSweep.vtu"
_SWEEP_PATH = _PROJECT_ROOT / "Example_TransientSweep.vtu"

# Legacy default kept for the --vtu-path CLI option
DEFAULT_VTU_PATH = _STATIONARY_PATH


# ---------------------------------------------------------------------------
# CLI option (legacy)
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vtu-path",
        default=str(DEFAULT_VTU_PATH),
        help="Absolute path to the VTU file used for legacy tests "
             f"(default: {DEFAULT_VTU_PATH})",
    )


# ---------------------------------------------------------------------------
# Helper: build a (path, vtu) fixture pair from a fixed file path
# ---------------------------------------------------------------------------

def _make_fixtures(path: Path, fixture_name: str):
    """Return a (path_fixture, vtu_fixture) pair for *path*."""

    @pytest.fixture(scope="session", name=f"{fixture_name}_path")
    def _path_fixture() -> Path:
        if not path.exists():
            pytest.skip(f"VTU file not found: {path}")
        return path

    @pytest.fixture(scope="session", name=fixture_name)
    def _vtu_fixture(request: pytest.FixtureRequest) -> ComsolVtu:
        _path = request.getfixturevalue(f"{fixture_name}_path")
        return ComsolVtu.from_file(_path)

    return _path_fixture, _vtu_fixture


# ---------------------------------------------------------------------------
# Per-study-type session fixtures
# ---------------------------------------------------------------------------

vtu_stationary_path, vtu_stationary = _make_fixtures(
    _STATIONARY_PATH, "vtu_stationary")
vtu_transient_path,  vtu_transient = _make_fixtures(
    _TRANSIENT_PATH,  "vtu_transient")
vtu_sweep_path,      vtu_sweep = _make_fixtures(
    _SWEEP_PATH,       "vtu_sweep_transient")


# ---------------------------------------------------------------------------
# Legacy fixtures (kept for backwards compatibility)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vtu_path(request: pytest.FixtureRequest) -> Path:
    """Resolved path to the VTU file under test (legacy CLI option)."""
    p = Path(request.config.getoption("--vtu-path"))
    if not p.exists():
        pytest.skip(f"VTU file not found: {p}")
    return p


@pytest.fixture(scope="session")
def vtu(vtu_path: Path) -> ComsolVtu:
    """Session-scoped ComsolVtu loaded from the CLI-configured path (legacy)."""
    return ComsolVtu.from_file(vtu_path)

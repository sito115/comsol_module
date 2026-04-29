"""
Pytest suite for ComsolVtu.
Simplified, industrial-strength tests covering Stationary, Transient, and Sweep files.
"""

import copy
import warnings
from typing import cast

import numpy as np
import pytest
import pyvista as pv

from comsol_module import ComsolVtu
from comsol_module.helper import ComsolKeyNames

# ===========================================================================
# Configuration & Helpers
# ===========================================================================


@pytest.fixture
def vtu_copy(request):
    """Returns a function-scoped deep copy of a session-scoped fixture."""
    fixture_name = request.param
    return copy.deepcopy(request.getfixturevalue(fixture_name))


# ===========================================================================
# Core Interface Tests (Parameterized)
# ===========================================================================


def _test_get_values(vtu: ComsolVtu):
    sweep_values = vtu.sweep_combos[0] if vtu._is_sweep else None

    f = vtu.exported_fields[0]
    field_name = vtu.format_field(f, -1, sweep_values)
    _ = vtu.get_values(field_name)
    field_name = vtu.format_field(
        f, vtu.time_keys[0] if vtu.time_keys else "0", sweep_values
    )
    _ = vtu.get_values(field_name)
    field_name = vtu.format_field(
        f, vtu.time_values[0] if vtu.time_values else 0.0, sweep_values
    )
    _ = vtu.get_values(field_name)


def _test_properties_and_mesh(vtu: ComsolVtu):
    assert len(vtu.exported_fields) > 0
    assert vtu.mesh.n_points > 0
    assert vtu.mesh.n_cells > 0
    assert vtu.vtu_path != ""
    assert repr(vtu).startswith("ComsolVtu(")


def _test_data_stores(vtu: ComsolVtu):
    assert vtu._data_store("point") == vtu.mesh.point_data
    assert vtu._data_store("cell") == vtu.mesh.cell_data
    assert vtu._n_values("point") == vtu.mesh.n_points
    assert vtu._n_values("cell") == vtu.mesh.n_cells


@pytest.mark.parametrize("vtu_name", ["vtu_stationary", "vtu_transient", "vtu_sweep"])
def test_vtu_basic_interface(vtu_name, request: pytest.FixtureRequest):
    """Verify basic properties, mesh integrity, and info() runs without error."""
    vtu = cast(ComsolVtu, request.getfixturevalue(vtu_name))

    # 1. Properties & Mesh
    _test_properties_and_mesh(vtu)
    # 2. info() runs
    vtu.info()

    # 3. Data stores
    _test_data_stores(vtu)

    _test_get_values(vtu)


@pytest.mark.parametrize("vtu_name", ["vtu_stationary", "vtu_transient", "vtu_sweep"])
def test_vtu_mesh_operations(vtu_name, request: pytest.FixtureRequest):
    """Verify mesh updates and consistency between from_file and from_mesh."""
    vtu = cast(ComsolVtu, request.getfixturevalue(vtu_name))

    # 1. from_mesh consistency
    vtu_from_mesh = ComsolVtu.from_mesh(vtu.mesh)
    assert set(vtu_from_mesh.exported_fields) == set(vtu.exported_fields)

    # 2. update_mesh
    tiny = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    updated = vtu.update_mesh(tiny)
    assert updated is not vtu
    assert updated.mesh is tiny
    assert updated.exported_fields == vtu.exported_fields


# ===========================================================================
# Study-Specific Tests (Stationary)
# ===========================================================================


def test_stationary_properties(vtu_stationary: ComsolVtu):
    vtu = vtu_stationary
    assert vtu._is_stationary is True
    assert vtu._is_sweep is False
    assert len(vtu.times) == 0
    assert "Temperature" in vtu.exported_fields


def test_stationary_get_values(vtu_stationary: ComsolVtu):
    vtu = vtu_stationary
    f = vtu.exported_fields[0]

    # Point data
    pts = vtu.get_values(f, location="point")
    assert pts.shape == (vtu.mesh.n_points,)

    # Cell data (via conversion)
    vtu_cell = copy.deepcopy(vtu)
    vtu_cell.convert_to_cell_data()
    cls = vtu_cell.get_values(f, location="cell")
    assert cls.shape == (vtu_cell.mesh.n_cells,)


def test_stationary_unsupported_ops(vtu_stationary: ComsolVtu):
    vtu = copy.deepcopy(vtu_stationary)
    with pytest.raises(NotImplementedError):
        vtu.unify_field("Temperature")
    with pytest.raises(NotImplementedError):
        vtu.delete_field("Temperature")
    with pytest.raises(NotImplementedError):
        vtu.merge_datasets(vtu)


# ===========================================================================
# Study-Specific Tests (Transient)
# ===========================================================================


def test_transient_properties(vtu_transient: ComsolVtu):
    vtu = vtu_transient
    assert vtu._is_stationary is False
    assert vtu._is_sweep is False
    assert len(vtu.times) > 1


def test_transient_get_array(vtu_transient: ComsolVtu):
    vtu = vtu_transient
    f = vtu.exported_fields[0]
    arr = vtu.get_array(f)
    assert arr.ndim == 2
    assert arr.shape == (len(vtu.times), vtu.mesh.n_points)


def test_transient_mutations(vtu_transient: ComsolVtu):
    """Test unify and delete which are supported for Transient-NonSweep."""
    vtu = copy.deepcopy(vtu_transient)
    f = vtu.exported_fields[0]

    vtu.unify_field(f)
    assert f in vtu.mesh.point_data

    vtu.delete_field(f)
    assert f not in vtu.exported_fields


def test_transient_merge(vtu_transient: ComsolVtu):
    """Verify merge_datasets works for transient non-sweep studies."""
    vtu_a = copy.deepcopy(vtu_transient)
    vtu_b = copy.deepcopy(vtu_transient)

    # Simulate a new timestep in vtu_b
    new_t = "9999.0"
    vtu_b.times[new_t] = 9999.0
    for field in vtu_b.exported_fields:
        # Get data from some existing timestep to replicate
        old_key = vtu_b.format_field(field, 0)
        vtu_b.mesh.point_data[f"{field}_@_t={new_t}"] = vtu_b.mesh.point_data[old_key]

    vtu_a.merge_datasets(vtu_b)
    assert new_t in vtu_a.times
    assert len(vtu_a.times) == len(vtu_transient.times) + 1


# ===========================================================================
# Study-Specific Tests (Sweep)
# ===========================================================================


def test_sweep_properties(vtu_sweep: ComsolVtu):
    vtu = vtu_sweep
    assert vtu._is_sweep is True
    assert len(vtu.times) > 0


def test_sweep_get_array(vtu_sweep: ComsolVtu):
    vtu = vtu_sweep
    f = vtu.exported_fields[0]
    arr = vtu.get_array(f)
    # Shape: (T, Sweep, N)
    assert arr.ndim == 3
    assert arr.shape[0] == len(vtu.times)
    assert arr.shape[1] == len(vtu.sweep_combos)
    assert arr.shape[2] == vtu.mesh.n_points


# ===========================================================================
# Feature Tests (Cross-cutting)
# ===========================================================================


def test_deprecated_get_point_values(vtu_stationary: ComsolVtu):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = vtu_stationary.get_point_values(vtu_stationary.exported_fields[0])
        assert len(w) > 0
        assert issubclass(w[-1].category, DeprecationWarning)
        assert res.shape == (vtu_stationary.mesh.n_points,)


def test_overwrite_domain(vtu_stationary: ComsolVtu):
    vtu = copy.deepcopy(vtu_stationary)
    f = vtu.exported_fields[0]

    # Create synthetic surface
    surf_pts = vtu.mesh.points[:5].copy()
    surf = pv.PolyData(surf_pts)
    surf.point_data["Color"] = np.full(5, 999.0)

    vtu.overwrite_domain_from_surface(surf, f, "Color")
    # Spot check first point
    assert vtu.get_values(f)[0] == pytest.approx(999.0)


def test_key_enum_access(vtu_stationary: ComsolVtu):
    """Verify ComsolKeyNames enum works for field access."""
    vtu = vtu_stationary
    # T and P are universally present in these test files
    assert vtu.get_values(ComsolKeyNames.T).shape == (vtu.mesh.n_points,)
    assert vtu.get_values(ComsolKeyNames.P).shape == (vtu.mesh.n_points,)

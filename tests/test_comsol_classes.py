"""
Pytest test suite for ComsolVtu.

Tests are organised around base classes (one per study type) so that adding
a new VTU file later only requires:
  1. A new session fixture in conftest.py that loads the file.
  2. A new concrete subclass of the appropriate base class pointing at that
     fixture (set ``vtu_fixture`` to the fixture name).

Files under test
----------------
Example_Stationary_NonSweep.vtu  — stationary, no sweep
  Fields (5): Effective_volumetric_heat_capacity, Temperature,
              Material_settings, Porosity, Pressure

Example_Transient_NonSweep.vtu   — transient, no sweep
  Fields (4): Effective_volumetric_heat_capacity, Temperature,
              Porosity, Pressure
  Times: 101 steps (0, 1E11, 2E11, …)

Example_TransientSweep.vtu       — transient, sweep
  Fields (5): Stress_tensor,_x-component, Stress_tensor,_y-component,
              Temperature, Stress_tensor,_z-component, Pore_pressure
  Times: 2 steps (0, 1E13)
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from comsol_module import ComsolVtu
from comsol_module.helper import ComsolKeyNames


# ===========================================================================
# Shared function-scoped copy fixture
# (needed for tests that mutate state — keeps the session fixture pristine)
# ===========================================================================

@pytest.fixture
def vtu_copy(vtu: ComsolVtu) -> ComsolVtu:
    """Deep-copy of the session VTU so mutating tests don't interfere."""
    return copy.deepcopy(vtu)


# ===========================================================================
# Base test classes
# Each class receives a `vtu` fixture via `self.vtu_fixture` (set per subclass)
# and a matching `vtu_copy` for mutating tests.
# ===========================================================================

class _BaseVtuTests:
    """
    Common tests valid for *every* study type.

    Subclasses must set:
        vtu_fixture     – name of the session fixture (string)
        n_fields        – expected number of exported fields (int)
        first_field     – a known field name guaranteed to be in the file (str)
    """
    vtu_fixture: str = "vtu"
    n_fields: int
    first_field: str
    field_name: str = ""

    # ---- helpers -----------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _inject_fixtures(self, request):
        self.vtu: ComsolVtu = request.getfixturevalue(self.vtu_fixture)
        self.vtu_copy: ComsolVtu = copy.deepcopy(self.vtu)
        sweep_values = self.vtu.sweep_combos[0] if self.vtu._is_sweep else None

        self.field_name = self.vtu.format_field(field_name=self.first_field,
                                                time=0,
                                                sweep_values=sweep_values)

    # ---- construction / repr -----------------------------------------------

    def test_fields_not_empty(self):
        assert len(self.vtu.exported_fields) == self.n_fields

    def test_known_field_present(self):
        assert self.first_field in self.vtu.exported_fields

    def test_mesh_has_points(self):
        assert self.vtu.mesh.n_points > 0

    def test_mesh_has_cells(self):
        assert self.vtu.mesh.n_cells > 0

    def test_repr_format(self):
        r = repr(self.vtu)
        assert r.startswith("ComsolVtu(")
        assert "fields=" in r

    def test_vtu_path_set(self):
        assert self.vtu.vtu_path != ""

    # ---- info() ------------------------------------------------------------

    def test_info_runs(self, capsys):
        self.vtu.info()
        out = capsys.readouterr().out
        assert self.first_field in out

    # ---- data store --------------------------------------------------------

    def test_data_store_point(self):
        ds = self.vtu._data_store("point")
        assert ds == self.vtu.mesh.point_data

    def test_n_values_point(self):
        assert self.vtu._n_values("point") == self.vtu.mesh.n_points

    def test_n_values_cell(self):
        assert self.vtu._n_values("cell") == self.vtu.mesh.n_cells

    def test_convert_to_cell_data(self):
        vtu = self.vtu_copy
        n_cells = vtu.mesh.n_cells
        vtu.convert_to_cell_data()
        assert vtu.mesh.n_cells == n_cells
        assert self.field_name in vtu.mesh.cell_data.keys()

    def test_convert_to_cell_data_no_pass_point(self):
        vtu = self.vtu_copy
        vtu.convert_to_cell_data(pass_point_data=False)
        assert len(vtu.mesh.point_data.keys()) == 0

    # ---- get_values --------------------------------------------------------

    def test_get_values_point_shape(self):
        arr = self.vtu.get_values(self.field_name, location="point")
        assert arr.shape == (self.vtu.mesh.n_points,)

    def test_get_values_default_is_point(self):
        arr_default = self.vtu.get_values(self.field_name)
        arr_point = self.vtu.get_values(self.field_name, location="point")
        np.testing.assert_array_equal(arr_default, arr_point)

    def test_get_values_cell_shape(self):
        vtu = self.vtu_copy
        vtu.convert_to_cell_data()
        arr = vtu.get_values(self.field_name, location="cell")
        assert arr.shape == (vtu.mesh.n_cells,)

    def test_get_point_values_deprecated(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.vtu.get_point_values(self.field_name)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
        np.testing.assert_array_equal(
            result, self.vtu.get_values(self.field_name)
        )

    # ---- update_mesh -------------------------------------------------------

    def test_update_mesh_returns_new_instance(self):
        tiny = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
        updated = self.vtu.update_mesh(tiny)
        assert updated is not self.vtu
        assert updated.mesh is tiny

    def test_update_mesh_preserves_metadata(self):
        vtu = self.vtu_copy
        vtu.name = "meta_check"
        tiny = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
        updated = vtu.update_mesh(tiny)
        assert updated.name == "meta_check"
        assert updated.exported_fields == vtu.exported_fields

    def test_update_mesh_original_unchanged(self):
        orig_pts = self.vtu.mesh.n_points
        tiny = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
        self.vtu.update_mesh(tiny)
        assert self.vtu.mesh.n_points == orig_pts

    # ---- from_file / from_mesh consistency ---------------------------------

    def test_from_file_vs_from_mesh_same_fields(self):
        vtu_from_mesh = ComsolVtu.from_mesh(self.vtu.mesh)
        assert set(vtu_from_mesh.exported_fields) == set(
            self.vtu.exported_fields)


class _StationaryVtuTests(_BaseVtuTests):
    """Additional tests specific to stationary studies."""

    def test_is_stationary(self):
        assert self.vtu._is_stationary is True

    def test_is_not_sweep(self):
        assert self.vtu._is_sweep is False

    def test_times_empty(self):
        assert len(self.vtu.times) == 0

    def test_info_says_stationary(self, capsys):
        self.vtu.info()
        out = capsys.readouterr().out
        assert "Stationary" in out

    # ---- format_field (stationary always returns bare name) ----------------

    def test_format_field_returns_field_name(self):
        result = self.vtu.format_field(self.first_field, 0)
        assert result == self.first_field

    # ---- get_array ---------------------------------------------------------

    def test_get_array_point_shape(self):
        arr = self.vtu.get_array(self.first_field)
        assert arr.ndim == 1
        assert arr.shape[0] == self.vtu.mesh.n_points

    def test_get_array_cell_shape(self):
        vtu = self.vtu_copy
        vtu.convert_to_cell_data()
        arr = vtu.get_array(self.first_field, location="cell")
        assert arr.ndim == 1
        assert arr.shape[0] == vtu.mesh.n_cells

    def test_get_array_consistent_with_get_values(self):
        arr = self.vtu.get_array(self.first_field)
        direct = self.vtu.get_values(self.first_field)
        np.testing.assert_array_equal(arr, direct)

    def test_get_array_unknown_field_raises(self):
        with pytest.raises(KeyError, match="not found in exported fields"):
            self.vtu.get_array("__nonexistent__")

    # ---- unify_field -- not supported for stationary -----------------------

    def test_unify_field_raises_for_stationary(self):
        with pytest.raises(NotImplementedError):
            self.vtu_copy.unify_field(self.first_field)

    # ---- delete_field -- not supported for stationary ----------------------

    def test_delete_field_raises_for_stationary(self):
        with pytest.raises(NotImplementedError):
            self.vtu_copy.delete_field(self.first_field)

    def test_delete_field_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.vtu_copy.delete_field("__nonexistent__")

    # ---- overwrite_domain_from_surface ------------------------------------

    def test_overwrite_domain_missing_3d_field_raises(self):
        surf = pv.PolyData(self.vtu.mesh.points[:2])
        surf.point_data["Color"] = np.zeros(2)
        with pytest.raises(KeyError, match="not found in 3D mesh"):
            self.vtu_copy.overwrite_domain_from_surface(
                surf, "__missing__", "Color")

    def test_overwrite_domain_missing_surface_field_raises(self):
        surf = pv.PolyData(self.vtu.mesh.points[:2])
        # No "Color" on the surface
        with pytest.raises(KeyError, match="not found in surface dataset"):
            self.vtu_copy.overwrite_domain_from_surface(
                surf, self.first_field, "Color"
            )

    def test_overwrite_domain_updates_values(self):
        """Values at surface points should be replaced in the 3D mesh."""
        vtu = self.vtu_copy
        surface_pts = vtu.mesh.points[:4].copy()
        target_val = 42.0
        surface = pv.PolyData(surface_pts)
        surface.point_data["Color"] = np.full(4, target_val)

        vtu.overwrite_domain_from_surface(surface, self.first_field, "Color")

        field_vals = vtu.mesh.point_data[self.first_field]
        for pt in surface_pts:
            idx = np.where(np.all(vtu.mesh.points == pt, axis=1))[0]
            if idx.size:
                assert field_vals[idx[0]] == pytest.approx(target_val)

    # ---- merge_datasets -- error paths only for stationary -----------------

    def test_merge_raises_for_stationary(self):
        with pytest.raises(NotImplementedError):
            self.vtu_copy.merge_datasets(self.vtu_copy)


class _TransientVtuTests(_BaseVtuTests):
    """Additional tests specific to transient (time-dependent) studies."""

    def test_is_not_stationary(self):
        assert self.vtu._is_stationary is False

    def test_is_not_sweep(self):
        assert self.vtu._is_sweep is False

    def test_times_not_empty(self):
        assert len(self.vtu.times) > 0

    def test_info_says_time_dependent(self, capsys):
        self.vtu.info()
        out = capsys.readouterr().out
        assert "Time-dependent" in out

    # ---- format_field ------------------------------------------------------

    def test_format_field_by_string_key(self):
        time_key = next(iter(self.vtu.times))
        result = self.vtu.format_field(self.first_field, time_key)
        assert self.first_field in result
        assert time_key in result

    def test_format_field_by_index(self):
        time_key = next(iter(self.vtu.times))
        result = self.vtu.format_field(self.first_field, 0)
        assert time_key in result

    def test_format_field_by_float_nearest(self):
        t_val = next(iter(self.vtu.times.values()))
        result = self.vtu.format_field(self.first_field, t_val)
        assert self.first_field in result

    def test_format_field_bad_string_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.vtu.format_field(self.first_field, "__badtime__")

    def test_format_field_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            self.vtu.format_field(self.first_field, len(self.vtu.times) + 100)

    def test_format_field_bad_type_raises(self):
        with pytest.raises(TypeError):
            self.vtu.format_field(self.first_field, [1.0])  # type: ignore

    # ---- get_array ---------------------------------------------------------

    def test_get_array_point_shape(self):
        arr = self.vtu.get_array(self.first_field)
        n_t = len(self.vtu.times)
        assert arr.shape == (n_t, self.vtu.mesh.n_points)

    def test_get_array_cell_shape(self):
        vtu = self.vtu_copy
        vtu.convert_to_cell_data()
        arr = vtu.get_array(self.first_field, location="cell")
        assert arr.shape == (len(vtu.times), vtu.mesh.n_cells)

    def test_get_array_unknown_field_raises(self):
        with pytest.raises(KeyError, match="not found in exported fields"):
            self.vtu.get_array("__nonexistent__")

    # ---- unify_field -------------------------------------------------------

    def test_unify_field_creates_unified_key(self):
        vtu = self.vtu_copy
        vtu.unify_field(self.first_field)
        assert self.first_field in vtu.mesh.point_data

    def test_unify_field_removes_timestep_keys(self):
        vtu = self.vtu_copy
        vtu.unify_field(self.first_field)
        remaining = vtu.mesh.point_data.keys()
        leftover = [
            k for k in remaining if "_@_t=" in k and self.first_field in k]
        assert len(leftover) == 0

    # ---- delete_field ------------------------------------------------------

    def test_delete_field_removes_from_exported_fields(self):
        vtu = self.vtu_copy
        vtu.delete_field(self.first_field)
        assert self.first_field not in vtu.exported_fields

    def test_delete_field_removes_data_keys(self):
        vtu = self.vtu_copy
        vtu.delete_field(self.first_field)
        leftover = [k for k in vtu.mesh.point_data.keys()
                    if self.first_field in k]
        assert len(leftover) == 0

    def test_delete_field_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.vtu_copy.delete_field("__nonexistent__")

    # ---- merge_datasets ----------------------------------------------------

    def test_merge_adds_new_times(self):
        """Merge a synthetic compatible dataset with a novel timestep."""
        vtu_a = self.vtu_copy
        new_mesh = vtu_a.mesh.copy()
        # Remove all existing keys and add one new timestep
        for k in list(new_mesh.point_data.keys()):
            new_mesh.point_data.remove(k)
        rng = np.random.default_rng(0)
        for field in vtu_a.exported_fields:
            new_mesh.point_data[f"{field}_@_t=9999.0"] = rng.random(
                new_mesh.n_points)
        vtu_b = ComsolVtu.from_mesh(new_mesh)

        vtu_a.merge_datasets(vtu_b)
        assert "9999.0" in vtu_a.times

    def test_merge_times_sorted_after_merge(self):
        vtu_a = self.vtu_copy
        new_mesh = vtu_a.mesh.copy()
        for k in list(new_mesh.point_data.keys()):
            new_mesh.point_data.remove(k)
        rng = np.random.default_rng(1)
        for field in vtu_a.exported_fields:
            new_mesh.point_data[f"{field}_@_t=9999.0"] = rng.random(
                new_mesh.n_points)
        vtu_b = ComsolVtu.from_mesh(new_mesh)

        vtu_a.merge_datasets(vtu_b)
        values = list(vtu_a.times.values())
        assert values == sorted(values)

    def test_merge_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self.vtu_copy.merge_datasets("not_a_vtu")  # type: ignore

    def test_merge_shape_mismatch_raises(self):
        small = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
        rng = np.random.default_rng(2)
        for field in self.vtu.exported_fields:
            small.point_data[f"{field}_@_t=9999.0"] = rng.random(
                small.n_points)
        vtu_small = ComsolVtu.from_mesh(small)
        with pytest.raises(ValueError, match="different point counts"):
            self.vtu_copy.merge_datasets(vtu_small)


# ===========================================================================
# Sweep-specific base class
# ===========================================================================

class _SweepVtuTests(_BaseVtuTests):
    """Additional tests specific to sweep (transient + parametric sweep) studies."""

    def test_is_not_stationary(self):
        assert self.vtu._is_stationary is False

    def test_is_sweep(self):
        assert self.vtu._is_sweep is True

    def test_times_not_empty(self):
        assert len(self.vtu.times) > 0

    def test_info_says_time_dependent(self, capsys):
        self.vtu.info()
        out = capsys.readouterr().out
        assert "Time-dependent" in out

    # ---- format_field ------------------------------------------------------

    def test_format_field_by_index(self):
        time_key = next(iter(self.vtu.times))
        result = self.vtu.format_field(
            self.first_field,  0, self.vtu.sweep_combos[0])
        assert time_key in result

    def test_format_field_bad_string_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.vtu.format_field(self.first_field, "__badtime__")

    def test_format_field_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            self.vtu.format_field(self.first_field, len(self.vtu.times) + 100)

    # ---- get_array ---------------------------------------------------------

    def test_get_array_point_shape(self):
        arr = self.vtu.get_array(self.first_field)
        n_t = len(self.vtu.times)
        assert arr.shape == (n_t, self.vtu.sweep_combos.shape[0],
                             self.vtu.mesh.n_points)

    def test_get_array_unknown_field_raises(self):
        with pytest.raises(KeyError, match="not found in exported fields"):
            self.vtu.get_array("__nonexistent__")

    # ---- unify_field -------------------------------------------------------

    def test_unify_field_raises_for_sweep(self):
        """Sweep vtus raise NotImplementedError for unify_field."""
        with pytest.raises(NotImplementedError):
            self.vtu_copy.unify_field(self.first_field)

    # ---- delete_field ------------------------------------------------------

    def test_delete_field_raises_for_sweep(self):
        """Sweep vtus raise NotImplementedError for delete_field."""
        with pytest.raises(NotImplementedError):
            self.vtu_copy.delete_field(self.first_field)

    def test_delete_field_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.vtu_copy.delete_field("__nonexistent__")

    # ---- merge_datasets ----------------------------------------------------

    def test_merge_raises_for_sweep(self):
        """Sweep vtus raise NotImplementedError for merge_datasets."""
        with pytest.raises(NotImplementedError):
            self.vtu_copy.merge_datasets(self.vtu_copy)


# ===========================================================================
# Concrete test classes — one per VTU file
# ===========================================================================

class TestExampleStationaryNonSweep(_StationaryVtuTests):
    """
    Tests for Example_Stationary_NonSweep.vtu
    Study: stationary, no sweep
    Fields (5): Effective_volumetric_heat_capacity, Temperature,
                Material_settings, Porosity, Pressure
    """
    vtu_fixture = "vtu_stationary"
    n_fields = 5
    first_field = "Temperature"

    @pytest.mark.parametrize("field_name", [
        "Temperature",
        "Pressure",
        "Porosity",
        "Effective_volumetric_heat_capacity",
        "Material_settings",
    ])
    def test_all_expected_fields_present(self, field_name):
        assert field_name in self.vtu.exported_fields

    def test_mesh_n_points(self):
        assert self.vtu.mesh.n_points == 384_000

    def test_mesh_n_cells(self):
        assert self.vtu.mesh.n_cells == 384_000

    def test_from_file_path_is_set(self, vtu_stationary_path):
        assert self.vtu.vtu_path == vtu_stationary_path

    def test_key_enum_temperature(self):
        arr = self.vtu.get_values(ComsolKeyNames.T)
        assert arr.shape[0] == self.vtu.mesh.n_points

    def test_key_enum_pressure(self):
        arr = self.vtu.get_values(ComsolKeyNames.P)
        assert arr.shape[0] == self.vtu.mesh.n_points


class TestExampleTransientNonSweep(_TransientVtuTests):
    """
    Tests for Example_Transient_NonSweep.vtu
    Study: transient, no sweep
    Fields (4): Effective_volumetric_heat_capacity, Temperature,
                Porosity, Pressure
    Times: 101 steps (0, 1E11, 2E11, …)
    """
    vtu_fixture = "vtu_transient"
    n_fields = 4
    first_field = "Temperature"

    @pytest.mark.parametrize("field_name", [
        "Temperature",
        "Pressure",
        "Porosity",
        "Effective_volumetric_heat_capacity",
    ])
    def test_all_expected_fields_present(self, field_name):
        assert field_name in self.vtu.exported_fields

    def test_n_timesteps(self):
        assert len(self.vtu.times) == 101

    def test_mesh_n_points(self):
        assert self.vtu.mesh.n_points == 1_000

    def test_mesh_n_cells(self):
        assert self.vtu.mesh.n_cells == 1_000

    def test_from_file_path_is_set(self, vtu_transient_path):
        assert self.vtu.vtu_path == vtu_transient_path


class TestExampleTransientSweep(_SweepVtuTests):
    """
    Tests for Example_TransientSweep.vtu
    Study: transient, parametric sweep
    Fields (5): Stress_tensor,_x-component, Stress_tensor,_y-component,
                Temperature, Stress_tensor,_z-component, Pore_pressure
    Times: 2 steps (0, 1E13)
    """
    vtu_fixture = "vtu_sweep_transient"
    n_fields = 5
    first_field = "Temperature"

    @pytest.mark.parametrize("field_name", [
        "Temperature",
        "Pore_pressure",
        "Stress_tensor,_x-component",
        "Stress_tensor,_y-component",
        "Stress_tensor,_z-component",
    ])
    def test_all_expected_fields_present(self, field_name):
        assert field_name in self.vtu.exported_fields

    def test_n_timesteps(self):
        assert len(self.vtu.times) == 2

    def test_mesh_n_points(self):
        assert self.vtu.mesh.n_points == 18_082

    def test_mesh_n_cells(self):
        assert self.vtu.mesh.n_cells == 88_975

    def test_from_file_path_is_set(self, vtu_sweep_transient_path):
        assert self.vtu.vtu_path == vtu_sweep_transient_path

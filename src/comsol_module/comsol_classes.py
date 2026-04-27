from .helper import (
    ComsolKeyNames,
    format_sweep_parameters,
    get_field_name_pattern,
    read_comsol_fields,
    determine_time_key
)
import logging
import warnings
from dataclasses import field, replace, dataclass
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import pyvista as pv

#: Selector for point-based or cell-based data access on a PyVista mesh.
DataLocation = Literal["point", "cell"]


@dataclass
class ComsolVtu:
    """Class to read and process exported simulation files from COMSOL."""

    mesh: pv.DataSet
    vtu_path: Path | str = ""
    name: str = ""

    exported_fields: list[str] = field(default_factory=list)
    times: dict[str, float] = field(default_factory=dict)
    sweep_keys: list[str] = field(default_factory=list)
    sweep_combos: np.ndarray = field(default_factory=lambda: np.array([]))

    _is_sweep: bool = False
    _is_stationary: bool = False
    field_pattern: str = ""

    @property
    def time_keys(self) -> list[str]:
        return list(self.times.keys())

    @property
    def time_values(self) -> list[float]:
        return list(self.times.values())

    @classmethod
    def from_file(cls, path: str | Path, is_clean_mesh: bool = False) -> Self:
        path = path if isinstance(path, Path) else Path(path)

        logging.debug("Reading VTU file...")
        mesh: pv.DataSet = cast(pv.DataSet, pv.wrap(pv.read(path)))

        if is_clean_mesh:
            mesh = mesh.clean()
            logging.info("Mesh cleaned successfully.")

        logging.debug("Finished reading VTU file.")

        # read_comsol_fields returns:
        # exported_fields, times, sweep_keys, sweep_combos
        fields, times, keys, combos = read_comsol_fields(mesh)

        is_sweep = len(keys) > 0
        is_stationary = len(times) <= 1
        field_pattern = get_field_name_pattern(is_stationary, is_sweep)

        return cls(
            vtu_path=path,
            mesh=mesh,
            exported_fields=fields,
            times=times,
            sweep_keys=keys,
            sweep_combos=combos,
            _is_sweep=is_sweep,
            _is_stationary=is_stationary,
            field_pattern=field_pattern,
        )

    @classmethod
    def from_mesh(cls, mesh: pv.DataSet) -> Self:
        fields, times, keys, combos = read_comsol_fields(mesh)

        is_sweep = len(keys) > 0
        is_stationary = len(times) <= 1
        field_pattern = get_field_name_pattern(is_stationary, is_sweep)

        return cls(
            mesh=mesh,
            exported_fields=fields,
            times=times,
            sweep_keys=keys,
            sweep_combos=combos,
            _is_sweep=is_sweep,
            _is_stationary=is_stationary,
            field_pattern=field_pattern,
        )

    def __repr__(self) -> str:
        return f"ComsolVtu(path='{self.vtu_path}', fields={len(self.exported_fields)})"

    def convert_to_cell_data(self, pass_point_data: bool = True):
        self.mesh = self.mesh.point_data_to_cell_data(
            pass_point_data=pass_point_data)

    def _data_store(
        self, location: DataLocation = "point"
    ) -> pv.DataSetAttributes:
        """Return the mesh data store for the requested *location*.

        When *location* is ``"cell"``, the mesh is first converted from
        point data to cell data via
        :pymeth:`pyvista.DataSet.point_data_to_cell_data` so that all
        exported fields are available in ``cell_data``.

        Args:
            location: ``"point"`` (default) or ``"cell"``.

        Returns:
            The corresponding :class:`pyvista.DataSetAttributes`.
        """
        match location:
            case "cell":
                return self.mesh.cell_data
            case "point":
                return self.mesh.point_data

    def _n_values(self, location: DataLocation = "point") -> int:
        """Return the number of entries (points or cells) for *location*."""
        return self.mesh.n_cells if location == "cell" else self.mesh.n_points

    def info(self):
        """Print detailed information about the COMSOL dataset."""
        display_name = self.name or (
            self.vtu_path.name if isinstance(
                self.vtu_path, Path) else self.vtu_path
        )
        print(f"Dataset: {display_name}")
        print(f"Path: {self.vtu_path}")
        print(
            f"Study Type: {'Stationary' if self._is_stationary else 'Time-dependent'}"
        )

        if not self._is_stationary:
            t_values = self.time_values
            print(
                f"Timesteps: {len(self.times)} (from {min(t_values):.3e} to {max(t_values):.3e})"
            )

        print(f"Mesh Bounds: {self.mesh.bounds}")
        print(f"Points: {self.mesh.n_points}, Cells: {self.mesh.n_cells}")

        print("Available Fields:")
        for idx, field in enumerate(sorted(self.exported_fields), start=1):
            print(f"  {idx:2d}: {field}")

        if self._is_sweep:
            print(f"Parametric Sweep Detected: {self.sweep_keys}")
            print(f"Total Sweep Combinations: {len(self.sweep_combos)}")

    def get_values(
        self, field_name: str, location: DataLocation = "point"
    ) -> np.ndarray:
        """Get data for a specific field name.

        Args:
            field_name: Name of the field to retrieve.
            location: ``"point"`` (default) or ``"cell"``.
        """
        return self._data_store(location)[field_name]

    def get_point_values(self, field_name: str) -> np.ndarray:
        """Get point data for a specific field name.

        .. deprecated::
            Use :meth:`get_values` instead.
        """
        warnings.warn(
            "get_point_values is deprecated, use get_values(field, location='point') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_values(field_name, location="point")

    def unify_field(
        self,
        field_name: str | ComsolKeyNames,
        location: DataLocation = "point",
    ) -> None:
        """Merge all timestep entries for *field_name* into a single field.

        Useful for quantities that do not change over time to save memory.

        Args:
            field_name: Base field name to unify.
            location: ``"point"`` (default) or ``"cell"``.
        """
        if self._is_sweep or self._is_stationary:
            raise NotImplementedError(
                "unify_field not yet supported for sweeps or stationary studies."
            )

        data = self._data_store(location)
        first_time_key = self.time_keys[0]
        pattern_field = self.field_pattern.format(field_name, first_time_key)
        data[field_name] = data[pattern_field]

        for key in self.times.keys():
            try:
                data.remove(self.field_pattern.format(field_name, key))
            except KeyError:
                pass

    def format_field(
        self, field_name: str, time: str | float | int, sweep_values: list[float | int] | None = None
    ) -> str:
        """
        Get the internal COMSOL field name for a given field, time, and sweep combination.

        Args:
            field_name (str): The base field name.
            time (Union[str, float, int]): Time string, value, or index.
            sweep_values (list, optional): Values for the parametric sweep.

        Returns:
            str: The formatted COMSOL field name.
        """
        if self._is_stationary and not self._is_sweep:  # Stationary case, non sweep
            return field_name

        time_key = determine_time_key(time, self.times)  # Non-stationary case

        if not self._is_sweep:  # Non-sweep case
            return self.field_pattern.format(field_name, time_key)

        if sweep_values is None:  # Sweep case
            raise ValueError(
                "sweep_values must be provided for parametric sweeps.")

        if len(sweep_values) != len(self.sweep_keys):
            raise ValueError(
                f"Expected {len(self.sweep_keys)} sweep values, got {len(sweep_values)}."
            )

        formatted_sweep = format_sweep_parameters(
            self.sweep_keys, np.array(sweep_values)
        )
        return self.field_pattern.format(field_name, time_key, formatted_sweep)

    def overwrite_domain_from_surface(
        self,
        surface: pv.DataSet,
        field_name_3d: str,
        field_name_2d: str = "Color",
        location: DataLocation = "point",
    ) -> None:
        """Overwrite 3D domain values with values from a 2D surface subset.

        Args:
            surface: The 2D surface mesh to read from.
            field_name_3d: Field name in the 3D mesh to overwrite.
            field_name_2d: Field name in *surface* to read from.
            location: ``"point"`` (default) or ``"cell"``.
        """
        data = self._data_store(location)
        if field_name_3d not in data:
            raise KeyError(f"Field '{field_name_3d}' not found in 3D mesh.")
        if field_name_2d not in surface.point_data:
            raise KeyError(
                f"Field '{field_name_2d}' not found in surface dataset.")

        # Vectorized lookup using point coordinates
        def structured_view(arr: np.ndarray) -> np.ndarray:
            return arr.view([("", arr.dtype)] * arr.shape[1])

        view_3d = structured_view(self.mesh.points)
        view_2d = structured_view(surface.points)

        mask_3d = np.isin(view_3d, view_2d).flatten()
        points_3d_masked = self.mesh.points[mask_3d]

        # Build mapping for faster lookup
        vals_3d = np.zeros(np.sum(mask_3d))

        # Simple coordinate-based mapping
        for i, pt in enumerate(points_3d_masked):
            idx_2d = np.where(np.all(surface.points == pt, axis=1))[0]
            if len(idx_2d) > 0:
                vals_3d[i] = surface.point_data[field_name_2d][idx_2d[0]]

        data[field_name_3d][mask_3d] = vals_3d

    def get_array(
        self,
        field: str | ComsolKeyNames,
        location: DataLocation = "point",
    ) -> np.ndarray:
        """Assemble field data into a NumPy array.

        Args:
            field: Field name to extract.
            location: ``"point"`` (default) or ``"cell"``.

        Returns:
            np.ndarray:
            - Stationary study: ``(N_VALUES,)``
            - Transient study: ``(N_TIME_STEPS, N_VALUES)``
            - Transient + Sweep: ``(N_TIME_STEPS, N_SWEEP_COMBOS, N_VALUES)``
        """
        if field not in self.exported_fields:
            raise KeyError(f"Field '{field}' not found in exported fields.")

        data = self._data_store(location)
        n_values = self._n_values(location)

        if self._is_sweep and not self._is_stationary:
            shape = (len(self.times), len(self.sweep_combos), n_values)
            matrix = np.zeros(shape)

            for i, time_key in enumerate(self.times.keys()):
                for j, combo in enumerate(self.sweep_combos):
                    field_key = self.format_field(field, time_key, list(combo))
                    matrix[i, j] = data[field_key]

            return matrix

        if not self._is_sweep and not self._is_stationary:
            return np.array(
                [
                    data[self.format_field(field, time_key)]
                    for time_key in self.times.keys()
                ]
            )

        if not self._is_sweep and self._is_stationary:
            return data[field]

        raise NotImplementedError(
            "get_array currently only supports transient studies "
            "(with or without sweeps)."
        )

    def merge_datasets(self, *others: "ComsolVtu") -> None:
        """Merge other ComsolVtu datasets into this one."""
        if self._is_sweep or self._is_stationary:
            raise NotImplementedError(
                "merge_datasets not yet supported for sweeps or stationary studies."
            )

        for other in others:
            if not isinstance(other, ComsolVtu):
                raise TypeError(f"Expected ComsolVtu, got {type(other)}")

            # Basic compatibility checks
            if self.mesh.points.shape != other.mesh.points.shape:
                raise ValueError(
                    "Meshes have different point counts or coordinates.")

            self.times.update(other.times)
            self.mesh.point_data.update(other.mesh.point_data)
            # Re-sort times
            self.times = dict(sorted(self.times.items(), key=lambda x: x[1]))

    def delete_field(
        self,
        field_name: str | ComsolKeyNames,
        location: DataLocation = "point",
    ) -> None:
        """Delete a field from the mesh and tracking lists.

        Args:
            field_name: Base field name to delete.
            location: ``"point"`` (default) or ``"cell"``.
        """
        if field_name not in self.exported_fields:
            raise KeyError(f"Field '{field_name}' not found.")

        if self._is_sweep or self._is_stationary:
            raise NotImplementedError(
                "delete_field not yet supported for sweeps or stationary studies."
            )

        data = self._data_store(location)
        for time_key in self.times.keys():
            internal_name = self.format_field(field_name, time_key)
            if internal_name in data:
                data.remove(internal_name)

        self.exported_fields.remove(field_name)

    def update_mesh(self, new_mesh: pv.DataSet) -> Self:
        """Return a copy of the class instance with a new mesh.

        Args:
            new_mesh (pv.DataSet):

        Returns:
            Self: ComsolVtu
        """
        return replace(self, mesh=new_mesh)

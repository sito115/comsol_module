import logging
from pathlib import Path
from typing import cast

import numpy as np
import pyvista as pv
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass

from .helper import (
    ComsolKeyNames,
    ensure_pathlib_path,
    format_sweep_parameters,
    get_field_name_pattern,
    read_comsol_fields,
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ComsolVtu:
    """Class to read and process exported simulation files from COMSOL.

    Attributes:
        vtu_path (Path): Path to the VTU file.
        name (str): Optional name for the dataset.
        is_clean_mesh (bool): Whether to clean the mesh upon loading.
        mesh (pv.DataSet): The PyVista dataset object.
        times (dict[str, float]): Sorted dictionary of exported times.
        exported_fields (list[str]): All exported field names.
        sweep_keys (list[str]): Names of parametric sweep variables.
        sweep_combos (np.ndarray): Unique combinations of sweep parameters.
    """

    vtu_path: Path | str
    name: str | None = ""
    is_clean_mesh: bool | None = False

    @field_validator("vtu_path", mode="before")
    @classmethod
    def check_path_exists(cls, vtu_path: Path | str) -> Path:
        """Validate that the given path exists."""
        resolved_path = ensure_pathlib_path(vtu_path)
        if isinstance(resolved_path, list):
            # The current implementation of ensure_pathlib_path can return a list
            # but we expect a single path here.
            if len(resolved_path) > 0:
                resolved_path = resolved_path[0]
            else:
                raise ValueError("vtu_path resolved to an empty list.")

        if not resolved_path.exists():
            raise ValueError(f"Given path does not exist: {resolved_path}.")
        return resolved_path

    def __post_init__(self):
        logging.debug("Reading vtu file...")
        self.mesh: pv.DataSet = cast(pv.DataSet, pv.wrap(pv.read(self.vtu_path)))
        if self.is_clean_mesh:
            self.mesh = self.mesh.clean()
            logging.info("Mesh cleaned successfully.")

        logging.debug("Finished reading vtu file.")
        # read_comsol_fields returns 4 values: exported_fields, times, sweep_keys, sweep_combos
        fields, times, keys, combos = read_comsol_fields(self.mesh)
        self.exported_fields: list[str] = fields
        self.times: dict[str, float] = times
        self.sweep_keys: list[str] = keys
        self.sweep_combos: np.ndarray = combos

        self._is_sweep: bool = len(self.sweep_keys) > 0
        self._is_stationary: bool = len(self.times) <= 1
        self.field_pattern: str = get_field_name_pattern(
            self._is_stationary, self._is_sweep
        )

    def __repr__(self) -> str:
        return f"ComsolVtu(path='{self.vtu_path}', fields={len(self.exported_fields)})"

    def info(self):
        """Print detailed information about the COMSOL dataset."""
        display_name = self.name or (
            self.vtu_path.name if isinstance(self.vtu_path, Path) else self.vtu_path
        )
        print(f"Dataset: {display_name}")
        print(f"Path: {self.vtu_path}")
        print(
            f"Study Type: {'Stationary' if self._is_stationary else 'Time-dependent'}"
        )

        if not self._is_stationary:
            t_values = list(self.times.values())
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

    def get_point_values(self, field_name: str) -> np.ndarray:
        """Get point data for a specific field name."""
        return self.mesh.point_data[field_name]

    def unify_field(self, field_name: str | ComsolKeyNames) -> None:
        """
        Merge all entries from COMSOL into one field.
        Useful for quantities that do not change over time to save memory.
        """
        if self._is_sweep or self._is_stationary:
            raise NotImplementedError(
                "unify_field not yet supported for sweeps or stationary studies."
            )

        first_time_key = list(self.times.keys())[0]
        pattern_field = self.field_pattern.format(field_name, first_time_key)
        self.mesh.point_data[field_name] = self.mesh.point_data[pattern_field]

        for key in self.times.keys():
            try:
                self.mesh.point_data.remove(self.field_pattern.format(field_name, key))
            except KeyError:
                pass

    def format_field(
        self, field_name: str, time: str | float | int, sweep_values: list | None = None
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
        if isinstance(time, str):
            if time not in self.times:
                raise ValueError(f"Time '{time}' not found in dataset.")
            time_key = time
        elif isinstance(time, (float, np.floating)):
            time_key = min(self.times, key=lambda k: abs(self.times[k] - float(time)))
        elif isinstance(time, (int, np.integer)):
            if time >= len(self.times):
                raise IndexError(
                    f"Time index {time} out of bounds for {len(self.times)} steps."
                )
            time_key = list(self.times.keys())[time]
        else:
            raise TypeError(f"Unsupported time type: {type(time)}")

        if not self._is_sweep:
            return self.field_pattern.format(field_name, time_key)

        if sweep_values is None:
            raise ValueError("sweep_values must be provided for parametric sweeps.")

        if len(sweep_values) != len(self.sweep_keys):
            raise ValueError(
                f"Expected {len(self.sweep_keys)} sweep values, got {len(sweep_values)}."
            )

        formatted_sweep = format_sweep_parameters(
            self.sweep_keys, np.array(sweep_values)
        )
        return self.field_pattern.format(field_name, time_key, formatted_sweep)

    def overwrite_domain_from_surface(
        self, surface: pv.DataSet, field_name_3d: str, field_name_2d: str = "Color"
    ) -> None:
        """
        Overwrite 3D domain point values with values from a 2D surface subset.
        """
        if field_name_3d not in self.mesh.point_data:
            raise KeyError(f"Field '{field_name_3d}' not found in 3D mesh.")
        if field_name_2d not in surface.point_data:
            raise KeyError(f"Field '{field_name_2d}' not found in surface dataset.")

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

        self.mesh.point_data[field_name_3d][mask_3d] = vals_3d

    def get_array(
        self,
        field: str | ComsolKeyNames,
        is_cell_data: bool = False,
    ) -> np.ndarray:
        """
        Assemble field data into a NumPy array.

        Args:
            field: Field name to extract.
            is_cell_data: If True, read from ``mesh.cell_data``.
                Otherwise read from ``mesh.point_data``.

        Returns:
            np.ndarray:
                - Transient study: (N_TIME_STEPS, N_VALUES)
                - Transient + Sweep: (N_TIME_STEPS, N_SWEEP_COMBOS, N_VALUES)
        """
        if field not in self.exported_fields:
            raise KeyError(f"Field '{field}' not found in exported fields.")

        data = self.mesh.cell_data if is_cell_data else self.mesh.point_data
        n_values = self.mesh.n_cells if is_cell_data else self.mesh.n_points

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
                raise ValueError("Meshes have different point counts or coordinates.")

            self.times.update(other.times)
            self.mesh.point_data.update(other.mesh.point_data)
            # Re-sort times
            self.times = dict(sorted(self.times.items(), key=lambda x: x[1]))

    def delete_field(self, field_name: str | ComsolKeyNames) -> None:
        """Delete a field from the mesh and tracking lists."""
        if field_name not in self.exported_fields:
            raise KeyError(f"Field '{field_name}' not found.")

        if self._is_sweep or self._is_stationary:
            raise NotImplementedError(
                "delete_field not yet supported for sweeps or stationary studies."
            )

        for time_key in self.times.keys():
            internal_name = self.format_field(field_name, time_key)
            if internal_name in self.mesh.point_data:
                self.mesh.point_data.remove(internal_name)

        self.exported_fields.remove(field_name)

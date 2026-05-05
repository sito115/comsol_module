"""Classes for reading and processing COMSOL Multiphysics VTU exports.

This module provides two main dataclasses:

* :class:`ComsolMetaData` — Lightweight container for field names, time
  steps, and parametric-sweep metadata parsed from a COMSOL VTU file.
* :class:`ComsolVtu` — High-level wrapper around a PyVista mesh that exposes
  convenient accessors for COMSOL simulation data (stationary, transient,
  and parametric-sweep studies).

Typical usage::

    from comsol_module.comsol_classes import ComsolVtu

    # Load a simulation file
    vtu = ComsolVtu.from_file("simulation.vtu")
    
    # Print summary information
    vtu.info()
    
    # Extract data as a NumPy array
    temperature = vtu.get_array("Temperature")
"""

import logging
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import pyvista as pv

from .helper import (
    ComsolKeyNames,
    determine_time_key,
    format_sweep_parameters,
    get_field_name_pattern,
    read_comsol_fields,
)

#: Selector for point-based or cell-based data access on a PyVista mesh.
DataLocation = Literal["point", "cell"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class ComsolMetaData:
    """Metadata extracted from a COMSOL VTU export.

    Stores the base field names, time-step mapping, parametric-sweep
    parameters, and derived flags that describe the study type.

    Attributes:
        exported_fields: Base field names found in the export
            (e.g., ``["Temperature", "Pressure"]``).
        times: Mapping of time-key strings to their float values,
            sorted in ascending order.
        sweep_keys: Parameter names of a parametric sweep
            (empty list when no sweep is present).
        sweep_combos: Array of unique sweep-parameter combinations,
            shape ``(N_COMBOS, N_PARAMS)``.
        is_sweep: ``True`` when the dataset contains a parametric sweep.
        is_stationary: ``True`` when the dataset has at most one time step.
        field_pattern: :meth:`str.format`-compatible pattern used to
            reconstruct internal field names from base name, time key,
            and (optionally) sweep segment.

    """

    exported_fields: list[str] = field(default_factory=list)
    times: dict[str, float] = field(default_factory=dict)
    sweep_keys: list[str] = field(default_factory=list)
    sweep_combos: np.ndarray = field(default_factory=lambda: np.array([]))

    is_sweep: bool = False
    is_stationary: bool = False
    field_pattern: str = ""

    # -- convenience properties --------------------------------------------

    @property
    def time_keys(self) -> list[str]:
        """Time-step keys in export order."""
        return list(self.times.keys())

    @property
    def time_values(self) -> list[float]:
        """Time-step values (floats) in export order."""
        return list(self.times.values())

    # -- factory -----------------------------------------------------------

    @classmethod
    def from_mesh(cls, mesh: pv.DataSet) -> Self:
        """Build metadata by parsing the field names of *mesh*.

        This delegates to :func:`~comsol_module.helper.read_comsol_fields`
        and derives ``is_sweep``, ``is_stationary``, and ``field_pattern``
        automatically.

        Args:
            mesh: The PyVista dataset to parse.

        Returns:
            A new :class:`ComsolMetaData` instance.

        """
        fields, times, keys, combos = read_comsol_fields(mesh)

        is_sweep = len(keys) > 0
        is_stationary = len(times) <= 1
        field_pattern = get_field_name_pattern(is_stationary, is_sweep)

        return cls(
            exported_fields=fields,
            times=times,
            sweep_keys=keys,
            sweep_combos=combos,
            is_sweep=is_sweep,
            is_stationary=is_stationary,
            field_pattern=field_pattern,
        )

    # -- guards ------------------------------------------------------------

    def _require_transient_non_sweep(self, operation: str) -> None:
        """Check if the study type supports a transient-only operation.

        Args:
            operation: Name of the operation being attempted.

        Raises:
            NotImplementedError: If the study is a sweep or stationary.

        """
        if self.is_sweep or self.is_stationary:
            raise NotImplementedError(
                f"{operation} is not yet supported for sweeps or stationary studies."
            )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@dataclass
class ComsolVtu:
    """High-level wrapper for COMSOL VTU simulation exports.

    Construct instances via the :meth:`from_file` or :meth:`from_mesh`
    class methods rather than calling the dataclass constructor directly.

    Attributes:
        mesh: The underlying PyVista mesh holding point/cell data.
        vtu_path: Path to the source ``.vtu`` file (``None`` when created
            from an in-memory mesh).
        name: Optional human-readable label (used by :meth:`info`).
        metadata: Parsed :class:`ComsolMetaData` for this dataset.

    Example:
        >>> from comsol_module import ComsolVtu
        >>> vtu = ComsolVtu.from_file("simulation.vtu")
        >>> vtu.info()
        Dataset: simulation.vtu
        Path: simulation.vtu
        Study Type: Time-dependent
        Timesteps: 40 (from 0.000e+00 to 3.000e+13)
        Mesh Bounds: (-7.415000000000242e-06, 249999.99999999978, -1.9414062500000573e-06, 250000.00000000036, -7500.000000000005, 84.3298467930108)
        Points: 81782, Cells: 301702
        Available Fields:
        1: Pressure
        2: Temperature
        >>> T = vtu.get_array("Temperature")

    """

    mesh: pv.DataSet
    vtu_path: Path | None = None
    name: str = ""

    metadata: ComsolMetaData = field(default_factory=ComsolMetaData)

    # -- delegating properties ---------------------------------------------

    @property
    def exported_fields(self) -> list[str]:
        """Base field names available in the export."""
        return self.metadata.exported_fields

    @property
    def sweep_keys(self) -> list[str]:
        """Parametric-sweep parameter names (empty if none)."""
        return self.metadata.sweep_keys

    @property
    def sweep_combos(self) -> np.ndarray:
        """Unique sweep-parameter combinations."""
        return self.metadata.sweep_combos

    @property
    def time_keys(self) -> list[str]:
        """Time-step keys in export order."""
        return self.metadata.time_keys

    @property
    def time_values(self) -> list[float]:
        """Time-step values (floats) in export order."""
        return self.metadata.time_values

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, is_clean_mesh: bool = False) -> Self:
        """Read a ``.vtu`` file and build a :class:`ComsolVtu`.

        Args:
            path: Path to the COMSOL VTU export file.
            is_clean_mesh: If ``True``, call :meth:`pyvista.DataSet.clean`
                on the mesh after reading.

        Returns:
            A new instance of :class:`ComsolVtu`.

        """
        path = Path(path)

        logging.debug("Reading VTU file...")
        mesh: pv.DataSet = cast(pv.DataSet, pv.wrap(pv.read(path)))

        if is_clean_mesh:
            mesh = mesh.clean()
            logging.info("Mesh cleaned successfully.")

        logging.debug("Finished reading VTU file.")

        return cls(
            vtu_path=path,
            mesh=mesh,
            metadata=ComsolMetaData.from_mesh(mesh),
        )

    @classmethod
    def from_mesh(cls, mesh: pv.DataSet) -> Self:
        """Create a :class:`ComsolVtu` from an existing PyVista mesh.

        Args:
            mesh: The PyVista dataset to wrap.

        Returns:
            A new instance of :class:`ComsolVtu`.

        """
        return cls(
            mesh=mesh,
            metadata=ComsolMetaData.from_mesh(mesh),
        )

    # -- dunder methods ----------------------------------------------------

    def __repr__(self) -> str:
        return f"ComsolVtu(path='{self.vtu_path}', fields={len(self.exported_fields)})"

    # -- data access helpers -----------------------------------------------

    def convert_to_cell_data(self, pass_point_data: bool = True) -> None:
        """Convert point data to cell data in-place.

        Args:
            pass_point_data: If ``True``, copy point data arrays into the
                converted cell-data mesh.

        """
        self.mesh = self.mesh.point_data_to_cell_data(
            pass_point_data=pass_point_data)

    def _data_store(self, location: DataLocation = "point") -> pv.DataSetAttributes:
        """Return the mesh data store for the requested *location*.

        Args:
            location: The data location to access. Either ``"point"`` (default)
                or ``"cell"``.

        Returns:
            The corresponding :class:`pyvista.DataSetAttributes`.

        """
        match location:
            case "cell":
                return self.mesh.cell_data
            case "point":
                return self.mesh.point_data

    def _n_values(self, location: DataLocation = "point") -> int:
        """Return the number of entries (points or cells) for *location*.

        Args:
            location: Either ``"point"`` or ``"cell"``.

        Returns:
            The number of points or cells in the mesh.

        """
        return self.mesh.n_cells if location == "cell" else self.mesh.n_points

    # -- inspection --------------------------------------------------------

    def info(self) -> None:
        """Print a human-readable summary of the dataset."""
        display_name = self.name or (
            self.vtu_path.name if isinstance(
                self.vtu_path, Path) else self.vtu_path
        )
        meta = self.metadata

        print(f"Dataset: {display_name}")
        print(f"Path: {self.vtu_path}")
        print(
            f"Study Type: {'Stationary' if meta.is_stationary else 'Time-dependent'}"
        )

        if not meta.is_stationary:
            t_vals = self.time_values
            print(
                f"Timesteps: {len(meta.times)} "
                f"(from {min(t_vals):.3e} to {max(t_vals):.3e})"
            )

        print(f"Mesh Bounds: {self.mesh.bounds}")
        print(f"Points: {self.mesh.n_points}, Cells: {self.mesh.n_cells}")

        print("Available Fields:")
        for idx, f in enumerate(sorted(self.exported_fields), start=1):
            print(f"  {idx:2d}: {f}")

        if meta.is_sweep:
            print(f"Parametric Sweep Detected: {self.sweep_keys}")
            print(f"Total Sweep Combinations: {len(self.sweep_combos)}")

    # -- value retrieval ---------------------------------------------------

    def get_values(
        self, field_name: str, location: DataLocation = "point"
    ) -> np.ndarray:
        """Get the raw data array for a single internal field key.

        Args:
            field_name: Full internal field key (use :meth:`format_field` to
                build it from base name + time + sweep values).
            location: The data location (``"point"`` or ``"cell"``).

        Returns:
            NumPy array containing the field data.

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

    # -- field manipulation ------------------------------------------------

    def unify_field(
        self,
        field_name: str | ComsolKeyNames,
        location: DataLocation = "point",
    ) -> None:
        """Merge all timestep entries for *field_name* into a single field.

        Useful for quantities that do not change over time (e.g. geometry
        indicators) to reduce memory usage.

        Args:
            field_name: Base field name to unify.
            location: ``"point"`` (default) or ``"cell"``.

        """
        self.metadata._require_transient_non_sweep("unify_field")

        data = self._data_store(location)
        first_time_key = self.time_keys[0]
        pattern_field = self.metadata.field_pattern.format(
            field_name, first_time_key)
        data[field_name] = data[pattern_field]

        for key in self.metadata.times:
            try:
                data.remove(
                    self.metadata.field_pattern.format(field_name, key))
            except KeyError:
                pass

    def format_field(
        self,
        field_name: str,
        time: str | float | int,
        sweep_values: list[float | int] | None = None,
    ) -> str:
        """Build the internal COMSOL field key for a given query.

        Args:
            field_name: The base field name (e.g., ``"Temperature"``).
            time: Time step as a string key, float value, or integer index.
            sweep_values: Values for each sweep parameter. Required when the
                dataset contains a parametric sweep.

        Returns:
            The fully-qualified internal field name stored in the mesh.

        """
        meta = self.metadata

        if meta.is_stationary and not meta.is_sweep:
            return field_name

        time_key = determine_time_key(time, meta.times)

        if not meta.is_sweep:
            return meta.field_pattern.format(field_name, time_key)

        # --- parametric sweep ---
        if sweep_values is None:
            raise ValueError(
                "sweep_values must be provided for parametric sweeps.")

        if len(sweep_values) != len(self.sweep_keys):
            raise ValueError(
                f"Expected {len(self.sweep_keys)} sweep values, got {len(sweep_values)}."
            )

        formatted_sweep = format_sweep_parameters(
            self.sweep_keys, np.array(sweep_values)
        )
        return meta.field_pattern.format(field_name, time_key, formatted_sweep)

    def delete_field(
        self,
        field_name: str | ComsolKeyNames,
        location: DataLocation = "point",
    ) -> None:
        """Remove all timestep entries and the tracking record for a field.

        Args:
            field_name: Base field name to delete.
            location: ``"point"`` (default) or ``"cell"``.

        """
        if field_name not in self.exported_fields:
            raise KeyError(f"Field '{field_name}' not found.")

        self.metadata._require_transient_non_sweep("delete_field")

        data = self._data_store(location)
        for time_key in self.metadata.times:
            internal_name = self.format_field(field_name, time_key)
            if internal_name in data:
                data.remove(internal_name)

        self.metadata.exported_fields.remove(field_name)

    # -- array assembly ----------------------------------------------------

    def get_array(
        self,
        field: str | ComsolKeyNames,
        location: DataLocation = "point",
    ) -> np.ndarray:
        """Assemble field data across all timesteps/sweeps into a NumPy array.

        Args:
            field: Base field name to extract.
            location: ``"point"`` (default) or ``"cell"``.

        Returns:
            np.ndarray with shape depending on study type:

            * Stationary: ``(N_VALUES,)``
            * Transient: ``(N_TIMESTEPS, N_VALUES)``
            * Transient + Sweep: ``(N_TIMESTEPS, N_SWEEP_COMBOS, N_VALUES)``

        """
        if field not in self.exported_fields:
            raise KeyError(f"Field '{field}' not found in exported fields.")

        meta = self.metadata
        data = self._data_store(location)
        n_values = self._n_values(location)

        # Stationary (no sweep)
        if meta.is_stationary and not meta.is_sweep:
            return data[field]

        # Transient without sweep
        if not meta.is_sweep and not meta.is_stationary:
            return np.array(
                [
                    data[self.format_field(field, time_key)]
                    for time_key in meta.times
                ]
            )

        # Transient with sweep
        if meta.is_sweep and not meta.is_stationary:
            shape = (len(meta.times), len(self.sweep_combos), n_values)
            matrix = np.zeros(shape)

            for i, time_key in enumerate(meta.times):
                for j, combo in enumerate(self.sweep_combos):
                    field_key = self.format_field(field, time_key, list(combo))
                    matrix[i, j] = data[field_key]

            return matrix

        raise NotImplementedError(
            "get_array currently only supports stationary and transient studies "
            "(with or without sweeps)."
        )

    # -- dataset operations ------------------------------------------------

    def overwrite_domain_from_surface(
        self,
        surface: pv.DataSet,
        field_name_3d: str,
        field_name_2d: str = "Color",
        location: DataLocation = "point",
    ) -> None:
        """Overwrite 3D domain values with values from a 2D surface subset.

        For each point in *surface* that also exists in the 3D mesh, the
        value of *field_name_2d* on the surface is written to
        *field_name_3d* on the 3D mesh.

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

        # Vectorized lookup using structured coordinate views
        def structured_view(arr: np.ndarray) -> np.ndarray:
            return arr.view([("", arr.dtype)] * arr.shape[1])

        view_3d = structured_view(self.mesh.points)
        view_2d = structured_view(surface.points)

        mask_3d = np.isin(view_3d, view_2d).flatten()
        points_3d_masked = self.mesh.points[mask_3d]

        vals_3d = np.zeros(np.sum(mask_3d))

        for i, pt in enumerate(points_3d_masked):
            idx_2d = np.where(np.all(surface.points == pt, axis=1))[0]
            if len(idx_2d) > 0:
                vals_3d[i] = surface.point_data[field_name_2d][idx_2d[0]]

        data[field_name_3d][mask_3d] = vals_3d

    def merge_datasets(self, *others: "ComsolVtu") -> None:
        """Merge other :class:`ComsolVtu` instances into this one.

        Combines time steps and point data from *others* into the current
        dataset. Only supported for transient, non-sweep studies.

        Args:
            others: One or more :class:`ComsolVtu` objects to merge.

        Raises:
            NotImplementedError: If the current study is a sweep or stationary.
            TypeError: If any element of *others* is not a :class:`ComsolVtu`.
            ValueError: If mesh geometries are incompatible.

        """
        self.metadata._require_transient_non_sweep("merge_datasets")

        for other in others:
            if not isinstance(other, ComsolVtu):
                raise TypeError(f"Expected ComsolVtu, got {type(other)}")

            if self.mesh.points.shape != other.mesh.points.shape:
                raise ValueError(
                    "Meshes have different point counts or coordinates.")

            self.metadata.times.update(other.metadata.times)
            self.mesh.point_data.update(other.mesh.point_data)

        # Re-sort times after all merges
        self.metadata.times = dict(
            sorted(self.metadata.times.items(), key=lambda x: x[1])
        )

    def update_mesh(self, new_mesh: pv.DataSet) -> Self:
        """Return a shallow copy of this instance with a different mesh.

        The metadata and other attributes are shared with the original;
        only the mesh reference is replaced.

        Args:
            new_mesh: The replacement PyVista mesh.

        Returns:
            A new :class:`ComsolVtu` with the updated mesh.

        """
        return replace(self, mesh=new_mesh)

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pyvista as pv

from .helper import get_field_name_pattern, read_comsol_fields


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

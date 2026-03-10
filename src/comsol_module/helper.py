from enum import StrEnum
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pyvista as pv


def ensure_pathlib_path(path: str | Path | list) -> list[Path] | Path:
    """Ensure that the input is a Path object or a list of Path objects."""
    if isinstance(path, list):
        return [Path(v) if not isinstance(v, Path) else v for v in path]
    return Path(path) if isinstance(path, str) else path


def read_comsol_fields(
    mesh: pv.DataSet,
) -> Tuple[list[str], dict[str, float], list[str], np.ndarray]:
    """
    Parse COMSOL field names from mesh point data.
    Field names typically follow patterns like:
    - Time-dependent: "FIELDNAME_@_t=TIME"
    - Time + Sweep: "FIELDNAME_@_t=TIME,_PARAM1=VAL1,_PARAM2=VAL2"

    Returns:
        tuple containing:
            - list[str]: Exported base field names (e.g., ['Temperature', 'Pressure'])
            - dict[str, float]: Mapping of time strings to float values.
            - list[str]: List of sweep parameter keys found.
            - np.ndarray: Unique combinations of sweep parameter values.
    """
    times_raw = []
    all_vars_dicts = []
    base_fields = set()

    for key in mesh.point_data.keys():
        if "_@_" not in key:
            continue

        name, vars_str = key.split("_@_", 1)
        base_fields.add(name)

        vars_list = vars_str.split(",")
        vars_dict = {}
        for var in vars_list:
            var = var.strip()
            if "=" in var:
                key_val, val_str = var.split("=", 1)
                key_val = key_val.strip().lstrip("_")
                val_str = val_str.strip()

                if key_val == "t":
                    times_raw.append(val_str)
                else:
                    try:
                        vars_dict[key_val] = float(val_str)
                    except ValueError:
                        vars_dict[key_val] = val_str

        all_vars_dicts.append(vars_dict)

    # Process times: unique, sorted by float value
    unique_times = sorted(set(times_raw), key=float)
    times_map = {t: float(t) for t in unique_times}

    # Process sweep parameters
    sweep_keys = []
    sweep_combos = np.array([])

    if all_vars_dicts:
        # Assume all fields have the same sweep keys if any exist
        # Filter out empty dicts (if some fields aren't part of sweep)
        non_empty_vars = [d for d in all_vars_dicts if d]
        if non_empty_vars:
            sweep_keys = list(non_empty_vars[0].keys())
            tuples = [tuple(d.get(k) for k in sweep_keys) for d in non_empty_vars]
            sweep_combos = np.unique(tuples, axis=0)

    return list(base_fields), times_map, sweep_keys, sweep_combos


def get_field_name_pattern(is_stationary: bool, is_sweep: bool) -> str:
    """Return the naming pattern used by COMSOL exports."""
    if is_stationary:
        if is_sweep:
            # TODO: Verify stationary sweep pattern in COMSOL VTU export
            return "{}_@_{}"
        return "{}"  # Simple stationary field might not have _@_

    if is_sweep:
        return "{}_@_t={},{}"  # Name, Time, FormattedSweep
    return "{}_@_t={}"  # Name, Time


def format_value(
    x: Any, sig: int = 4, sci_threshold: Tuple[float, float] = (1e-4, 1e6)
) -> str:
    """
    Format a value with significant digits and optional scientific notation.
    """
    try:
        val = float(x)
    except (ValueError, TypeError):
        return str(x)

    abs_x = abs(val)
    if abs_x != 0 and (abs_x < sci_threshold[0] or abs_x >= sci_threshold[1]):
        return f"{val:.{sig}e}"
    return f"{val:.{sig}g}"


def format_sweep_parameters(sweep_keys: List[str], values: np.ndarray) -> str:
    """Format sweep keys and values into the COMSOL string segment: _k1=v1,_k2=v2..."""
    return ",".join(f"_{k}={format_value(v)}" for k, v in zip(sweep_keys, values))


class ComsolKeyNames(StrEnum):
    """Standard COMSOL field names for convenience."""

    T = "Temperature"
    T_GRAD_X = "Temperature_gradient,_x-component"
    T_GRAD_Y = "Temperature_gradient,_y-component"
    T_GRAD_Z = "Temperature_gradient,_z-component"
    T_GRAD_MAG = "Temperature_gradient_magnitude"
    DARCY_X = "Total_Darcy_velocity_field,_x-component"
    DARCY_Y = "Total_Darcy_velocity_field,_y-component"
    DARCY_Z = "Total_Darcy_velocity_field,_z-component"
    DARCY_MAG = "Total_Darcy_velocity_magnitude"
    P = "Pressure"

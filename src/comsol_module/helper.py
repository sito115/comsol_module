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
    mesh: pv.DataSet, is_cell_data: bool
) -> Tuple[list[str], dict[str, float], list[str], np.ndarray]:
    """
    Parse COMSOL field names from mesh point or cell data.

    Field naming patterns:
        FIELDNAME
        FIELDNAME_@_t=TIME
        FIELDNAME_@_t=TIME,_PARAM1=VAL1,_PARAM2=VAL2

    Returns
    -------
    base_fields : list[str]
        Exported base field names (e.g. ['Temperature', 'Pressure'])

    times_map : dict[str, float]
        Mapping of time strings to float values

    sweep_keys : list[str]
        Sweep parameter names

    sweep_combos : np.ndarray
        Unique combinations of sweep parameter values
    """
    invalid_field_names = ["Data"]
    keys = mesh.cell_data.keys() if is_cell_data else mesh.point_data.keys()

    base_fields: set[str] = set()
    times_raw: list[str] = []
    sweep_dicts: list[dict] = []

    for key in keys:
        # Split base field name and variable string
        if "_@_" in key:
            name, vars_str = key.split("_@_", 1)
        elif "," in key:
            name, vars_str = key.split(",", 1)
        else:
            name, vars_str = key, ""

        if name in invalid_field_names:
            raise ValueError(
                f"{name} is not a valid field name and indicates en empty field name description in Comsol. Please add a field name."
            )
        base_fields.add(name)

        vars_dict = {}
        for var in vars_str.split(","):
            if "=" not in var:
                continue

            k, v = var.split("=", 1)
            k = k.strip().lstrip("_")
            v = v.strip()

            if k == "t":
                times_raw.append(v)
            else:
                try:
                    vars_dict[k] = float(v)
                except ValueError:
                    vars_dict[k] = v

        if vars_dict:
            sweep_dicts.append(vars_dict)

    # --- Times ---
    unique_times = sorted(set(times_raw), key=float)
    times_map = {t: float(t) for t in unique_times}

    # --- Sweep parameters ---
    sweep_keys: list[str] = []
    sweep_combos = np.array([])

    if sweep_dicts:
        sweep_keys = list(sweep_dicts[0].keys())
        tuples = [tuple(d.get(k) for k in sweep_keys) for d in sweep_dicts]
        sweep_combos = np.unique(np.array(tuples, dtype=float), axis=0)

    return list(base_fields), times_map, sweep_keys, sweep_combos


def get_field_name_pattern(is_stationary: bool, is_sweep: bool) -> str:
    """Return the naming pattern used by COMSOL exports."""
    if is_stationary:
        if is_sweep:
            # TODO: Verify stationary sweep pattern in COMSOL VTU export
            return "{}_@_{}"
        return "{}"  # Simple stationary field might not have _@_

    if is_sweep:
        return "{}_@_t={},{}"  # transient sweep: Name, Time, FormattedSweep
    return "{}_@_t={}"  # transient:  Name, Time


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

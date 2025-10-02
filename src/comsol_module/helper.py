import pyvista as pv
from typing import Union, List
import numpy as np
from pathlib import Path   
from enum import StrEnum 

def ensure_pathlib_path(path: Union[str,Path, List]) -> Union[List[Path], Path]:
    if isinstance(path, List):
        return [Path(v) if not isinstance(v, Path) else v for v in path]
    else:
        return Path(path) if isinstance(path, str) else path


def read_comsol_fields(mesh:pv.DataSet) -> tuple[list[str], dict[str, float], list[dict]]:
    """Field names in COMSOL are FIELDNAME_@_tTIME.
        - For time dependent studies, e.g. "Temperature_@_t=0"
        - For time dependent sweeps, e.g.  "Temperature_@_t=0,_SH_grad=15,_Sh_grad=15,_host_rho=1500,_host_E=5"
        - For stationary sweeps, ??? TODO

    Args:
        mesh (pv.DataSet):
                               
    Returns:
        tuple[pv.DataSet,list[str], dict[str, float]]: Tuple with three elements:
            - list[str] containing exported fields (e.g. Temperature, Pressure, etc.)
            -  dict[str, float] containing increasing time steps in original string format and respective numerical values
            -  list[dict] containing additional information of parameters (for sweeps) ordered.
    """    

    times = []
    add_vars_dict = []
    exported_fields = []
    for key in mesh.point_data.keys():
        name, vars_str = key.split("_@_") 

        vars_list = vars_str.split(",")
        vars_dict = {}
        for var in vars_list:
            if "=" in var:
                key_val, val = var.split("=")
                if key_val != "t":
                    vars_dict[key_val.strip().lstrip("_")] = float(val.strip())       
                else:
                    times.append(val.strip()) # store time as a str
        
        add_vars_dict.append(vars_dict)
        exported_fields.append(name)
    
    exported_fields : list[str] = list(set(exported_fields)) 

    sweep_keys = list(add_vars_dict[0].keys())
    if len(sweep_keys) == 0:
        sweep_combos = sweep_keys = []
    else:
        tuples = [tuple(d[k] for k in sweep_keys) for d in add_vars_dict]
        sweep_combos = np.unique(tuples, axis=0)

    times : dict[str:float]= {str(val): float(val) for val in sorted(np.unique(times), key=float)}  # Sort by float value
    return (exported_fields, times, sweep_keys, sweep_combos)  


def get_field_name_pattern(is_stationary: bool, is_sweep: bool) -> str:
    if is_stationary and is_sweep:
        raise NotImplementedError('Stationary studies not implemented yed')
    elif not is_stationary and not is_sweep:
        field_name_pattern = '{}_@_t={}'  # Temperature_@_t=0
    elif not is_stationary and is_sweep:
        field_name_pattern = '{}_@_t={},{}' # 'Temperature_@_t=0,_SH_grad=15,_Sh_grad=15,_host_rho=1500,_host_E=5',
    elif is_stationary and not is_sweep:
        raise NotImplementedError('Stationary studies not implemented yed')
    return field_name_pattern


def format_value(x, sig=4, sci_threshold=(1e-4, 1e6)):
    """
    Format x with sig significant digits.
    Use scientific notation if outside sci_threshold.
    """
    abs_x = abs(x)
    if abs_x != 0 and (abs_x < sci_threshold[0] or abs_x >= sci_threshold[1]):
        return f"{x:.{sig}e}"  # scientific
    else:
        return f"{x:.{sig}g}"  # general/fixed
    

def format_sweep_parameters(sweep_keys: List[str], values: np.ndarray) -> str:
    return ",".join(f"_{k}={format_value(v)}" for k, v in zip(sweep_keys, values))


class ComsolKeyNames(StrEnum):
    "Temperature_@_t={key}"
    T = 'Temperature' 
    T_grad_x = 'Temperature_gradient,_x-component'
    T_grad_y = 'Temperature_gradient,_y-component'
    T_grad_z = 'Temperature_gradient,_z-component'
    T_grad_L2 = 'Temperature_gradient_magnitude'
    darcy_x = 'Total_Darcy_velocity_field,_x-component'
    darcy_y = 'Total_Darcy_velocity_field,_y-component'
    darcy_z = 'Total_Darcy_velocity_field,_z-component',
    darcy_total = 'Total_Darcy_velocity_magnitude'
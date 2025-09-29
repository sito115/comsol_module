import pyvista as pv
from typing import Union, List
import re
from pathlib import Path    

def ensure_pathlib_path(path: Union[str,Path, List]) -> Union[List[Path], Path]:
    if isinstance(path, List):
        return [Path(v) if not isinstance(v, Path) else v for v in path]
    else:
        return Path(path) if isinstance(path, str) else path

def initilise_plotter(mesh: pv.DataSet, mp4_file: Path, cmap) -> pv.Plotter:
    b = mesh.bounds # x_min, x_max, y_min, y_max, z_min, z_max
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(mp4_file)
    plotter.add_axes()
    # plotter.add_mesh(mesh.outline_corners())
    # plotter.add_ruler(pointa =[b[0], b[2], 0],
    #               pointb =[b[1], b[2], 0],
    #               title = 'x [m]',
    #               flip_range = True,
    #               label_format = '%g',
    #             #   number_labels = 12
    #             )  
    plotter.add_ruler(pointa =[-1200, b[2], 0],
                    pointb =[-1200, b[3], 0],
                    title = 'y [m]',
                    label_format = '%g',
                    flip_range = True
                    # number_labels = 12,
                    )  
    plotter.add_ruler(pointa =[-1500, b[3] , 0],
                    pointb =[-1500, b[3] , b[-2]],
                    title = 'z [m]',
                    label_format = '%g',
                    # number_labels = 12,
                    ) 
    return plotter

def read_comsol_fields(mesh:pv.DataSet) -> tuple[list[str], dict[str, float], list[dict]]:
    """Field names in COMSOL are FIELDNAME_@_tTIME.

    Args:
        mesh (pv.DataSet): 
        field_pattern (_type_): regex to find field names in pyvista dataset,
                               
        time_pattern (_type_): regex to find time in pyvista dataset,

    Returns:
        tuple[pv.DataSet,list[str], dict[str, float]]: _description_
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
                    vars_dict[key_val.strip().lstrip("_")] = val.strip()

                
        time = vars_dict.pop("t", None)
        times.append(time)
        add_vars_dict.append(vars_dict)
        exported_fields.append(name)
    
    exported_fields : list[str] = list(set(exported_fields)) 
    # Sort the times and map them back to the original string values
    # assure that it is a field from COMSOL (usually contains an @)
    times : dict[str:float]= {str(val): float(val) for val in sorted(np.unique(times), key=float)}  # Sort by float value
    return (exported_fields, times, add_vars_dict)  
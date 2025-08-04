import pyvista as pv
from typing import Union, List
import re
from pathlib import Path    

def ensure_pathlib_path(path: Union[str,Path, List]) -> Union[List[Path], Path]:
    if isinstance(path, List):
        return [Path(v) if not isinstance(v, Path) else v for v in path]
    else:
        return Path(path) if isinstance(path, str) else path

def initilise_plotter(mesh: pv.DataSet, mp4_file: Path) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True)
    plotter.open_movie(mp4_file)
    plotter.add_mesh(mesh.outline_corners())
    plotter.add_axes()
    plotter.show_bounds(mesh)
    return plotter

def read_comsol_fields(mesh:pv.DataSet, field_pattern, time_pattern) -> tuple[list[str], dict[str, float]]:
    """Field names in COMSOL are FIELDNAME_@_tTIME.

    Args:
        mesh (pv.DataSet): 
        field_pattern (_type_): regex to find field names in pyvista dataset,
                               
        time_pattern (_type_): regex to find time in pyvista dataset,

    Returns:
        tuple[pv.DataSet,list[str], dict[str, float]]: _description_
    """    
    # assure that it is a field from COMSOL (usually contains an @)
    exported_fields : list[str] = list(set([re.search(field_pattern, key).group(1) for key in mesh.point_data.keys() if "@" in key])) 
    # Sort the times and map them back to the original string values
    # assure that it is a field from COMSOL (usually contains an @)
    time_map : dict[str:float] = {re.search(time_pattern, key).group(1): float(re.search(time_pattern, key).group(1)) for key in mesh.point_data.keys() if "@" in key}
    times : dict[str:float]= dict(sorted(time_map.items(), key=lambda x: x[1]))  # Sort by float value
    return (exported_fields, times)  
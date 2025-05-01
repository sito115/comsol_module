import numpy as np
import pyvista as pv
from typing import Optional, Union, List
from pydantic import BaseModel
import scipy
import re
from pathlib import Path

def calculate_normal(dip: float, strike: float) -> np.ndarray:
    """_summary_

    Args:
        dip (float): deg
        strike (float): deg

    Returns:
        tuple[float]: (x_normal, y_normal, z_normal)
    """
    dip    = np.deg2rad(dip)
    strike = np.deg2rad(strike)
    x_normal = -np.sin(dip)*np.sin(strike) 
    y_normal =  np.sin(dip)*np.cos(strike)
    z_normal = -np.cos(dip)
    return np.array([x_normal, y_normal, z_normal])
    

class ModelData(BaseModel):
    alpha: Optional[float] = None
    rho0: Optional[float] = None
    c_f: Optional[float] = None
    T_c: Optional[float] = None
    T_h: Optional[float] = None
    H: Optional[float] = None
    mu0: Optional[float] = None
    lambda_m: Optional[float] = None  # Allow lambda_m to be a part of the model for easy access after calculation
    k_m: Optional[Union[float, np.ndarray]] = None 
    
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like np.ndarray
    
    
    def calculate_lambda_m(self, lambda_f, lambda_s, phi) -> float:
        """ 
        Calculate the effective thermal conductivity of the porous medium (lambda_m).

        Args:
            lambda_f (float): Thermal conductivity of the fluid [W/mK].
            lambda_s (float): Thermal conductivity of the solid phase [W/mK].
            phi (float): Porosity of the material [-].

        Returns:
            float: Effective thermal conductivity of the porous medium [W/mK].

        """
        self.lambda_m = phi * lambda_f + (1 - phi) * lambda_s
        
        return self.lambda_m
    
    def calculate_T0(self) -> float:
        return 0.5 * (self.T_c + self.T_h) 


    def calculate_rayleigh_number(self) -> float:
        """
        Calculate the Rayleigh number for convection in a porous medium.
        - alpha = Thermal expansion coefficient [1/°C]
        - rho0 = Density of the fluid [kg/m3]
        - c_f = Specific heat of the fluid [J/kg/°C]
        - g = Gravitational constant [m2/s]
        - delta_T = Temperature difference [K]
        - K_m = Permeability of the porous material [m2]
        - H = Box size [m]
        - mu0 = Viscosity of the fluid [Pa/s]
        - lambda_m = Effective thermal conductivity of the porous 

        Returns:
            float: Rayleigh number [-].
        """
        
        for key, val in iter(self):
            if val is None:
                raise ValueError(f"Missing value for {key}")
        
        rayleigh_number =  self.k_m * self.rho0**2 * self.c_f * scipy.constants.g * self.alpha * (self.T_h - self.T_c) *  self.H / (self.mu0 * self.lambda_m) # 
        
        return rayleigh_number
    
    
def compute_surface_normal_vector(bounds: pv.DataSet.bounds) -> np.ndarray:
    """Computes surface normal vector from the bounds of the 2D surface.

    Args:
        bounds (pv.DataSet.bounds): _description_

    Returns:
        np.ndarray: _description_
    """        
    point1 = np.array([bounds[1], bounds[2], bounds[4]]) # base point bottom
    point2 = np.array([bounds[1], bounds[3], bounds[4]]) # lateral extent (y-dir)
    point3 = np.array([bounds[0], bounds[2], bounds[5]]) # vertical extent (z-dir)
    vector1 = point2 - point1  
    vector2 = point3 - point1  
    return  np.cross(vector1, vector2), point1 # normal vector from crossproduct


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
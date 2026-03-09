from pydantic import  field_validator
from pydantic.dataclasses import dataclass
from typing import Optional
import numpy as np  
from pathlib import Path
from typing import Union
import pyvista as pv
import logging
from .helper import (ensure_pathlib_path,
                    read_comsol_fields,
                    get_field_name_pattern,
                    format_sweep_parameters,
                    ComsolKeyNames)


@dataclass
class COMSOL_VTU():
    """Class to read exported simulation files from COMSOL.

    Attributes:
        mesh (pv.DataSet): _description_
        times (dict(str:float)): Sorted dictionary of exported times. The key corresponds to the exact string in the field name.
        exported fields (list(str)): All exported field names.
    """    
    
    vtu_path: Union[Path, str]
    name : Optional[str] = ''
    is_clean_mesh : Optional[bool] = False 

    class Config:
        arbitrary_types_allowed = True # for numpy etc
    
    @field_validator("vtu_path")
    @classmethod
    def check_path_exists(cls, vtu_path: Union[Path, str]) -> Path:#
        vtu_path = ensure_pathlib_path(vtu_path)
        if not vtu_path.exists():
            raise ValueError(f'Given path does not exist: {vtu_path}.')
        return vtu_path
    
    def __post_init__(self):
        logging.debug('Reading vtu file...')
        self.mesh = pv.read(self.vtu_path)
        if self.is_clean_mesh:
            self.mesh = self.mesh.clean()
            logging.info('Finished')
 
        logging.debug('Finished')
        self.exported_fields, self.times, self.sweep_keys, self.sweep_combos = read_comsol_fields(self.mesh)
        self._is_sweep = len(self.sweep_keys) > 0
        self._is_stationary = not len(self.times) > 1
        self.field_name_pattern =  get_field_name_pattern(self._is_stationary, self._is_sweep)

    def info(self):
        print(f'{self.vtu_path=}')
        if self._is_stationary:
            print('Stationary study.')
        else:
            print('Time-dependent study.')
            print(f'{len(self.times)} timesteps from {min(self.times.values()):.3e} to {max(self.times.values()):.3e}')
        print(f'{self.mesh.bounds=}')
        print('Available fields in dataset:')
        for idx, field in enumerate(sorted(self.exported_fields), start=1):
            print('\t %d: %s' % (idx, field))
        if self._is_sweep:
            print(f'Detected parametric sweep for {self.sweep_keys}')
            print(f'In total {len(self.sweep_combos)} sweep simulations per parameter.')
        

    def get_point_values(self, field_name: str) -> np.ndarray:
        """Get point values for a specific field.

        Args:
            field_name (str): The name of the field.

        Returns:
            np.ndarray: The point values of the field.
        """
        return self.mesh.point_data[field_name]
    
    
    def unify_field(self, field_name:ComsolKeyNames) -> None:
        """Merge all entries from COMSOL into on field.
        Useful for quantities that do not change over time to save memory.

        Args:
            field_name (str): 
        """        
        if self._is_sweep or self._is_stationary:
            raise NotImplementedError()
        self.mesh.point_data[field_name] = self.mesh.point_data[self.field_name_pattern.format(field_name, list(self.times.keys())[0])]
        for key in self.times.keys():
            self.mesh.point_data.remove(self.field_name_pattern.format(field_name, key))
    
    
    def format_field(self, field_name: str, time: Union[str, float, int], sweep_values : list = None) -> str:
        """

        Args:
            field_name (str): Must be in self.exported_fields
            time (Union[str, float, int]): Can be a string "1E13", a float 1E13 or the integer index of time series.

        Returns:
            str: Returns formatted field name for fields exported via COMSOL ("field_name_@_time").
        """        
        # assert field_name in self.exported_fields
        if isinstance(time, str):
            assert time in self.times.keys()
        if isinstance(time, float):
            time = min(self.times, key=lambda k: abs(self.times[k] - time))
        if isinstance(time, int):
            assert time <= len(self.times)
            time = list(self.times.keys())[time]
            
        if not self._is_sweep and not self._is_stationary:
            return self.field_name_pattern.format(field_name, time)
        elif self._is_sweep and not self._is_stationary:
            assert sweep_values in self.sweep_combos, f"This value combination {sweep_values} is not part of the sweep for {self.sweep_keys}."
            assert len(sweep_values) == len(self.sweep_keys), f"Values must match length of {self.sweep_keys}"
            formatted_sweep_values = format_sweep_parameters(self.sweep_keys, sweep_values)
            return self.field_name_pattern.format(field_name, time, formatted_sweep_values)
        else:
            raise NotImplementedError()
            
                
    def overwrite_domain_from_surface(self, surface: pv.DataSet, field_name3D: str, field_name2D: str = 'Color') -> None:
        """Can be used, when the 2D surface is a subset of the 3D domain. For example, if a fracture is implemented in Comsol (dl.frac), its Darcy-Velocieties
        from an export in surface-plot will differ from data Export in 3D domain. This function overwrites the 3D domain with the values from the 2D surface.
        Args:
            surface (pv.DataSet): 
            field_name3D (str): Field name in COMSOL_VTU object in mesh.point_data
            field_name2D (str, optional): Field name in Surface DataSet. Defaults to 'Color'.

        Returns:
            _type_: None
        """
        assert field_name3D in self.mesh.point_data.keys()
        assert field_name2D in surface.point_data.keys()
        
        # Structured view helper function for comparing elements in surface and domain row-wise
        def structured_view(arr : np.ndarray) -> np.ndarray:
            return arr.view([('', arr.dtype)] * arr.shape[1])
    
        mask3d = np.isin(structured_view(self.mesh.points), structured_view(surface.points)).flatten()
        mask2d = np.isin(structured_view(surface.points), structured_view(self.mesh.points)).flatten()

        points_3d_masked = self.mesh.points[mask3d, :]
        vals_3d = np.zeros((np.sum(mask3d)))
        for tuple2d, val in zip(surface.points[mask2d, :], surface.point_data[field_name2D][mask2d]):
            mask = np.all(points_3d_masked == tuple2d, axis=1)
            vals_3d[mask] = val

        self.mesh.point_data[field_name3D][mask3d] = vals_3d
    
        
    def get_array(self, field: ComsolKeyNames) -> np.ndarray:
        """Return numpy matrix of given field name.


        Args:
            field (ComsolKeyNames): field_name
            is_cell_data (bool, optional): Return cell or point data . Defaults to False.

        Returns:
            np.ndarray:
                - For transient studies (N_TIME_STEP x N_POINTS)
                - For transient studies and sweep (N_TIME_STEP x SWEEP COMBOS x N_POINTS)
        """
        assert field in self.exported_fields, f"{field} not found."
        if self._is_sweep and not self._is_stationary:
            assembled_matrix = np.zeros((len(self.times), len(self.sweep_combos), self.mesh.n_points))
            for i, time_key in enumerate(self.times.keys()):
                for j, param_combo in enumerate(self.sweep_combos):
                    assembled_matrix[i,j] = self.mesh.point_data[self.format_field(field, time_key, param_combo)]
            return assembled_matrix
                    
        elif not self._is_sweep and not self._is_stationary:
            return np.array([self.mesh.point_data[self.format_field(field, key)] for key in self.times.keys()])
        else:
            raise NotImplementedError()
        
            
                
    def merge_datasets(self, *args) -> None:
        if self._is_sweep or self._is_stationary:
            raise NotImplementedError()
        for arg in args:
            assert isinstance(arg, COMSOL_VTU)
            assert set(arg.exported_fields).issubset(set(self.exported_fields))
            assert self.mesh.points.shape == arg.mesh.points.shape
            self.times.update(arg.times)
            self.mesh.point_data.update(arg.mesh.point_data)
            
            
            
    def delete_field(self, field_name: ComsolKeyNames):
        assert field_name in self.exported_fields
        if self._is_sweep or self._is_stationary:
            raise NotImplementedError()
        for time_key in self.times.keys():
            temp_field_name = self.format_field(field_name, time_key)
            self.mesh.point_data.remove(temp_field_name)
        self.exported_fields.remove(field_name)
        
        
        
    

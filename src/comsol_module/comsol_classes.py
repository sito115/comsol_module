from pydantic import  field_validator
from pydantic.dataclasses import dataclass
from typing import Optional
import numpy as np  
from pathlib import Path
from typing import Union, List
from enum import StrEnum
import pyvista as pv
import logging
from matplotlib import pyplot as plt
from tqdm import tqdm
from .helper import (ensure_pathlib_path,
                    read_comsol_fields,
                    initilise_plotter)
from .entropy import (calculate_S_therm,
                      calculate_S_visc)


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
    s_total = 'Total_Entropy Production Rate_W/(K*m^3*s)'
    s_therm = 'Thermal_Entropy Production Rate'
    s_visc = 'Viscous_Entropy Production Rate_W/(K*m^3*s)'
    s_therm_L2 = 'L2_Thermal_Entropy Production Rate_W/(K*m^3*s)'


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
    vtu_pattern =  '{}_@_t={}'                # container for finding values in pyvista.DataSet
    is_clean_mesh : Optional[bool] = True 

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
        logging.debug('Finished')
        time_pattern = r"@_t=([\d.]+(?:[Ee][+-]?\d+)?)" # find exported time steps
        field_pattern = r"^(.*?)_@_t"                    # find exported fields
        self.exported_fields, self.times = read_comsol_fields(self.mesh, field_pattern, time_pattern)

    def info(self):
        print(f'{self.vtu_path=}')
        print(f'{len(self.times)} timesteps from {min(self.times.values()):.3e} s to {max(self.times.values()):.3e} s')
        print(f'{self.mesh.bounds=}')
        print('Availabe fields in vtu dataset:')
        for idx, field in enumerate(sorted(self.exported_fields), start=1):
            print('\t %d: %s' % (idx, field))
        
    def export_mp4_movie(self, field: str, mp4_file: Path = None, **kwargs) -> None:
        """Exports a mp4 movie.

        Args:
            field (str): _description_
            mp4_file (Path, optional): _description_. Defaults to None.
        
        Keyword Args:
            is_diff (bool, optional): Subtract the value at the first time index from every iteration. Defaults to False.
            is_log (bool, optional): log10 display 
            is_ind_cmap (bool, optional): Display an individual colormap for each iteration. Defaults to False.
            t_grad (float, optional): Temperature gradient in the z-direction that is subtracted from every iteration (in [K/m]). 
            movie_field (str, optional): Name of the field to display above the colormap in the movie.
            bounds (tuple of float): Tuple of the form (xmin, xmax, ymin, ymax, zmin, zmax) for the clipped array.
            normal (np.ndarray): Normal vector for the plane to be clipped.
            origin (np.ndarray): Origin point for the plane to be clipped.
            mesh_kwargs (dict): Additional keyword arguments for `mesh.clip()` or `mesh.clip_box()`.
            add_mesh_kwargs (dict): Additional keyword arguments for `pv.add_mesh()`.
            is_min_max (bool, optional): Display min and max values in the colorbar. Defaults to False.
            param_string (str, optional) : Display Parameters
            title_string (str, optional): Title of the plotter window. Defaults to None.
            plot_last_frame (bool, optional): If True, the last frame will be saved as a screenshot. Defaults to True.
        Returns:
            _type_: None
        """
        
        my_cmap = kwargs.pop("cmap", plt.get_cmap("turbo", 15))
        movie_field = kwargs.pop('movie_field', field + "-mp4")
        if mp4_file is None:
            mp4_file = self.vtu_path.parent.joinpath(f'{self.vtu_path.stem}_{field}.mp4')
        print('Export path = %s' % str(mp4_file))
        plotter = initilise_plotter(self.mesh, mp4_file, my_cmap)



        bounds = kwargs.pop('bounds', None)
        normal = kwargs.pop('normal', None)
        mesh_kwargs = kwargs.pop('mesh_kwargs', {})
        if bounds is not None:
            mesh = self.mesh.clip_box(bounds, **mesh_kwargs)
        elif normal is not None:
            origin = kwargs.pop('origin', None)
            mesh = self.mesh.clip(normal=normal, origin=origin, **mesh_kwargs)
        else:
            mesh = self.mesh
        
        key0 = next(iter(self.times.keys()))
        val0 = mesh[self.vtu_pattern.format(field,key0)]
         
        t_grad = kwargs.pop('t_grad', None)
        if t_grad is not None:
            z = mesh.points[:, 2]
            val0 = t_grad['t0'] - t_grad['t_grad'] * z 
                
        is_diff = kwargs.pop('is_diff', False)
        is_log = kwargs.pop('is_log', False)
        if is_diff:
            mesh[movie_field] = val0 - val0
        elif is_log:
            mesh[movie_field] = np.log10(val0)
        else:
            mesh[movie_field] = val0
        
        add_mesh_kwargs = kwargs.pop('add_mesh_kwargs', {})
        actor = plotter.add_mesh(mesh,
                                 scalars = movie_field,
                                 cmap=my_cmap,
                                 **add_mesh_kwargs) # The sliced plane
        # Add the scalar bar to the plotter
        plotter.add_scalar_bar(title=movie_field, label_font_size=12, 
                       position_x=0.2, position_y=0.05)
        plotter.write_frame()
        
        is_ind_cmap = kwargs.pop('is_ind_cmap', False)
        is_min_max = kwargs.pop('is_min_max', False)
        if is_min_max:
            win_width, win_height = plotter.render_window.GetSize()
            x, y = plotter.scalar_bar.GetPosition()
            width, _ = plotter.scalar_bar.GetPosition2()
            min_text_position = ((x - 0.1)*win_width, y*win_height) 
            max_text_position = ((x + width + 0.02)*win_width, y*win_height)
            fs = plotter.scalar_bar.GetLabelTextProperty().GetFontSize()
            # label_format = plotter.scalar_bar.GetLabelFormat()
            min_text_actor = plotter.add_text(f"Min: {np.min(mesh[movie_field]):.2e}", font_size=0.6 * fs,
                             color="black", position=min_text_position)
            max_text_actor = plotter.add_text(f"Max: {np.min(mesh[movie_field]):.2e}", font_size=0.6 * fs,
                             color="black", position=max_text_position)

        param_string = kwargs.pop("param_string", "")
        title_string = kwargs.pop("title_string", "")
        for idx, (key, time) in tqdm(enumerate(self.times.items(), start = 1), desc=f'Processing frames for {field}', total = len(self.times)):
            if is_diff:
                mesh[movie_field] = mesh[self.vtu_pattern.format(field,key)] - val0
            elif is_log:
                mesh[movie_field] = np.log10(mesh[self.vtu_pattern.format(field,key)])
            else:
                mesh[movie_field] = mesh[self.vtu_pattern.format(field,key)]
            plotter.add_text(f"{title_string}Output {idx} @ {time:.3e} s",
                             name='time-label', font_size=14)
            plotter.add_text(param_string,
                viewport=True, 
                position=(0, 0.6),  #"left_edge",
                font_size=14,)
            if is_ind_cmap:
                actor.mapper.scalar_range = ( np.min(mesh[movie_field]), np.max(mesh[movie_field]) )
            if is_min_max:
                min_text_actor.input = f'Min: {np.min(mesh[movie_field]):.2e}'
                max_text_actor.input = f'Max: {np.max(mesh[movie_field]):.2e}'
            plotter.write_frame()
        
        plot_last_frame  = kwargs.pop('plot_last_frame', False)
        if plot_last_frame:
            plotter.screenshot(mp4_file.parent / f'{self.vtu_path.stem}_{field}_lastframe.png')
        plotter.close()



    def get_point_values(self, field: ComsolKeyNames, time_step : Union[int, str]) -> np.ndarray:
        """_summary_

        Args:
            field (ComsolKeyNames): _description_
            time_step (Union[int, str]): _description_

        Returns:
            np.ndarray: _description_
        """        
        if isinstance(time_step, int):
            key = list(self.times.keys())[time_step]
            logging.info(f'Time step {key}')
        else:
            key = time_step
        return self.mesh.point_data[self.vtu_pattern.format(field,key)]
    
    
    def unify_field(self, field_name:str) -> None:
        """Merge all entries from COMSOL into on field.
        Useful for quantities that do not change over time to save memory.

        Args:
            field_name (str): 
        """        
        self.mesh.point_data[field_name] = self.mesh.point_data[self.vtu_pattern.format(field_name, list(self.times.keys())[0])]
        for key in tqdm(self.times.keys(), f"Removing redundant fields '{field_name}'"):
            self.mesh.point_data.remove(self.vtu_pattern.format(field_name, key))
    
    def format_field(self, field_name: str, time: Union[str, float, int]) -> str:
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
        return self.vtu_pattern.format(field_name, time)
    
    def calculate_total_quantity(self, field_name: str, fields: list[str] = None) -> None:
        """Calculates L2 norm of given fields.

        Args:
            fields (list[str]): list of fields to calculate the L2 norm (must be in self.exported_fields)
            field_name (str): the new field name (will be appended to self.exported_fields)
        """        
        
        if fields is None:
            fields = ['Total_Darcy_velocity_field,_x-component',
                    'Total_Darcy_velocity_field,_y-component',
                    'Total_Darcy_velocity_field,_z-component',]
        
        self.exported_fields.append(field_name)
        for time_key in tqdm(self.times.keys(), desc='Processing...', total = len(self.times)): 
            quantities = np.zeros((len(fields), len(self.mesh.points)))
            for i_field, field in enumerate(fields):  # noqa: F402
                quantities[i_field] = self.mesh.point_data[self.vtu_pattern.format(field, time_key)]
            self.mesh.point_data[self.format_field(field_name, time_key)] = np.sqrt(np.sum(quantities**2, axis = 0))
        
    
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
    
    
    def get_cell_values(self, field: ComsolKeyNames, time_step :  Union[int, str]) -> np.ndarray:
        data = self.mesh.point_data_to_cell_data()
        if isinstance(time_step, int):
            key = list(self.times.keys())[time_step]
            logging.info(f'Time step {key}')
        else:
            key = time_step
        return data.cell_data[self.vtu_pattern.format(field,key)]
    
    
    def get_array(self, field: ComsolKeyNames, is_cell_data : bool = False) -> np.ndarray:
        """Return numpy matrix of given field name (N_TIME_STEP x N_POINTS)

        Args:
            field (ComsolKeyNames): field_name
            is_cell_data (bool, optional): Return cell or point data . Defaults to False.

        Returns:
            np.ndarray: 
        """
        assert field in self.exported_fields, f"{field} not found."
        if is_cell_data:
            cell_mesh = self.mesh.point_data_to_cell_data() 
            return np.array([cell_mesh[self.format_field(field, key)] for key in self.times.keys()])        
        return np.array([self.mesh.point_data[self.format_field(field, key)] for key in self.times.keys()])
            
            
    def calculate_total_entropy_per_vol(self,
                                        model_data: dict,
                                        time_steps: Union[list[int],int] = None,
                                        is_return_as_integration: bool = True) -> np.ndarray:
        """_summary_

        Args:
            model_data (ModelData): _description_
            time_steps (Union[list[int],int]): zero-indexed!

        Returns:
            np.ndarray: [N x 2] Entropy (therm, visc) in  W/(K * s)
        """
        # Default to all time steps if none provided
        if time_steps is None:
            time_steps = np.arange(len(self.times))
        
         # Ensure time_steps is a list
        if isinstance(time_steps, int):
            time_steps = [time_steps]
        
        required_model_keys = ['lambda_m', 'T0', 'mu0', 'k_m']
        missing = [k for k in required_model_keys if k not in model_data]
        if missing:
            print("Missing keys in 'model_data':", missing)
        
        # Extract time keys for the selected time steps
        time_keys = [list(self.times.keys())[i] for i in time_steps]
        
        # convert to cell data to match cell areas for Integration
        temp_mesh = self.mesh.copy() 
        temp_mesh.clear_data()
        
        # Initialize array for storing integrated entropy values (thermal, viscous)

        if is_return_as_integration:
            entropy = np.zeros((len(time_steps), 2))
        else:
            entropy = np.zeros((len(time_steps), self.mesh.n_points, 2))
    
        for idx, time_key in enumerate(time_keys):
            logging.debug(f'Time {time_key}')
            
            deriv = self.mesh.compute_derivative(scalars=self.format_field(ComsolKeyNames.T.value,time_key), preference = 'point')
            temp_gradient = deriv.point_data['gradient']
            # Retrieve Darcy velocities with default handling
        
            s_therm = calculate_S_therm(model_data['lambda_m'] , model_data['T0'] , temp_gradient)
            try:
                s_visc  = calculate_S_visc(model_data['mu0'], model_data['k_m'], model_data['T0'] , self.get_point_values(ComsolKeyNames.darcy_total.value, time_key))
            except KeyError:
                # logging.debug(f"Field '{ComsolKeyNames.darcy_total.value}' not found in exported fields. Returning zero viscous entropy.")
                s_visc = np.zeros_like(s_therm)
            
            if is_return_as_integration:
                temp_mesh.point_data['s_therm'] = s_therm
                temp_mesh.point_data['s_visc'] = s_visc
                integrated = temp_mesh.integrate_data()
                entropy[idx, 0] = integrated.point_data['s_therm'][0]
                entropy[idx, 1] = integrated.point_data['s_visc'][0]
            else:
                entropy[idx, :, 0] = s_therm
                entropy[idx, :, 1] = s_visc
            
        return entropy
    
    
    def merge_datasets(self, *args) -> None:
        for arg in args:
            assert isinstance(arg, COMSOL_VTU)
            assert set(arg.exported_fields).issubset(set(self.exported_fields))
            assert self.mesh.points.shape == arg.mesh.points.shape
            self.times.update(arg.times)
            self.mesh.point_data.update(arg.mesh.point_data)
            
            
    def delete_field(self, field_name: str):
        assert field_name in self.exported_fields
        for time_key in self.times.keys():
            temp_field_name = self.format_field(field_name, time_key)
            self.mesh.point_data.remove(temp_field_name)
        self.exported_fields.remove(field_name)
        
        
        
            

        
if __name__ == '__main__':
    
    # Set up basic configuration for logging
    logging.basicConfig(
    level=logging.DEBUG,               # Set the lowest level of logging to capture
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the format of log messages
    )
    
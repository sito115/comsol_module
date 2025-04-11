from typing import Union
import numpy as np
# from numba import jit


# @jit
def calculate_S_therm(lambda_m : float, T_0 : float, temp_gradient : np.ndarray) -> np.ndarray:
    """Calculate the thermal entropy generation rate per volume.

    Args:
        lambda_m (float): _description_
        T_0 (float): _description_
        temp_gradient (np.ndarray): [N x 3] matrix of temperature gradient components [K/m]

    Returns:
        np.ndarray: entropy generation rate per VOLUME [W/(K * m^3 * s)]
    """
    return lambda_m / T_0**2 * (temp_gradient[:, 0]**2 + temp_gradient[:, 1]**2 + temp_gradient[:, 2]**2) 

# @jit
def calculate_S_visc(mu_f : float, k_tensor : np.ndarray, T_0 : Union[float, np.ndarray], darcy_vel : np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        mu_f (float): _description_
        k_tensor (np.ndarray): _description_
        T_0 (float): _description_
        darcy_vel (np.ndarray): [3 x N] matrix of darcy velocity components [m/s]

    Returns:
        np.ndarray: entropy generation rate per VOLUME [W/(K * m^3 * s)]
    """
    
    return mu_f / (np.mean(k_tensor) * T_0) * (darcy_vel[0]**2 + darcy_vel[1]**2 + darcy_vel[2]**2) 



def calculate_S_total(temp_gradient : np.ndarray, darcy_vel : np.ndarray,  T_0 : Union[float, np.ndarray], lambda_m : float, mu_f : float, k_tensor : np.ndarray) -> np.ndarray:
    """Calculate the total entropy generation rate per volume composed of viscous and thermal term.

    Args:
        temp_gradient (np.ndarray): [N x 3] matrix of temperature gradient components [K/m]
        darcy_vel (np.ndarray): [3 x N] matrix of darcy velocity components [m/s]
        T_0 (float): _description_
        lambda_m (float): _description_
        mu_f (float): _description_
        k_tensor (np.ndarray): _description_

    Returns:
        np.ndarray: Tototal entropy generation rate per VOLUME [W/(K * m^3 * s)]
    """  
    
    S_therm = calculate_S_therm(lambda_m , T_0 , temp_gradient)
    S_visc  = calculate_S_visc(mu_f, k_tensor, T_0 , darcy_vel)
    
    return S_therm + S_visc

# @jit
def caluclate_entropy_gen_number(s_total : Union[float, np.ndarray], q_field: float, lambda_m :float, T_0: float, V: float):
    """_summary_

    Args:
        S_total (_type_): total entropy generation rate [W/(K * s)]
        q_field (_type_):     q is the total specific heat flow, represents the bulk heat flux between the lower and upper boundaries of the model [W/m^2]
        lambda_m (_type_): bulk thermal conductivity of the porous medium [W/mK]
        T_0 (_type_): _description_
        V (_type_): Volume of the porous medium [m^3]

    Returns:
        _type_: _description_
    """  
    return s_total * lambda_m * T_0**2 / (q_field**2 * V )
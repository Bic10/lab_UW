# src/LAB_UW_forward_modeling.py

import numpy as np
from numpy import linalg as LA
from plotting import plot_simulation_waveform
from helpers import *
from synthetic_data import *

from typing import Union

def DDS_UW_simulation(t_OBS: np.ndarray, waveform_OBS: np.ndarray, t_pulse: np.ndarray, pulse: np.ndarray, interval: slice,
                      sample_dimensions: tuple, freq_cut: float, x_transmitter: float, x_receiver: float, pzt_width: float, pmma_width: float,
                      c_max: float, c_gouge: Union[float, np.ndarray], c_pzt: float, c_pmma: float, normalize: bool = True, plotting: bool = False) -> float:
    '''
    Simulate ultrasonic wave propagation through a sample and compute the L2 norm between observed and synthetic waveforms.

    Args:
        t_OBS (np.ndarray): Time axis of observed waveform.
        waveform_OBS (np.ndarray): Observed waveform.
        t_pulse (np.ndarray): Time axis of the pulse.
        pulse (np.ndarray): Pulse waveform.
        interval (slice): Interval for L2 norm computation.
        sample_dimensions (tuple): Dimensions of the sample (side_block_1, gouge_1, central_block, gouge_2, side_block_2).
        freq_cut (float): Maximum frequency cut-off.
        x_trasmitter (float): Transmitter location.
        x_receiver (float): Receiver location.
        pzt_width (float): Width of the piezoelectric transducer.
        pmma_width (float): Width of the PMMA layer.
        c_max (float): Maximum velocity.
        c_gouge (float): Velocity in the gouge layer.
        c_pzt (float): Velocity in the piezoelectric transducer.
        c_pmma (float): Velocity in the PMMA layer.
        normalize (bool, optional): Whether to normalize the synthetic waveform. Defaults to True.
        plotting (bool, optional): Whether to plot the synthetic waveform. Defaults to False.

    Returns:
        float: L2 norm between observed and synthetic waveforms.
    '''

    global sp_field, sp_recorded, x, t

    # Compute the space grid and the time steps according to numerical dispersion and stability
    x, dx, nx = compute_grid(sample_dimensions, c_gouge, freq_cut)   
    c_model,_,_,_,_ = build_velocity_model(x=x, 
                                            sample_dimensions=sample_dimensions, 
                                            x_trasmitter=x_transmitter, 
                                            x_receiver=x_receiver, 
                                            pzt_width=pzt_width, 
                                            pmma_width=pmma_width, 
                                            csteel=c_max, 
                                            c_gouge=c_gouge, 
                                            c_pzt=c_pzt, 
                                            c_pmma=c_pmma,
                                            plotting=False)

    t, dt, nt = prepare_time_variables(t_OBS, dx, c_max)

    # Interpolate source time function on the simulation time steps
    interpolated_pulse = interpolate_source(t_pulse, pulse, dt)
    src_t = np.zeros(len(t))
    src_t[:np.size(interpolated_pulse)] = interpolated_pulse

    # generate soruce and receiver spatial funciton: synthetic for now, since we do not have it yet
    isx = np.argmin(np.abs(x - x_transmitter))
    irx = np.argmin(np.abs(x - x_receiver))
    sigma = pzt_width / 100
    src_x = synthetic_source_spatial_function(x, isx, sigma=sigma, plotting=False)
    rec_x = synthetic_source_spatial_function(x, irx, sigma=sigma, plotting=False)

    # actual computation of synthetic wavefield
    sp_field = pseudospectral_1D_forward(nx, dx, nt, dt, src_x, src_t, c_model)

    # The simulated wavefield recorded
    sp_simulated = np.sum(sp_field * rec_x, axis=-1)

    if normalize:
        sp_simulated = sp_simulated * (np.amax(waveform_OBS) / np.amax(sp_simulated))

    waveform_SYNT = np.interp(t_OBS, t, sp_simulated)
#    correction_3D = (sample_dimensions[0] / t_OBS) ** 2
#    waveform_SYNT_corrected = waveform_SYNT * correction_3D
 #   print(f"waveform_SYNT_corrected: {waveform_SYNT_corrected}")
    if plotting:
        plot_simulation_waveform(t_OBS, waveform_SYNT, waveform_OBS)
        plt.figure()
        plt.plot(src_t)
        plt.show()
    return waveform_SYNT

def pseudospectral_1D_forward(nx: int, dx: float, nt: int, dt: float, src_x: np.ndarray, src_t: np.ndarray, c_model: np.ndarray) -> np.ndarray:
    '''
    Perform pseudospectral forward modeling for 1D wave propagation.

    Args:
        nx (int): Number of spatial grid points.
        dx (float): Grid spacing.
        nt (int): Number of time steps.
        dt (float): Time step.
        src_x (np.ndarray): Spatial source function.
        src_t (np.ndarray): Time source function.
        c_model (np.ndarray): Velocity model.

    Returns:
        np.ndarray: Synthetic wavefield.
    '''
    sp = np.zeros(nx)
    spnew = sp
    spold = sp
    sp_field = np.zeros((nt, nx))
    scaling_factor = 1

    for it in range(nt):
        sd2p = fourier_derivative_2nd(sp, dx)
        spnew = 2 * sp - spold + c_model ** 2 * dt ** 2 * sd2p
        spnew = spnew + src_x * src_t[it] * dt ** 2
        spnew = scaling_factor * spnew
        spold, sp = scaling_factor * sp, spnew
        sp[1] = 0
        sp[nx - 1] = 0
        sp_field[it, :] = sp

    return sp_field

def compute_grid(sample_dimensions, c_gouge, freq_cut, ppt_for_minimum_len=10):
    """
    Compute the 1D spatial grid for the simulation based on the minimum velocity and frequency.

    Args:
        sample_dimensions (tuple): Dimensions of the sample.
        c_gouge (Union[float, np.ndarray]): Velocity in the gouge layer.
        freq_cut (float): Maximum frequency cut-off.
        ppt_for_minimum_len (int, optional): Points per minimum wavelength. Defaults to 10.

    Returns:
        tuple: A tuple containing the grid (x), grid spacing (dx), and number of grid points (nx).
    """
    cmin = c_gouge  # Minimum velocity
    grid_length = sum(sample_dimensions)
    x = make_grid_1D(grid_len=grid_length, cmin=cmin, fmax=freq_cut, ppt=ppt_for_minimum_len)
    dx = x[1] - x[0]
    nx = len(x)
    return x, dx, nx

def prepare_time_variables(t_OBS, dx, c_max, eps = 0.5):
    """
    Prepare the time variables for the simulation based on the grid and maximum velocity.
    This implementation try to roughly respect CFL stability criterium for 1D simulation

    Args:
        t_OBS (np.ndarray): Time axis of observed waveform.
        dx (float): Grid spacing.
        c_max (float): Maximum velocity.

    Returns:
        tuple: A tuple containing the time axis (t), time step (dt), and number of time steps (nt).
    """
    dt = eps * dx / c_max
    t = np.arange(0, t_OBS[-1], dt)
    nt = len(t)
    return t, dt, nt

def interpolate_source(t_pulse, pulse, dt):
    """
    Interpolate the source time function to match the time discretization.

    Args:
        t_pulse (np.ndarray): Time axis of the pulse.
        pulse (np.ndarray): Pulse waveform.
        dt (float): Time step for interpolation.

    Returns:
        np.ndarray: Interpolated source time function.
    """
    interpolated_t_pulse = np.arange(t_pulse[0], t_pulse[-1], dt)
    interpolated_pulse = np.interp(interpolated_t_pulse, t_pulse, pulse)
    return interpolated_pulse

def compute_misfit(waveform_OBS, waveform_SYNT, interval):
    """
    Compute misfit between observed and synthetic waveforms.
    At the moment is an l2 norm evalutated on the interval of simulated waveform 
    that can be compared with the observed waveform, 
    given the fact that the forward modeling is 1D

    Args:
        waveform_OBS (np.ndarray): Observed waveform.
        waveform_SYNT (np.ndarray): Synthetic waveform.
        interval (slice): Interval for L2 norm computation.

    Returns:
        float: The misfit.
    """

    return LA.norm(waveform_OBS[interval] - waveform_SYNT[interval], 2)
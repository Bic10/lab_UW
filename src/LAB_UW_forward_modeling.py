# src.LAB_UW_forward_modeling.py
import numpy as np
from numpy import linalg as LA
from plotting import plot_simulation_waveform
from helpers import *
from synthetic_data import *

def DDS_UW_simulation(t_OBS: np.ndarray, waveform_OBS: np.ndarray, t_pulse: np.ndarray, pulse: np.ndarray, interval: slice,
                      sample_dimensions: tuple, freq_cut: float, x_trasmitter: float, x_receiver: float, pzt_width: float, pmma_width: float,
                      cmax: float, c_gouge: float, cpzt: float, cpmma: float, normalize: bool = True, plotting: bool = False) -> float:
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
        cmax (float): Maximum velocity.
        c_gouge (float): Velocity in the gouge layer.
        cpzt (float): Velocity in the piezoelectric transducer.
        cpmma (float): Velocity in the PMMA layer.
        normalize (bool, optional): Whether to normalize the synthetic waveform. Defaults to True.
        plotting (bool, optional): Whether to plot the synthetic waveform. Defaults to False.

    Returns:
        float: L2 norm between observed and synthetic waveforms.
    '''

    global sp_field, sp_recorded, x, t

    ppt_for_minimum_len = 10
    cmin = c_gouge
    x = make_grid_1D(grid_len=sum(sample_dimensions), cmin=cmin, fmax=freq_cut, ppt=ppt_for_minimum_len)
    dx = x[1] - x[0]
    nx = len(x)

    isx = np.argmin(np.abs(x - x_trasmitter))
    irx = np.argmin(np.abs(x - x_receiver))

    c_model, idx_gouge_1, idx_gouge_2, idx_pzt_1, idx_pzt_2 = build_velocity_model(x, sample_dimensions, x_trasmitter, x_receiver, pzt_width, pmma_width, cmax, c_gouge, cpzt, cpmma, plotting=False)

    eps = 0.5
    dt = eps * dx / cmax
    t = np.arange(0, t_OBS[-1], dt)
    nt = len(t)

    interpolated_t_pulse = np.arange(t_pulse[0], t_pulse[-1], dt)
    interpolated_pulse = np.interp(interpolated_t_pulse, t_pulse, pulse)

    src_t = np.zeros(len(t))
    src_t[:np.size(interpolated_pulse)] = interpolated_pulse

    sigma = pzt_width / 100
    src_x = synthetic_source_spatial_function(x, isx, sigma=sigma, plotting=False)
    rec_x = synthetic_source_spatial_function(x, irx, sigma=sigma, plotting=False)

    sp_field = pseudospectral_1D_forward(nx, dx, nt, dt, src_x, src_t, c_model)
    sp_recorded = np.sum(sp_field * rec_x, axis=-1)

    if normalize:
        sp_recorded = sp_recorded * (np.amax(waveform_OBS) / np.amax(sp_recorded))

    if plotting:
        plot_simulation_waveform(t, sp_recorded)

    waveform_SYNT = np.interp(t_OBS, t, sp_recorded)
    correction_3D = (sample_dimensions[0] / t_OBS) ** 2
    waveform_SYNT = waveform_SYNT * correction_3D

    L2norm = LA.norm(waveform_OBS[interval] - waveform_SYNT[interval], 2)

    return L2norm

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

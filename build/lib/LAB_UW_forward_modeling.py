# src.LAB_UW_forward_modeling.py
import numpy as np
from numpy import linalg as LA
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

from src.plotting import *
from src.helpers import *
from src.synthetic_data import *
from src.LAB_UW_forward_modeling import *
# Define the color palette for the plots
colors = {
    'reseda_green': '#788054',
    'dutch_white': '#E0D6B4',
    'khaki': '#CABB9E',
    'platinum': '#E7E5E2',
    'black_olive': '#322D1E'}


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
    # DEFINE GRID:
    ppt_for_minimum_len = 10       #  implement as a possible parameter option?
    # minimum among the various velocity
    # cmin=  min([i for i in [cmax,c_gouge,cpzt,cpmma] if i!=0])
    cmin = c_gouge
    x = make_grid_1D(grid_len = sum(sample_dimensions), cmin = cmin, fmax=freq_cut, ppt=ppt_for_minimum_len)
    dx = x[1]-x[0]       # [cm/mus] spacing between the point of the x axis     
    nx = len(x)          # number of samples in the simulated x axes. Must be even for pseudo spectral computation. So the if above

    # SOURCE LOCATION COORDINATE INDEX 
    # for now just the index closer to the true trasmission position. Check the error!
    isx = np.argmin(np.abs(x - x_trasmitter))

    # RECEIVER LOCATION COORDINATE INDEX 
    # for now just the index closer to the true RECEIVER position. Check the error!
    irx = np.argmin(np.abs(x - x_receiver))

    # BUILD VELCOTIY MODEL
    c_model = build_velocity_model(x,sample_dimensions,x_trasmitter,x_receiver, pzt_width, pmma_width, cmax,c_gouge,cpzt,cpmma, plotting= False)

    # DEFINE TIME AX USING CFL CRITERION: https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
    eps   = 0.5     # stability limit, should be under 1. Implement as a possible parameter option?
    dt = eps*dx/cmax; # calculate time step from stability criterion
    t = np.arange(0, t_OBS[-1],dt)   
    nt = len(t)

    # BUILD SOURCE TIME FUNCTION

    # Interpolate time axes of the pulse on the time axes of the simulation
    interpolated_t_pulse = np.arange(t_pulse[0], t_pulse[-1], dt)
    interpolated_pulse = np.interp(interpolated_t_pulse, t_pulse, pulse)
    # interpolated_pulse = np.hanning(len(interpolated_pulse))*interpolated_pulse
    # plt.plot(interpolated_t_pulse,interpolated_pulse)

    src_t = np.zeros(len(t)) 
    src_t[:np.size(interpolated_pulse)] = interpolated_pulse

    # BUILD SOURCE AND RECEIVER SPACIAL FUNCTION
    sigma = pzt_width/100           # is there a smarter way to define it?
                    # width of the gaussian bell representing the pzt spatial profile
                        # with a sigma bigger than this, you are going to see the finite size of the pzt in the waveforms
                        # as a douplet on top of some picks. Investigating it?
    src_x = synthetic_source_spatial_function(x, isx, sigma=sigma, plotting=False)
    # src_x[:isx] = 0
    # well, the trasmitter spatially is identical to the receiver, so
    rec_x = synthetic_source_spatial_function(x, irx, sigma=sigma, plotting=False)
    # src_x[irx+1:] = 0

    # THIS IS THE ACTUAL COMPUTATION OF THE WAVE EQUATION WE RECORD
    sp_field = pseudospectral_1D_forward(nx,dx,nt,dt,src_x,src_t, c_model)       
    sp_recorded = np.sum(sp_field*rec_x, axis=-1)    # to account for spacial dimension of the receiver

    if normalize:
        sp_recorded = sp_recorded * (np.amax(waveform_OBS)/np.amax(sp_recorded))

    if plotting:
        ax = plt.gca()
        # ax.set_facecolor(colors['khaki'])  # You can choose any color you like
        ax.plot(t,sp_recorded, label="UW simulated", color=colors['platinum'], linewidth=4.0)
        # plt.plot(t_OBS,waveform_SYNT*(norm/np.amax(waveform_SYNT)) - 15*(idx+1), label=f"pulse {idx}")
        
    waveform_SYNT = np.interp(t_OBS,t,sp_recorded)   # back to the same dt of the observed data
    correction_3D = (sample_dimensions[0] / t_OBS) ** 2
    waveform_SYNT = waveform_SYNT * correction_3D

    L2norm = LA.norm(waveform_OBS[interval]-waveform_SYNT[interval],2)

    
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
    # pressure fields Initialization
    sp = np.zeros(nx); spnew = sp; spold = sp; sd2p = sp; sdp = sp 
    sp_field = np.zeros((nt,nx))

    # # Calculate distance from source for each point
    # distance_from_source = np.abs(x - x_trasmitter)
    # # Calculate the scaling factor based on the inverse square law
    # calculate_scaling_factor(distance_from_source)
    scaling_factor = 1 # 

    # Function to update the plot for each frame
    for it in range(nt):
        # global sp, spold, spnew  # get them from outside

        # pseudospectral 1D-wave equation solver
        # ----------------------------------------
        sd2p  = fourier_derivative_2nd(sp, dx)      # 2nd space derivative     
        spnew = 2*sp - spold + c_model**2 * dt**2 *sd2p  # Time Extrapolation   
        spnew = spnew + src_x*src_t[it]*dt**2       # Add sources   
        spnew = scaling_factor * spnew              # Apply geometrical spreading correction 1D to 3D
        spold, sp = scaling_factor*sp, spnew                       # Time levels
        sp[1] = 0; sp[nx-1] = 0                     # Set boundaries pressure free
        sp_field[it,:] = sp

    return sp_field

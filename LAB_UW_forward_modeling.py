#!/usr/bin/env python3

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from LAb_UW_functions import *
import json

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

    global sp_field, x
    # DEFINE GRID:
    ppt_for_minimum_len = 10       #  implement as a possible parameter option?
    x = make_grid_1D(grid_len = sum(sample_dimensions), cmin = c_gouge, fmax=freq_cut, ppt=ppt_for_minimum_len)
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
        plt.plot(t,sp_recorded)
        # plt.plot(t_OBS,waveform_SYNT*(norm/np.amax(waveform_SYNT)) - 15*(idx+1), label=f"pulse {idx}")
        
    waveform_SYNT = np.interp(t_OBS,t,sp_recorded)   # back to the same dt of the observed data


    L2norm = LA.norm(waveform_OBS[interval]-waveform_SYNT[interval],2)

    
    return L2norm


def load_waveform_json(infile_path: str) -> tuple:
    '''
    Load data and metadata from a JSON file.

    Args:
        infile_path (str): Path to the input JSON file.

    Returns:
        tuple: Tuple containing data and metadata.
    '''
    with open(infile_path, "r") as json_file:
        data_dict = json.load(json_file)

    # Retrieve metadata and data from the loaded dictionary
    data = np.array(data_dict["data"])
    metadata = data_dict["metadata"]

    return data,metadata


def calculate_scaling_factor(distance_from_source: float) -> float:
    scaling_factor = 1 / (4 * np.pi * (distance_from_source +1))
    scaling_factor = scaling_factor/np.amax(scaling_factor)
    return scaling_factor


def make_grid_1D(cmin: float, fmax: float, grid_len: float, ppt: int) -> np.ndarray:
    '''
    Create a 1D grid.

    Args:
        cmin (float): Minimum velocity.
        fmax (float): Maximum frequency.
        grid_len (float): Length of the grid.
        ppt (int): Points for the shortest wavelength.

    Returns:
        np.ndarray: 1D grid.
    '''
    lambda_min =  cmin/(fmax)               # [cm] minimum wavelength of the simulation
    dx = (lambda_min/ppt)                   # dx spacing x axes
    x  = np.arange(0,grid_len,dx)           # [cm] space coordinates
    if len(x)%2:                                    
        x = np.append(x,grid_len+dx)

    return x

def build_velocity_model(x: np.ndarray, sample_dimensions: tuple, x_trasmitter: float, x_receiver: float, pzt_width: float, pmma_width: float,
                         cmax: float, cgouge: float, cpzt: float, cpmma: float, plotting: bool = True) -> np.ndarray:
    global idx_gouge_1,idx_gouge_1,idx_grooves_side1,idx_grooves_side2, idx_grooves_central1, idx_grooves_central2
    # unpack sample dimensions
    side_block_1 = sample_dimensions[0]
    gouge_1 = sample_dimensions[1]
    central_block = sample_dimensions[2]
    gouge_2 = sample_dimensions[3]
    side_block_2 = sample_dimensions[4]

    h_grove = 0.1                           # [cm] grooves hight

    # find the indeces inside
    idx_gouge_1 =np.where((x>side_block_1) & (x<side_block_1 + gouge_1))[0]
    idx_gouge_2 =np.where((x>side_block_1+gouge_1+central_block) & (x<side_block_1 + gouge_1+central_block+gouge_2))[0]

    idx_grooves_side1 = np.where((x>side_block_1) & (x<side_block_1 + h_grove))[0]
    idx_grooves_central1 = np.where((x>side_block_1 + gouge_1 - h_grove) & (x<side_block_1 + gouge_1))[0]
    idx_grooves_central2 = np.where((x>side_block_1 + gouge_1 + central_block) & (x<side_block_1 + gouge_1 + central_block + h_grove))[0]    
    idx_grooves_side2 = np.where((x>side_block_1 + gouge_1 + central_block + gouge_2-h_grove) & (x<side_block_1 + gouge_1 + central_block + gouge_2))[0]

    idx_pzt_1 = np.where((x>x_trasmitter-pzt_width) & (x<x_trasmitter))[0]
    idx_pzt_2 = np.where((x>x_receiver) & (x<x_receiver + pzt_width))[0]

    idx_pmma_1 = np.where((x>x_trasmitter-pzt_width-pmma_width) & (x<x_trasmitter-pzt_width))[0]
    idx_pmma_2 = np.where((x>x_receiver+pzt_width) & (x<x_receiver + pzt_width + pmma_width))[0]

    idx_air_1 = np.where((x<x_trasmitter-pzt_width-pmma_width))[0]
    idx_air_2 = np.where((x>x_receiver+pzt_width+pmma_width))[0]

    c = cmax * np.ones(x.shape)   
    c[idx_gouge_1] = cgouge
    c[idx_gouge_2] = cgouge

    c[idx_grooves_side1] = 0.5*(cgouge + cmax)    # grooves are trinagles...
    c[idx_grooves_central1] = 0.5*(cgouge + cmax)
    c[idx_grooves_central2] = 0.5*(cgouge + cmax)
    c[idx_grooves_side2] = 0.5*(cgouge + cmax)

    c[idx_pmma_1] = cpmma
    c[idx_pmma_2] = cpmma

    c[idx_pzt_1] = cpzt
    c[idx_pzt_2] = cpzt

    c[idx_air_1] = 0
    c[idx_air_2] = 0

    if plotting:
        plt.figure()
        plt.plot(x,c)
        plt.close()
    return c

def synthetic_source_time_function(t: np.ndarray) -> np.ndarray:
    '''
    Generate a synthetic time function.

    Args:
        t (np.ndarray): Time axis.

    Returns:
        np.ndarray: Synthetic time function.
    '''
    winlen = 100
    amp_square = 5
    mask = amp_square * np.hanning(winlen)

    frequency = 12
    amplitude = 2
    sin_signal = amplitude*np.sin(2 * np.pi * frequency * t[:winlen])

    mask =  mask * sin_signal
    mask = np.diff(mask)

    src = np.zeros(len(t)) 
    src[0:np.size(mask)] = mask

    return (src)

def synthetic_source_spatial_function(x: np.ndarray, isx: int, sigma: float, plotting: bool = True) -> np.ndarray:
    '''
    Generate a synthetic spatial function.

    Args:
        x (np.ndarray): Spatial axis.
        isx (int): Index of the source.
        sigma (float): Sigma parameter for Gaussian.
        plotting (bool, optional): Whether to plot the spatial function. Defaults to True.

    Returns:
        np.ndarray: Synthetic spatial function.
    '''
    dx = x[1]-x[0]
    x0 = x[isx-1]
    src_x = np.exp(-1/sigma**2 *(x - x0)**2); src_x = src_x/np.amax(src_x)
    if plotting:
        fig = plt.figure()
        plt.plot(x,src_x)
    return src_x


def fourier_derivative_2nd(f: np.ndarray, dx: float) -> np.ndarray:
    '''
    Compute second spatial derivatives using Fourier transform.

    Args:
        f (np.ndarray): Input function.
        dx (float): Grid spacing.

    Returns:
        np.ndarray: Second spatial derivative.
    '''
    # Length of vector f
    nx = np.size(f)
    # Initialize k vector up to Nyquist wavenumber 
    kmax = np.pi / dx
    dk = kmax / (nx / 2)
    k = np.arange(float(nx))
    k[: int(nx/2)] = k[: int(nx/2)] * dk
     
    k[int(nx/2) :] = k[: int(nx/2)] - kmax
    
    # Fourier derivative
    ff = np.fft.fft(f)
    ff = (1j*k)**2 * ff
    df_num = np.real(np.fft.ifft(ff))
    return df_num

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


def make_movie(side_block_1: float, gouge_1: float, central_block: float, gouge_2: float, side_block_2: float) -> None:
    # MAKE A MOVIE FROM THE RECORDED FIELD
    movie_sampling = 10             # downsampling of the snapshot to speed up movie

    # Initialize figure and axes
    fig, ax = plt.subplots()
    ylim = 1.3*np.amax(sp_field)   # this is going to be around the higher value of pressure field
    ax.set(xlim=[x[0],x[-1]], ylim=[-ylim, ylim], xlabel='DDS Sample length [cm]', ylabel='Shear Wave Amplitude [.]')

    # this lines are just to shadow in different colours steel and gouge
    # idx_gouge_1 =np.where((x>side_block_1) & (x<side_block_1 + gouge_1))[0]
    # idx_gouge_2 =np.where((x>side_block_1+gouge_1+central_block) & (x<side_block_1 + gouge_1+central_block+gouge_2))[0]
    # ax.axvspan(x[0],x[idx_gouge_1[0]], color="lightsteelblue", alpha=0.5)
    # ax.axvspan(x[idx_gouge_1[-1]],x[idx_gouge_2[0]], color="lightsteelblue", alpha=0.5)
    # ax.axvspan(x[idx_gouge_2[-1]],x[-1], color="lightsteelblue", alpha=0.5)

    # Function to initialize the plot for movie
    def init():
        line.set_data([], [])
        return line,


    def movie_uw(frame):
        # Plot x against sp__movie
        line.set_xdata(x)
        line.set_ydata(sp_movie[frame])
        return line,
    # Initialize empty line object
    line, = ax.plot([], [], 'k', lw=1.5)


    sp_movie = sp_field[::movie_sampling]
    # Define the number of frames
    num_frames = len(sp_movie)

    # Create animation
    ani = animation.FuncAnimation(fig, movie_uw, frames=num_frames, blit=True, interval=20)

    # Save the animation as an MP4 file
    ani.save('simulation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


def compute_max_and_min_travel_time(side_block_1, x_transmitter, c_steel, gouge_1, c_min, c_max, central_block):
    """
    # TO MODIFY INCLUDING X_RECEIVER COORDINATES and DIFFERENCE IN LAYER THICKNESS BETWEEN BLOCKS
    Compute the maximum and minimum travel times for a signal transmitted through the sample assembly (DDS).

    Parameters:
        side_block_1 (float): Thickness of the side blocks used in the DDS configuration in [cm]
        x_transmitter (float): thickness of the trasnmitter PZT [cm]
        c_steel (float): Speed of wave propagation in steel (units: distance per time).
        gouge_1 (float): The thickness of the gouge layer.
        c_min (float): Minimum speed of signal propagation in the gouge layer.
        c_max (float): Maximum speed of signal propagation in the gouge layer.
        central_block (float): The thickness of the central block.

    Returns:
        tuple: A tuple containing the maximum and minimum travel times.

    The function computes the maximum and minimum travel times for a signal transmitted through
    a DDS configurations (2 side blocks, 1 central block, 2 gouge layers), where the signal
    travels from a transmitter located within the side block to a receiver located in the other.
    The travel times are calculated based on the given parameters such as
    distances, material properties, and signal propagation speeds.

    The maximum travel time is calculated considering the minimum speed of propagation in the gouge layer,
    while the minimum travel time is calculated considering the maximum speed of propagation in the gouge layer.
    """

    # Compute the maximum travel time
    max_travel_time = 2 * (side_block_1 - x_transmitter) / c_steel + 2 * gouge_1 / c_min + central_block / c_steel
    # Compute the minimum travel time
    min_travel_time = 2 * (side_block_1 - x_transmitter) / c_steel + 2 * gouge_1 / c_max + central_block / c_steel

    return max_travel_time, min_travel_time
# src/synthetic_data.py
import numpy as np
import matplotlib.pyplot as plt

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
    global idx_gouge_1,idx_gouge_1,idx_grooves_side1,idx_grooves_side2, idx_grooves_central1, idx_grooves_central2, idx_pzt_1, idx_pzt_2
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

# functions for testing

def synthetic_wavelets_in_noise(nsamples: int = 1000, noise_amplitude: float = 0.5, winlen: int = 100,
                                 square_start: int = 400, amp_square: float = 5, frequency: int = 4,
                                 amplitude: int = 1) -> np.ndarray:
    """
    Generate synthetic wavelets in noise.

    Args:
        nsamples (int): Number of samples.
        noise_amplitude (float): Amplitude of the noise.
        winlen (int): Length of the window.
        square_start (int): Start index of the square wave.
        amp_square (float): Amplitude of the square wave.
        frequency (int): Frequency of the sine wave.
        amplitude (int): Amplitude of the sine wave.

    Returns:
        np.ndarray: Synthetic waveform with wavelets in noise.
    """
    time = np.linspace(0, 10, nsamples)

    noise = np.random.normal(0, noise_amplitude, nsamples)

    signal = np.zeros(noise.shape)

    mask = amp_square * np.hanning(winlen)
    signal[square_start:square_start + winlen] = mask

    sin_signal = np.sin(2 * np.pi * frequency * time[square_start:square_start + winlen])
    signal[square_start:square_start + winlen] = mask * sin_signal

    waveform = signal + noise

    return waveform

# Example usage:
# waveform = synthetic_wavelets_in_noise()

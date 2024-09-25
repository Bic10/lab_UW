# src/synthetic_data.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple

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

def build_velocity_model(
    x: np.ndarray,
    sample_dimensions: Tuple[float, float, float, float, float],
    x_trasmitter: float,
    x_receiver: float,
    pzt_width: float,
    pmma_width: float,
    csteel: float,
    c_gouge: float,
    c_pzt: float,
    c_pmma: float,
    plotting: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 1D velocity model for a given sample configuration.

    This function constructs a 1D velocity profile based on the given sample dimensions,
    transmitter and receiver positions, and material properties. It assigns different
    velocities to different regions of the sample based on the materials present.

    Parameters:
    ----------
    x : np.ndarray
        The array representing the spatial coordinates (in cm) along the length of the sample.
    sample_dimensions : Tuple[float, float, float, float, float]
        A tuple containing the dimensions of different regions of the sample in cm:
        (side_block_1, gouge_1, central_block, gouge_2, side_block_2).
    x_trasmitter : float
        The position of the transmitter (in cm).
    x_receiver : float
        The position of the receiver (in cm).
    pzt_width : float
        The width of the piezoelectric transducers (in cm).
    pmma_width : float
        The width of the PMMA (polymethyl methacrylate) layers (in cm).
    csteel : float
        The velocity (in cm/s) of the steel blocks, assumed as the same for each block.
    c_gouge : Union[float, np.ndarray]
        The velocity (in cm/s) in the gouge regions. Can be a single float or a numpy array.
    c_pzt : float
        The velocity (in cm/s) in the piezoelectric transducer (PZT) regions.
    c_pmma : float
        The velocity (in cm/s) in the PMMA regions.
    plotting : bool, optional
        Whether to plot the velocity model (default is True).

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - c: np.ndarray - The velocity model array with assigned velocities.
        - idx_gouge_1: np.ndarray - Indices corresponding to the first gouge region.
        - idx_gouge_2: np.ndarray - Indices corresponding to the second gouge region.
        - idx_pzt_1: np.ndarray - Indices corresponding to the first PZT region.
        - idx_pzt_2: np.ndarray - Indices corresponding to the second PZT region.
    """
    
    # Unpack sample dimensions
    side_block_1, gouge_1, central_block, gouge_2, side_block_2 = sample_dimensions

    h_grove = 0.1  # [cm] grooves height. THIS TOO MUST BECOME AN INPUT, AS SOON AS WE HAVE A DATABASE THAT DESCRIBE THE GEOMETRY OF THE BLOCKS TOO

    # Find the indices inside
    idx_gouge_1 = np.where((x > side_block_1) & (x < side_block_1 + gouge_1))[0]
    idx_gouge_2 = np.where((x > side_block_1 + gouge_1 + central_block) & (x < side_block_1 + gouge_1 + central_block + gouge_2))[0]
    idx_grooves_side1 = np.where((x > side_block_1) & (x < side_block_1 + h_grove))[0]
    idx_grooves_central1 = np.where((x > side_block_1 + gouge_1 - h_grove) & (x < side_block_1 + gouge_1))[0]
    idx_grooves_central2 = np.where((x > side_block_1 + gouge_1 + central_block) & (x < side_block_1 + gouge_1 + central_block + h_grove))[0]
    idx_grooves_side2 = np.where((x > side_block_1 + gouge_1 + central_block + gouge_2 - h_grove) & (x < side_block_1 + gouge_1 + central_block + gouge_2))[0]
    idx_pzt_1 = np.where((x > x_trasmitter - pzt_width) & (x < x_trasmitter))[0]
    idx_pzt_2 = np.where((x > x_receiver) & (x < x_receiver + pzt_width))[0]
    idx_pmma_1 = np.where((x > x_trasmitter - pzt_width - pmma_width) & (x < x_trasmitter - pzt_width))[0]
    idx_pmma_2 = np.where((x > x_receiver + pzt_width) & (x < x_receiver + pzt_width + pmma_width))[0]
    idx_air_1 = np.where((x < x_trasmitter - pzt_width - pmma_width))[0]
    idx_air_2 = np.where((x > x_receiver + pzt_width + pmma_width))[0]

    c = csteel * np.ones(x.shape)

#   Build homogeneus model
    c[idx_gouge_1] = c_gouge
    c[idx_gouge_2] = c_gouge
    c[idx_grooves_side1] = 0.5 * (c_gouge + csteel)  # grooves are approximately rectangular triangles...
    c[idx_grooves_central1] = 0.5 * (c_gouge + csteel)
    c[idx_grooves_central2] = 0.5 * (c_gouge + csteel)
    c[idx_grooves_side2] = 0.5 * (c_gouge + csteel)

#   This part regard the piezoelectric sensors
    c[idx_pmma_1] = c_pmma
    c[idx_pmma_2] = c_pmma
    c[idx_pzt_1] = c_pzt
    c[idx_pzt_2] = c_pzt
    c[idx_air_1] = 0
    c[idx_air_2] = 0

    if plotting:
        plt.figure()
        plt.plot(x, c)
        plt.close()
    return c, idx_gouge_1, idx_gouge_2, idx_pzt_1, idx_pzt_2

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
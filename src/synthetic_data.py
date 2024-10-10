# src/synthetic_data.py

import numpy as np
from typing import Union, Tuple
from plotting import plot_velocity_model, plot_synthetic_spatial_function

def make_grid_1D(cmin: float, fmax: float, grid_len: float, ppt: int) -> np.ndarray:
    '''
    Create a 1D grid.

    Args:
        cmin (float): Minimum velocity.
        fmax (float): Maximum frequency.
        grid_len (float): Length of the grid.
        ppt (int): Points per the shortest wavelength.

    Returns:
        np.ndarray: 1D grid.
    '''
    lambda_min = cmin / fmax               # [cm] minimum wavelength of the simulation
    dx = lambda_min / ppt                  # dx spacing x axes
    x = np.arange(0, grid_len, dx)         # [cm] space coordinates
    if len(x) % 2:
        x = np.append(x, grid_len + dx)

    return x

def build_velocity_model(
    x: np.ndarray,
    sample_dimensions: Tuple[float, float, float, float, float],
    x_transmitter: float,
    x_receiver: float,
    pzt_layer_width: float,
    pmma_layer_width: float,
    h_groove_side: float,
    h_groove_central: float,
    steel_velocity: float,
    gouge_velocity: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    pzt_velocity: float,
    pmma_velocity: float,
    plotting: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Build a 1D velocity model for a given sample configuration, with smoothing zones around
    the transmitter and receiver positions between pzt_velocity and steel_velocity.

    Parameters:
    ----------
    x : np.ndarray
        The array representing the spatial coordinates (in cm) along the length of the sample.
        x = 0 corresponds to the external edge of Side Block 1.
    sample_dimensions : Tuple[float, float, float, float, float]
        A tuple containing the dimensions of different regions of the sample in cm:
        (side_block_1_total, gouge_1_length, central_block_total, gouge_2_length, side_block_2_total).
        The block sizes exclude the heights of their grooves.
    x_transmitter : float
        The position of the transmitter (in cm) from the external edge of Side Block 1.
        It corresponds to the contact between the PZT layer and Side Block 1.
    x_receiver : float
        The position of the receiver (in cm) from the external edge of Side Block 1.
    pzt_layer_width : float
        The width of the piezoelectric transducers (in cm).
    pmma_layer_width : float
        The width of the PMMA layers (in cm).
    h_groove_side : float
        The height of the grooves (in cm).
    h_groove_central : float
        The height of the central block's grooves (in cm).
    steel_velocity : float
        The velocity (in cm/μs) of the steel blocks, assumed to be the same for each block.
    gouge_velocity : Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]
        A tuple containing the velocities (in cm/μs) in the gouge regions.
        Each element can be a single float or a numpy array representing the 1D velocity profile.
    pzt_velocity : float
        The velocity (in cm/μs) in the piezoelectric transducer (PZT) regions.
    pmma_velocity : float
        The velocity (in cm/μs) in the PMMA regions.
    plotting : bool, optional
        Whether to plot the velocity model (default is True).

    Returns:
    -------
    Tuple[np.ndarray, dict]
        - c: np.ndarray - The velocity model array with assigned velocities.
        - idx_dict: dict - A dictionary containing indices for different regions.
    """
    # Unpack sample dimensions
    side_block_1_total, gouge_1_length, central_block_total, gouge_2_length, side_block_2_total = sample_dimensions

    # Adjust block lengths to exclude groove heights
    side_block_1 = side_block_1_total - h_groove_side
    side_block_2 = side_block_2_total - h_groove_side
    central_block = central_block_total - 2 * h_groove_central  # Subtract grooves on both sides

    # Compute cumulative positions along the sample
    layer_thicknesses = [
        pmma_layer_width,
        pzt_layer_width,
        side_block_1 - x_transmitter,
        h_groove_side,
        gouge_1_length,
        h_groove_central,
        central_block,
        h_groove_central,
        gouge_2_length,
        h_groove_side,
        side_block_2 - x_receiver,
        pzt_layer_width,
        pmma_layer_width
    ]
    layer_starts = np.concatenate(([0.0], np.cumsum(layer_thicknesses)))

    # Define indices for each region
    idx_dict = {}

    idx_dict['pmma_1'] = np.where((x >= layer_starts[0]) & (x < layer_starts[1]))[0]
    idx_dict['pzt_1'] = np.where((x >= layer_starts[1]) & (x < layer_starts[2]))[0]
    idx_dict['side_block_1'] = np.where((x >= layer_starts[2]) & (x < layer_starts[3]))[0]
    idx_dict['groove_sb1'] = np.where((x >= layer_starts[3]) & (x < layer_starts[4]))[0]
    idx_dict['gouge_1'] = np.where((x >= layer_starts[4]) & (x < layer_starts[5]))[0]
    idx_dict['groove_cb1'] = np.where((x >= layer_starts[5]) & (x < layer_starts[6]))[0]
    idx_dict['central_block'] = np.where((x >= layer_starts[6]) & (x < layer_starts[7]))[0]
    idx_dict['groove_cb2'] = np.where((x >= layer_starts[7]) & (x < layer_starts[8]))[0]
    idx_dict['gouge_2'] = np.where((x >= layer_starts[8]) & (x < layer_starts[9]))[0]
    idx_dict['groove_sb2'] = np.where((x >= layer_starts[9]) & (x < layer_starts[10]))[0]
    idx_dict['side_block_2'] = np.where((x >= layer_starts[10]) & (x < layer_starts[11]))[0]
    idx_dict['pzt_2'] = np.where((x >= layer_starts[11]) & (x < layer_starts[12]))[0]
    idx_dict['pmma_2'] = np.where((x >= layer_starts[12]) & (x <= layer_starts[13]))[0]

    # Initialize velocity model
    c = steel_velocity * np.ones_like(x)

    # Assign velocities
    c[idx_dict['pmma_1']] = pmma_velocity
    c[idx_dict['pzt_1']] = pzt_velocity
    # Side Block 1 remains steel_velocity

    # Assign velocities in Groove SB1
    groove_sb1_length = len(idx_dict['groove_sb1'])
    if groove_sb1_length > 0:
        groove_sb1_start_vel = steel_velocity
        groove_sb1_end_vel = gouge_velocity[0][0] if isinstance(gouge_velocity[0], np.ndarray) else gouge_velocity[0]
        c[idx_dict['groove_sb1']] = np.linspace(groove_sb1_start_vel, groove_sb1_end_vel, groove_sb1_length)

    # Assign velocities in Gouge Layer 1
    if isinstance(gouge_velocity[0], np.ndarray):
        if len(gouge_velocity[0]) != len(idx_dict['gouge_1']):
            raise ValueError("Length of gouge_velocity[0] does not match the size of gouge_1 region.")
        c[idx_dict['gouge_1']] = gouge_velocity[0]
    else:
        c[idx_dict['gouge_1']] = gouge_velocity[0]

    # Assign velocities in Groove CB1
    groove_cb1_length = len(idx_dict['groove_cb1'])
    if groove_cb1_length > 0:
        groove_cb1_start_vel = gouge_velocity[0][-1] if isinstance(gouge_velocity[0], np.ndarray) else gouge_velocity[0]
        groove_cb1_end_vel = steel_velocity
        c[idx_dict['groove_cb1']] = np.linspace(groove_cb1_start_vel, groove_cb1_end_vel, groove_cb1_length)

    # Assign velocities in Central Block
    c[idx_dict['central_block']] = steel_velocity

    # Assign velocities in Groove CB2
    groove_cb2_length = len(idx_dict['groove_cb2'])
    if groove_cb2_length > 0:
        groove_cb2_start_vel = steel_velocity
        groove_cb2_end_vel = gouge_velocity[1][0] if isinstance(gouge_velocity[1], np.ndarray) else gouge_velocity[1]
        c[idx_dict['groove_cb2']] = np.linspace(groove_cb2_start_vel, groove_cb2_end_vel, groove_cb2_length)

    # Assign velocities in Gouge Layer 2
    if isinstance(gouge_velocity[1], np.ndarray):
        if len(gouge_velocity[1]) != len(idx_dict['gouge_2']):
            raise ValueError("Length of gouge_velocity[1] does not match the size of gouge_2 region.")
        c[idx_dict['gouge_2']] = gouge_velocity[1]
    else:
        c[idx_dict['gouge_2']] = gouge_velocity[1]

    # Assign velocities in Groove SB2
    groove_sb2_length = len(idx_dict['groove_sb2'])
    if groove_sb2_length > 0:
        groove_sb2_start_vel = gouge_velocity[1][-1] if isinstance(gouge_velocity[1], np.ndarray) else gouge_velocity[1]
        groove_sb2_end_vel = steel_velocity
        c[idx_dict['groove_sb2']] = np.linspace(groove_sb2_start_vel, groove_sb2_end_vel, groove_sb2_length)

    # Side Block 2 remains steel_velocity
    c[idx_dict['pzt_2']] = pzt_velocity
    c[idx_dict['pmma_2']] = pmma_velocity

    # Apply smoothing between pzt_velocity and steel_velocity around transmitter and receiver
    # Transmitter smoothing
    transmitter_smoothing_start = layer_starts[1]  # Start of PZT layer
    transmitter_smoothing_end = layer_starts[2] + pzt_layer_width  # End of smoothing region
    transmitter_indices = np.where((x >= transmitter_smoothing_start) & (x < transmitter_smoothing_end))[0]
    if len(transmitter_indices) > 0:
        c[transmitter_indices] = np.linspace(pzt_velocity, steel_velocity, len(transmitter_indices))

    # Receiver smoothing
    receiver_smoothing_start = layer_starts[11]  # Start of PZT layer
    receiver_smoothing_end = layer_starts[12] + pzt_layer_width  # End of smoothing region
    receiver_indices = np.where((x >= receiver_smoothing_start) & (x < receiver_smoothing_end))[0]
    if len(receiver_indices) > 0:
        c[receiver_indices] = np.linspace(steel_velocity, pzt_velocity, len(receiver_indices))

    # Plotting with shaded areas
    # Plotting with shaded areas
    if plotting:
        plot_velocity_model(
            x=x,
            c=c,
            layer_starts=layer_starts,
            pzt_layer_width=pzt_layer_width,
            pmma_layer_width=pmma_layer_width,
            outfile_path=None,  # You can specify a path if you want to save the plot
        )

    return c, idx_dict

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
    sin_signal = amplitude * np.sin(2 * np.pi * frequency * t[:winlen])

    mask = mask * sin_signal
    mask = np.diff(mask)

    src = np.zeros(len(t))
    src[0:np.size(mask)] = mask

    return src

def synthetic_spatial_function(
    x: np.ndarray,
    position: float,
    sigma: float,
    normalize: bool = True,
    plotting: bool = False,
) -> np.ndarray:
    '''
    Generate a synthetic spatial function using a Gaussian.

    Args:
        x (np.ndarray): Spatial axis.
        position (float): Physical position of the source in the same units as x.
        sigma (float): Sigma parameter for the Gaussian, in the same units as x.
        normalize (bool, optional): Whether to normalize the maximum amplitude to 1. Defaults to True.
        plotting (bool, optional): Whether to plot the spatial function. Defaults to True.

    Returns:
        np.ndarray: Synthetic spatial function.
    '''
    spatial_gaussian = np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
    if normalize:
        spatial_gaussian = spatial_gaussian / np.amax(np.abs(spatial_gaussian))
    if plotting:
        plot_synthetic_spatial_function(
                                    x=x,
                                    spatial_function=spatial_gaussian,
                                    outfile_path=None,  # Specify if needed
                                )

    return spatial_gaussian


def synthetic_wavelets_in_noise(
    nsamples: int = 1000,
    noise_amplitude: float = 0.5,
    winlen: int = 100,
    square_start: int = 400,
    amp_square: float = 5,
    frequency: int = 4,
    amplitude: int = 1
) -> np.ndarray:
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

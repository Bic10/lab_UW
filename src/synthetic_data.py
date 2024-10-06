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
    h_groove: float,
    steel_velocity: float,
    gouge_velocity: Union[float, np.ndarray],
    pzt_velocity: float,
    pmma_velocity: float,
    plotting: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Build a 1D velocity model for a given sample configuration.

    Parameters:
    ----------
    x : np.ndarray
        The array representing the spatial coordinates (in cm) along the length of the sample.
        x = 0 corresponds to the external edge of Side Block 1.
    sample_dimensions : Tuple[float, float, float, float, float]
        A tuple containing the dimensions of different regions of the sample in cm:
        (side_block_1_total, gouge_1, central_block_total, gouge_2, side_block_2_total).
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
    h_groove : float
        The height of the grooves (in cm).
    steel_velocity : float
        The velocity (in cm/μs) of the steel blocks, assumed to be the same for each block.
    gouge_velocity : Union[float, np.ndarray]
        The velocity (in cm/μs) in the gouge regions. Can be a single float or a numpy array.
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
    side_block_1_total, gouge_1, central_block_total, gouge_2, side_block_2_total = sample_dimensions

    # Adjust block lengths to exclude groove heights
    side_block_1 = side_block_1_total - h_groove
    side_block_2 = side_block_2_total - h_groove
    central_block = central_block_total - 2 * h_groove  # Subtract grooves on both sides

    # Compute cumulative positions along the sample
    layer_thicknesses = [
        pmma_layer_width,
        pzt_layer_width,
        side_block_1 - x_transmitter,
        h_groove,
        gouge_1,
        h_groove,
        central_block,
        h_groove,
        gouge_2,
        h_groove,
        side_block_2 - x_receiver,
        pzt_layer_width,
        pmma_layer_width
    ]
    layer_starts = np.concatenate(([0.0], np.cumsum(layer_thicknesses)))

    # Positions of different layers 
    x_pmma_1_start = layer_starts[0]
    x_pmma_1_end = layer_starts[1]

    x_pzt_1_start = layer_starts[1]
    x_pzt_1_end = layer_starts[2]

    x_side_block_1_start = layer_starts[2]
    x_side_block_1_end = layer_starts[3]

    x_groove_sb1_start = layer_starts[3]
    x_groove_sb1_end = layer_starts[4]

    x_gouge_1_start = layer_starts[4]
    x_gouge_1_end = layer_starts[5]

    x_groove_cb1_start = layer_starts[5]
    x_groove_cb1_end = layer_starts[6]

    x_central_block_start = layer_starts[6]
    x_central_block_end = layer_starts[7]

    x_groove_cb2_start = layer_starts[7]
    x_groove_cb2_end = layer_starts[8]

    x_gouge_2_start = layer_starts[8]
    x_gouge_2_end = layer_starts[9]

    x_groove_sb2_start = layer_starts[9]
    x_groove_sb2_end = layer_starts[10]

    x_side_block_2_start = layer_starts[10]
    x_side_block_2_end = layer_starts[11]

    x_pzt_2_start = layer_starts[11]
    x_pzt_2_end = layer_starts[12]

    x_pmma_2_start = layer_starts[12]
    x_pmma_2_end = layer_starts[13]

    # Initialize velocity model
    c = steel_velocity * np.ones_like(x)

    # Define indices for each region
    idx_dict = {}

    # Indices for PMMA and PZT layers inside Side Block 1
    idx_dict['pmma_1'] = np.where((x >= x_pmma_1_start) & (x < x_pmma_1_end))[0]
    idx_dict['pzt_1'] = np.where((x >= x_pzt_1_start) & (x < x_pzt_1_end))[0]

    # Indices for Side Block 1
    idx_dict['side_block_1'] = np.where((x >= x_side_block_1_start) & (x < x_side_block_1_end))[0]

    # Groove of Side Block 1
    idx_dict['groove_sb1'] = np.where((x >= x_groove_sb1_start) & (x < x_groove_sb1_end))[0]

    # Gouge Layer 1
    idx_dict['gouge_1'] = np.where((x >= x_gouge_1_start) & (x < x_gouge_1_end))[0]

    # Groove of Central Block (First Side)
    idx_dict['groove_cb1'] = np.where((x >= x_groove_cb1_start) & (x < x_groove_cb1_end))[0]

    # Central Block
    idx_dict['central_block'] = np.where((x >= x_central_block_start) & (x < x_central_block_end))[0]

    # Groove of Central Block (Second Side)
    idx_dict['groove_cb2'] = np.where((x >= x_groove_cb2_start) & (x < x_groove_cb2_end))[0]

    # Gouge Layer 2
    idx_dict['gouge_2'] = np.where((x >= x_gouge_2_start) & (x < x_gouge_2_end))[0]

    # Groove of Side Block 2
    idx_dict['groove_sb2'] = np.where((x >= x_groove_sb2_start) & (x < x_groove_sb2_end))[0]

    # Indices for Side Block 2
    idx_dict['side_block_2'] = np.where((x >= x_side_block_2_start) & (x < x_side_block_2_end))[0]

    # Indices for PMMA and PZT layers inside Side Block 2
    idx_dict['pzt_2'] = np.where((x >= x_pzt_2_start) & (x < x_pzt_2_end))[0]
    idx_dict['pmma_2'] = np.where((x >= x_pmma_2_start) & (x < x_pmma_2_end))[0]

    # Assign velocities
    c[idx_dict['pmma_1']] = pmma_velocity
    c[idx_dict['pzt_1']] = pzt_velocity
    # Side Block 1 remains steel_velocity
    c[idx_dict['groove_sb1']] = 0.5 * (gouge_velocity + steel_velocity)
    c[idx_dict['gouge_1']] = gouge_velocity
    c[idx_dict['groove_cb1']] = 0.5 * (gouge_velocity + steel_velocity)
    c[idx_dict['central_block']] = steel_velocity
    c[idx_dict['groove_cb2']] = 0.5 * (gouge_velocity + steel_velocity)
    c[idx_dict['gouge_2']] = gouge_velocity
    c[idx_dict['groove_sb2']] = 0.5 * (gouge_velocity + steel_velocity)
    # Side Block 2 remains steel_velocity
    c[idx_dict['pzt_2']] = pzt_velocity
    c[idx_dict['pmma_2']] = pmma_velocity

    # Plotting with shaded areas
    if plotting:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x, c, label='Velocity Model', color='black')

        layers = [
            {'name': 'PMMA Layer 1', 'start': x_pmma_1_start, 'end': x_pmma_1_end, 'color': 'lightblue'},
            {'name': 'PZT Layer 1', 'start': x_pzt_1_start, 'end': x_pzt_1_end, 'color': 'violet'},
            {'name': 'Side Block 1', 'start': x_side_block_1_start, 'end': x_side_block_1_end, 'color': 'grey'},
            {'name': 'Groove SB1', 'start': x_groove_sb1_start, 'end': x_groove_sb1_end, 'color': 'lightgrey'},
            {'name': 'Gouge Layer 1', 'start': x_gouge_1_start, 'end': x_gouge_1_end, 'color': 'sandybrown'},
            {'name': 'Groove CB1', 'start': x_groove_cb1_start, 'end': x_groove_cb1_end, 'color': 'lightgrey'},
            {'name': 'Central Block', 'start': x_central_block_start, 'end': x_central_block_end, 'color': 'grey'},
            {'name': 'Groove CB2', 'start': x_groove_cb2_start, 'end': x_groove_cb2_end, 'color': 'lightgrey'},
            {'name': 'Gouge Layer 2', 'start': x_gouge_2_start, 'end': x_gouge_2_end, 'color': 'sandybrown'},
            {'name': 'Groove SB2', 'start': x_groove_sb2_start, 'end': x_groove_sb2_end, 'color': 'lightgrey'},
            {'name': 'Side Block 2', 'start': x_side_block_2_start, 'end': x_side_block_2_end, 'color': 'grey'},
            {'name': 'PZT Layer 2', 'start': x_pzt_2_start, 'end': x_pzt_2_end, 'color': 'violet'},
            {'name': 'PMMA Layer 2', 'start': x_pmma_2_start, 'end': x_pmma_2_end, 'color': 'lightblue'},
        ]

        labels_used = set()

        for layer in layers:
            # Avoid duplicate labels in legend
            if layer['name'] in labels_used:
                label = None
            else:
                label = layer['name']
                labels_used.add(layer['name'])
            ax.axvspan(layer['start'], layer['end'], color=layer['color'], alpha=0.3, label=label)

        # Plot transmitter and receiver positions
        ax.axvline(pzt_layer_width + pmma_layer_width, color="red", linestyle='-', label='Transmitter')
        ax.axvline(x[-1] -pzt_layer_width - pmma_layer_width, color="green", linestyle='-', label='Receiver')

        ax.set_title("Velocity Model with Layers")
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Velocity (cm/μs)")
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.show()

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

def synthetic_source_spatial_function(
    x: np.ndarray,
    source_index: int,
    sigma: float,
    plotting: bool = True
) -> np.ndarray:
    '''
    Generate a synthetic spatial function.

    Args:
        x (np.ndarray): Spatial axis.
        source_index (int): Index of the source.
        sigma (float): Sigma parameter for Gaussian.
        plotting (bool, optional): Whether to plot the spatial function. Defaults to True.

    Returns:
        np.ndarray: Synthetic spatial function.
    '''
    dx = x[1] - x[0]
    x0 = x[source_index - 1]
    src_x = np.exp(-1 / sigma ** 2 * (x - x0) ** 2)
    src_x = src_x / np.amax(src_x)
    if plotting:
        plt.figure()
        plt.plot(x, src_x)
        plt.title("Synthetic Spatial Function")
        plt.xlabel("Position (cm)")
        plt.ylabel("Amplitude")
        plt.show()
    return src_x

# Functions for testing

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

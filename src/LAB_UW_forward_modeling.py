# src/LAB_UW_forward_modeling.py

import numpy as np
from numpy import linalg as LA
from plotting import plot_simulation_waveform
from typing import Union
from scipy.signal.windows import kaiser

from helpers import *
from synthetic_data import *


def DDS_UW_simulation(
    observed_time: np.ndarray,
    observed_waveform: np.ndarray,
    pulse_time: np.ndarray,
    pulse_waveform: np.ndarray,
    sample_dimensions: tuple,
    h_groove: float,
    frequency_cutoff: float,
    transmitter_position: float,
    receiver_position: float,
    pzt_layer_width: float,
    pmma_layer_width: float,
    steel_velocity: float,
    gouge_velocity: Union[float, np.ndarray],
    pzt_velocity: float,
    pmma_velocity: float,
    normalize_waveform: bool = True,
    enable_plotting: bool = False,
    make_movie: bool = False,
    movie_settings: dict = None,
    movie_output_path: str = "DDS_simulation_movie.mp4"
) -> np.ndarray:
    """
    Simulate ultrasonic wave propagation through a sample and compute the synthetic waveform.
    """

    # Compute the spatial grid. Must add the actual length of the useful grid: 
    # the sensors are elastic media and the hole behind the sensors are of air 
    # the trasmitter and receiver position should be passed as their "pzt_depth" respect to the external edge of their block
    total_length = np.sum(sample_dimensions) + 2*pmma_layer_width + 2*pzt_layer_width -  (transmitter_position + receiver_position)

    # Cread grid according to numerical dispersion criteria for minimum wavelength
    spatial_axis, dx, num_x = compute_grid(
        total_length_to_simulate = total_length,
        min_velocity=gouge_velocity,
        frequency_cutoff=frequency_cutoff,
    )

    # Build the velocity model
    velocity_model, idx_dict = build_velocity_model(
        x=spatial_axis,
        sample_dimensions=sample_dimensions,
        h_groove=h_groove,
        x_transmitter=transmitter_position,
        x_receiver=receiver_position,
        pzt_layer_width=pzt_layer_width,
        pmma_layer_width=pmma_layer_width,
        steel_velocity=steel_velocity,
        gouge_velocity=gouge_velocity,
        pzt_velocity=pzt_velocity,
        pmma_velocity=pmma_velocity,
        plotting=enable_plotting
    )

    # Prepare time variables
    simulation_time, dt, num_t = prepare_time_variables(
        observed_time=observed_time,
        dx=dx,
        max_velocity=steel_velocity
    )

    # Interpolate source time function
    interpolated_pulse = interpolate_source(
        pulse_time=pulse_time,
        pulse_waveform=pulse_waveform,
        dt=dt
    )
    src_time_function = np.zeros(len(simulation_time))
    src_time_function[:len(interpolated_pulse)] = interpolated_pulse

    # Create source and receiver spatial functions
    # For the way the spatial axis is created now and the positions are passed, the transmitter and the receiver are a bit tricki
    transmitter_position_respect_to_spatial_axis = 0.5*pzt_layer_width + pmma_layer_width
    src_spatial_function = arbitrary_position_filter(
        spatial_axis=spatial_axis,
        dx=dx,
        position=transmitter_position_respect_to_spatial_axis,
        radius=10
    )
    # since we already use the receiver_position to build the grid, its place is just at the edge between pzt and block...
    receiver_position_respect_to_spatial_axis = total_length - 0.5*pzt_layer_width - pmma_layer_width

    rec_spatial_function = arbitrary_position_filter(
        spatial_axis=spatial_axis,
        dx=dx,
        position=receiver_position_respect_to_spatial_axis,
        radius=10
    )


    # Compute synthetic wavefield
    synthetic_field = pseudospectral_1D(
        num_x=num_x,
        delta_x=dx,
        num_t=num_t,
        delta_t=dt,
        source_spatial=src_spatial_function,
        source_time=src_time_function,
        velocity_model=velocity_model
    )

    # Record the simulated wavefield at the receiver position
    simulated_waveform = np.sum(synthetic_field * rec_spatial_function, axis=-1)

    if normalize_waveform:
        amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
        simulated_waveform *= amplitude_scale

    # Interpolate the synthetic waveform onto the observed time axis
    synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

    if enable_plotting:
        plot_simulation_waveform(observed_time, synthetic_waveform, observed_waveform)
        # plt.figure()
        # plt.plot(src_time_function)
        # plt.title("Source Time Function")
        # plt.xlabel("Time [μs]")
        # plt.ylabel("Amplitude")
        # plt.show()

        plt.figure()
        plt.plot(spatial_axis,src_spatial_function)
        plt.plot(spatial_axis,rec_spatial_function)
        plt.show()

    if make_movie:
        # Create and save the movie using the simulation outputs
        make_movie_from_simulation(
            outfile_path=movie_output_path,
            x=spatial_axis,
            t=simulation_time,
            sp_field=synthetic_field,
            sp_recorded=simulated_waveform,
            sample_dimensions=sample_dimensions,
            idx_dict=idx_dict,
            settings=movie_settings
        )

    return synthetic_waveform


def pseudospectral_1D(
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial: np.ndarray,
    source_time: np.ndarray,
    velocity_model: np.ndarray,
    compute_derivative: bool = False,
    reverse_time: bool = False,
) -> Union[np.ndarray, tuple]:
    """
    Perform pseudospectral modeling for 1D wave propagation (forward or adjoint).
    """
    # Initialize wavefield arrays
    wavefield_current = np.zeros(num_x)
    wavefield_future = np.zeros(num_x)
    wavefield_past = np.zeros(num_x)
    wavefield = np.zeros((num_t, num_x))
    
    if compute_derivative:
        derivative_wavefield = np.zeros(wavefield.shape)  # Derivative wavefield

    # Time-stepping loop
    time_steps = range(num_t)
    if reverse_time:
        time_steps = reversed(time_steps)

    for time_step in time_steps:
        # Second spatial derivative using Fourier method
        second_derivative = fourier_derivative_2nd(wavefield_current, delta_x)

        # Update wavefield using the finite difference time stepping
        wavefield_future = (
            2 * wavefield_current - wavefield_past
            + (velocity_model ** 2) * (delta_t ** 2) * second_derivative
        )

        # Add source contribution
        wavefield_future += source_spatial * source_time[time_step] * (delta_t ** 2)

        # Update wavefield states for next iteration
        wavefield_past = wavefield_current.copy()
        wavefield_current = wavefield_future.copy()

        # Apply boundary conditions (e.g., Dirichlet boundaries)
        wavefield_current[0] = 0
        wavefield_current[-1] = 0

        # Store wavefield at current time step
        if reverse_time:
            wavefield[num_t - 1 - time_step, :] = wavefield_current
        else:
            wavefield[time_step, :] = wavefield_current

        # Compute derivative wavefield if required
        if compute_derivative:
            derivative = (wavefield_future - 2 * wavefield_current + wavefield_past) / (delta_t ** 2)
            if reverse_time:
                derivative_wavefield[num_t - 1 - time_step, :] = derivative
            else:
                derivative_wavefield[time_step, :] = derivative

    if compute_derivative:
        return wavefield, derivative_wavefield
    else:
        return wavefield


def compute_grid(
    total_length_to_simulate: float,
    min_velocity: Union[float, np.ndarray],
    frequency_cutoff: float,
    points_per_min_wavelength: int = 10,
    x_start: float = None,
    x_end: float = None
) -> tuple:
    """
    Compute the 1D spatial grid for the simulation based on the minimum velocity and frequency.
    """
    if x_start is None:
        x_start = 0.0  # Default start position
    if x_end is None:
        x_end = total_length_to_simulate
    grid_length = x_end - x_start

    # Use make_grid_1D function, adjusted to grid_length
    spatial_axis_relative = make_grid_1D(
        grid_len=grid_length,
        cmin=min_velocity,
        fmax=frequency_cutoff,
        ppt=points_per_min_wavelength
    )
    # Shift spatial_axis to start at x_start
    spatial_axis = spatial_axis_relative + x_start
    dx = spatial_axis[1] - spatial_axis[0]
    num_x = len(spatial_axis)
    return spatial_axis, dx, num_x


def prepare_time_variables(
    observed_time: np.ndarray,
    dx: float,
    max_velocity: float,
    cfl_factor: float = 0.5
) -> tuple:
    """
    Prepare the time variables for the simulation based on the spatial grid and maximum velocity.
    """
    # Calculate the raw time step based on CFL condition
    dt_raw = cfl_factor * dx / max_velocity

    # Extract the data sampling rate from observed_time
    dt_obs = observed_time[1] - observed_time[0]

    # Find the largest submultiple of dt_obs smaller than dt_raw
    dt = dt_obs / np.ceil(dt_obs / dt_raw)

    # Create the time axis for the simulation
    simulation_time = np.arange(0, observed_time[-1], dt)
    num_t = len(simulation_time)

    return simulation_time, dt, num_t


def interpolate_source(
    pulse_time: np.ndarray,
    pulse_waveform: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Interpolate the source time function to match the simulation time discretization.
    """
    interpolated_pulse_time = np.arange(pulse_time[0], pulse_time[-1], dt)
    interpolated_pulse = np.interp(interpolated_pulse_time, pulse_time, pulse_waveform)
    return interpolated_pulse


def compute_misfit(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    misfit_interval: slice
) -> float:
    """
    Compute the L2 norm misfit between observed and synthetic waveforms over a specified interval.
    """
    return LA.norm(observed_waveform[misfit_interval] - synthetic_waveform[misfit_interval], 2)


def arbitrary_position_filter(
    spatial_axis: np.ndarray,
    dx: float,
    position: float,
    radius: int
) -> np.ndarray:
    """
    Create a Kaiser-windowed sinc filter for arbitrary source/receiver positioning on a 1D grid.
    """
    # Normalize positions to grid indices
    grid_indices = spatial_axis / dx
    num_points = len(grid_indices)
    position_index = position / dx

    # Create sinc function centered at the arbitrary position
    sinc_function = np.sinc(grid_indices - position_index)

    # Apply Kaiser window to the sinc function
    beta = 6.0  # Kaiser window parameter
    kaiser_window = kaiser(2 * radius + 1, beta)

    # Find the grid node closest to the desired position
    closest_node_index = np.argmin(np.abs(grid_indices - position_index))

    # Apply windowed sinc filter centered on the position_index
    start_idx = max(0, closest_node_index - radius)
    end_idx = min(num_points, closest_node_index + radius + 1)
    
    windowed_sinc = np.zeros_like(sinc_function)
    windowed_sinc[start_idx:end_idx] = sinc_function[start_idx:end_idx] * kaiser_window[:end_idx - start_idx]

    return windowed_sinc


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

def make_movie_from_simulation(
    outfile_path: str,
    x: np.ndarray,
    t: np.ndarray,
    sp_field: np.ndarray,
    sp_recorded: np.ndarray,
    sample_dimensions: tuple,
    idx_dict: dict,
    settings: dict = None
):
    if settings is None:
        settings = {'figure_size': (12, 6), 'fontsize_title': 16, 'fontsize_labels': 14}

    movie_sampling = 10  # Downsampling of the snapshot to speed up movie

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=settings['figure_size'], gridspec_kw={'width_ratios': [10, 1]})
    ylim = 1.3 * np.amax(np.abs(sp_field))

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([-ylim, ylim])
    ax.set_title("Ultrasonic Wavefield in DDS Experiment", fontsize=settings['fontsize_title'], color='darkslategray')
    ax.set_xlabel("Sample Length [cm]", fontsize=settings['fontsize_labels'], color='darkslategray')
    ax.set_ylabel('Relative Shear Wave Amplitude', fontsize=settings['fontsize_labels'], color='darkslategray')

    # Shading layers based on indices in idx_dict
    layers = [
        {'name': 'Gouge Layer 1', 'idx': idx_dict['gouge_1'], 'color': 'sandybrown'},
        {'name': 'Gouge Layer 2', 'idx': idx_dict['gouge_2'], 'color': 'sandybrown'},
        {'name': 'PZT Layer 1', 'idx': idx_dict['pzt_1'], 'color': 'violet'},
        {'name': 'PZT Layer 2', 'idx': idx_dict['pzt_2'], 'color': 'violet'},
        {'name': 'Grooves', 'idx': np.concatenate([idx_dict['groove_sb1'], idx_dict['groove_cb1'], idx_dict['groove_cb2'], idx_dict['groove_sb2']]), 'color': 'lightgrey'},
        {'name': 'Steel Blocks', 'idx': np.concatenate([idx_dict['side_block_1'], idx_dict['central_block'], idx_dict['side_block_2']]), 'color': 'lightsteelblue'},
    ]

    for layer in layers:
        ax.axvspan(x[layer['idx'][0]], x[layer['idx'][-1]], color=layer['color'], alpha=0.3, label=layer['name'])

    # Plot transmitter and receiver positions
    x_tr = x[idx_dict['pzt_1'][-1]]
    y_tr = 0
    pzt_width = x[idx_dict['pzt_1'][-1]] - x[idx_dict['pzt_1'][0]]
    pzt_height = 4 * pzt_width
    ax.add_patch(Rectangle((x_tr -  pzt_width, y_tr - pzt_height / 2), pzt_width, pzt_height, color='teal'))
    ax.text(x_tr - pzt_width / 2, y_tr - pzt_height, 'Transmitter', ha='center', fontsize=settings['fontsize_labels'], color='darkslategray')

    x_rc = x[idx_dict['pzt_2'][0]]
    y_rc = 0
    ax.add_patch(Rectangle((x_rc +  pzt_width, y_rc - pzt_height / 2), pzt_width, pzt_height, color='teal'))
    ax.text(x_rc + pzt_width, y_rc - pzt_height, 'Receiver', ha='center', fontsize=settings['fontsize_labels'], color='darkslategray')

    # Configure ax2 for the recorded signal
    ax2.set_ylim([t[0], t[-1]])
    ax2.set_xlim([-1, 1])  # Set x-limits to small range around zero
    ax2.set_title("Recorded Signal", fontsize=settings['fontsize_title'], color='darkslategray')
    ax2.axis('off')
    ax2.invert_yaxis()

    fig.tight_layout()

    # Initialize lines for animation
    line_wavefield, = ax.plot([], [], color='darkslategray', lw=1.5)
    line_wavefield.set_linewidth(3.0)
    line_recorded, = ax2.plot([], [], color='darkslategray', lw=1.5)
    line_recorded.set_linewidth(3.0)

    # Prepare data for animation
    sp_movie = sp_field[::movie_sampling]
    sp_recorded_movie = sp_recorded[::movie_sampling] / np.amax(np.abs(sp_recorded))
    t_recorded_movie = t[::movie_sampling]
    num_frames = len(sp_movie)

    def update_frame(frame):
        line_wavefield.set_data(x, sp_movie[frame])
        line_recorded.set_data(sp_recorded_movie[:frame], t_recorded_movie[:frame])
        return line_wavefield, line_recorded

    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=True, interval=20)
    ani.save(outfile_path, fps=30, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)

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
    misfit_interval: slice,
    sample_dimensions: tuple,
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
    enable_plotting: bool = False
) -> np.ndarray:
    """
    Simulate ultrasonic wave propagation through a sample and compute the synthetic waveform.

    Args:
        observed_time (np.ndarray): Time axis of the observed waveform.
        observed_waveform (np.ndarray): Observed waveform.
        pulse_time (np.ndarray): Time axis of the pulse.
        pulse_waveform (np.ndarray): Pulse waveform.
        misfit_interval (slice): Interval for L2 norm computation.
        sample_dimensions (tuple): Dimensions of the sample layers.
        frequency_cutoff (float): Maximum frequency cut-off.
        transmitter_position (float): Transmitter location.
        receiver_position (float): Receiver location.
        pzt_layer_width (float): Width of the piezoelectric transducer layer.
        pmma_layer_width (float): Width of the PMMA layer.
        steel_velocity (float): Velocity of the blocks holding the gouge.
        gouge_velocity (Union[float, np.ndarray]): Velocity in the gouge layer.
        pzt_velocity (float): Velocity in the piezoelectric transducer.
        pmma_velocity (float): Velocity in the PMMA layer.
        normalize_waveform (bool, optional): Normalize the synthetic waveform to match the observed amplitude. Defaults to True.
        enable_plotting (bool, optional): Enable plotting of the synthetic and observed waveforms. Defaults to False.

    Returns:
        np.ndarray: Synthetic waveform interpolated on the observed time axis.
    """

    # Compute the spatial grid
    spatial_axis, dx, num_x = compute_grid(
        sample_dimensions=sample_dimensions,
        min_velocity=gouge_velocity,
        frequency_cutoff=frequency_cutoff
    )

    # Build the velocity model
    velocity_model, _, _, _, _ = build_velocity_model(
        x=spatial_axis,
        sample_dimensions=sample_dimensions,
        x_transmitter=transmitter_position,
        x_receiver=receiver_position,
        pzt_layer_width=pzt_layer_width,
        pmma_layer_width=pmma_layer_width,
        steel_velocity=steel_velocity,
        gouge_velocity=gouge_velocity,
        pzt_velocity=pzt_velocity,
        pmma_velocity=pmma_velocity,
        plotting=False
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
    src_spatial_function = arbitrary_position_filter(
        spatial_axis=spatial_axis,
        dx=dx,
        position=transmitter_position,
        radius=10
    )
    rec_spatial_function = arbitrary_position_filter(
        spatial_axis=spatial_axis,
        dx=dx,
        position=receiver_position,
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
        plt.figure()
        plt.plot(src_time_function)
        plt.title("Source Time Function")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

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
    reverse_time: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Perform pseudospectral modeling for 1D wave propagation (forward or adjoint).

    Args:
        num_x (int): Number of spatial grid points.
        delta_x (float): Grid spacing.
        num_t (int): Number of time steps.
        delta_t (float): Time step.
        source_spatial (np.ndarray): Spatial source function.
        source_time (np.ndarray): Time source function.
        velocity_model (np.ndarray): Velocity model.
        compute_derivative (bool, optional): Compute derivative wavefield (for gradient calculation). Defaults to False.
        reverse_time (bool, optional): Reverse time axis (for adjoint modeling). Defaults to False.

    Returns:
        np.ndarray: Wavefield of shape (num_t, num_x).
        (Optional) np.ndarray: Derivative wavefield if `compute_derivative` is True.
    """
    # Initialize wavefield arrays
    wavefield_current = np.zeros(num_x)
    wavefield_future = np.zeros(num_x)
    wavefield_past = np.zeros(num_x)
    wavefield = np.zeros((num_t, num_x))
    
    if compute_derivative:
        derivative_wavefield = np.zeros((num_x, num_t))  # Derivative wavefield

    # Time-stepping loop
    for time_step in range(num_t):
        # Second spatial derivative using Fourier method
        second_derivative = fourier_derivative_2nd(wavefield_current, delta_x)

        # Update wavefield using the finite difference time stepping
        wavefield_future = (
            2 * wavefield_current - wavefield_past
            + (velocity_model ** 2) * (delta_t ** 2) * second_derivative
        )

        # Add source contribution
        wavefield_future += source_spatial * source_time[time_step] * (delta_t ** 2)

        # Compute derivative wavefield if required
        if compute_derivative:
            derivative_wavefield[:, time_step] = (
                (wavefield_future - 2 * wavefield_current + wavefield_past) / delta_t ** 2
            )

        # Update wavefield states for next iteration
        wavefield_past = wavefield_current.copy()
        wavefield_current = wavefield_future.copy()

        # Apply boundary conditions (e.g., Dirichlet boundaries)
        wavefield_current[0] = 0
        wavefield_current[-1] = 0

        # Store wavefield at current time step
        wavefield[time_step, :] = wavefield_current

    # Reverse time axis for adjoint modeling
    if reverse_time:
        wavefield = wavefield[::-1, :]

    if compute_derivative:
        return wavefield, derivative_wavefield  # Return both wavefield and its derivative
    
    return wavefield  # Return wavefield only


def compute_grid(
    sample_dimensions: tuple,
    min_velocity: Union[float, np.ndarray],
    frequency_cutoff: float,
    points_per_min_wavelength: int = 10
) -> tuple:
    """
    Compute the 1D spatial grid for the simulation based on the minimum velocity and frequency.

    Args:
        sample_dimensions (tuple): Dimensions of the sample layers.
        min_velocity (Union[float, np.ndarray]): Minimum velocity in the model.
        frequency_cutoff (float): Maximum frequency cut-off.
        points_per_min_wavelength (int, optional): Points per minimum wavelength. Defaults to 10.

    Returns:
        tuple: (spatial_axis, dx, num_x) where spatial_axis is the grid, dx is grid spacing, num_x is number of grid points.
    """
    grid_length = sum(sample_dimensions)
    spatial_axis = make_grid_1D(
        grid_len=grid_length,
        cmin=min_velocity,
        fmax=frequency_cutoff,
        ppt=points_per_min_wavelength
    )
    dx = spatial_axis[1] - spatial_axis[0]
    num_x = len(spatial_axis)
    return spatial_axis, dx, num_x


def prepare_time_variables(
    observed_time: np.ndarray,
    dx: float,
    max_velocity: float,
    cfl_factor: float = 0.6
) -> tuple:
    """
    Prepare the time variables for the simulation based on the spatial grid and maximum velocity.
    The time step is calculated to satisfy the CFL stability criterion.

    Args:
        observed_time (np.ndarray): Time axis of the observed waveform.
        dx (float): Grid spacing.
        max_velocity (float): Maximum velocity in the model.
        cfl_factor (float, optional): CFL safety factor. Defaults to 0.6.

    Returns:
        tuple: (simulation_time, dt, num_t) where simulation_time is the time axis for simulation, dt is time step, num_t is number of time steps.
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

    Args:
        pulse_time (np.ndarray): Time axis of the pulse.
        pulse_waveform (np.ndarray): Pulse waveform.
        dt (float): Time step for the simulation.

    Returns:
        np.ndarray: Interpolated source time function.
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

    Args:
        observed_waveform (np.ndarray): Observed waveform.
        synthetic_waveform (np.ndarray): Synthetic waveform.
        misfit_interval (slice): Interval over which to compute the misfit.

    Returns:
        float: The computed L2 norm misfit.
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

    Args:
        spatial_axis (np.ndarray): Spatial axis (grid points).
        dx (float): Grid spacing.
        position (float): Source or receiver position.
        radius (int): Radius for the Kaiser window (in grid points).

    Returns:
        np.ndarray: Filter approximating a Dirac delta function at the arbitrary position.
    """
    # Normalize positions to grid indices
    grid_indices = spatial_axis / dx
    num_points = len(grid_indices)
    position_index = position / dx

    # Find the grid node closest to the desired position
    closest_node_index = np.argmin(np.abs(grid_indices - position_index))

    if grid_indices[closest_node_index] == position_index:
        # If the position coincides with a grid node
        filter_array = np.zeros(num_points)
        filter_array[closest_node_index] = 1
        return filter_array

    # Create sinc function centered at the arbitrary position
    sinc_function = np.sinc(grid_indices - position_index)

    # Apply Kaiser window to the sinc function
    beta = 6.0  # Kaiser window parameter
    kaiser_window = kaiser(2 * radius + 1, beta)

    # Apply windowed sinc filter centered on the position_index
    start_idx = max(0, closest_node_index - radius)
    end_idx = min(num_points, closest_node_index + radius + 1)
    
    windowed_sinc = np.zeros_like(sinc_function)
    windowed_sinc[start_idx:end_idx] = sinc_function[start_idx:end_idx] * kaiser_window[:end_idx-start_idx]

    return windowed_sinc

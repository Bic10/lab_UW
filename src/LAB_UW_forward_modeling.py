# src/LAB_UW_forward_modeling.py

import numpy as np
from numpy import linalg as LA
from typing import Union
from scipy.signal.windows import kaiser

from helpers import *
from synthetic_data import *
from plotting import *

def DDS_UW_simulation(
    observed_time: np.ndarray,
    observed_waveform: np.ndarray,
    pulse_time: np.ndarray,
    pulse_waveform: np.ndarray,
    sample_dimensions: tuple,
    h_groove_side: float,
    h_groove_central: float,
    frequency_cutoff: float,
    transmitter_position: float,
    receiver_position: float,
    pzt_layer_width: float,
    pmma_layer_width: float,
    steel_velocity: float,
    gouge_velocity: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    pzt_velocity: float,
    pmma_velocity: float,
    misfit_interval: np.ndarray,
    fixed_minimum_velocity: float = None,
    iterative_gradient_descent: bool = False,
    n_iterations: int = 100,
    dc_max_start: float = 500 * 1e-4,
    reduce_factor: float = 0.5,
    dc_threshold: float = 1 * 1e-4 ,
    normalize_waveform: bool = True,
    enable_plotting: bool = False,
    make_movie: bool = False,
    plot_output_path: str = None,
    movie_output_path: str = "simulation.mp4"
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate ultrasonic wave propagation and perform gradient-based inversion using adaptive step size.
    
    Returns:
        synthetic_waveform: np.ndarray
            The synthetic waveform simulated.
        updated_gouge_velocities: Tuple[np.ndarray, np.ndarray]
            The updated velocity models for the two gouge layers.
    """
    # Unpack gouge velocities
    gouge_velocity_1, gouge_velocity_2 = gouge_velocity

    # Initialize velocity model and min_velocity for grid construction
    min_velocity = fixed_minimum_velocity if fixed_minimum_velocity is not None else min(np.amin(gouge_velocity_1), np.amin(gouge_velocity_2))

    # Create grid according to numerical dispersion criteria for minimum wavelength
    # the sensors are elastic media and the hole behind the sensors are of air 
    # the trasmitter and receiver position should be passed as their "pzt_depth" respect to the external edge of their block
    total_length = np.sum(sample_dimensions) + 2*pmma_layer_width + 2*pzt_layer_width -  (transmitter_position + receiver_position)

    spatial_axis, dx, num_x = compute_grid(
        total_length_to_simulate= total_length,
        min_velocity=min_velocity,
        frequency_cutoff=frequency_cutoff,
    )

    # Build the initial velocity model
    velocity_model, idx_dict = build_velocity_model(
        x=spatial_axis,
        sample_dimensions=sample_dimensions,
        x_transmitter=transmitter_position,
        x_receiver=receiver_position,
        pzt_layer_width=pzt_layer_width,
        pmma_layer_width=pmma_layer_width,
        h_groove_side=h_groove_side,
        h_groove_central=h_groove_central,
        steel_velocity=steel_velocity,
        gouge_velocity=(gouge_velocity_1, gouge_velocity_2),
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
    # For the way the spatial axis is created now and the positions are passed, the transmitter and the receiver are a bit tricki
    transmitter_position_respect_to_spatial_axis =  1*pzt_layer_width + pmma_layer_width
    radius_transmitter = 2*len(idx_dict['pzt_1'])
    src_spatial_function = arbitrary_position_filter(
        spatial_axis=spatial_axis,
        dx=dx,
        position=transmitter_position_respect_to_spatial_axis,
        flip_side=None,
        radius=radius_transmitter
    )
    # since we already use the receiver_position to build the grid, its place is just at the edge between pzt and block...
    receiver_position_respect_to_spatial_axis = total_length - 1*pzt_layer_width - pmma_layer_width 

    radius_receiver = 2*len(idx_dict['pzt_2'])
    rec_spatial_function = arbitrary_position_filter(
                                            spatial_axis=spatial_axis,
                                            dx=dx,
                                            position=receiver_position_respect_to_spatial_axis,
                                            flip_side=None,
                                            radius=radius_receiver
                                        )

    # Simulate without inversion
    wavefield_forward = pseudospectral_1D(
        num_x=num_x,
        delta_x=dx,
        num_t=num_t,
        delta_t=dt,
        source_spatial=src_spatial_function,
        source_time=src_time_function,
        velocity_model=velocity_model,
        compute_derivative=False
    )

    # Record the simulated wavefield at the receiver position
    simulated_waveform = np.sum(wavefield_forward * rec_spatial_function, axis=1)

    if normalize_waveform:
        amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
        simulated_waveform *= amplitude_scale

    # Interpolate the synthetic waveform onto the observed time axis
    synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

    # Compute residuals and misfit
    residual = np.zeros_like(observed_waveform)
    residual[misfit_interval] = synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]

    if iterative_gradient_descent:
        initial_misfit = np.sqrt(np.sum(residual ** 2))
        print(f"Initial Misfit: {initial_misfit}")
        # Initialize gradient descent parameters
        dc_max = dc_max_start
        misfit_prec = initial_misfit

        # Set the best_velocity_model to the initial velocity model
        best_velocity_model = velocity_model.copy()

        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")

            # Forward modeling with derivative computation using the current velocity model
            wavefield_forward, derivative_wavefield_forward = pseudospectral_1D(
                num_x=num_x,
                delta_x=dx,
                num_t=num_t,
                delta_t=dt,
                source_spatial=src_spatial_function,
                source_time=src_time_function,
                velocity_model=velocity_model,  # Use the current velocity model
                compute_derivative=True
            )

            # Record the simulated wavefield at the receiver position
            simulated_waveform = np.sum(wavefield_forward * rec_spatial_function, axis=1)
            if normalize_waveform:
                amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
                simulated_waveform *= amplitude_scale

            # Interpolate the synthetic waveform onto the observed time axis
            synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

            # Compute residuals and misfit
            residual = np.zeros_like(observed_waveform)
            residual[misfit_interval] = synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]
            misfit = np.sqrt(np.sum(residual ** 2))
            print(f"Misfit: {misfit}")

            # Early stopping based on misfit threshold
            if dc_max < dc_threshold:
                print(f"Gradient reduced to minimum updating step: ({dc_threshold}). Stopped")
                break

            if misfit <= misfit_prec:
                # Misfit decreased, update best model and proceed
                misfit_prec = misfit
                best_velocity_model = velocity_model.copy()
                print(f"Misfit decreased, saving the current model as the best.")

                # Adjoint modeling for gradient calculation
                adj_src_time_function = 2 * residual[::-1]  # Reverse time
                adj_src_time_function = interpolate_source(
                    pulse_time=observed_time,
                    pulse_waveform=adj_src_time_function,
                    dt=dt
                )

                wavefield_adjoint = pseudospectral_1D(
                    num_x=num_x,
                    delta_x=dx,
                    num_t=num_t,
                    delta_t=dt,
                    source_spatial=rec_spatial_function,
                    source_time=adj_src_time_function,
                    velocity_model=velocity_model,  # Use the current velocity model
                    compute_derivative=False
                )

                # Compute the gradient
                gradient = np.zeros_like(velocity_model)
                for t_step in range(num_t):
                    gradient += (2 / (velocity_model ** 3)) * wavefield_adjoint[t_step, :] * derivative_wavefield_forward[t_step, :]

                gradient *= dt

                # dE_max = np.max(np.abs(gradient))
                # velocity_model -= (dc_max/dE_max) * gradient

                plt.plot(velocity_model)
                plt.show()

                # Apply gradient only to gouge layers
                gradient_update = np.zeros(num_x)
                # gradient_update[idx_dict['gouge_1']] = gradient[idx_dict['gouge_1']]
                # gradient_update[idx_dict['gouge_2']] = gradient[idx_dict['gouge_2']]
                
                gradient_update[idx_dict['pmma_1']] = gradient[idx_dict['pmma_1']]
                gradient_update[idx_dict['pmma_2']] = gradient[idx_dict['pmma_2']]
                gradient_update[idx_dict['pzt_1']] = gradient[idx_dict['pzt_1']]
                gradient_update[idx_dict['pzt_2']] = gradient[idx_dict['pzt_2']]

                # Update velocity model with scaled gradient
                dE_max = np.max(np.abs(gradient_update))
                # velocity_model[idx_dict['gouge_1']] -= (dc_max/dE_max) * gradient_update[idx_dict['gouge_1']]
                # velocity_model[idx_dict['gouge_2']] -= (dc_max/dE_max) * gradient_update[idx_dict['gouge_2']]

                velocity_model[idx_dict['pmma_1']] -= (dc_max/dE_max) * gradient_update[idx_dict['pmma_1']] 
                velocity_model[idx_dict['pmma_2']] -= (dc_max/dE_max) * gradient_update[idx_dict['pmma_2']] 
                velocity_model[idx_dict['pzt_1']] -= (dc_max/dE_max) * gradient_update[idx_dict['pzt_1']] 
                velocity_model[idx_dict['pzt_2']] -= (dc_max/dE_max) * gradient_update[idx_dict['pzt_2']] 


            else:
                # Misfit increased, reduce step size and revert to best model
                dc_max *= reduce_factor
                print(f"Misfit increased, reducing maximum gradient magnitude to: {dc_max}")
                velocity_model = best_velocity_model.copy()  # Revert to the best velocity model

    else:
        best_velocity_model = velocity_model.copy()

    # At the end of the optimization, best_velocity_model contains the best model
    gouge_velocity_1_updated = best_velocity_model[idx_dict['gouge_1']]  # Use the best model for output
    gouge_velocity_2_updated = best_velocity_model[idx_dict['gouge_2']]

    # Build the final velocity model using the best velocities
    velocity_model, idx_dict = build_velocity_model(
        x=spatial_axis,
        sample_dimensions=sample_dimensions,
        x_transmitter=transmitter_position,
        x_receiver=receiver_position,
        pzt_layer_width=pzt_layer_width,
        pmma_layer_width=pmma_layer_width,
        h_groove_side=h_groove_side,
        h_groove_central=h_groove_central,
        steel_velocity=steel_velocity,
        gouge_velocity=(gouge_velocity_1_updated, gouge_velocity_2_updated),
        pzt_velocity=pzt_velocity,
        pmma_velocity=pmma_velocity,
        plotting=False
    )

    if enable_plotting:
        plot_simulation_waveform(observed_time, synthetic_waveform, observed_waveform, misfit_interval, outfile_path=plot_output_path)

    if make_movie:
        make_movie_from_simulation(
            outfile_path=movie_output_path,
            x=spatial_axis,
            t=simulation_time,
            sp_field=wavefield_forward,
            sp_recorded=simulated_waveform,
            sample_dimensions=sample_dimensions,
            idx_dict=idx_dict,
        )

    return synthetic_waveform, (gouge_velocity_1_updated, gouge_velocity_2_updated)

def pseudospectral_1D(
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial: np.ndarray,
    source_time: np.ndarray,
    velocity_model: np.ndarray,
    compute_derivative: bool = False,
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
        wavefield[time_step, :] = wavefield_current

        # Compute derivative wavefield if required
        if compute_derivative:
            derivative = (wavefield_future - 2 * wavefield_current + wavefield_past) / (delta_t ** 2)
            derivative_wavefield[num_t - 1 - time_step, :] = derivative

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
    return LA.norm(synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval], 2)


import numpy as np
from numpy import kaiser

def arbitrary_position_filter(
    spatial_axis: np.ndarray,
    dx: float,
    position: float,
    radius: int,
    flip_side: str = None  # Either 'left', 'right', or None
) -> np.ndarray:
    """
    Create a Kaiser-windowed sinc filter for arbitrary source/receiver positioning on a 1D grid.
    If flip_side is specified ('left' or 'right'), the values of the windowed sinc function on that side
    of the closest grid node are flipped and added to the values on the opposite side.
    
    Parameters:
    - spatial_axis (np.ndarray): The spatial axis of the grid.
    - dx (float): Spatial step size.
    - position (float): Exact position of the source/receiver.
    - radius (int): Radius of the windowed sinc function (number of grid points).
    - flip_side (str): 'left' or 'right' to indicate which side to flip and fold.
    
    Returns:
    - windowed_sinc (np.ndarray): The windowed sinc filter adjusted for the free surface.
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
    window_indices = np.arange(start_idx, end_idx)
    windowed_sinc[window_indices] = sinc_function[window_indices] * kaiser_window[:end_idx - start_idx]

    # Implement the flip and fold
    if flip_side == 'right':
        # Indices on the left side
        left_indices = np.arange(start_idx, closest_node_index)
        num_left = len(left_indices)
        # Indices on the right side
        right_indices = np.arange(closest_node_index, closest_node_index + num_left)
        # Adjust right_indices to not exceed end_idx
        right_indices = right_indices[right_indices < end_idx]

        # Flip the left values
        flipped_left_values = windowed_sinc[left_indices][::-1]
        flipped_left_values = flipped_left_values[:len(right_indices)]  # Adjust length

        # Add to the right side
        windowed_sinc[right_indices] += flipped_left_values

        # Zero out the left side
        windowed_sinc[left_indices] = 0.0

    elif flip_side == 'left':
        # Indices on the right side
        right_indices = np.arange(closest_node_index + 1, end_idx)
        num_right = len(right_indices)
        # Indices on the left side
        left_indices = np.arange(closest_node_index - num_right, closest_node_index)
        left_indices = left_indices[left_indices >= start_idx]  # Ensure within bounds

        # Flip the right values
        flipped_right_values = windowed_sinc[right_indices][::-1]
        flipped_right_values = flipped_right_values[:len(left_indices)]  # Adjust length

        # Add to the left side
        windowed_sinc[left_indices] += flipped_right_values

        # Zero out the right side
        windowed_sinc[right_indices] = 0.0

    return windowed_sinc



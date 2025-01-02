# lab_uw/forward_modeling.py

import numpy as np
from numpy import linalg as LA
from typing import Union, Tuple, Optional, Dict

from lab_uw.simulation_setup import Grid1D, VelocityModel1D, Source1D, Receiver1D, SimulationTime
from lab_uw.plotting import Plotter
from lab_uw.signal_processing import SignalProcessor

class ForwardModeler:
    """
    Class for simulating ultrasonic wave propagation and performing gradient-based inversion.
    """
    def __init__(self, plotter: Optional[Plotter] = None):
        """
        Initialize the ForwardModeler.

        Args:
            plotter (Plotter, optional): An instance of the Plotter class for plotting results.
        """
        self.plotter = plotter or Plotter()

    def DDS_UW_simulation(
        self,
        observed_time: np.ndarray,
        observed_waveform: np.ndarray,
        stf_time: np.ndarray,
        stf_waveform: np.ndarray,
        sample_dimensions: Tuple[float, float, float],
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
        fixed_minimum_velocity: Optional[float] = None,
        n_iterations: int = 100,
        dc_max_start: float = 500 * 1e-4,
        reduce_factor: float = 0.5,
        dc_threshold: float = 10 * 1e-4,
        normalize_waveform: bool = True,
        enable_plotting: bool = False,
        make_movie: bool = False,
        plot_output_path: Optional[str] = None,
        movie_output_path: Optional[str] = "simulation_movie.mp4",
        iterative_gradient_descent: bool = False,
        initial_velocity_model: Optional[np.ndarray] = None,
        idx_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Simulate ultrasonic wave propagation and perform gradient-based inversion.

        Args:
            observed_time (np.ndarray): Time array of the observed waveform.
            observed_waveform (np.ndarray): Observed waveform data.
            stf_time (np.ndarray): Time array of the source stf.
            stf_waveform (np.ndarray): Source stf waveform.
            sample_dimensions (Tuple[float, float, float]): Dimensions of the sample layers.
            h_groove_side (float): Height of the side grooves.
            h_groove_central (float): Height of the central groove.
            frequency_cutoff (float): Frequency cutoff for the simulation.
            transmitter_position (float): Position of the transmitter.
            receiver_position (float): Position of the receiver.
            pzt_layer_width (float): Width of the PZT layer.
            pmma_layer_width (float): Width of the PMMA layer.
            steel_velocity (float): Velocity in the steel blocks.
            gouge_velocity (Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]): Velocities in the gouge layers.
            pzt_velocity (float): Velocity in the PZT layers.
            pmma_velocity (float): Velocity in the PMMA layers.
            misfit_interval (np.ndarray): Indices defining the misfit interval.
            fixed_minimum_velocity (float, optional): Minimum velocity for grid calculation.
            n_iterations (int): Number of iterations for gradient descent.
            dc_max_start (float): Initial maximum gradient step size.
            reduce_factor (float): Factor to reduce the gradient step size.
            dc_threshold (float): Threshold for the gradient step size to stop iterations.
            normalize_waveform (bool): Whether to normalize the synthetic waveform amplitude.
            enable_plotting (bool): Whether to enable plotting of results.
            make_movie (bool): Whether to create an animation of the simulation.
            plot_output_path (str, optional): Path to save the simulation waveform plot.
            movie_output_path (str, optional): Path to save the simulation movie.
            iterative_gradient_descent (bool): Whether to perform iterative gradient descent inversion.
            invert_pzt_regions (bool): Whether to invert velocities in the PZT regions.
            initial_velocity_model (np.ndarray, optional): Initial velocity model for inversion.
            idx_dict (Dict[str, np.ndarray], optional): Dictionary of indices for different layers.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
                - synthetic_waveform: The synthetic waveform simulated.
                - best_velocity_model: The updated 1D velocity model after inversion.
                - idx_dict: Dictionary of indices for different regions.
        """
        # Input validation
        if observed_time.ndim != 1 or observed_waveform.ndim != 1:
            raise ValueError("observed_time and observed_waveform must be 1D numpy arrays.")
        if stf_time.ndim != 1 or stf_waveform.ndim != 1:
            raise ValueError("stf_time and stf_waveform must be 1D numpy arrays.")
        if len(observed_time) != len(observed_waveform):
            raise ValueError("observed_time and observed_waveform must have the same length.")
        if len(stf_time) != len(stf_waveform):
            raise ValueError("stf_time and stf_waveform must have the same length.")
        if misfit_interval.ndim != 1:
            raise ValueError("misfit_interval must be a 1D numpy array of indices.")

        # Unpack gouge velocities
        gouge_velocity_1, gouge_velocity_2 = gouge_velocity

        # Compute the spatial grid
        total_length = np.sum(sample_dimensions) + 2 * pmma_layer_width + 2 * pzt_layer_width - (transmitter_position + receiver_position)

        # Use provided initial_velocity_model and idx_dict, or build them
        if initial_velocity_model is not None and idx_dict is not None:
            velocity_model = initial_velocity_model.copy()
            num_x = len(velocity_model)
            spatial_axis = np.linspace(start=0, stop=total_length, num=num_x)
            dx = spatial_axis[1] - spatial_axis[0]
        else:
        # Build the initial velocity model using Grid1D
            if fixed_minimum_velocity is None:
                raise ValueError("fixed_minimum_velocity must be provided if initial_velocity_model is not given.")

            # Prepare the spatial grid
            grid = Grid1D(
                cmin=fixed_minimum_velocity,
                fmax=frequency_cutoff,
                grid_len=total_length,
                ppt=10  # Points per wavelength; adjust as needed
            )
            spatial_axis = grid.spatial_axis
            dx = grid.dx
            num_x = grid.total_grid_points

            # Prepare the simulation time 
            sim_time_handler = SimulationTime(
                observed_time=observed_time,
                dx=dx,
                max_velocity=steel_velocity
            )
            simulation_time = sim_time_handler.simulation_time
            dt = sim_time_handler.dt
            num_t = sim_time_handler.num_t

            velocity_model_handler = VelocityModel1D(              
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
                plotting=False)

        velocity_model = velocity_model_handler.values
        idx_dict = velocity_model_handler.idx_dict

        # Initialize Source
        transmitter_position_relative = pzt_layer_width + pmma_layer_width
        radius_transmitter = 2 * len(idx_dict['pzt_1'])
        source = Source1D(
            stf_time=stf_time,
            stf_waveform=stf_waveform,
            position=transmitter_position_relative,
            radius=radius_transmitter
    )        
        source.interpolate_time_function(dt=dt, simulation_time=simulation_time)
        source.create_spatial_function(spatial_axis=spatial_axis, dx=dx, flip_side=None)

        # Initialize Receiver
        receiver_position_relative = total_length - pzt_layer_width - pmma_layer_width
        radius_receiver = 2 * len(idx_dict['pzt_2'])
        receiver = Receiver1D(
            position=receiver_position_relative,
            radius=radius_receiver
        )
        receiver.create_spatial_function(spatial_axis=spatial_axis, dx=dx, flip_side=None)

        # Simulate without inversion
        wavefield_forward = pseudospectral_1D(
            num_x= num_x,
            delta_x= dx,
            num_t= num_t,
            delta_t= dt,
            source_spatial_function= source.spatial_function,
            source_time_function= source.time_function,
            velocity_model= velocity_model,
            compute_derivative= False
        )

        # Record the simulated wavefield at the receiver position
        simulated_waveform = np.sum(wavefield_forward * receiver.spatial_function, axis=1)

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
                    source_spatial_function= source.spatial_function,
                    source_time_function= source.time_function,
                    velocity_model=velocity_model,  # Use the current velocity model
                    compute_derivative=True
                )

                # Record the simulated wavefield at the receiver position
                simulated_waveform = np.sum(wavefield_forward * receiver.spatial_function, axis=1)
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

                    # Create an adjoint source at the receiver's position
                    adjoint_source = Source1D(
                        stf_time=observed_time,
                        stf_waveform=adj_src_time_function,
                        position=receiver.position,  # Use the receiver's position
                        radius=receiver.radius       # Use the receiver's radius
                    )
                    # Interpolate the adjoint source time function
                    adjoint_source.interpolate_time_function(dt=dt, simulation_time=simulation_time)
                    # Create the spatial function for the adjoint source
                    adjoint_source.create_spatial_function(spatial_axis=spatial_axis, dx=dx, flip_side=None)

                    # Perform the adjoint wavefield simulation
                    wavefield_adjoint = pseudospectral_1D(
                        num_x=num_x,
                        delta_x=dx,
                        num_t=num_t,
                        delta_t=dt,
                        source_spatial=adjoint_source.spatial_function,
                        source_time=adjoint_source.time_function,
                        velocity_model=velocity_model,  # Use the current velocity model
                        compute_derivative=False
                    )

                    # Compute the gradient
                    gradient = np.zeros_like(velocity_model)
                    for t_step in range(num_t):
                        gradient += (2 / (velocity_model ** 3)) * wavefield_adjoint[t_step, :] * derivative_wavefield_forward[t_step, :]

                    gradient *= dt

                    # Apply gradient only to specified regions
                    gradient_update = np.zeros(num_x)

                    dc_max = dc_max_start / 5
                    dc_threshold = dc_threshold / 5

                    # Apply gradient to selected regions
                    regions_to_update = np.concatenate([idx_dict['gouge_1'], idx_dict['gouge_2']])
                    gradient_update[regions_to_update] = gradient[regions_to_update]

                    # Scale gradient
                    dE_max = np.max(np.abs(gradient_update[regions_to_update]))
                    velocity_model[regions_to_update] -= (dc_max / dE_max) * gradient_update[regions_to_update]

                    # Clip velocities to physical bounds
                    velocity_min = fixed_minimum_velocity  # Minimum velocity
                    velocity_max = steel_velocity  # Maximum velocity is steel_velocity
                    velocity_model[regions_to_update] = np.clip(velocity_model[regions_to_update], velocity_min, velocity_max)

                else:
                    # Misfit increased, reduce step size and revert to best model
                    dc_max *= reduce_factor
                    print(f"Misfit increased, reducing maximum gradient magnitude to: {dc_max}")
                    velocity_model = best_velocity_model.copy()  # Revert to the best velocity model

        else:
            best_velocity_model = velocity_model.copy()

        if enable_plotting:
            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=plot_output_path
            )

        if make_movie:
            self.plotter.make_movie_from_simulation(
                outfile_path=movie_output_path,
                x=spatial_axis,
                t=simulation_time,
                sp_field=wavefield_forward,
                sp_recorded=simulated_waveform,
                sample_dimensions=sample_dimensions,
                idx_dict=idx_dict,
            )

        return synthetic_waveform, best_velocity_model, idx_dict

def pseudospectral_1D(
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial_function: np.ndarray,
    source_time_function: np.ndarray,
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
        second_derivative = SignalProcessor().fourier_derivative_2nd(wavefield_current, delta_x)

        # Update wavefield using the finite difference time stepping
        wavefield_future = (
            2 * wavefield_current - wavefield_past
            + (velocity_model ** 2) * (delta_t ** 2) * second_derivative
        )

        # Add source contribution
        wavefield_future += source_spatial_function * source_time_function[time_step] * (delta_t ** 2)

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

def compute_misfit(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    misfit_interval: slice
) -> float:
    """
    Compute the L2 norm misfit between observed and synthetic waveforms over a specified interval.
    """
    return LA.norm(synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval], 2)

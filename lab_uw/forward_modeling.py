# lab_uw/forward_modeling.py

import numpy as np
from numpy import linalg as LA
from typing import Union, Tuple, Optional, Dict

from lab_uw.simulation_setup import Grid1D, VelocityModel1D, Source1D, Receiver1D, SimulationTime
from lab_uw.plotting import Plotter
from lab_uw.signal_processing import SignalProcessor

class ForwardModeler:
    """
    Class for simulating ultrasonic wave propagation and optionally
    performing gradient-based local inversion.

    The forward modeling is done in a dedicated method (`forward_simulation`),
    and the local gradient descent is in `run_local_inversion`.

    This way, you can reuse `forward_simulation` for any global (or other)
    inversion method you choose to implement elsewhere.
    """

    def __init__(self, plotter: Optional["Plotter"] = None):
        """
        Initialize the ForwardModeler.

        Args:
            plotter (Plotter, optional): An instance of a Plotter class for plotting results.
        """
        self.plotter = plotter or Plotter()

    # -------------------------------------------------------------------------
    # 1) Forward Modeling Only
    # -------------------------------------------------------------------------
    def forward_simulation(
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
        pla_layer_width: float,
        steel_velocity: float,
        gouge_velocity: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
        pzt_velocity: float,
        pla_velocity: float,
        misfit_interval: np.ndarray,
        # The next four are relevant for building a brand-new velocity model if needed:
        fixed_minimum_velocity: Optional[float] = None,
        initial_velocity_model: Optional[np.ndarray] = None,
        idx_dict: Optional[Dict[str, np.ndarray]] = None,
        # Some general optional flags:
        normalize_waveform: bool = True,
        enable_plotting: bool = False,
        make_movie: bool = False,
        plot_output_path: Optional[str] = None,
        movie_output_path: Optional[str] = "simulation_movie.mp4",
    ) -> Tuple[np.ndarray,np.ndarray,Dict[str, np.ndarray],np.ndarray, np.ndarray, float,int, np.ndarray,float,int,Source1D,Receiver1D]:
        """
        Perform the forward modeling (one pass) of ultrasonic wave propagation
        in a 1D layered medium.

        Args:
            observed_time, observed_waveform: 1D arrays of the measured data.
            stf_time, stf_waveform: 1D arrays defining the source time function.
            sample_dimensions: (thickness_steel1, thickness_gouge, thickness_steel2)
            h_groove_side, h_groove_central: geometric groove parameters.
            frequency_cutoff: frequency cutoff for building grid spacing.
            transmitter_position, receiver_position: for geometry offset.
            pzt_layer_width, pla_layer_width: widths of PZT and PLA layers.
            steel_velocity, gouge_velocity, pzt_velocity, pla_velocity: wave speeds in each region.
            misfit_interval: 1D array of indices over which we compute the misfit.
            fixed_minimum_velocity: if no initial velocity is provided, we need the min velocity to build the grid.
            initial_velocity_model, idx_dict: can be provided to skip building a new velocity model.
            normalize_waveform: if True, amplitude-scale synthetic to match observed.
            enable_plotting, make_movie: if True, produce output via Plotter.
            plot_output_path, movie_output_path: specify file paths for saving plots/movies.

        Returns:
            A 12-tuple of:
                (np.ndarray,  # synthetic_waveform
                np.ndarray,  # velocity_model
                Dict[str, np.ndarray],  # idx_dict
                np.ndarray,  # simulation_time
                np.ndarray,  # wavefield_forward
                float,       # dt
                int,         # num_t
                np.ndarray,  # spatial_axis
                float,       # dx
                int,         # num_x
                Source1D,  # source
                Receiver1D # receiver)
        """

        # Validate shape of arrays
        if observed_time.ndim != 1 or observed_waveform.ndim != 1:
            raise ValueError("observed_time and observed_waveform must be 1D numpy arrays.")
        if stf_time.ndim != 1 or stf_waveform.ndim != 1:
            raise ValueError("stf_time and stf_waveform must be 1D numpy arrays.")
        if len(observed_time) != len(observed_waveform):
            raise ValueError("observed_time and observed_waveform must have the same length.")
        if len(stf_time) != len(stf_waveform):
            raise ValueError("stf_time and stf_waveform must have the same length.")
        if misfit_interval.ndim != 1:
            raise ValueError("misfit_interval must be a 1D array of indices.")

        # Unpack gouge velocities
        gouge_velocity_1, gouge_velocity_2 = gouge_velocity

        # Compute total length for 1D domain
        total_length = (
            np.sum(sample_dimensions)
            + 2 * pla_layer_width
            + 2 * pzt_layer_width
            - (transmitter_position + receiver_position)
        )

        # ---------------------------------------------------------------------
        # Build or reuse velocity model
        # ---------------------------------------------------------------------
        # If the user gave us a velocity_model and idx_dict, we just reuse it:
        if (initial_velocity_model is not None) and (idx_dict is not None):
            velocity_model = initial_velocity_model.copy()
            num_x = len(velocity_model)
            spatial_axis = np.linspace(start=0, stop=total_length, num=num_x)
            dx = spatial_axis[1] - spatial_axis[0]

        # Otherwise, build from scratch
        else:
            if fixed_minimum_velocity is None:
                raise ValueError(
                    "Must provide `fixed_minimum_velocity` if building a new velocity model."
                )
            # EXACT CODE FROM YOUR SETUP: create the 1D grid
            grid = Grid1D(
                cmin=fixed_minimum_velocity,
                fmax=frequency_cutoff,
                grid_len=total_length,
                ppt=10  # points per wavelength, or another suitable choice
            )
            spatial_axis = grid.spatial_axis
            dx = grid.dx
            num_x = grid.total_grid_points

            # EXACT CODE FROM YOUR SETUP: define sim time
            sim_time_handler = SimulationTime(
                observed_time=observed_time,
                dx=dx,
                max_velocity=steel_velocity
            )
            simulation_time = sim_time_handler.simulation_time
            dt = sim_time_handler.dt
            num_t = sim_time_handler.num_t

            # EXACT CODE FROM YOUR SETUP: build velocity model
            velocity_model_handler = VelocityModel1D(
                x=spatial_axis,
                sample_dimensions=sample_dimensions,
                x_transmitter=transmitter_position,
                x_receiver=receiver_position,
                pzt_layer_width=pzt_layer_width,
                pla_layer_width=pla_layer_width,
                h_groove_side=h_groove_side,
                h_groove_central=h_groove_central,
                steel_velocity=steel_velocity,
                gouge_velocity=(gouge_velocity_1, gouge_velocity_2),
                pzt_velocity=pzt_velocity,
                pla_velocity=pla_velocity,
            )
            velocity_model = velocity_model_handler.values
            idx_dict = velocity_model_handler.idx_dict

        # Initialize Source
        transmitter_position_relative = pzt_layer_width + pla_layer_width
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
        receiver_position_relative = total_length - pzt_layer_width - pla_layer_width
        radius_receiver = 2 * len(idx_dict['pzt_2'])
        receiver = Receiver1D(
            position=receiver_position_relative,
            radius=radius_receiver
        )
        receiver.create_spatial_function(spatial_axis=spatial_axis, dx=dx, flip_side=None)

        # ---------------------------------------------------------------------
        # Forward modeling (single pass)
        # ---------------------------------------------------------------------
        wavefield_forward = pseudospectral_1D(
            num_x=num_x,
            delta_x=dx,
            num_t=num_t,
            delta_t=dt,
            source_spatial_function=source.spatial_function,
            source_time_function=source.time_function,
            velocity_model=velocity_model,
            compute_derivative=False
        )

        # Record the simulated wavefield at the receiver position
        simulated_waveform = np.sum(wavefield_forward * receiver.spatial_function, axis=1)

        if normalize_waveform:
            amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
            simulated_waveform *= amplitude_scale

        # Interpolate synthetic waveform onto the observed time axis
        synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

        if enable_plotting:
            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=plot_output_path
            )

            velocity_model_handler.plot(outfile_path=plot_output_path + "_velocity_model")

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

        return (
            synthetic_waveform,
            velocity_model,
            idx_dict,
            simulation_time,
            wavefield_forward,
            dt,
            num_t,
            spatial_axis,
            dx,
            num_x,
            source,
            receiver
        )

def run_local_inversion(
    self,
    observed_time: np.ndarray,
    observed_waveform: np.ndarray,
    misfit_interval: np.ndarray,
    # Hyperparameters
    n_iterations: int,
    dc_max_start: float,
    reduce_factor: float,
    dc_threshold: float,
    fixed_minimum_velocity: float,
    steel_velocity: float,
    # The same forward-simulation inputs for consistency
    **forward_args
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform iterative gradient-based local inversion using repeated calls
    to `forward_simulation`. This method simply orchestrates the optimization.
    """

    # ---------------------------
    # 1) Forward pass (no deriv)
    # ---------------------------
    (
        synthetic_waveform,
        velocity_model,
        idx_dict,
        simulation_time,
        dt,
        num_t,
        spatial_axis,
        dx,
        num_x,
        source,
        receiver
    ) = self.forward_simulation(
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        misfit_interval=misfit_interval,
        **forward_args
    )

    # Compute residuals and misfit
    residual = np.zeros_like(observed_waveform)
    residual[misfit_interval] = synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]

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

    final_synthetic_waveform = synthetic_waveform.copy()
    return final_synthetic_waveform, best_velocity_model

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

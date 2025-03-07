# lab_uw/forward_modeling.py

import sys
from math import floor
import numpy as np
from numpy import linalg as LA
from typing import Union, Tuple, Optional, Dict, Any

from lab_uw.simulation_setup import (
    Grid1D,
    VelocityModel1D,
    VelocityModel1D_SingleBlock,
    Source1D,
    Receiver1D,
    SimulationTime
)
from lab_uw.plotting import Plotter
from lab_uw.signal_processing import SignalProcessor
from lab_uw.data_io import UltrasonicDataHandler


class ForwardModeler:
    """
    Class for simulating ultrasonic wave propagation under different 1D geometries:
     - "dds": double-direct-shear geometry (two gouge layers)
     - "block": single-block geometry
     - possibly extend in the future (e.g. "single_direct_shear", "bare_rock", etc.)
    """

    def __init__(self, plotter: Optional["Plotter"] = None):
        """
        Initialize the ForwardModeler.

        Args:
            plotter (Plotter, optional): For plotting results.
        """
        self.plotter = plotter or Plotter()

    def forward_simulation(
        self,
        geometry_type      : str,                        # "dds" or "block"
        observed_time      : np.ndarray,
        observed_waveform  : np.ndarray,
        stf_handler        : UltrasonicDataHandler,
        frequency_cutoff   : float,
        minimum_velocity   : Optional[float],
        maximum_velocity   : Optional[float],
        assembly_dict      : Dict[str, Any],
        misfit_interval    : np.ndarray,
        montecarlo         : Optional[Dict[str, Any]] = None,
        normalize_waveform : bool = True,
        enable_plotting    : bool = False,
        make_movie         : bool = False,
        plot_output_path   : Optional[str] = None,
        movie_output_path  : Optional[str] = "simulation_movie.mp4",
    ) -> Tuple[
        np.ndarray,  # synthetic_waveform
        np.ndarray,  # wavefield_forward
        Union[VelocityModel1D, VelocityModel1D_SingleBlock],  # velocity_model_handler
        SimulationTime,
        Grid1D,
        Source1D,
        Receiver1D
    ]:
        """
        Unifies forward modeling under different 1D geometries ("dds" or "block").
        The method unpacks geometry-specific parameters, builds the velocity model,
        runs the pseudo-spectral simulation, and returns the final results.

        Args:
            geometry_type (str): "dds" or "block" (extendable for other setups).
            observed_time, observed_waveform: Measured time axis and waveform.
            stf_handler: Provides source time function waveforms.
            frequency_cutoff: High-end frequency for building the grid spacing.
            assembly_dict: Dictionary containing geometry and velocity info.
            misfit_interval: Indices over which we might compute misfit (for plotting).
            gouge_velocity: Only needed if geometry_type=="dds"; otherwise unused.
            montecarlo: Optional dictionary for randomizing geometry or source/receiver.
            minimum_velocity, maximum_velocity: If None, attempt to compute from data.
            normalize_waveform: If True, amplitude-scale synthetic to match observed.
            enable_plotting, make_movie: Produce plots/movies via self.plotter if True.
            plot_output_path, movie_output_path: File paths for saving outputs.

        Returns:
            A tuple of:
              ( synthetic_waveform,
                wavefield_forward,
                velocity_model_handler,
                sim_time_handler,
                grid_handler,
                source,
                receiver )
        """
        # -------------------------------------------------------
        # 1) COMMON: UNPACK SOURCE TIME FUNCTION FROM stf_handler
        # -------------------------------------------------------
        stf_waveform = stf_handler.waveform_data
        stf_time     = stf_handler.metadata["time_ax_waveform"]

        # -------------------------------------------------------
        # 3) MONTECARLO OR DEFAULT PARAMS FOR SOURCE/RECEIVER
        # -------------------------------------------------------
        if montecarlo:
            spreading_factor_transmitter = montecarlo["spreading_factor_transmitter"]
            spreading_factor_receiver    = montecarlo["spreading_factor_receiver"]
            position2edge_transmitter    = montecarlo["position2edge_transmitter"]
            position2edge_receiver       = montecarlo["position2edge_receiver"]
            radius_factor_transmitter    = montecarlo["radius_factor_transmitter"]
            radius_factor_receiver       = montecarlo["radius_factor_receiver"]
        else:
            spreading_factor_transmitter = 1
            spreading_factor_receiver    = 1
            position2edge_transmitter    = -0.5
            position2edge_receiver       = -0.5
            radius_factor_transmitter    = 0.1
            radius_factor_receiver       = 0.1

        # Assembly dictionary must have the
        wave_type            = assembly_dict["wave_type"]
        sample_dimensions    = assembly_dict["sample_dimensions"]
        transmitter_position = assembly_dict["transmitter_position"]
        receiver_position    = assembly_dict["receiver_position"]

        ################# duct-taper!
        ########## must move these layers to into sample dimensions
        if geometry_type.lower() == "dds":
            # =========== Double-Direct Shear ==============
            side1_params      = assembly_dict["side1_params"] 
            side2_params      = assembly_dict["side2_params"] 
            pzt_layer_width  = side1_params["pzt_layer_width"]
            pla_layer_width  = side1_params["pla_layer_width"]

        elif geometry_type.lower() == "block":
            # =========== Single Block ==============
            pzt_layer_width      = assembly_dict["pzt_layer_width"]     
            pla_layer_width      = assembly_dict["pla_layer_width"] 
        ##############################################################

        # -------------------------------------------------------
        # 4) CREATE THE 1D GRID
        # -------------------------------------------------------
        total_length = (np.sum(sample_dimensions) 
                        + 2 * pla_layer_width
                        + 2 * pzt_layer_width
        )
        
        grid_handler = Grid1D(
            cmin=minimum_velocity,
            fmax=frequency_cutoff,
            grid_len=total_length,
            ppt=10  # points per wavelength
        )
        spatial_axis = grid_handler.spatial_axis
        dx           = grid_handler.dx
        num_x        = grid_handler.total_grid_points

        # -------------------------------------------------------
        # 5) DEFINE TIME AXIS
        # -------------------------------------------------------
        sim_time_handler = SimulationTime(
            observed_time=observed_time,
            dx=dx,
            max_velocity=maximum_velocity
        )
        simulation_time = sim_time_handler.simulation_time
        dt              = sim_time_handler.dt
        num_t           = sim_time_handler.num_t

        # -------------------------------------------------------
        # 2) GEOMETRY-SPECIFIC: UNPACK ASSEMBLY, BUILD MODEL
        # -------------------------------------------------------
        if geometry_type.lower() == "dds":
            # =========== Double-Direct Shear ==============
            # (gouge_velocity must be provided)
            try:
                gouge_velocity_1 = assembly_dict["gouge_velocity_1"]
                gouge_velocity_2 = assembly_dict["gouge_velocity_1"]
            except:
                raise ValueError("`gouge_velocity` is required for DDS geometry.")

            # 2a) Unpack geometry from assembly_dict
            central_params    = assembly_dict["central_params"] 

            h_groove_central = central_params["h_grooves"]
            h_groove_side    = side1_params["h_grooves"]

            steel_velocity   = side1_params["velocity" + wave_type]
            pzt_velocity     = side1_params["pzt_velocity" + wave_type]
            pla_velocity     = side1_params["pla_velocity" + wave_type]

            # 2d) Build velocity model
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

        elif geometry_type.lower() == "block":
            # =========== Single Block ==============
            pzt_layer_width      = assembly_dict["pzt_layer_width"]     
            pla_layer_width      = assembly_dict["pla_layer_width"]     
            steel_velocity       = assembly_dict["velocity" + wave_type]
            pzt_velocity         = assembly_dict["pzt_velocity" + wave_type]
            pla_velocity         = assembly_dict["pla_velocity" + wave_type]

            velocity_model_handler = VelocityModel1D_SingleBlock(
                x=spatial_axis,
                sample_dimensions=sample_dimensions,
                x_transmitter=transmitter_position,
                x_receiver=receiver_position,
                pzt_layer_width=pzt_layer_width,
                pla_layer_width=pla_layer_width,
                steel_velocity=steel_velocity,
                pzt_velocity=pzt_velocity,
                pla_velocity=pla_velocity,
            )
        else:
            raise ValueError(f"Unknown geometry_type: {geometry_type}. "
                            f"Must be 'dds' or 'block'.")
        # -------------------------------------------------------
        # 6) FILL IN VELOCITY MODEL X-AXIS, EXTRACT ARRAYS
        # -------------------------------------------------------
        velocity_model_handler.x = spatial_axis
        velocity_model = velocity_model_handler.values
        idx_dict       = velocity_model_handler.idx_dict

        # -------------------------------------------------------
        # 7) BUILD SOURCE
        # -------------------------------------------------------
        transmitter_position_relative = (
            pzt_layer_width
            + position2edge_transmitter * pzt_layer_width
            + pla_layer_width
        )
        radius_transmitter = floor(radius_factor_transmitter * (pzt_layer_width / 2) / dx)
        extension_transmitter = spreading_factor_transmitter * pzt_layer_width

        source = Source1D(
            stf_time=stf_time,
            stf_waveform=stf_waveform,
            position=transmitter_position_relative,
            radius=radius_transmitter,
            extension=extension_transmitter,
            pzt_layer_width=pzt_layer_width
        )
        source.interpolate_time_function(dt=dt, simulation_time=simulation_time)
        source.create_spatial_function(spatial_axis=spatial_axis, dx=dx)

        # -------------------------------------------------------
        # 8) BUILD RECEIVER
        # -------------------------------------------------------
        receiver_position_relative = (
            total_length
            - pzt_layer_width
            - position2edge_receiver * pzt_layer_width
            - pla_layer_width
        )
        radius_receiver    = floor(radius_factor_receiver * (pzt_layer_width / 2) / dx)
        extension_receiver = spreading_factor_receiver * pzt_layer_width

        receiver = Receiver1D(
            position=receiver_position_relative,
            radius=radius_receiver,
            extension=extension_receiver,
            pzt_layer_width=pzt_layer_width
        )
        receiver.create_spatial_function(spatial_axis=spatial_axis, dx=dx)
        print(receiver_position_relative)
        # -------------------------------------------------------
        # 9) FORWARD MODEL: PSEUDO-SPECTRAL
        # -------------------------------------------------------
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

        # -------------------------------------------------------
        # 10) EXTRACT SYNTHETIC SIGNAL AT RECEIVER
        # -------------------------------------------------------
        simulated_waveform = np.sum(wavefield_forward * receiver.spatial_function, axis=1)

        # -------------------------------------------------------
        # 11) NORMALIZE TO MATCH OBSERVED, IF REQUESTED
        # -------------------------------------------------------
        if normalize_waveform and np.max(simulated_waveform) != 0:
            amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
            simulated_waveform *= amplitude_scale

        # -------------------------------------------------------
        # 12) INTERPOLATE ONTO OBSERVED TIME AXIS
        # -------------------------------------------------------
        synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

        # -------------------------------------------------------
        # 13) OPTIONAL PLOTTING
        # -------------------------------------------------------
        if enable_plotting and plot_output_path:
            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=plot_output_path
            )
            # also plot velocity model
            model_output_name = plot_output_path.name + "_velocity_model"
            model_output_path = plot_output_path.parent / model_output_name
            velocity_model_handler.plot(outfile_path=model_output_path)

        # -------------------------------------------------------
        # 14) OPTIONAL MOVIE
        # -------------------------------------------------------
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

        # -------------------------------------------------------
        # 15) RETURN RESULTS
        # -------------------------------------------------------
        return (
            synthetic_waveform,
            wavefield_forward,
            velocity_model_handler,
            sim_time_handler,
            grid_handler,
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
        minimum_velocity: float,
        steel_velocity: float,
        # The same forward-simulation inputs for consistency
        **forward_args
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform iterative gradient-based local inversion using repeated calls
        to `forward_simulation`. This method simply orchestrates the optimization.
        """
        (
        synthetic_waveform,
        wavefield_forward,
        velocity_model_handler,
        sim_time_handler,
        grid_handler,
        source,
        receiver
        ) = self.forward_simulation(
            observed_time=observed_time,
            observed_waveform=observed_waveform,
            misfit_interval=misfit_interval,
            **forward_args
        )

        num_x = grid_handler.total_grid_points
        delta_x=grid_handler.dx
        spatial_axis = grid_handler.spatial_axis
        delta_t = sim_time_handler.dt
        num_t = sim_time_handler.num_t
        source_spatial_function= source.spatial_function
        receiver_spatial_function = receiver.spatial_function
        source_time_function = source.time_function
        velocity_model = velocity_model_handler.velocity_model
        idx_dict = velocity_model_handler.idx_dict

        # Compute residuals and misfit
        initial_misfit = compute_misfit(
            observed_waveform=observed_waveform,
            synthetic_waveform=synthetic_waveform,
            misfit_interval=misfit_interval
        )
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
                num_x = num_x,
                delta_x=delta_x,
                delta_t = delta_t,
                num_t = num_t,
                source_spatial_function= source_spatial_function,
                source_time_function= source_time_function,
                velocity_model=velocity_model,  # Use the current velocity model
                compute_derivative=True
            )

            # Record the simulated wavefield at the receiver position
            simulated_waveform = np.sum(wavefield_forward * receiver_spatial_function, axis=1)
            if normalize_waveform:
                amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
                simulated_waveform *= amplitude_scale

            # Interpolate the synthetic waveform onto the observed time axis
            simulation_time = sim_time_handler
            synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

            # Compute misfit
            misfit = compute_misfit(
                observed_waveform=observed_waveform,
                synthetic_waveform=synthetic_waveform,
                misfit_interval=misfit_interval
            )
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
                residual = synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]
                adj_src_time_function = 2 * residual[::-1]  # Reverse time

                # Create an adjoint source at the receiver's position
                adjoint_source = Source1D(
                    stf_time=observed_time,
                    stf_waveform=adj_src_time_function,
                    position=receiver.position,  # Use the receiver's position
                    radius=receiver.radius       # Use the receiver's radius
                )
                # Interpolate the adjoint source time function
                adjoint_source.interpolate_time_function(dt=delta_t, simulation_time=simulation_time)
                # Create the spatial function for the adjoint source
                adjoint_source.create_spatial_function(spatial_axis=spatial_axis, dx=delta_x)

                # Perform the adjoint wavefield simulation
                wavefield_adjoint = pseudospectral_1D(
                    num_x=num_x,
                    delta_x=delta_x,
                    num_t=num_t,
                    delta_t=delta_t,
                    source_spatial=adjoint_source.spatial_function,
                    source_time=adjoint_source.time_function,
                    velocity_model=velocity_model,  # Use the current velocity model
                    compute_derivative=False
                )

                # Compute the gradient
                gradient = np.zeros_like(velocity_model)
                for t_step in range(num_t):
                    gradient += (2 / (velocity_model ** 3)) * wavefield_adjoint[t_step, :] * derivative_wavefield_forward[t_step, :]
                gradient *= delta_t

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
                velocity_min = minimum_velocity  # Minimum velocity
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

# lab_uw/forward_modeling.py

import sys
from math import floor
import numpy as np
from numpy import linalg as LA
from typing import Union, Tuple, Optional, Dict, Any

from lab_uw.simulation_setup import (
    Grid1D,
    VelocityModel1D_DDS,
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
    """

    def __init__(self, plotter: Optional["Plotter"] = None):
        """
        Initialize the ForwardModeler.

        Args:
            plotter (Plotter, optional): For plotting results.
        """
        self.plotter = plotter or Plotter()
        self._last_forward_results = None  # Will store the most recent forward-sim results

    def forward_simulation(
        self,
        geometry_type      : str,                        
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
        Union[VelocityModel1D_DDS, VelocityModel1D_SingleBlock],  # velocity_model_handler
        SimulationTime,
        Grid1D,
        Source1D,
        Receiver1D
    ]:
        """
        Unifies forward modeling under different 1D geometries ("dds" or "block").
        The method unpacks geometry-specific parameters, builds the velocity model,
        runs the pseudo-spectral simulation, and returns the final results.
        """
        # COMMON: UNPACK SOURCE TIME FUNCTION
        stf_waveform = stf_handler.waveform_data
        stf_time     = stf_handler.metadata["time_ax_waveform"]

        # MONTECARLO OR DEFAULT PARAMS
        if montecarlo:
            spreading_factor_transmitter = montecarlo["spreading_factor_transmitter"]
            spreading_factor_receiver    = montecarlo["spreading_factor_receiver"]
            position2edge_transmitter    = montecarlo["position2edge_transmitter"]
            position2edge_receiver       = montecarlo["position2edge_receiver"]
            radius_factor_transmitter    = montecarlo["radius_factor_transmitter"]
            radius_factor_receiver       = montecarlo["radius_factor_receiver"]
        else:
            spreading_factor_transmitter = 0.00001
            spreading_factor_receiver    = 0.00001
            position2edge_transmitter    = 0
            position2edge_receiver       = 0
            radius_factor_transmitter    = 1
            radius_factor_receiver       = 1

        # ASSEMBLY DICT MUST HAVE THE
        wave_type            = assembly_dict["wave_type"]
        sample_dimensions    = assembly_dict["sample_dimensions"]
        transmitter_position = assembly_dict["transmitter_position"]
        receiver_position    = assembly_dict["receiver_position"]

        # For either "dds" or "block", we retrieve pzt and pla widths:
        if geometry_type.lower() == "dds":
            # Double-Direct-Shear geometry
            side1_params      = assembly_dict["side1_params"] 
            side2_params      = assembly_dict["side2_params"] 
            pzt_layer_width   = side1_params["pzt_layer_width"]
            pla_layer_width   = side1_params["pla_layer_width"]
        elif geometry_type.lower() == "block":
            # Single-block geometry
            pzt_layer_width      = assembly_dict["pzt_layer_width"]     
            pla_layer_width      = assembly_dict["pla_layer_width"] 
        else:
            raise ValueError(f"Unknown geometry_type: {geometry_type}.")

        # CREATE THE 1D GRID
        total_length = (
            np.sum(sample_dimensions) 
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

        # DEFINE TIME AXIS
        sim_time_handler = SimulationTime(
            observed_time=observed_time,
            dx=dx,
            max_velocity=maximum_velocity
        )
        simulation_time = sim_time_handler.simulation_time
        dt              = sim_time_handler.dt
        num_t           = sim_time_handler.num_t

        # -------------------------------------------------------
        # GEOMETRY-SPECIFIC: BUILD MODEL
        # -------------------------------------------------------
        if geometry_type.lower() == "dds":
            try:
                gouge_velocity_1 = assembly_dict["gouge_velocity_1"]
                gouge_velocity_2 = assembly_dict["gouge_velocity_2"]
            except KeyError:
                raise ValueError("`gouge_velocity_1` and `gouge_velocity_2` are required for DDS geometry.")
            
            central_params   = assembly_dict["central_params"]
            side1_params     = assembly_dict["side1_params"] 
            h_groove_central = central_params["h_grooves"]
            h_groove_side    = side1_params["h_grooves"]

            steel_velocity = side1_params["velocity" + wave_type]
            pzt_velocity   = side1_params["pzt_velocity" + wave_type]
            pla_velocity   = side1_params["pla_velocity" + wave_type]

            velocity_model_handler = VelocityModel1D_DDS(
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
            steel_velocity = assembly_dict["velocity" + wave_type]
            pzt_velocity   = assembly_dict["pzt_velocity" + wave_type]
            pla_velocity   = assembly_dict["pla_velocity" + wave_type]

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

        # Store references
        velocity_model_handler.x = spatial_axis
        velocity_model = velocity_model_handler.values
        idx_dict       = velocity_model_handler.idx_dict

        # BUILD SOURCE
        transmitter_position_relative = (
            pzt_layer_width
            + position2edge_transmitter * pzt_layer_width
            + pla_layer_width
        )
        radius_transmitter = floor(radius_factor_transmitter * (pzt_layer_width / 2) / dx)
        extension_transmitter = spreading_factor_transmitter * pzt_layer_width

        source_handler = Source1D(
            stf_time=stf_time,
            stf_waveform=stf_waveform,
            position=transmitter_position_relative,
            radius=radius_transmitter,
            extension=extension_transmitter,
            pzt_layer_width=pzt_layer_width
        )
        
        source_handler.interpolate_time_function(dt=dt, simulation_time=simulation_time)
        source_handler.create_spatial_function(spatial_axis=spatial_axis, dx=dx)

        # BUILD RECEIVER
        receiver_position_relative = (
            total_length
            - pzt_layer_width
            - position2edge_receiver * pzt_layer_width
            - pla_layer_width
        )
        radius_receiver    = floor(radius_factor_receiver * (pzt_layer_width / 2) / dx)
        extension_receiver = spreading_factor_receiver * pzt_layer_width

        receiver_handler = Receiver1D(
            position=receiver_position_relative,
            radius=radius_receiver,
            extension=extension_receiver,
            pzt_layer_width=pzt_layer_width
        )
        receiver_handler.create_spatial_function(spatial_axis=spatial_axis, dx=dx)

        # FORWARD MODEL: PSEUDO-SPECTRAL
        wavefield_forward = pseudospectral_1D(
            num_x=num_x,
            delta_x=dx,
            num_t=num_t,
            delta_t=dt,
            source_spatial_function=source_handler.spatial_function,
            source_time_function=source_handler.time_function,
            velocity_model=velocity_model,
            compute_derivative=False
        )

        # EXTRACT SYNTHETIC SIGNAL AT RECEIVER
        simulated_waveform = np.sum(wavefield_forward * receiver_handler.spatial_function, axis=1)

        # NORMALIZE IF REQUESTED
        if normalize_waveform and np.max(simulated_waveform) != 0:
            amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
            simulated_waveform *= amplitude_scale

        # INTERPOLATE ONTO OBSERVED TIME AXIS
        synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

        # Optionally do plotting or movie
        if enable_plotting:
            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=plot_output_path
            )
            model_output_name = plot_output_path.name + "_velocity_model"
            model_output_path = plot_output_path.parent / model_output_name
            velocity_model_handler.plot(outfile_path=model_output_path)

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

        # Store results in the instance so we can reuse them
        self.forward_results = {
            "synthetic_waveform":       synthetic_waveform,
            "wavefield_forward":        wavefield_forward,
            "velocity_model_handler":   velocity_model_handler,
            "sim_time_handler":         sim_time_handler,
            "grid_handler":             grid_handler,
            "source_handler":           source_handler,
            "receiver_handler":         receiver_handler,
        }

    def run_local_inversion(
        self,
        observed_time       : np.ndarray,
        observed_waveform   : np.ndarray,
        misfit_interval     : np.ndarray,
        n_iterations        : int,
        dc_max_start        : float,
        reduce_factor       : float,
        dc_threshold        : float,
        minimum_velocity    : float,
        maximum_velocity    : float,
        normalize_waveform  : bool = True,
        enable_plotting     : bool = False,
        plot_output_path    : str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform iterative gradient-based local inversion using repeated calls
        to the pseudo-spectral solver, *without* re-calling forward_simulation.
        
        Instead, we rely on the user having already called `forward_simulation(...)`,
        so that `self._last_forward_results` is populated.
        """
        # Check we do have forward-simulation results available:
        if self.forward_results is None:
            raise RuntimeError(
                "No forward-simulation results found. Please call `forward_simulation()` first "
                "so that `run_local_inversion` can use the same wavefield, velocity_model, etc."
            )

        # Unpack the stored forward-simulation results
        synthetic_waveform      = self.forward_results["synthetic_waveform"]
        wavefield_forward       = self.forward_results["wavefield_forward"]
        velocity_model_handler  = self.forward_results["velocity_model_handler"]
        sim_time_handler        = self.forward_results["sim_time_handler"]
        grid_handler            = self.forward_results["grid_handler"]
        source_handler          = self.forward_results["source_handler"]
        receiver_handler        = self.forward_results["receiver_handler"]

        # Basic references
        num_x   = grid_handler.total_grid_points
        delta_x = grid_handler.dx
        delta_t = sim_time_handler.dt
        num_t   = sim_time_handler.num_t
        spatial_axis = grid_handler.spatial_axis

        source_spatial_function   = source_handler.spatial_function
        source_time_function      = source_handler.time_function
        receiver_spatial_function = receiver_handler.spatial_function

        velocity_model = velocity_model_handler.values
        idx_dict       = velocity_model_handler.idx_dict

        # Compute initial misfit
        initial_misfit = compute_misfit(
            observed_waveform=observed_waveform,
            synthetic_waveform=synthetic_waveform,
            misfit_interval=misfit_interval
        )
        print(f"Initial Misfit: {initial_misfit}")

        # Initialize gradient descent parameters
        dc_max       = dc_max_start
        misfit_prec  = initial_misfit
        best_velocity_model = velocity_model.copy()

        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")

            # Forward modeling with derivative (for gradient)
            wavefield_forward, derivative_wavefield_forward = pseudospectral_1D(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial_function=source_spatial_function,
                source_time_function=source_time_function,
                velocity_model=velocity_model,
                compute_derivative=True
            )

            # Record the simulated wavefield at the receiver
            simulated_waveform = np.sum(wavefield_forward * receiver_spatial_function, axis=1)
            if normalize_waveform and np.max(simulated_waveform) != 0:
                amplitude_scale = np.amax(observed_waveform) / np.amax(simulated_waveform)
                simulated_waveform *= amplitude_scale

            # Interpolate onto observed-time axis
            simulation_time   = sim_time_handler.simulation_time
            synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

            # Compute misfit
            misfit = compute_misfit(
                observed_waveform=observed_waveform,
                synthetic_waveform=synthetic_waveform,
                misfit_interval=misfit_interval
            )
            print(f"Misfit: {misfit}")

            # Check threshold
            if dc_max < dc_threshold:
                print(f"Gradient step reduced below threshold ({dc_threshold}). Stopping.")
                break

            if misfit <= misfit_prec:
                # If misfit is better, update best model
                misfit_prec        = misfit
                best_velocity_model = velocity_model.copy()
                best_synthetic_waveform = synthetic_waveform.copy()
                print("Misfit decreased, updating best model and best synthetic waveform.")

                # Create the adjoint source from residual
                residual = synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]
                residual_norm = np.sqrt(np.sum(residual**2))
                if residual_norm != 0:
                    adj_src_time_function = np.flipud(residual / residual_norm)
                else:
                    adj_src_time_function = np.zeros_like(residual)

                # adj_src_time_function = np.flipud(residual)  # Reverse in time

                # Adjoint source at the receiver
                adjoint_source_handler = Source1D(
                    stf_time=observed_time,
                    stf_waveform=adj_src_time_function,
                    position=receiver_handler.position,
                    radius=receiver_handler.radius,
                    extension= receiver_handler.extension,
                    pzt_layer_width= receiver_handler.pzt_layer_width
                )

                adjoint_source_handler.interpolate_time_function(dt=delta_t, simulation_time=simulation_time)
                adjoint_source_handler.create_spatial_function(spatial_axis=spatial_axis, dx=delta_x)
                
                # Adjoint wavefield
                wavefield_adjoint = pseudospectral_1D(
                    num_x=num_x,
                    delta_x=delta_x,
                    num_t=num_t,
                    delta_t=delta_t,
                    source_spatial_function=adjoint_source_handler.spatial_function,
                    source_time_function=adjoint_source_handler.time_function,
                    velocity_model=velocity_model,
                    compute_derivative=False
                )

                # Compute gradient
                gradient = np.zeros_like(velocity_model)
                for t_step in range(num_t):
                    gradient += (
                        (2.0 / (velocity_model ** 3.0))
                        * wavefield_adjoint[t_step, :]
                        * derivative_wavefield_forward[t_step, :]
                    )
                gradient *= delta_t

                # Zero out everything but the gouge regions
                gradient_update = np.zeros(num_x)
                try: 
                    regions_to_update = np.concatenate([idx_dict["gouge_1"], idx_dict["gouge_2"]])
                except:
                    regions_to_update = np.where(spatial_axis>=spatial_axis[0])  # this is very stupid, must fix it

                gradient_update[regions_to_update] = gradient[regions_to_update]

                # Rescale step
                dE_max = np.max(np.abs(gradient_update[regions_to_update]))
                velocity_model[regions_to_update] -= (dc_max / dE_max) * gradient_update[regions_to_update]

                # Clip velocities to physical bounds
                velocity_model[regions_to_update] = np.clip(
                    velocity_model[regions_to_update],
                    a_min=minimum_velocity,
                    a_max=maximum_velocity,
                )

            else:
                # Misfit got worse: reduce step and revert
                dc_max = dc_max/reduce_factor
                print(f"Misfit increased, reducing gradient step to {dc_max}")
                velocity_model = best_velocity_model.copy()

        if enable_plotting:
            plot_output_name = plot_output_path.name + "_local_inversion"
            plot_output_path = plot_output_path.parent / plot_output_name
            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=best_synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=plot_output_path
            )
            model_output_name = plot_output_path.name + "_velocity_model"
            model_output_path = plot_output_path.parent / model_output_name
            velocity_model_handler.plot(outfile_path=model_output_path)
        
        return best_synthetic_waveform, best_velocity_model


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

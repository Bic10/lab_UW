# lab_uw/forward_modeling.py

import sys
from math import floor
import numpy as np
from numpy import linalg as LA
from typing import Union, Tuple, Optional, Dict, Any

from lab_uw.simulation_setup import Grid1D, VelocityModel1D, Source1D, Receiver1D, SimulationTime, VelocityModel1D_SingleBlock
from lab_uw.plotting import Plotter
from lab_uw.signal_processing import SignalProcessor
from lab_uw.data_io import UltrasonicDataHandler

class ForwardModeler:
    """
    Class for simulating ultrasonic wave propagation and optionally
    performing gradient-based local inversion.
    """

    def __init__(self, plotter: Optional["Plotter"] = None):
        """
        Initialize the ForwardModeler.

        Args:
            plotter (Plotter, optional): An instance of a Plotter class for plotting results.
        """
        self.plotter = plotter or Plotter()

    def dds_forward_simulation(
        self,
        observed_time       : np.ndarray,
        observed_waveform   : np.ndarray,
        stf_handler         : UltrasonicDataHandler,
        frequency_cutoff    : float,
        assembly_dict       : Dict[str,Any],
        gouge_velocity      : Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
        misfit_interval     : np.ndarray,
        minimum_velocity    : Optional[float] = None,
        maximum_velocity    : Optional[float] = None,
        montecarlo          : Optional[Dict[str,Any]] = None,        
        idx_dict            : Optional[Dict[str, np.ndarray]] = None,
        # Some general optional flags:
        normalize_waveform  : bool = True,
        enable_plotting     : bool = False,
        make_movie          : bool = False,
        plot_output_path    : Optional[str] = None,
        movie_output_path   : Optional[str] = "simulation_movie.mp4",
    ) -> Tuple[np.ndarray,np.ndarray,Dict[str, np.ndarray],np.ndarray, np.ndarray, float,int, np.ndarray,float,int,Source1D,Receiver1D]:
        """
        Perform the forward modeling (one pass) of ultrasonic wave propagation
        in a 1D layered medium.

        Args:
            observed_time, observed_waveform: 1D arrays of the measured data.
            stf_time, stf_waveform: 1D arrays defining the source time function.
            frequency_cutoff: frequency cutoff for building grid spacing.
            misfit_interval: 1D array of indices over which we compute the misfit.
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

        # Unpack Source Time Function handler
        stf_waveform = stf_handler.waveform_data
        stf_time     = stf_handler.metadata["time_ax_waveform"]

        # Unpack gouge velocities
        gouge_velocity_1, gouge_velocity_2 = gouge_velocity

        # Unpack assembly parameters
        gouge_thickness_1    = assembly_dict["thickness_gouge_1"]  # in cm
        gouge_thickness_2    = assembly_dict["thickness_gouge_2"]  # in cm
        wave_type            = assembly_dict["wave_type"]
        side1_params         = assembly_dict["side1_params"] 
        side2_params         = assembly_dict["side2_params"] 
        central_params       = assembly_dict["central_params"] 
        transmitter_position = assembly_dict["transmitter_position"]
        receiver_position    = assembly_dict["receiver_position"]
        h_groove_central     = central_params["h_grooves"]
        h_groove_side        = side1_params["h_grooves"]
        pla_layer_width      = side1_params["pla_layer_width"]     
        pzt_layer_width      = side1_params["pzt_layer_width"]     
        steel_velocity       = side1_params["velocity" + wave_type]
        pzt_velocity         = side1_params["pzt_velocity" + wave_type]
        pla_velocity         = side1_params["pla_velocity" + wave_type]
        
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
            position2edge_transmitter    = 0.5
            position2edge_receiver       = 0.5
            radius_factor_transmitter    = 0.1
            radius_factor_receiver       = 0.1

        sample_dimensions = [
            side1_params["z"],
            gouge_thickness_1,
            central_params["z"],
            gouge_thickness_2,
            side2_params["z"]
        ]

        # Compute total length for 1D domain
        total_length = (
            np.sum(sample_dimensions)
            + 2 * pla_layer_width
            + 2 * pzt_layer_width
            - (transmitter_position + receiver_position)
        )

        if not minimum_velocity:
            minimum_velocity = min(gouge_velocity_1,gouge_velocity_2,pzt_velocity,steel_velocity)
        if not maximum_velocity:
            maximum_velocity = max(gouge_velocity_1,gouge_velocity_2,pzt_velocity,steel_velocity)

        # create the 1D grid space axis
        grid = Grid1D(
            cmin=minimum_velocity,
            fmax=frequency_cutoff,
            grid_len=total_length,
            ppt=10  # points per wavelength
        )
        spatial_axis = grid.spatial_axis
        dx = grid.dx
        num_x = grid.total_grid_points

        # Define time axis
        sim_time_handler = SimulationTime(
            observed_time=observed_time,
            dx=dx,
            max_velocity=maximum_velocity
        )
        simulation_time = sim_time_handler.simulation_time
        dt = sim_time_handler.dt
        num_t = sim_time_handler.num_t

        # build velocity model
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

        transmitter_position_relative = pzt_layer_width + position2edge_transmitter*pzt_layer_width + pla_layer_width
        radius_transmitter = floor(radius_factor_transmitter * (pzt_layer_width/2) / dx)
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

        # Initialize Receiver
        receiver_position_relative = total_length - pzt_layer_width - position2edge_receiver*pzt_layer_width - pla_layer_width
        radius_receiver = floor(radius_factor_receiver * (pzt_layer_width/2) / dx)
        extension_receiver = spreading_factor_receiver * (pzt_layer_width)

        receiver = Receiver1D(
            position=receiver_position_relative,
            radius=radius_receiver,
            extension=extension_receiver,
            pzt_layer_width=pzt_layer_width
        )
        receiver.create_spatial_function(spatial_axis=spatial_axis, dx=dx)
            
        # Forward modeling 
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

            model_output_name = plot_output_path.name + "_velocity_model"
            model_output_path = plot_output_path.parent / model_output_name
            print(f"model output path: {model_output_name}")
            velocity_model_handler.plot(outfile_path=model_output_path )

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
            spatial_axis,
            wavefield_forward,
            source,
            receiver
        )

    def block_forward_simulation(
        self,
        observed_time: np.ndarray,
        observed_waveform: np.ndarray,
        stf_handler : UltrasonicDataHandler,
        frequency_cutoff: float,
        assembly_dict: Dict[str,Any],
        montecarlo: Dict[str,Any],
        misfit_interval: np.ndarray,
        minimum_velocity: Optional[float] = None,
        maximum_velocity: Optional[float] = None,
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
            frequency_cutoff: frequency cutoff for building grid spacing.
            misfit_interval: 1D array of indices over which we compute the misfit.
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

        # Unpack Source Time Function handler
        stf_waveform = stf_handler.waveform_data
        stf_time     = stf_handler.metadata["time_ax_waveform"]

        # Unpack assembly parameters
        wave_type = assembly_dict["wave_type"]
        transmitter_position = assembly_dict["transmitter_position"]
        receiver_position    = assembly_dict["receiver_position"]
        pla_layer_width      = assembly_dict["pla_layer_width"]     
        pzt_layer_width      = assembly_dict["pzt_layer_width"]     
        pla_velocity         = assembly_dict["pla_velocity" + wave_type]
        steel_velocity       = assembly_dict["velocity" + wave_type]
        pzt_velocity         = assembly_dict["pzt_velocity" + wave_type]
        
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
            position2edge_transmitter    = 0.5
            position2edge_receiver       = 0.5
            radius_factor_transmitter    = 0.1
            radius_factor_receiver       = 0.1

        sample_dimensions = [assembly_dict["z"]]

        # Compute total length for 1D domain
        total_length = (
            np.sum(sample_dimensions)
            + 2 * pla_layer_width
            + 2 * pzt_layer_width
        )

        # create the 1D grid space axis
        grid = Grid1D(
            cmin=minimum_velocity,
            fmax=frequency_cutoff,
            grid_len=total_length,
            ppt=10  # points per wavelength
        )
        spatial_axis = grid.spatial_axis
        dx = grid.dx
        num_x = grid.total_grid_points

        # Define time axis
        sim_time_handler = SimulationTime(
            observed_time=observed_time,
            dx=dx,
            max_velocity=maximum_velocity
        )
        simulation_time = sim_time_handler.simulation_time
        dt = sim_time_handler.dt
        num_t = sim_time_handler.num_t

        # build velocity model
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
        velocity_model = velocity_model_handler.values
        idx_dict = velocity_model_handler.idx_dict

        transmitter_position_relative = pzt_layer_width + position2edge_transmitter*pzt_layer_width + pla_layer_width
        radius_transmitter = floor(radius_factor_transmitter * (pzt_layer_width/2) / dx)
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

        # Initialize Receiver
        receiver_position_relative = total_length - pzt_layer_width - position2edge_receiver*pzt_layer_width - pla_layer_width
        radius_receiver = floor(radius_factor_receiver * (pzt_layer_width/2) / dx)
        extension_receiver = spreading_factor_receiver * (pzt_layer_width)

        receiver = Receiver1D(
            position=receiver_position_relative,
            radius=radius_receiver,
            extension=extension_receiver,
            pzt_layer_width=pzt_layer_width
        )
        receiver.create_spatial_function(spatial_axis=spatial_axis, dx=dx)
        # import matplotlib.pyplot as plt
        # plt.plot(spatial_axis,receiver.spatial_function)
        # plt.plot(spatial_axis,source.spatial_function)
        # plt.show()
        # Forward modeling 
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

            model_output_name = plot_output_path.name + "_velocity_model"
            model_output_path = plot_output_path.parent / model_output_name
            velocity_model_handler.plot(outfile_path=model_output_path )

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
            spatial_axis,
            wavefield_forward,
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
            adjoint_source.create_spatial_function(spatial_axis=spatial_axis, dx=dx)

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

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

            velocity_model_handler.apply_smoothing_between("pzt_1", "steel_block", len(velocity_model_handler.idx_dict["pzt_1"]))
            velocity_model_handler.apply_smoothing_between("steel_block", "pzt_2", len(velocity_model_handler.idx_dict["pzt_1"]))

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
        )

        # EXTRACT SYNTHETIC SIGNAL AT RECEIVER
        simulated_waveform = np.sum(wavefield_forward * receiver_handler.spatial_function, axis=1)
        # INTERPOLATE ONTO OBSERVED TIME AXIS
        synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

        # NORMALIZE IF REQUESTED
        if normalize_waveform and np.max(synthetic_waveform) != 0:
            start_A0 = np.searchsorted(observed_time, 15)
            end_A0 = np.searchsorted(observed_time, 25)
            A0 = np.amax(observed_waveform[start_A0:end_A0])
            A1 = np.amax(observed_waveform[3*start_A0:3*start_A0+end_A0])
            
            amplitude_scale_A0 = np.amax(synthetic_waveform) / A0
            amplitude_scale_A1 = np.amax(synthetic_waveform) / A1

            synthetic_waveform[start_A0:end_A0] /= amplitude_scale_A0
            synthetic_waveform[3*start_A0:3*end_A0+end_A0] /= amplitude_scale_A1

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
        observed_time:         np.ndarray,
        observed_waveform:     np.ndarray,
        misfit_interval:       np.ndarray,
        n_iterations:          int,
        dc_max_start:          float,
        reduce_factor:         float,
        dc_threshold:          float,
        minimum_velocity:      float,
        maximum_velocity:      float,
        normalize_waveform:    bool = True,
        enable_plotting:       bool = False,
        plot_output_path:      str   = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Local inversion that adjusts the velocity_model of a 1D wave equation.

        - Always compute gradient (forward + adjoint) on every iteration.
        - Accept or revert based on improvement in the misfit.

        Returns
        -------
        best_synthetic_waveform : np.ndarray
            The 1D waveform (sampled at the receiver) for the best model.
        best_velocity_model : np.ndarray
            The best velocity model found during the inversion.
        """

        if self.forward_results is None:
            raise RuntimeError(
                "No forward-simulation results found. Please call `forward_simulation()` first."
            )

        # Unpack forward-simulation results
        initial_synthetic_waveform = self.forward_results["synthetic_waveform"]
        initial_wavefield_forward  = self.forward_results["wavefield_forward"]

        velocity_model_handler     = self.forward_results["velocity_model_handler"]
        sim_time_handler           = self.forward_results["sim_time_handler"]
        grid_handler               = self.forward_results["grid_handler"]
        source_handler             = self.forward_results["source_handler"]
        receiver_handler           = self.forward_results["receiver_handler"]

        # Basic references
        num_x           = grid_handler.total_grid_points
        delta_x         = grid_handler.dx
        delta_t         = sim_time_handler.dt
        num_t           = sim_time_handler.num_t
        simulation_time = sim_time_handler.simulation_time
        spatial_axis    = grid_handler.spatial_axis

        # Source & receiver spatial weighting
        source_spatial_function   = source_handler.spatial_function
        receiver_spatial_function = receiver_handler.spatial_function

        # We start with the same velocity model used in forward_results
        initial_velocity_model = velocity_model_handler.values.copy()
        idx_dict               = velocity_model_handler.idx_dict

        # Regions that we allow to update 
        try: 
            # Zero out everything but the gouge regions
            regions_to_update = np.concatenate([idx_dict["gouge_1"], idx_dict["gouge_2"]])
        except:
            # regions_to_update = np.where(spatial_axis>=spatial_axis[0])  # this is very stupid, must fix it
            regions_to_update = np.concatenate([
                idx_dict["pzt_1"],
                idx_dict["steel_block"][:len(idx_dict["pzt_1"])],
                idx_dict["steel_block"][-len(idx_dict["pzt_1"]):],
                idx_dict["pzt_2"]
            ])

        # -----------------------------------------------------------------------
        # (B) COMPUTE THE "BEST" MODEL VARIABLES (same as initial at iteration 0)
        # -----------------------------------------------------------------------
        # Best model: velocity, wavefield, derivative wavefield, waveform, misfit
        best_velocity_model     = initial_velocity_model.copy()
        best_wavefield_forward  = initial_wavefield_forward
        best_derivative_wavefield_forward = compute_time_derivative(best_wavefield_forward, delta_t)

        # Extract the best synthetic waveform (already in forward_results)
        best_synthetic_waveform = initial_synthetic_waveform.copy()
        best_misfit = compute_misfit(
            observed_waveform=observed_waveform,
            synthetic_waveform=best_synthetic_waveform,
            misfit_interval=misfit_interval,
        )

        print(f"Initial misfit: {best_misfit}")

        dc_max = dc_max_start
        updating = True
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")

            # Check threshold
            if dc_max < dc_threshold:
                print("Step size dropped below threshold; stopping.")
                break

            # Re-initialize the gradient
            updated_velocity_model = best_velocity_model.copy()
            if updating:
                # -------------------------------------------------------------------
                # 1) Build the "residual" for the adjoint source
                # -------------------------------------------------------------------
                # (Compare best synthetic waveform to the observed data)
                residual = best_synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]

                # Time-reversed residual → adjoint source
                adj_src_time_function = np.flipud(residual)

                # Create an adjoint source at the receiver
                adjoint_source_handler = Source1D(
                    stf_time=observed_time,
                    stf_waveform=adj_src_time_function,
                    position=receiver_handler.position,
                    radius=receiver_handler.radius,
                    extension=receiver_handler.extension,
                    pzt_layer_width=receiver_handler.pzt_layer_width
                )
                adjoint_source_handler.interpolate_time_function(dt=delta_t, simulation_time=simulation_time)
                adjoint_source_handler.create_spatial_function(spatial_axis=spatial_axis, dx=delta_x)

                wavefield_adjoint = pseudospectral_1D(
                    num_x=num_x,
                    delta_x=delta_x,
                    num_t=num_t,
                    delta_t=delta_t,
                    source_spatial_function=adjoint_source_handler.spatial_function,
                    source_time_function=adjoint_source_handler.time_function,
                    velocity_model=best_velocity_model
                )

                # -------------------------------------------------------------------
                # 2) Compute the gradient wrt velocity (cross-correlation approach)
                # -------------------------------------------------------------------
                gradient_vel = np.zeros_like(best_velocity_model)
                for t_step in range(num_t):
                    # wavefield_adjoint[t_step, :]   <----> backward wavefield
                    # best_derivative_wavefield_forward[t_step, :] <----> forward derivative
                    gradient_vel += (
                        (2.0 / best_velocity_model ** 3.0)
                        * wavefield_adjoint[t_step, :]
                        * best_derivative_wavefield_forward[num_t - 1 - t_step, :]
                    )

                gradient_vel *= delta_t
                # We do a maximum step scale based on the largest gradient
                dE_max = np.max(np.abs(gradient_vel)) 

            # -------------------------------------------------------------------
            # 3) Form the "updated" model by stepping from the best model
            # -------------------------------------------------------------------
            step_size = dc_max / dE_max
            updated_velocity_model[regions_to_update] -= step_size * gradient_vel[regions_to_update]

            # Clip to physical limits
            updated_velocity_model[regions_to_update] = np.clip(
                updated_velocity_model[regions_to_update],
                a_min=minimum_velocity,
                a_max=maximum_velocity
            )

            # -------------------------------------------------------------------
            # 4) Forward modeling with "updated" velocity
            # -------------------------------------------------------------------
            wavefield_forward_updated = pseudospectral_1D(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial_function=source_spatial_function,  # same wavelet
                source_time_function=source_handler.time_function, 
                velocity_model=updated_velocity_model,
            )
            derivative_wavefield_forward_updated = compute_time_derivative(wavefield_forward_updated, delta_t)

            # Extract updated synthetic waveform at the receiver
            updated_simulated_waveform = np.sum(
                wavefield_forward_updated * receiver_spatial_function, axis=1
            )

            updated_synthetic_waveform = np.interp(observed_time, simulation_time, updated_simulated_waveform)

            if normalize_waveform and np.max(updated_synthetic_waveform) != 0:
                start_A0 = np.searchsorted(observed_time, 15)
                end_A0 = np.searchsorted(observed_time, 25)
                A0 = np.amax(observed_waveform[start_A0:end_A0])
                A1 = np.amax(observed_waveform[3*start_A0:3*start_A0+end_A0])
                
                amplitude_scale_A0 = np.amax(updated_synthetic_waveform) / A0
                amplitude_scale_A1 = np.amax(updated_synthetic_waveform) / A1

                updated_synthetic_waveform[start_A0:end_A0] /= amplitude_scale_A0
                updated_synthetic_waveform[3*start_A0:3*end_A0+end_A0] /= amplitude_scale_A1

            # Compute updated misfit
            updated_misfit = compute_misfit(
                observed_waveform=observed_waveform,
                synthetic_waveform=updated_synthetic_waveform,
                misfit_interval=misfit_interval
            )
            print(f"    Updated Misfit: {updated_misfit}")

            # -------------------------------------------------------------------
            # 5) Accept or reject update
            # -------------------------------------------------------------------
            if updated_misfit < best_misfit:
                updating = True
                # Accept: "updated" becomes the new "best"
                print("    ✓ Misfit decreased. Accepting update.")
                best_velocity_model                 = updated_velocity_model
                best_wavefield_forward              = wavefield_forward_updated
                best_derivative_wavefield_forward   = derivative_wavefield_forward_updated
                best_synthetic_waveform             = updated_synthetic_waveform
                best_misfit                         = updated_misfit
            else:
                updating = False
                # Reject: revert to best and reduce step
                dc_max /= reduce_factor
                print(f"    ✗ Misfit did not improve. Reverting and reducing step to {dc_max}")

        if enable_plotting:
            if plot_output_path is not None:
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
                velocity_model_handler.values = best_velocity_model
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
) -> np.ndarray:
    """
    Returns the wavefield as a 2D array of shape (num_t, num_x).
    """
    import numpy as np

    wavefield_current = np.zeros(num_x)
    wavefield_future = np.zeros(num_x)
    wavefield_past = np.zeros(num_x)
    wavefield = np.zeros((num_t, num_x))

    for time_step in range(num_t):
        # (Placeholder) Compute second spatial derivative with your method
        second_derivative = np.gradient(np.gradient(wavefield_current, delta_x), delta_x)
        
        # Time stepping
        wavefield_future = (
            2 * wavefield_current
            - wavefield_past
            + (velocity_model ** 2) * (delta_t ** 2) * second_derivative
        )
        
        # Source
        wavefield_future += source_spatial_function * source_time_function[time_step] * (delta_t ** 2)

        # Shift wavefields
        wavefield_past = wavefield_current.copy()
        wavefield_current = wavefield_future.copy()

        # Boundary conditions
        wavefield_current[0] = 0
        wavefield_current[-1] = 0

        wavefield[time_step, :] = wavefield_current

    return wavefield

def compute_time_derivative(
    wavefield: np.ndarray,
    delta_t: float
) -> np.ndarray:
    """
    Compute the second time derivative of the wavefield.
    wavefield.shape == (num_t, num_x).
    
    Returns a 2D array (num_t, num_x), where derivative[t, x] is the
    second time derivative at time t, position x.
    """
    import numpy as np

    num_t, num_x = wavefield.shape
    derivative = np.zeros_like(wavefield)

    # For each t in [1, num_t-2], compute the 2nd derivative.
    # This is a typical finite difference: d2u/dt2 ~ (u[t+1] - 2u[t] + u[t-1]) / (delta_t^2)
    for t in range(1, num_t - 1):
        derivative[t, :] = (
            wavefield[t + 1, :]
            - 2 * wavefield[t, :]
            + wavefield[t - 1, :]
        ) / (delta_t ** 2)

    # Depending on your choice of boundary conditions for the time derivative:
    # derivative[0, :] and derivative[num_t-1, :] might remain zero or might
    # need a one-sided difference. It depends on your modeling setup.

    return derivative

def compute_misfit(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    misfit_interval: slice
) -> float:
    """
    Compute the L2 norm misfit between observed and synthetic waveforms over a specified interval.
    """
    if np.max(synthetic_waveform) != 0:
        return LA.norm(synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval], 2)
    else:
        # workaround in case something went wrong: misfit is so high this way, this wrong simulation never become the minimum for local inverison
        return 3 * LA.norm(synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval], 2)

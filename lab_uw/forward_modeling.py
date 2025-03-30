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
from lab_uw.signal_processing import SignalProcessor

class UltrasonicModeler:
    """
    Class for simulating ultrasonic wave propagation under different 1D geometries:
     - "dds": double-direct-shear geometry (two gouge layers)
     - "block": single-block geometry
    """

    def __init__(self, plotter: Optional["Plotter"] = None):
        """
        Initialize the UltrasonicModeler.

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

        self.assembly_dict = assembly_dict
        self.stf_handler = stf_handler
        self.frequency_cutoff = frequency_cutoff

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

        # ASSEMBLY DICT MUST HAVE AT LEAST:
        self.wave_type            = assembly_dict["wave_type"]
        self.sample_dimensions    = assembly_dict["sample_dimensions"]
        self.transmitter_position = assembly_dict["transmitter_position"]
        self.receiver_position    = assembly_dict["receiver_position"]

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
            np.sum(self.sample_dimensions) 
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

            steel_velocity = side1_params["velocity" + self.wave_type]
            pzt_velocity   = side1_params["pzt_velocity" + self.wave_type]
            pla_velocity   = side1_params["pla_velocity" + self.wave_type]

            velocity_model_handler = VelocityModel1D_DDS(
                x=spatial_axis,  
                sample_dimensions=self.sample_dimensions,
                x_transmitter=self.transmitter_position,
                x_receiver=self.receiver_position,
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
            steel_velocity = assembly_dict["velocity" + self.wave_type]
            pzt_velocity   = assembly_dict["pzt_velocity" + self.wave_type]
            pla_velocity   = assembly_dict["pla_velocity" + self.wave_type]

            velocity_model_handler = VelocityModel1D_SingleBlock(
                x=spatial_axis,
                sample_dimensions=self.sample_dimensions,
                x_transmitter=self.transmitter_position,
                x_receiver=self.receiver_position,
                pzt_layer_width=pzt_layer_width,
                pla_layer_width=pla_layer_width,
                steel_velocity=steel_velocity,
                pzt_velocity=pzt_velocity,
                pla_velocity=pla_velocity,
            )

            # velocity_model_handler.apply_smoothing_between("pzt_1", "steel_block", len(velocity_model_handler.idx_dict["pzt_1"]))
            # velocity_model_handler.apply_smoothing_between("steel_block", "pzt_2", len(velocity_model_handler.idx_dict["pzt_1"]))

        # Store references
        velocity_model_handler.x = spatial_axis
        velocity_model = velocity_model_handler.values
        idx_dict       = velocity_model_handler.idx_dict

        # BUILD SOURCE
        self.transmitter_position_relative = (
            pzt_layer_width
            + position2edge_transmitter * pzt_layer_width
            + pla_layer_width
        )
        radius_transmitter = floor(radius_factor_transmitter * (pzt_layer_width / 2) / dx)
        extension_transmitter = spreading_factor_transmitter * pzt_layer_width

        source_handler = Source1D(
            stf_time=stf_handler.metadata["time_ax_waveform"],
            stf_waveform=stf_handler.waveform_data,
            position=self.transmitter_position_relative,
            radius=radius_transmitter,
            extension=extension_transmitter,
            pzt_layer_width=pzt_layer_width
        )
        
        source_handler.interpolate_time_function(dt=dt, simulation_time=simulation_time)
        source_handler.create_spatial_function(spatial_axis=spatial_axis, dx=dx)

        # BUILD RECEIVER
        self.receiver_position_relative = (
            total_length
            - pzt_layer_width
            - position2edge_receiver * pzt_layer_width
            - pla_layer_width
        )
        radius_receiver    = floor(radius_factor_receiver * (pzt_layer_width / 2) / dx)
        extension_receiver = spreading_factor_receiver * pzt_layer_width

        receiver_handler = Receiver1D(
            position=self.receiver_position_relative,
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
            first_arrival = self.assembly_dict["z"] / self.assembly_dict["velocity" + self.wave_type]
            stf_duration = self.stf_handler.metadata["time_ax_waveform"][-1]-self.stf_handler.metadata["time_ax_waveform"][0]

            start_A0 = np.searchsorted(observed_time, first_arrival)
            end_A0 = np.searchsorted(observed_time, first_arrival + stf_duration)
            A0 = np.amax(observed_waveform[start_A0:end_A0])
            A1 = np.amax(observed_waveform[3*start_A0:3*start_A0+end_A0])

            synthetic_waveform[3*start_A0:3*end_A0+end_A0] /= A0/A1
            synthetic_waveform *= np.max(np.abs(observed_waveform)/np.max(np.abs(synthetic_waveform))) 

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
                sample_dimensions=self.sample_dimensions,
                idx_dict=idx_dict,
            )

        # Store results in the instance so we can reuse them
        self.synthetic_waveform     = synthetic_waveform
        self.wavefield_forward      = wavefield_forward
        self.velocity_model_handler = velocity_model_handler
        self.sim_time_handler       = sim_time_handler
        self.grid_handler           = grid_handler
        self.source_handler         = source_handler
        self.receiver_handler       = receiver_handler
        

    def run_local_inversion(
        self,
        observed_time:         np.ndarray,
        observed_waveform:     np.ndarray,
        misfit_interval:       np.ndarray,
        n_iterations:          int,
        dc_max_start:          float,
        reduce_factor:         float,
        dc_threshold:          float,
        dw_max_start:          float,
        ds_max_start:          float,
        minimum_velocity:      float,
        maximum_velocity:      float,
        normalize_waveform:    bool = True,
        enable_plotting:       bool = False,
        plot_output_path:      str   = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Jointly invert for:
        1) velocity_model(x)
        2) source_time_function(t)
        3) source_spatial_function(x)

        Uses the same local-inversion approach from your velocity-only method,
        but extends it to handle source-time and source-spatial updates in parallel.
        
        Returns
        -------
        best_synthetic_waveform : np.ndarray
            The 1D waveform at the receiver for the best overall model.
        best_velocity_model : np.ndarray
        best_source_time_function : np.ndarray
        best_source_spatial_function : np.ndarray
        """

        if self.synthetic_waveform is None:
            raise RuntimeError("No forward-simulation results. Call `forward_simulation()` first.")

        # === UNPACK FORWARD-SIMULATION RESULTS ===
        # Store results in the instance so we can reuse them
        initial_synthetic_waveform = self.synthetic_waveform
        initial_wavefield_forward  = self.wavefield_forward

        velocity_model_handler     = self.velocity_model_handler
        sim_time_handler           = self.sim_time_handler
        grid_handler               = self.grid_handler
        source_handler             = self.source_handler
        receiver_handler           = self.receiver_handler

        # Basic references
        num_x           = grid_handler.total_grid_points
        delta_x         = grid_handler.dx
        delta_t         = sim_time_handler.dt
        num_t           = sim_time_handler.num_t
        simulation_time = sim_time_handler.simulation_time
        spatial_axis    = grid_handler.spatial_axis

        # -- We store the initial source time/spatial functions from the forward run --
        initial_source_time_function   = source_handler.time_function.copy()
        initial_source_spatial_function = source_handler.spatial_function.copy()
        
        initial_receiver_spatial_function = receiver_handler.spatial_function.copy()
        # We start with the same velocity model used in forward_results
        initial_velocity_model = velocity_model_handler.values.copy()
        idx_dict               = velocity_model_handler.idx_dict

        # Regions where velocity is allowed to update
        try: 
            regions_to_update = np.concatenate([idx_dict["gouge_1"], idx_dict["gouge_2"]])
        except:
            regions_to_update = np.concatenate([
                idx_dict["pzt_1"],
                idx_dict["steel_block"][:len(idx_dict["pzt_1"])],
                idx_dict["steel_block"][-len(idx_dict["pzt_1"]):],
                idx_dict["pzt_2"]
            ])

        stf_duration_idx = np.where(initial_source_time_function!=0)[-1][-1]
        stf_extension_idx = np.where(initial_source_spatial_function!=0)[-1][-1]
        rx_extension_idx = np.where(initial_receiver_spatial_function!=0)[0][0]

        # -----------------------------------------------------------------------
        # Store "best" parameters at iteration 0
        # -----------------------------------------------------------------------
        best_velocity_model              = initial_velocity_model.copy()
        best_source_time_function        = initial_source_time_function.copy()
        best_source_spatial_function     = initial_source_spatial_function.copy()

        best_receiver_spatial_function     = initial_receiver_spatial_function.copy()

        best_wavefield_forward           = initial_wavefield_forward
        best_derivative_wavefield_forward = compute_time_derivative(best_wavefield_forward, delta_t)
        best_synthetic_waveform          = initial_synthetic_waveform.copy()

        best_misfit = compute_misfit(
            observed_waveform=observed_waveform,
            synthetic_waveform=best_synthetic_waveform,
            misfit_interval=misfit_interval,
        )
        print(f"Initial misfit: {best_misfit}")

        # Step-size management
        dc_max = dc_max_start
        dw_max = dw_max_start
        ds_max = ds_max_start

        updating = True
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")

            # Check threshold
            if dc_max < dc_threshold:
                print("Step size dropped below threshold; stopping.")
                break

            # Prepare updated arrays from 'best'
            updated_velocity_model            = best_velocity_model.copy()
            updated_source_time_function      = best_source_time_function.copy()
            updated_source_spatial_function   = best_source_spatial_function.copy()
            updated_receiver_spatial_function = best_receiver_spatial_function.copy()

            if updating:
                # -------------------------------------------------------------------
                # Build "residual" => adjoint source
                # -------------------------------------------------------------------
                residual = best_synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]
                adj_src_time_function = np.flipud(residual)
                adj_src_time_function = np.interp(simulation_time, observed_time, adj_src_time_function)

                adj_src_spatial_function = updated_receiver_spatial_function
                # Solve adjoint wavefield with the CURRENT best velocity
                wavefield_adjoint = pseudospectral_1D(
                    num_x=num_x,
                    delta_x=delta_x,
                    num_t=num_t,
                    delta_t=delta_t,
                    source_spatial_function=adj_src_spatial_function,
                    source_time_function=adj_src_time_function,
                    velocity_model=best_velocity_model
                )

                # -------------------------------------------------------------------
                # GRADIENT wrt velocity c(x)
                # -------------------------------------------------------------------
                if dc_max: 
                    gradient_vel = np.zeros_like(best_velocity_model)
                    for t_step in range(num_t):
                        # Cross-correlate adjoint[t] with forward[T-1 - t].
                        gradient_vel += (
                            (2.0 / best_velocity_model**3.0)
                            * wavefield_adjoint[t_step, :]
                            * best_derivative_wavefield_forward[num_t - 1 - t_step, :]
                        )
                    gradient_vel *= delta_t
                    max_vel_grad = np.max(np.abs(gradient_vel))

                if dw_max:
                    # -------------------------------------------------------------------
                    # GRADIENT wrt source-time function w(t)
                    # -------------------------------------------------------------------
                    gradient_w = np.zeros_like(best_source_time_function)
                    for t_step in range(num_t):
                        integrand = wavefield_adjoint[t_step, :] * best_source_spatial_function
                        gradient_w[num_t-t_step-1] = np.sum(integrand)
                    # Multiply by delta_t to approximate integral in continuous form (optional):
                    gradient_w *= delta_t
                    gradient_w[stf_duration_idx:] = 0
                    max_w_grad   = np.max(np.abs(gradient_w))
                
                if ds_max:
                    # -------------------------------------------------------------------
                    # GRADIENT wrt source-spatial function s(x)
                    # -------------------------------------------------------------------
                    gradient_s = np.zeros_like(best_source_spatial_function)
                    for x_idx in range(num_x):
                        # wavefield_adjoint[:, x_idx] is at location x
                        # We multiply by w(t) and sum over t
                        integrand = wavefield_adjoint[:, x_idx] * best_source_time_function
                        gradient_s[num_x-x_idx-1] = np.sum(integrand)
                    # Multiply by delta_t if you want an integral in time:
                    gradient_s *= delta_x
                    gradient_s[stf_extension_idx:] = 0
                    max_s_grad   = np.max(np.abs(gradient_s))


                    # -------------------------------------------------------------------
                    # GRADIENT wrt receiver-spatial function r(x)
                    # -------------------------------------------------------------------
                    gradient_r = np.zeros_like(best_receiver_spatial_function)
                    for x_idx in range(num_x):
                        integrand = wavefield_adjoint[:, x_idx] * best_wavefield_forward[:,num_x-x_idx-1]
                        gradient_r[num_x-x_idx-1] = np.sum(integrand)
                    # Multiply by delta_t if you want an integral in time:
                    gradient_r *= delta_x
                    gradient_r[:rx_extension_idx] = 0
                    max_r_grad   = np.max(np.abs(gradient_r))

            # -------------------------------------------------------------------
            # Form the "updated" parameters by stepping from the best
            # -------------------------------------------------------------------
            if dc_max:
                # -- update velocity only in the selected region --
                step_size_vel = dc_max / (max_vel_grad + 1e-15)
                updated_velocity_model[regions_to_update] -= step_size_vel * gradient_vel[regions_to_update]
                updated_velocity_model[regions_to_update] = np.clip(
                    updated_velocity_model[regions_to_update],
                    a_min=minimum_velocity,
                    a_max=maximum_velocity
                )

            if dw_max:
                # -- update wavelet w(t) --
                step_size_w   = dw_max / (max_w_grad + 1e-15)
                updated_source_time_function -= step_size_w * gradient_w

                signal_processor = SignalProcessor()
                updated_source_time_function, _ = signal_processor.signal2noise_separation_lowpass(
                        waveform_data=updated_source_time_function,
                        metadata=self.stf_handler.metadata,
                        freq_cut=self.frequency_cutoff
                    )
                updated_source_time_function = updated_source_time_function - updated_source_time_function[0]

            if ds_max:
                # -- update spatial distribution s(x) --
                step_size_s = ds_max / (max_s_grad + 1e-15)
                updated_source_spatial_function -= step_size_s * gradient_s

                step_size_r = ds_max / (max_r_grad + 1e-15)
                updated_receiver_spatial_function -= step_size_r * gradient_r

            # -------------------------------------------------------------------
            # Forward modeling with updated parameters
            # -------------------------------------------------------------------
            # Use updated velocity, wavelet, and spatial distribution
            wavefield_forward_updated = pseudospectral_1D(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial_function=updated_source_spatial_function,
                source_time_function=updated_source_time_function,
                velocity_model=updated_velocity_model,
            )
            derivative_wavefield_forward_updated = compute_time_derivative(wavefield_forward_updated, delta_t)

            # Extract updated synthetic waveform
            updated_simulated_waveform = np.sum(
                wavefield_forward_updated * updated_receiver_spatial_function, axis=1
            )
            updated_synthetic_waveform = np.interp(observed_time, simulation_time, updated_simulated_waveform)

            if normalize_waveform and np.max(updated_synthetic_waveform) != 0:
                first_arrival = self.assembly_dict["z"] / self.assembly_dict["velocity" + self.wave_type]
                stf_duration = self.stf_handler.metadata["time_ax_waveform"][-1]-self.stf_handler.metadata["time_ax_waveform"][0]
                start_A0 = np.searchsorted(observed_time, first_arrival)
                end_A0 = np.searchsorted(observed_time, first_arrival + stf_duration)
                A0 = np.amax(observed_waveform[start_A0:end_A0])
                A1 = np.amax(observed_waveform[3*start_A0:3*start_A0+end_A0])

                updated_synthetic_waveform[3*start_A0:3*end_A0+end_A0] /= A0/A1
                updated_synthetic_waveform *= np.max(np.abs(observed_waveform)/np.max(np.abs(updated_synthetic_waveform))) 

            # Compute updated misfit
            updated_misfit = compute_misfit(
                observed_waveform=observed_waveform,
                synthetic_waveform=updated_synthetic_waveform,
                misfit_interval=misfit_interval
            )
            print(f"    Updated Misfit: {updated_misfit}")

            # -------------------------------------------------------------------
            # Accept or reject
            # -------------------------------------------------------------------
            if updated_misfit < best_misfit:
                updating = True
                print("    ✓ Misfit decreased. Accepting update.")
                best_velocity_model                 = updated_velocity_model
                best_source_time_function           = updated_source_time_function
                best_source_spatial_function        = updated_source_spatial_function
                best_receiver_spatial_function      = updated_receiver_spatial_function

                best_wavefield_forward              = wavefield_forward_updated
                best_derivative_wavefield_forward   = derivative_wavefield_forward_updated
                best_synthetic_waveform             = updated_synthetic_waveform
                best_misfit                         = updated_misfit

            else:
                updating = False
                dc_max /= reduce_factor
                dw_max /= reduce_factor
                ds_max /= reduce_factor
                print(f"    ✗ No improvement. Reverting & reducing step")

        self.synthetic_waveform                = best_synthetic_waveform
        self.velocity_model_handler.values     = best_velocity_model
        self.source_handler.time_function      = best_source_time_function
        self.source_handler.spatial_function   = best_source_spatial_function
        self.receiver_handler.spatial_function = best_receiver_spatial_function         

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

                self.velocity_model_handler.plot(outfile_path=model_output_path)

                stf_output_name = plot_output_path.name + "_best_STF"
                stf_output_path = plot_output_path.parent / stf_output_name
                self.plotter.plot_original_vs_updated_stf(
                    t=simulation_time,
                    stf_updated=best_source_time_function,
                    stf_original=initial_source_time_function,
                    min_time=simulation_time[0],
                    max_time=simulation_time[stf_duration_idx],
                    outfile_path=stf_output_path) 
    
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
        # second_derivative = np.gradient(np.gradient(wavefield_current, delta_x), delta_x)
        second_derivative = SignalProcessor().fourier_derivative_2nd(wavefield_current,delta_x)

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

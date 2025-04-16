# lab_uw/forward_modeling.py

import sys
import matplotlib.pyplot as plt
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

    def forward_simulation(
        self,
        geometry_type      : str,                        
        observed_time      : np.ndarray,
        observed_waveform  : np.ndarray,
        stf_handler        : UltrasonicDataHandler,
        frequency_cutoff   : float,
        minimum_velocity   : float,
        maximum_velocity   : float,
        assembly_dict      : Dict[str, Any],
        misfit_interval    : np.ndarray,
        absorbing          : bool = False,
        maximum_damping    : float = None,
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

        # Explicitly assign each argument to `self`
        self.geometry_type       = geometry_type
        self.observed_time       = observed_time
        self.observed_waveform   = observed_waveform
        self.stf_handler         = stf_handler
        self.frequency_cutoff    = frequency_cutoff
        self.minimum_velocity    = minimum_velocity
        self.maximum_velocity    = maximum_velocity
        self.assembly_dict       = assembly_dict
        self.misfit_interval     = misfit_interval
        self.absorbing           = absorbing
        self.maximum_damping     = maximum_damping
        self.montecarlo          = montecarlo
        self.normalize_waveform  = normalize_waveform
        self.enable_plotting     = enable_plotting
        self.make_movie          = make_movie
        self.plot_output_path    = plot_output_path
        self.movie_output_path   = movie_output_path

        # MONTECARLO OR DEFAULT PARAMS
        if montecarlo:
            spreading_factor_transmitter = montecarlo["spreading_factor_transmitter"]
            spreading_factor_receiver    = montecarlo["spreading_factor_receiver"]
            position2edge_transmitter    = montecarlo["position2edge_transmitter"]
            position2edge_receiver       = montecarlo["position2edge_receiver"]
            radius_factor_transmitter    = montecarlo["radius_factor_transmitter"]
            radius_factor_receiver       = montecarlo["radius_factor_receiver"]
            multiplier_STF               = montecarlo["multiplier_STF"]

        else:
            spreading_factor_transmitter = 1
            spreading_factor_receiver    = 1
            position2edge_transmitter    = -0.5
            position2edge_receiver       = -0.5
            radius_factor_transmitter    = 1
            radius_factor_receiver       = 1
            multiplier_STF               = 1

        # ASSEMBLY DICT MUST HAVE AT LEAST:
        self.wave_type            = assembly_dict["wave_type"]
        self.sample_dimensions    = assembly_dict["sample_dimensions"]
        self.transmitter_position = assembly_dict["transmitter_position"]
        self.receiver_position    = assembly_dict["receiver_position"]

        # For either "dds" or "block", we retrieve pzt and pla widths:
        if self.geometry_type.lower() == "dds":
            # Double-Direct-Shear geometry
            side1_params      = assembly_dict["side1_params"] 
            side2_params      = assembly_dict["side2_params"] 
            pzt_layer_width   = side1_params["pzt_layer_width"]
            pla_layer_width   = side1_params["pla_layer_width"]
            self.acquisition_time = self.assembly_dict["acquisition_time"] # contain the time, referred to the start of the experiment, when the waveforms are acquired

        elif self.geometry_type.lower() == "block":
            # Single-block geometry
            pzt_layer_width      = assembly_dict["pzt_layer_width"]     
            pla_layer_width      = assembly_dict["pla_layer_width"] 
            maximum_damping      = 0.
            self.acquisition_time= 0.

        else:
            raise ValueError(f"Unknown geometry_type: {self.geometry_type}.")

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
            observed_time = observed_time,
            dx=dx,
            max_velocity  = maximum_velocity,
            max_alpha     = maximum_damping
        )
        simulation_time = sim_time_handler.simulation_time
        dt              = sim_time_handler.dt
        num_t           = sim_time_handler.num_t

        # -------------------------------------------------------
        # GEOMETRY-SPECIFIC: BUILD MODEL
        # -------------------------------------------------------
        if self.geometry_type.lower() == "dds":
            try:
                gouge_velocity_1 = assembly_dict["gouge_velocity_1"]
                gouge_velocity_2 = assembly_dict["gouge_velocity_2"]
                gouge_damping_1  = assembly_dict["gouge_damping_1"]
                gouge_damping_2  = assembly_dict["gouge_damping_2"]
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
                gouge_damping=(gouge_damping_1, gouge_damping_2),
                pzt_velocity=pzt_velocity,
                pla_velocity=pla_velocity,
            )

        elif self.geometry_type.lower() == "block":
            steel_velocity = assembly_dict["velocity" + self.wave_type]
            pzt_velocity   = assembly_dict["pzt_velocity" + self.wave_type]
            pla_velocity   = assembly_dict["pla_velocity" + self.wave_type]
            gouge_velocity_1 = 0.
            gouge_velocity_2 = 0.
            gouge_damping_1  = 0.
            gouge_damping_2  = 0.

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

        velocity_model = velocity_model_handler.velocity_array
        damping_model  = velocity_model_handler.damping_array if geometry_type == "dds" else 0
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
            stf_waveform=stf_handler.waveform_data * multiplier_STF ,
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
        wavefield_forward = pseudospectral_1D_damped(
            num_x=num_x,
            delta_x=dx,
            num_t=num_t,
            delta_t=dt,
            source_spatial_function=source_handler.spatial_function,
            source_time_function=source_handler.time_function,
            velocity_model = velocity_model,
            damping_model  = damping_model,
            absorbing      = absorbing
        )

        # EXTRACT SYNTHETIC SIGNAL AT RECEIVER
        simulated_waveform = np.sum(wavefield_forward * receiver_handler.spatial_function, axis=1)
        # INTERPOLATE ONTO OBSERVED TIME AXIS
        synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

        # NORMALIZE IF REQUESTED
        if normalize_waveform and np.max(synthetic_waveform) != 0:
            if self.geometry_type == "block":
                first_arrival = self.assembly_dict["z"] / self.assembly_dict["velocity" + self.wave_type]
                stf_duration = self.stf_handler.metadata["time_ax_waveform"][-1]-self.stf_handler.metadata["time_ax_waveform"][0]
                start_A0 = np.searchsorted(observed_time, first_arrival)
                end_A0 = np.searchsorted(observed_time, first_arrival + stf_duration)
                A0 = np.amax(np.abs(observed_waveform[start_A0:end_A0]))
                A0_synth = np.amax(np.abs(synthetic_waveform[start_A0:end_A0]))
                multiplier_factor = A0/A0_synth
                synthetic_waveform *= multiplier_factor
                print(multiplier_factor)

                try:
                    A1 = np.amax(np.abs(observed_waveform[3*start_A0:3*start_A0+end_A0]))
                    synthetic_waveform[3*start_A0:3*end_A0+end_A0] *= A1/A0
                except:
                    pass
            elif self.geometry_type == "dds":
                synthetic_waveform *= np.sum(np.abs(observed_waveform[misfit_interval]))/np.sum(np.abs(synthetic_waveform[misfit_interval])) 

        #-------------------------------------------
        # COMPUTE MISFIT
        #-------------------------------------------
        misfit = self.compute_misfit(
            observed_waveform   = observed_waveform,
            synthetic_waveform  = synthetic_waveform,
            misfit_interval     = misfit_interval
        )

        # Store results in the instance so we can reuse them
        self.synthetic_waveform     = synthetic_waveform
        self.wavefield_forward      = wavefield_forward
        self.velocity_model_handler = velocity_model_handler
        self.sim_time_handler       = sim_time_handler
        self.grid_handler           = grid_handler
        self.source_handler         = source_handler
        self.receiver_handler       = receiver_handler
        self.misfit                 = misfit

        if self.geometry_type == "dds":
            acq_time_label   = str(round(self.acquisition_time,5)).replace(".",",")
            damping_label    = str(round(gouge_damping_1,5)).replace(".",",")
            velocity_label   = str(round(1e4*gouge_velocity_1,5)).replace(".",",")  
            label = f"_acq_time_{acq_time_label}_vel_{velocity_label}_damping_{damping_label}_global_search_{misfit:.0f}_waveform"
    
        else:
            pzt_vel_label     = str(round(1e4*pzt_velocity)).replace(".",",")
            steel_vel_label   = str(round(1e4*steel_velocity)).replace(".",",")  
            label = f"_pzt_{pzt_vel_label}_vel_{steel_vel_label}_global_search_{misfit:.0f}_waveform"

        if enable_plotting:
            plot_output_name = plot_output_path.name + label
            plot_output_path = plot_output_path.parent / plot_output_name
            self.plotter.plot_simulation_waveform(
            t=observed_time,
            sp_simulated=synthetic_waveform,
            sp_recorded=observed_waveform,
            misfit_interval=misfit_interval,
            outfile_path=plot_output_path
        )

        # if make_movie:
        #     self.plotter.make_movie_from_simulation(
        #         outfile_path=movie_output_path,
        #         x=spatial_axis,
        #         t=simulation_time,
        #         sp_field=wavefield_forward,
        #         sp_recorded=simulated_waveform,
        #         sample_dimensions=self.sample_dimensions,
        #         idx_dict=idx_dict,
        #     )


    def compute_misfit(self,
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

    def compute_amplitude_and_phase_spectrum(self, 
                                    observed_time     : np.ndarray,
                                    synthetic_waveform: np.ndarray,
                                    ) -> tuple:
            
            n_samples = len(observed_time)
            dt = observed_time[1]-observed_time[0]
            self.frequencies = np.fft.rfftfreq(n_samples, d=dt)

            # Compute the FFT along the sample axis
            fft_data = np.fft.rfft(synthetic_waveform)

            # Compute amplitude and phase
            self.amplitude_spectrum = np.abs(fft_data)
            self.phase_spectrum = np.angle(fft_data)

            return self.frequencies, self.amplitude_spectrum, self.phase_spectrum
    
    def run_local_inversion(
        self,
        n_iterations:          int,
        dc_max_start:          float = 0,
        dc_threshold:          float = 0,
        da_max_start:          float = 0,
        da_threshold:          float = 0,
        dw_max_start:          float = 0,
        dw_threshold:          float = 0,
        ds_max_start:          float = 0,
        ds_threshold:          float = 0,
        reduce_factor:         float = 1/2,
        misfit_thresold    :   float = 0.5,
        normalize_waveform :   bool = True,
        enable_plotting    :   bool = True,
        make_movie         :   bool = False,
        plot_output_path   :   Optional[str] = None,
        movie_output_path  :   Optional[str] = "simulation_movie.mp4",

    ) -> None:
        """
        Jointly invert for:
        1) velocity_model(x)
        2) source_time_function(t)
        3) source_spatial_function(x)
        """

        if self.synthetic_waveform is None:
            raise RuntimeError("No forward-simulation results. Call `forward_simulation()` first.")

        # === UNPACK FORWARD-SIMULATION RESULTS ===
        # Store results in the instance so we can reuse them
        absorbing                  = self.absorbing

        initial_synthetic_waveform = self.synthetic_waveform
        initial_wavefield_forward  = self.wavefield_forward     
        velocity_model_handler     = self.velocity_model_handler
        sim_time_handler           = self.sim_time_handler
        grid_handler               = self.grid_handler
        source_handler             = self.source_handler
        receiver_handler           = self.receiver_handler

        observed_time              = self.observed_time
        observed_waveform          = self.observed_waveform
        misfit_interval            = self.misfit_interval

        initial_misfit             = self.misfit

        print(f"Initial misfit: {initial_misfit}")

        # Basic references
        num_x           = grid_handler.total_grid_points
        delta_x         = grid_handler.dx
        delta_t         = sim_time_handler.dt
        num_t           = sim_time_handler.num_t
        simulation_time = sim_time_handler.simulation_time
        spatial_axis    = grid_handler.spatial_axis

        # Initialize the inversion variables 
        initial_source_time_function        = source_handler.time_function.copy()
        initial_source_spatial_function     = source_handler.spatial_function.copy()
        initial_receiver_spatial_function   = receiver_handler.spatial_function.copy()
        initial_velocity_model              = velocity_model_handler.velocity_array.copy()
        initial_damping_model               = velocity_model_handler.damping_array.copy()
        idx_dict                            = velocity_model_handler.idx_dict

        # Regions where velocity is allowed to update
        if self.geometry_type == "dds": 
            # regions_to_update = np.arange(num_x)
            regions_to_update = np.concatenate([
                                                idx_dict["pzt_1"],
                                                idx_dict["groove_sb1"], 
                                                idx_dict["gouge_1"],
                                                idx_dict["groove_cb1"],
                                                idx_dict["groove_cb2"],
                                                idx_dict["gouge_2"],
                                                idx_dict["groove_sb2"],
                                                idx_dict["pzt_2"]
                                            ])
            
            grooves = np.concatenate([idx_dict["groove_sb1"], 
                                    idx_dict["groove_cb1"],
                                    idx_dict["groove_cb2"],
                                    idx_dict["groove_sb2"]
                                ])
            
            gouge = np.concatenate([idx_dict["gouge_1"], idx_dict["gouge_2"]])

            
        elif self.geometry_type == "block":
            regions_to_update = np.concatenate([
                idx_dict["pzt_1"],
                idx_dict["steel_block"][:len(idx_dict["pzt_1"])],
                idx_dict["steel_block"][-len(idx_dict["pzt_1"]):],
                idx_dict["pzt_2"]
            ])

        stf_duration_idx  = np.where(initial_source_time_function!=0)[-1][-1]
        stf_extension_idx = np.where(initial_source_spatial_function!=0)[-1][-1]
        rx_extension_idx  = np.where(initial_receiver_spatial_function!=0)[0][0]

        # -----------------------------------------------------------------------
        # Store "best" parameters at iteration 0
        # -----------------------------------------------------------------------
        best_velocity_model              = initial_velocity_model.copy()
        best_damping_model               = initial_damping_model.copy()
        best_source_time_function        = initial_source_time_function.copy()
        best_source_spatial_function     = initial_source_spatial_function.copy()
        best_receiver_spatial_function   = initial_receiver_spatial_function.copy()
        best_wavefield_forward           = initial_wavefield_forward.copy()
        best_laplacian_wavefield         = compute_spatial_laplacian(best_wavefield_forward, delta_x)
        best_first_derivative_laplacian  = compute_time_derivative_of_laplacian(best_laplacian_wavefield, delta_t)

        best_synthetic_waveform          = initial_synthetic_waveform.copy()
        best_misfit                      = initial_misfit
        updated_misfit                   = initial_misfit
        previous_misfit                  = initial_misfit  # let's speed: if 2 misfit differ for less than another threshold value, stop
        # Step-size management
        dc_max = dc_max_start
        da_max = da_max_start
        dw_max = dw_max_start
        ds_max = ds_max_start

        updating = True
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")

            # Check threshold
            if (dc_max < dc_threshold) or (dw_max < dw_threshold) or ((ds_max < ds_threshold)) or (da_max < da_threshold):
                print("Step size dropped below threshold; stopping.")
                break
            
            # Prepare updated arrays from 'best'
            updated_velocity_model            = best_velocity_model.copy()
            updated_damping_model             = best_damping_model.copy()
            updated_source_time_function      = best_source_time_function.copy()
            updated_source_spatial_function   = best_source_spatial_function.copy()
            updated_receiver_spatial_function = best_receiver_spatial_function.copy()

            if updating:
                # -------------------------------------------------------------------
                # Build "residual" => adjoint source
                # -------------------------------------------------------------------
                adj_src_time_function = np.zeros(observed_waveform.shape)
                residual = best_synthetic_waveform[misfit_interval] - observed_waveform[misfit_interval]
                adj_src_time_function[misfit_interval] = np.flipud(residual)

                adj_src_time_function = np.interp(simulation_time, observed_time, adj_src_time_function)

                adj_src_spatial_function = updated_receiver_spatial_function
                # Solve adjoint wavefield with the CURRENT best velocity
                wavefield_adjoint = pseudospectral_1D_damped(
                    num_x                   = num_x,
                    delta_x                 = delta_x,
                    num_t                   = num_t,
                    delta_t                 = delta_t,
                    source_spatial_function = adj_src_spatial_function,
                    source_time_function    = adj_src_time_function,
                    velocity_model          = best_velocity_model,
                    damping_model           = best_damping_model,
                    absorbing=absorbing
                )
                
                if dc_max: 
                    # -------------------------------------------------------------------
                    # GRADIENT wrt velocity c(x)
                    # -------------------------------------------------------------------
                    gradient_vel = np.zeros_like(best_velocity_model)
                    # temp = wavefield_adjoint
                    temp = -(2.0 * best_velocity_model[None, :]) * np.flipud(wavefield_adjoint) * (best_laplacian_wavefield)
                    gradient_vel = np.sum(temp, axis=0) 
                    max_vel_grad = np.max(np.abs(gradient_vel[regions_to_update]))

                    # -------------------------------------------------------------------
                    # GRADIENT wrt damping a(x)
                    # -------------------------------------------------------------------
                if da_max:
                    gradient_damp = np.zeros_like(best_damping_model)
                    # temp = wavefield_adjoint
                    temp =  -np.flipud(wavefield_adjoint) * best_first_derivative_laplacian
                    gradient_damp = np.sum(temp, axis=0) 
                    max_damp_grad = np.max(np.abs(gradient_damp[regions_to_update]))

                if dw_max:
                    # -------------------------------------------------------------------
                    # GRADIENT wrt source-time function w(t)
                    # -------------------------------------------------------------------
                    gradient_w = np.zeros_like(best_source_time_function)
                    temp = wavefield_adjoint @ best_source_spatial_function  # shape => (num_t,)
                    gradient_w = np.flipud(temp)
                    # Multiply by delta_t to approximate integral in continuous form (optional):
                    gradient_w[stf_duration_idx:] = 0
                    max_w_grad   = np.max(np.abs(gradient_w))
                
                if ds_max:
                    # -------------------------------------------------------------------
                    # GRADIENT wrt source-spatial function s(x)
                    # -------------------------------------------------------------------
                    gradient_s = np.zeros_like(best_source_spatial_function)
                    # wavefield_adjoint => shape (num_t, num_x)
                    temp = wavefield_adjoint.T @ best_source_time_function  # shape => (num_x,)
                    gradient_s = np.flipud(temp) * delta_x

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

                # updated_velocity_model[grooves] = np.clip(updated_velocity_model[grooves],
                #                                           a_min=None,
                #                                           a_max=updated_velocity_model[idx_dict["central_block"]][0])


            if da_max:
                # -- update velocity only in the selected region --
                step_size_damp = da_max / (max_damp_grad + 1e-15)
                updated_damping_model[regions_to_update] -= step_size_damp * gradient_damp[regions_to_update]

                # updated_damping_model[grooves] = np.clip(updated_damping_model[grooves],
                #                                           a_min=updated_damping_model[idx_dict["central_block"]][0],
                #                                           a_max=None)

            if dw_max:
                # -- update wavelet w(t) --
                step_size_w   = dw_max / (max_w_grad + 1e-15)
                updated_source_time_function -= step_size_w * gradient_w

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
            wavefield_forward_updated = pseudospectral_1D_damped(
                num_x                   = num_x,
                delta_x                 = delta_x,
                num_t                   = num_t,
                delta_t                 = delta_t,
                source_spatial_function = updated_source_spatial_function,
                source_time_function    = updated_source_time_function,
                velocity_model          = updated_velocity_model,
                damping_model           = updated_damping_model,
                absorbing               = absorbing
            )

            laplacian_wavefield_updated         = compute_spatial_laplacian(wavefield_forward_updated, delta_x)
            first_derivative_laplacian_updated  = compute_time_derivative_of_laplacian(laplacian_wavefield_updated, delta_t)

            # Extract updated synthetic waveform
            updated_simulated_waveform = np.sum(
                wavefield_forward_updated * updated_receiver_spatial_function, axis=1
            )
            updated_synthetic_waveform = np.interp(observed_time, simulation_time, updated_simulated_waveform)

            if normalize_waveform and np.max(updated_synthetic_waveform) != 0:
                if self.geometry_type == "block":
                    first_arrival = self.assembly_dict["z"] / self.assembly_dict["velocity" + self.wave_type]
                    stf_duration = self.stf_handler.metadata["time_ax_waveform"][-1]-self.stf_handler.metadata["time_ax_waveform"][0]
                    start_A0 = np.searchsorted(observed_time, first_arrival)
                    end_A0 = np.searchsorted(observed_time, first_arrival + stf_duration)            
                    A0 = np.amax(np.abs(observed_waveform[start_A0:end_A0]))
                    A0_synth = np.amax(np.abs(updated_synthetic_waveform[start_A0:end_A0]))
                    multiplier_factor = A0/A0_synth
                    updated_synthetic_waveform *= multiplier_factor
                    try:
                        A1 = np.amax(np.abs(observed_waveform[3*start_A0:3*start_A0+end_A0]))
                        updated_synthetic_waveform[3*start_A0:3*end_A0+end_A0] *=A1/A0
                    except:
                        pass

                elif self.geometry_type == "dds":
                    updated_synthetic_waveform *= np.sum(abs(observed_waveform[misfit_interval]))/np.sum(np.abs(updated_synthetic_waveform[misfit_interval]))

            # Compute updated misfit
            updated_misfit = self.compute_misfit(
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
                best_damping_model                  = updated_damping_model

                best_source_time_function           = updated_source_time_function
                best_source_spatial_function        = updated_source_spatial_function
                best_receiver_spatial_function      = updated_receiver_spatial_function

                best_wavefield_forward              = wavefield_forward_updated
                best_laplacian_wavefield            = laplacian_wavefield_updated
                best_first_derivative_laplacian     = first_derivative_laplacian_updated

                best_synthetic_waveform             = updated_synthetic_waveform
                best_simulated_waveform             = updated_simulated_waveform

                best_misfit                         = updated_misfit

            elif (updated_misfit > best_misfit) and (abs(previous_misfit-updated_misfit) < misfit_thresold):
                print("Misfit updating is below threshold. Stopping!")
                break
            
            else:
                updating = False
                dc_max /= reduce_factor
                da_max /= reduce_factor
                dw_max /= reduce_factor
                ds_max /= reduce_factor
                print(f"    ✗ No improvement. Reverting & reducing step")

            previous_misfit = updated_misfit

        self.synthetic_waveform                    = best_synthetic_waveform

        self.source_handler.time_function          = best_source_time_function
        self.source_handler.spatial_function       = best_source_spatial_function
        self.receiver_handler.spatial_function     = best_receiver_spatial_function         
        self.misfit                                = best_misfit

        if self.geometry_type == "dds":
            self.velocity_model_handler.velocity_array = best_velocity_model
            self.velocity_model_handler.damping_array  = best_damping_model
            self.average_gouge_damping    = np.mean(self.velocity_model_handler.damping_array[gouge]) 
            self.average_gouge_velocity   = np.mean(self.velocity_model_handler.velocity_array[gouge]) 
            acq_time_label   = str(round(self.acquisition_time,5)).replace(".",",")
            damping_label    = str(round(self.average_gouge_damping,5)).replace(".",",")
            velocity_label   = str(round(1e4*self.average_gouge_velocity)).replace(".",",")  
            label = f"_acq_time_{acq_time_label}_vel_{velocity_label}_damping_{damping_label}_FWI_misfit_{best_misfit:.0f}_waveform"
    
        else:
            label = ""

        if enable_plotting:
            if plot_output_path:
                plot_output_name = plot_output_path.name + label
                plot_output_path = plot_output_path.parent / plot_output_name

                self.plotter.plot_simulation_waveform(
                    t=observed_time,
                    sp_simulated=best_synthetic_waveform,
                    sp_recorded=observed_waveform,
                    misfit_interval=misfit_interval,
                    outfile_path=plot_output_path
                )

                if dc_max_start:
                    model_output_name = plot_output_path.name.replace("_waveform","_velocity_model")
                    model_output_path = plot_output_path.parent / model_output_name
                    self.velocity_model_handler.plot(model=self.velocity_model_handler.velocity_array, outfile_path=model_output_path)

                if da_max_start:
                    model_output_name = plot_output_path.name.replace("_waveform","_damping_model")
                    model_output_path = plot_output_path.parent / model_output_name
                    self.velocity_model_handler.plot(model=self.velocity_model_handler.damping_array, outfile_path=model_output_path)

                if dw_max_start:
                    stf_output_name = plot_output_path.name + "_best_STF"
                    stf_output_path = plot_output_path.parent / stf_output_name
                    self.plotter.plot_original_vs_updated_stf(
                        t=simulation_time,
                        stf_updated=best_source_time_function,
                        stf_original=initial_source_time_function,
                        min_time=simulation_time[0],
                        max_time=simulation_time[stf_duration_idx],
                        outfile_path=stf_output_path) 
                    
        if (make_movie) and (initial_misfit != best_misfit):
            movie_output_name = movie_output_path.name + "_local_inversion.mp4"
            movie_output_path = movie_output_path.parent / movie_output_name

            self.plotter.make_movie_from_simulation(
                outfile_path=movie_output_path,
                x=spatial_axis,
                t=simulation_time,
                sp_field=best_wavefield_forward,
                sp_recorded = best_simulated_waveform,
                sample_dimensions=self.sample_dimensions,
                idx_dict=idx_dict,
            )
    
def pseudospectral_1D_damped(
    num_x: int,          # original interior size
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial_function: np.ndarray,  # length num_x
    source_time_function: np.ndarray,     # length num_t
    velocity_model: np.ndarray,           # length num_x
    damping_model: np.ndarray = None,     # length num_x for Kelvin-Voigt alpha
    absorbing: np.ndarray = False, # length (num_x+2*N_pad). If None, no boundary damping.
    N_pad: int = 200,     # number of points to pad on each side
):
    """
    1D wavefield solution on an extended domain so that waves leaving the 
    interior do not reflect but get damped in the outer zones.
    
    Returns wavefield of shape (num_t, num_x) containing just the interior slice.
    
    PDE:  u_tt = v(x)^2 * u_xx  +  alpha(x)* d/dt[u_xx],
    with optional absorbing zone in the extension.
    """

    if damping_model is None:
        damping_model = np.zeros(num_x)

    if absorbing:
            # Extended size
        N_ext = num_x + 2*N_pad
        
        # Extend velocity, damping, and allocate wavefields
        velocity_ext = extend_model(velocity_model, N_pad)  
        damping_ext  = extend_model(damping_model, N_pad)   

        absorbing_boundary_taper = np.ones(N_ext)
        z = np.linspace(-1, 6, N_pad)
        sigma_ext = sigmoid(z)
        absorbing_boundary_taper[0:N_pad] = sigma_ext
        absorbing_boundary_taper[-N_pad:] = np.flip(sigma_ext)

        # plt.plot(absorbing_boundary_taper)
        # plt.scatter(N_pad,absorbing_boundary_taper[N_pad])
        # plt.show()
        
    else:
        N_pad = 0
        N_ext = num_x
        velocity_ext = velocity_model
        damping_ext = damping_model

    # Storage arrays in the extended domain
    wave_past    = np.zeros(N_ext)
    wave_current = np.zeros(N_ext)
    wave_future  = np.zeros(N_ext)
    
    # For storing the interior portion at each time step
    wavefield_out = np.zeros((num_t, num_x))
    
    # Keep track of second derivative at previous time
    # Create or retrieve the SignalProcessor once
    sp = SignalProcessor()  
    second_deriv_past = np.zeros(N_ext)
        
    # Time loop
    for itime in range(num_t):
        # Compute 2nd derivative in extended domain
        second_deriv_current = sp.fourier_derivative_2nd(
            wave_current, delta_x
        )        

        # Standard wave update
        wave_future = (
            2.0 * wave_current
            - wave_past
            + (velocity_ext**2) * (delta_t**2) * second_deriv_current
        )

        # Add source term *in the interior*
        # If your source_spatial_function is length num_x, 
        wave_future[N_pad:N_pad+num_x] += (
            source_spatial_function * source_time_function[itime] * (delta_t**2)
        )
        
        # Kelvin–Voigt damping term
        wave_future += damping_ext * delta_t * (second_deriv_current - second_deriv_past)
    
        if absorbing:
            # Absorbing boundary: multiply by exp(-sigma[i] * dt) at all i
            wave_future *= absorbing_boundary_taper

        else:
            # Zero out the edges (Dirichlet), totally reflected boundaries
            wave_future[0]  = 0.0
            wave_future[-1] = 0.0

        # Shift old arrays
        wave_past[:] = wave_current
        wave_current[:] = wave_future
        second_deriv_past[:] = second_deriv_current
        
        # Save the interior slice to output
        wavefield_out[itime, :] = wave_current[N_pad:N_pad+num_x]
    
    return wavefield_out
    
def compute_time_derivatives(wavefield_out, delta_t):
    """
    Returns wavefield_tt of shape (num_t, num_x), 
    where wavefield_tt[i, j] ~ second derivative in time of the wavefield.
    """
    num_t, num_x = wavefield_out.shape
    wavefield_tt = np.zeros_like(wavefield_out)
    
    for ix in range(num_x):
        # 1) Take a first derivative in time
        # 2) Then a second derivative in time
        first_derivative = np.gradient(wavefield_out[:, ix], delta_t)
        second_derivative = np.gradient(first_derivative, delta_t)
        
        wavefield_tt[:, ix] = second_derivative
    
    return wavefield_tt

def compute_spatial_laplacian(wavefield_out, delta_x):
    """
    Returns wavefield_lap of shape (num_t, num_x),
    where wavefield_lap[i, :] ~ second derivative in x of wavefield_out[i, :].
    """
    sp = SignalProcessor()

    num_t, num_x = wavefield_out.shape
    wavefield_lap = np.zeros_like(wavefield_out)
    
    for it in range(num_t):
        wavefield_lap[it, :] = sp.fourier_derivative_2nd(
            wavefield_out[it, :], delta_x
        )
    
    return wavefield_lap

def compute_time_derivative_of_laplacian(wavefield_lap, delta_t):
    """
    Returns wavefield_lap_dt of shape (num_t, num_x),
    where wavefield_lap_dt[i, j] = d/dt( wavefield_lap[i, j] ).
    """
    num_t, num_x = wavefield_lap.shape
    wavefield_lap_dt = np.zeros_like(wavefield_lap)
    
    for ix in range(num_x):
        wavefield_lap_dt[:, ix] = np.gradient(wavefield_lap[:, ix], delta_t)
    
    return wavefield_lap_dt


def extend_model(original_model, N_pad):
    """
    Extend a 1D array 'original_model' by N_pad on each side
    by simply copying boundary values. 
    """
    N = len(original_model)
    N_ext = N + 2*N_pad
    extended = np.zeros(N_ext)
    
    # Fill interior
    extended[N_pad:N_pad+N] = original_model[:]
    
    # Left padding
    for i in range(N_pad):
        extended[N_pad - 1 - i] = original_model[0]  # copy left boundary value
    
    # Right padding
    for i in range(N_pad):
        extended[N_pad+N + i] = original_model[-1]   # copy right boundary value
    
    return extended

def sigmoid(z):
    return 1/(1 + np.exp(-z))


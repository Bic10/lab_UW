# lab_uw/forward_modeling.py

import sys
import matplotlib.pyplot as plt
from math import floor
import numpy as np
from numpy import linalg as LA
from typing import Union, Tuple, Optional, Dict, Any
from scipy.sparse import csr_matrix

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
from lab_uw.utils import is_compact

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

        elif self.geometry_type.lower() == "block":
            # Single-block geometry
            pzt_layer_width      = assembly_dict["pzt_layer_width"]     
            pla_layer_width      = assembly_dict["pla_layer_width"] 
            maximum_damping      = 0.

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

        # plt.plot(source_handler.time_function)
        # plt.show()

        # EXTRACT SYNTHETIC SIGNAL AT RECEIVER
        simulated_waveform = np.sum(wavefield_forward * receiver_handler.spatial_function, axis=1)
        # INTERPOLATE ONTO OBSERVED TIME AXIS
        synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)

        # plt.plot(observed_waveform)
        # plt.plot(synthetic_waveform)
        # plt.show()

        if not is_compact(misfit_interval):
            # where are the gaps?
            gaps = np.where(np.diff(np.sort(misfit_interval)) > 1)[0]
            segments = np.split(np.sort(misfit_interval), gaps + 1)
            direct   = segments[0]
            reflect  = segments[1]
            A0 = np.amax(np.abs(observed_waveform[direct]))
            # A0_synth = np.amax(np.abs(synthetic_waveform[direct]))
            # multiplier_factor = A0/A0_synth
            # synthetic_waveform *= multiplier_factor
            A1 = np.amax(np.abs(observed_waveform[reflect]))
            synthetic_waveform[reflect] *= A1/A0
            A1 = np.amax(np.abs(observed_waveform[reflect]))
  

        # NORMALIZE IF REQUESTED
        if normalize_waveform and np.max(synthetic_waveform) != 0:
            if self.geometry_type == "block":

                synthetic_waveform *= np.amax(np.abs(observed_waveform[misfit_interval]))/np.amax(np.abs(synthetic_waveform[misfit_interval])) 
                    
            elif self.geometry_type == "dds":
                # synthetic_waveform *= np.sum(np.abs(observed_waveform[misfit_interval]))/np.sum(np.abs(synthetic_waveform[misfit_interval])) 
                synthetic_waveform *= np.amax(np.abs(observed_waveform[misfit_interval]))/np.amax(np.abs(synthetic_waveform[misfit_interval])) 

        # plt.plot(observed_waveform)
        # plt.plot(synthetic_waveform)
        # plt.show()
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
            damping_label    = str(round(gouge_damping_1,5)).replace(".",",")
            velocity_label   = str(round(1e4*gouge_velocity_1,5)).replace(".",",")  
            label = f"_vel_{velocity_label}_damping_{damping_label}_global_search_{misfit:.3f}_waveform"
    
        else:
            pzt_vel_label     = str(round(1e4*pzt_velocity)).replace(".",",")
            steel_vel_label   = str(round(1e4*steel_velocity)).replace(".",",")  
            label = f"_pzt_{pzt_vel_label}_vel_{steel_vel_label}_global_search_{misfit:.3f}_waveform"

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
            observed_waveform : np.ndarray,
            synthetic_waveform: np.ndarray,
            misfit_interval   : slice | np.ndarray,
            *,
            equal_segment_weight: bool = False,     # <- NEW FLAG
            eps: float = 1e-12                      #   (to avoid divide‑by‑zero)
    ) -> float:
        """
        L2‑norm misfit between observed and synthetic waveforms on a given interval.

        Parameters
        ----------
        observed_waveform, synthetic_waveform : 1‑D ndarray
        misfit_interval : slice | 1‑D ndarray of indices
            Either a simple slice (contiguous window) or the array of indices returned
            by `compute_misfit_interval`, which may contain several disjoint segments.
        equal_segment_weight : bool, optional
            *False* (default) – the classic behaviour: every individual **sample**
            inside `misfit_interval` is equally weighted.
            *True*  – every **segment** (direct arrival, reflection, …) is given the
            same total weight, so a low‑amplitude reflection can influence the
            inversion as much as the stronger direct wave.
        eps : float, optional
            Small constant to protect against divisions by zero.
        """
        # ------------------------------------------------------------------ helpers
        def _split_into_segments(idxs: np.ndarray) -> list[np.ndarray]:
            """Split an index array into a list of contiguous blocks."""
            idxs = np.sort(np.unique(idxs))
            gaps = np.where(np.diff(idxs) > 1)[0]
            return np.split(idxs, gaps + 1)

        # ------------------------------------------------------------------ guards
        if synthetic_waveform.ndim != 1 or observed_waveform.ndim != 1:
            raise ValueError("Waveforms must be 1‑D arrays")
        if synthetic_waveform.size != observed_waveform.size:
            raise ValueError("Synthetic and observed waveforms must have same length")
        if not np.any(misfit_interval):
            raise ValueError("misfit_interval is empty")

        # ------------------------------------------------------------------- logic
        diff = synthetic_waveform - observed_waveform

        if isinstance(misfit_interval, slice):
            # ------------------ contiguous window --------------------------
            win = diff[misfit_interval]
            norm = LA.norm(win, 2)

        else:
            # ------------------ 1 or more segments -------------------------
            idx = np.asarray(misfit_interval, dtype=int)

            if equal_segment_weight:
                # Each segment contributes the SAME weight
                norms = []
                for seg in _split_into_segments(idx):
                    err_seg = diff[seg]
                    ref_seg = observed_waveform[seg]
                    # normalise by the energy of the observed signal in that segment
                    seg_norm = LA.norm(err_seg, 2) / (LA.norm(ref_seg, 2) + eps)
                    norms.append(seg_norm)
                norm = np.mean(norms)                   # equal weight per segment
            else:
                # Classical behaviour: equal weight per *sample*
                norm = LA.norm(diff[idx], 2)

        # ----------------------------------------------------------------- scaling
        if np.max(np.abs(synthetic_waveform)) == 0:
            # Something went very wrong – penalise heavily so it never becomes best fit
            norm *= 3

        return norm
      

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
        misfit_threshold    :   float = 1,
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

        initial_misfit             = self.compute_misfit(
            observed_waveform   = observed_waveform,
            synthetic_waveform  = initial_synthetic_waveform,
            misfit_interval     = misfit_interval
        )

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
                # idx_dict["pzt_1"],
                idx_dict["groove_sb1"], 
                idx_dict["gouge_1"],
                idx_dict["groove_cb1"],
                idx_dict["groove_cb2"],
                idx_dict["gouge_2"],
                idx_dict["groove_sb2"],
                # idx_dict["pzt_2"]
            ])
            
            grooves = np.concatenate([
                idx_dict["groove_sb1"], 
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
        best_velocity_model              = initial_velocity_model
        best_damping_model               = initial_damping_model
        best_source_time_function        = initial_source_time_function
        best_source_spatial_function     = initial_source_spatial_function
        best_receiver_spatial_function   = initial_receiver_spatial_function
        best_wavefield_forward           = initial_wavefield_forward
        best_laplacian_wavefield         = compute_laplacian(best_wavefield_forward, delta_x)
        best_first_derivative_laplacian  = compute_dt_laplacian(best_laplacian_wavefield, delta_t)

        best_synthetic_waveform          = initial_synthetic_waveform
        best_misfit                      = initial_misfit
        updated_misfit                   = initial_misfit
        previous_misfit                  = initial_misfit  # let's speed: if 2 misfit differ for less than another threshold value, stop
 
        # Step-size management
        dc_max = dc_max_start
        da_max = da_max_start
        dw_max = dw_max_start
        ds_max = ds_max_start

        # ------------------------------------------------------------------
        # 0. pre–compute constant objects *before* the FWI loop
        # ------------------------------------------------------------------
        mask  = np.zeros_like(observed_waveform, dtype=bool)
        mask[misfit_interval] = True

        # split misfit only once
        if not is_compact(misfit_interval):
            gaps      = np.where(np.diff(np.sort(misfit_interval)) > 1)[0]
            segments  = np.split(np.sort(misfit_interval), gaps + 1)
            direct, reflect = segments
        else:
            segments = [misfit_interval]

        # 1‑D interpolation as a sparse matrix (much faster than np.interp)
        row  = np.arange(simulation_time.size)
        col  = np.searchsorted(observed_time, simulation_time)
        data = np.ones_like(row, dtype=float)
        interp_mat = csr_matrix((data, (row, col)), shape=(row.size, observed_time.size))

        updating        = True
        end_of_the_game = False
        activate_damping= False
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")
            
            # Prepare updated arrays from 'best'
            updated_velocity_model            = best_velocity_model.copy()
            updated_damping_model             = best_damping_model.copy()
            updated_source_time_function      = best_source_time_function.copy()
            updated_source_spatial_function   = best_source_spatial_function.copy()
            updated_receiver_spatial_function = best_receiver_spatial_function.copy()

            if updating:
                # ----------------------------------------------------------
                # Residual / adjoint source
                # ----------------------------------------------------------
                residual = (best_synthetic_waveform - observed_waveform) * mask
                adj_src_time_function = residual[::-1]            # global flip
                adj_src_time_function = interp_mat @ adj_src_time_function  # sparse dot

                # spatial part unchanged
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
                    absorbing               = absorbing
                )
                wav_adj_flipped   = np.flipud(wavefield_adjoint)   # ONE global flip

                # ----------------------------------------------------------
                # GRADIENTS
                # ----------------------------------------------------------
                if dc_max:
                    temp = +(2.0 * best_velocity_model) * wav_adj_flipped * best_laplacian_wavefield
                    gradient_vel = temp.sum(0)                     # no extra copy
                    max_vel_grad = np.abs(gradient_vel[regions_to_update]).max()

                if da_max and activate_damping:
                    temp = +wav_adj_flipped * best_first_derivative_laplacian
                    gradient_damp = temp.sum(0)
                    max_damp_grad = np.abs(gradient_damp[regions_to_update]).max()

                if dw_max:
                    gradient_w = wav_adj_flipped @ best_source_spatial_function
                    gradient_w[stf_duration_idx:] = 0.0
                    max_w_grad = np.abs(gradient_w).max()

                if ds_max:
                    gradient_s = (wavefield_adjoint.T @ best_source_time_function)[::-1] * delta_x
                    gradient_s[stf_extension_idx:] = 0.0
                    max_s_grad = np.abs(gradient_s).max()

                    gradient_r = (wavefield_adjoint * best_wavefield_forward[:, ::-1]).sum(0)[::-1] * delta_x
                    gradient_r[:rx_extension_idx] = 0.0
                    max_r_grad = np.abs(gradient_r).max()                

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

            if da_max and activate_damping:
                # -- update velocity only in the selected region --
                step_size_damp = da_max / (max_damp_grad + 1e-15)
                updated_damping_model[regions_to_update] -= step_size_damp * gradient_damp[regions_to_update]

                # updated_damping_model[grooves] = np.clip(updated_damping_model[grooves],
                #                                           a_min=updated_damping_model[idx_dict["central_block"]][0],
                #                                           a_max=None)

                # plt.plot(updated_damping_model)
                # plt.show()
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

            laplacian_wavefield_updated         = compute_laplacian(wavefield_forward_updated, delta_x)
            first_derivative_laplacian_updated  = compute_dt_laplacian(laplacian_wavefield_updated, delta_t)

            # Extract updated synthetic waveform
            updated_simulated_waveform = np.sum(
                wavefield_forward_updated * updated_receiver_spatial_function, axis=1
            )
            updated_synthetic_waveform = np.interp(observed_time, simulation_time, updated_simulated_waveform)

            if not is_compact(misfit_interval):
                # where are the gaps?
                gaps = np.where(np.diff(np.sort(misfit_interval)) > 1)[0]
                segments = np.split(np.sort(misfit_interval), gaps + 1)
                direct   = segments[0]
                reflect  = segments[1]
                A0 = np.sum(np.abs(observed_waveform[direct]))
                # A0_synth = np.sum(np.abs(updated_synthetic_waveform[direct]))
                # multiplier_factor = A0/A0_synth
                # updated_synthetic_waveform *= multiplier_factor
                A1 = np.sum(np.abs(observed_waveform[reflect]))
                updated_synthetic_waveform[reflect] *= A1/A0

            # NORMALIZE IF REQUESTED
            if normalize_waveform and np.max(updated_synthetic_waveform) != 0:
                if self.geometry_type == "block":
                    updated_synthetic_waveform *= np.amax(observed_waveform[misfit_interval])/np.amax(updated_synthetic_waveform[misfit_interval])
                        
                elif self.geometry_type == "dds":
                    # updated_synthetic_waveform *= np.sum(np.abs(observed_waveform[misfit_interval]))/np.sum(np.abs(updated_synthetic_waveform[misfit_interval])) 
                    updated_synthetic_waveform *= np.amax(np.abs(observed_waveform[misfit_interval]))/np.amax(np.abs(updated_synthetic_waveform[misfit_interval])) 

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

            else:
                updating = False
                dc_max /= reduce_factor
                da_max /= reduce_factor
                dw_max /= reduce_factor
                ds_max /= reduce_factor
                print(f"    ✗ No improvement. Reverting & reducing step")

            #-------------------------------------
            # evaluate if stop the processing
            #------------------------------------
            if (updated_misfit > best_misfit) and (abs(previous_misfit-updated_misfit) < misfit_threshold):
                if end_of_the_game:
                    print("Misfit updating is below threshold. Stopping!")
                    break

            # Check threshold
            if (ds_max < ds_threshold) or (dw_max < dw_threshold):
                print("Step size dropped below threshold; stopping.")
                break

            if (dc_max < dc_threshold):
                activate_damping = True
                print("Damping Activated!")
                da_max = da_max_start
                if (da_max < da_threshold):
                    print("Step size dropped below threshold.")
                    end_of_the_game = True
                break

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
            damping_label    = str(round(self.average_gouge_damping,5)).replace(".",",")
            velocity_label   = str(round(1e4*self.average_gouge_velocity)).replace(".",",")  
            label = f"_vel_{velocity_label}_damping_{damping_label}_FWI_misfit_{best_misfit:.0f}"
    
        else:
            label = f"FWI_{best_misfit:.0f}_waveform"

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
                    model_output_name = plot_output_path.name + "_velocity_model"
                    model_output_path = plot_output_path.parent / model_output_name
                    self.velocity_model_handler.plot(model=self.velocity_model_handler.velocity_array, outfile_path=model_output_path)

                if da_max_start:
                    model_output_name = plot_output_path.name + "_damping_model"
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
    
# ------------------------------------------------------------------
def pseudospectral_1D_damped(
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial_function: np.ndarray,
    source_time_function: np.ndarray,
    velocity_model: np.ndarray,
    damping_model: np.ndarray = None,
    absorbing: np.ndarray    = False,
    N_pad: int = 200,
):
    """vectorised + FFT‑optimised replacement of the original routine."""
    # ------------------------- set‑up ---------------------------------
    if damping_model is None:
        damping_model = np.zeros_like(velocity_model)

    if absorbing:
        N_ext = num_x + 2 * N_pad
        velocity_ext  = extend_model(velocity_model,  N_pad)
        damping_ext   = extend_model(damping_model,  N_pad)

        z = np.linspace(-1, 6, N_pad)
        sigma_ext = sigmoid(z)
        absorbing_taper = np.ones(N_ext, np.float64)
        absorbing_taper[:N_pad]     = sigma_ext
        absorbing_taper[-N_pad:]    = sigma_ext[::-1]
    else:
        N_pad = 0
        N_ext = num_x
        velocity_ext  = velocity_model
        damping_ext   = damping_model
        absorbing_taper = None      # sentinel

    # --------------- pre‑compute constants / arrays -------------------
    dt2     = delta_t * delta_t
    vel2dt2 = (velocity_ext ** 2) * dt2

    src_x_term = np.zeros(N_ext)
    src_x_term[N_pad:N_pad + num_x] = source_spatial_function * dt2  # used each step

    # k‑vector for real FFT:  k = 2π * n / L
    L     = N_ext * delta_x
    k     = 2.0 * np.pi * np.fft.rfftfreq(N_ext, d=delta_x)
    k2    = -(k ** 2)                         # minus sign already included

    # time integrator buffers
    u_past    = np.zeros(N_ext)
    u_curr    = np.zeros(N_ext)
    u_fut     = np.zeros(N_ext)
    dxx_past  = np.zeros(N_ext)
    dxx_curr  = np.zeros(N_ext)               # reused every step

    wave_out  = np.empty((num_t, num_x))      # final result

    # -------------------------- time loop -----------------------------
    for it in range(num_t):
        # ∂²/∂x² via spectral multiplier  (≈ 60 % of runtime)
        _spectral_dxx(u_curr, k2, dxx_curr)

        # u_tt update (all in‑place, no temporaries)
        u_fut[:] = (2.0 * u_curr
                    - u_past
                    + vel2dt2 * dxx_curr)

        # add source   s(x)*w(t)
        u_fut[N_pad:N_pad + num_x] += src_x_term[N_pad:N_pad + num_x] * source_time_function[it]

        # Kelvin–Voigt damping term
        u_fut += damping_ext * delta_t * (dxx_curr - dxx_past)

        # boundary treatment
        if absorbing_taper is not None:
            u_fut *= absorbing_taper
        else:                                  # hard walls (Dirichlet)
            u_fut[0] = u_fut[-1] = 0.0

        # roll buffers (fast view swap – no copy)
        u_past, u_curr, u_fut = u_curr, u_fut, u_past
        dxx_past, dxx_curr    = dxx_curr, dxx_past

        # store interior slice
        wave_out[it] = u_curr[N_pad:N_pad + num_x]

    return wave_out

import numpy as np
from numpy.fft import rfft, irfft   # real FFT → half the work

# ------------------------------------------------------------------
# Helper: 2‑nd derivative via *pre‑tabulated* spectral multiplier
# ------------------------------------------------------------------
def _spectral_dxx(u, k2, out):
    """
    in‑place 2‑nd spatial derivative of a 1‑D real signal.

    Parameters
    ----------
    u   : (N,) float64 array
    k2  : (N//2+1,) float64    # - (2π k / L)^2  pre‑computed
    out : (N,) float64 array   # target (can share memory with u)
    """
    # forward/ inverse rFFT allocate temporaries inside NumPy's C, so
    # no Python‑level overhead; they reuse work buffers between calls.
    out[:] = irfft(rfft(u) * k2, n=u.size)
    return out
    
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

# jit_kernels.py
import numpy as np
from numba import njit, prange
# ------------------------------------------------------------
# Spatial Laplacian  ∂²u/∂x²   (shape = (num_t, num_x))
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Second spatial derivative for each time slice.
    """
    nt, nx = u.shape
    coeff  = 1.0 / (dx * dx)
    out    = np.empty_like(u)

    for it in prange(nt):                     # ← parallel over time
        # Dirichlet copy boundaries (cheap and OK for FWI)
        out[it, 0]      = 0.0
        out[it, nx - 1] = 0.0
        for ix in range(1, nx - 1):
            out[it, ix] = (u[it, ix + 1] - 2.0 * u[it, ix] + u[it, ix - 1]) * coeff
    return out


# ------------------------------------------------------------
# Time derivative of Laplacian  ∂/∂t (∂²u/∂x²)
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_dt_laplacian(lap: np.ndarray, dt: float) -> np.ndarray:
    """
    First‑order time derivative (central differences) of a (num_t, num_x) array.
    """
    nt, nx = lap.shape
    coeff  = 1.0 / (2.0 * dt)
    out    = np.empty_like(lap)

    for ix in prange(nx):                     # ← parallel over space
        out[0, ix]       = 0.0
        out[nt - 1, ix]  = 0.0
        for it in range(1, nt - 1):
            out[it, ix] = (lap[it + 1, ix] - lap[it - 1, ix]) * coeff
    return out


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


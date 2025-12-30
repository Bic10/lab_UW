# lab_uw/local_optimization_velocity_homogeneus.py

import sys
from pathlib import Path
import pickle
import time as tm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import Any, Dict
from itertools import product
import matplotlib.pyplot as plt

from lab_uw.data_io.data_io import UltrasonicDataHandler, BlockMetadataHandler, MechanicalDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.simulation_setup import *
from lab_uw.forward_modeling import *
from lab_uw.plotting import Plotter
from lab_uw.forward_modeling import UltrasonicModeler
from lab_uw.utils import *

import os

# 1) Make Matplotlib non-interactive (no X server use)
os.environ["MPLBACKEND"] = "Agg"

# 2) Prevent Numba from using TBB (which is the one complaining)
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
# optional: cap threads to avoid oversubscription
# os.environ["NUMBA_NUM_THREADS"] = "4"

def _set_start_method():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # avoid fork
    except RuntimeError:
        pass  # already set

_set_start_method()

def build_time_paired_waveform_mech_table(
    observed_waveform_data: np.ndarray,
    waveform_metadata: Dict[str, Any],
    mechanical_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair UW waveforms to mechanical samples using time in seconds.

    Assumptions:
      - waveform_metadata["time_ax_acquisition"] is in seconds (as shown by 0, 0.1, ..., 999)
      - mechanical_dataframe["time_s"] is in seconds

    The UW axis is treated as relative to the UW file start; we shift it by the first
    mechanical time (start of synced slice) to get absolute seconds.
    """

    n_uw = len(observed_waveform_data)

    if "time_s" not in mechanical_dataframe.columns:
        raise ValueError("mechanical_dataframe must contain a 'time_s' column in seconds.")

    mech_df = (
        mechanical_dataframe.sort_values("time_s")
        .reset_index()
        .rename(columns={"index": "Index"})
        .reset_index(drop=True)
    )

    t_uw_rel_s = waveform_metadata.get("time_ax_acquisition", None)
    if t_uw_rel_s is None:
        # Fallback: distribute across mech window
        mech_time = mech_df["time_s"].to_numpy()
        if mech_time.size < 2:
            raise ValueError("Mechanical dataframe has insufficient time_s samples for pairing.")
        t_uw_abs_s = np.linspace(mech_time[0], mech_time[-1], n_uw)
    else:
        t_uw_rel_s = np.asarray(t_uw_rel_s, dtype=float)
        if t_uw_rel_s.size != n_uw:
            raise ValueError(
                f"time_ax_acquisition length ({t_uw_rel_s.size}) != number of waveforms ({n_uw})."
            )
        t0_mech_s = float(mech_df["time_s"].iloc[0])
        t_uw_abs_s = t_uw_rel_s + t0_mech_s

    uw_df = (
        pd.DataFrame({"uw_time_s": t_uw_abs_s, "uw_idx": np.arange(n_uw, dtype=int)})
        .sort_values("uw_time_s")
        .reset_index(drop=True)
    )

    # Tolerance: half the mechanical sampling interval (this is the right scale)
    mech_time = mech_df["time_s"].to_numpy()
    if mech_time.size > 2:
        mech_dt = float(np.median(np.diff(mech_time)))
        tol_s = 0.5 * mech_dt if np.isfinite(mech_dt) and mech_dt > 0 else None
    else:
        tol_s = None

    paired = pd.merge_asof(
        uw_df,
        mech_df,
        left_on="uw_time_s",
        right_on="time_s",
        direction="nearest",
        tolerance=tol_s,   # optional; can remove if you prefer never skipping
    )

    paired["match_error_s"] = np.abs(paired["uw_time_s"] - paired["time_s"])
    return paired

def load_pickle_log(path):
    out = []
    if not path.exists():
        return out
    with open(path, "rb") as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
    return out

def min_assembly_velocity(assembly_dict: Dict[str, Any]):
    return min(assembly_dict["gouge_velocity_1"], 
            assembly_dict["gouge_velocity_2"], 
            assembly_dict["side1_params"]["velocity" + wave_type],
            assembly_dict["side1_params"]["pzt_velocity" + wave_type],
            assembly_dict["side1_params"]["pla_velocity" + wave_type])
    
def max_assembly_velocity(assembly_dict: Dict[str, Any]):
    return max(assembly_dict["gouge_velocity_1"], 
            assembly_dict["gouge_velocity_2"], 
            assembly_dict["side1_params"]["velocity" + wave_type],
            assembly_dict["side1_params"]["pzt_velocity" + wave_type],
            assembly_dict["side1_params"]["pla_velocity" + wave_type])
            
def update_assembly_dict_with_mech_data(
    assembly_dict: Dict[str, Any],
    mech_data: Tuple,
) -> None:
    """
    Update assembly_dict with per-waveform mechanical data entry
    """
    assembly_dict["gouge_thickness_1"]      = mech_data.rgt_lt_mm / 10.0     # thickness in mm -> convert to cm
    assembly_dict["gouge_thickness_2"]      = mech_data.rgt_lt_mm / 10.0     # for now, layers are assumed to have same thickness
    assembly_dict["normal_stress"]          = mech_data.normal_stress_MPa
    assembly_dict["shear_stress"]           = mech_data.shear_stress_MPa
    assembly_dict["ec_disp_mm"]             = mech_data.ec_disp_mm
    assembly_dict["acquisition_time"]       = mech_data.time_s
    assembly_dict["rec_n"]                  = int(mech_data.records_na)
    assembly_dict['idx_wave']               = mech_data.Index
    # define assembly sample_dimensions all together. It is usefull for handling forward modeling 
    assembly_dict["sample_dimensions"] = [
                                            side1_params["z_pzt2grove"],
                                            assembly_dict["gouge_thickness_1"] ,
                                            central_params["z"],
                                            assembly_dict["gouge_thickness_2"],
                                            side2_params["z_pzt2grove"],
                                        ]
        
def set_assembly_dict_guessed_with_gouge_velocity_and_damping(
    assembly_dict: Dict[str, Any],
    gouge_velocity_tuple: Union[Tuple[float],Tuple[np.ndarray]],
    gouge_damping_tuple: Union[Tuple[float],Tuple[np.ndarray]],

) -> None:
    assembly_dict_guessed = assembly_dict.copy()
    assembly_dict_guessed["gouge_velocity_1"], assembly_dict_guessed["gouge_velocity_2"] = gouge_velocity_tuple
    assembly_dict_guessed["gouge_damping_1"], assembly_dict_guessed["gouge_damping_2"] = gouge_damping_tuple
    return assembly_dict_guessed

def mechdata_slice4uw_processing(mech_data, sync_peaks, chosen_uw_file, num_waveform2process=None):      
    from math import ceil  
    start_sync      = sync_peaks[2 * chosen_uw_file]
    end_sync        = sync_peaks[2 * chosen_uw_file + 1]
    mech_data_slice = mech_data.iloc[start_sync : end_sync].copy()
    if num_waveform2process:
        downsampling    = max(1, ceil(len(mech_data_slice)/ num_waveform2process)) 
        return mech_data_slice.iloc[:: downsampling].reset_index(drop=True)
    else:
        return mech_data_slice

def compute_dds_travel_time(
    assembly_dict: Dict[str, Any],
    v_gouge_1: float = None,
    v_gouge_2: float = None,
    wave_type: str = None
) -> float:
    """
    Compute the travel time in a Double Direct Shear (DDS) assembly
    consisting of side1, side2, and a central block, plus two gouge layers.

    The function:
      1) Computes the steel-only travel time.
      2) Adds the gouge thickness contributions.
      3) Adds the "groove mixing" travel time terms.

    Parameters
    ----------
    assembly_dict : Dict[str, Any]
        Must contain:
          - "side1_params", "side2_params", "central_params" (dicts with keys
            e.g. "z", "z_pzt2grove", "h_grooves", "velocity_p" or velocity_s, etc.)
          - "wave_type" (e.g., "_p" or "_s")
          - "gouge_thickness_1", "gouge_thickness_2" (in cm)
        Possibly set by 'update_assembly_dict_with_mech_data' and the block metadata.
    v_gouge_1, v_gouge_2 : float
        Velocities in the two gouge layers (in cm/μs, or consistent units).

    Returns
    -------
    float
        Total travel time (in microseconds, if velocities are in cm/μs).
    """

    side1_params    = assembly_dict["side1_params"]
    side2_params    = assembly_dict["side2_params"]
    central_params  = assembly_dict["central_params"]

    if not wave_type:
        wave_type = assembly_dict["wave_type"]

    # Compute the steel-only portion
    steel_only_time = (
        (side1_params["z_pzt2grove"] - side1_params["h_grooves"]) / side1_params["velocity" + wave_type]
      + (side2_params["z_pzt2grove"] - side2_params["h_grooves"]) / side2_params["velocity" + wave_type]
      + (central_params["z"] - 2 * central_params["h_grooves"]) / central_params["velocity" + wave_type]
    )

    if (v_gouge_1 is not None) and (v_gouge_2 is not None):
        # Gouge thickness contributions
        thick_g1 = assembly_dict["gouge_thickness_1"]  # in cm
        thick_g2 = assembly_dict["gouge_thickness_2"]  # in cm

        # Groove mixing contributions
        groove_time = (
          side1_params["h_grooves"] / (side1_params["velocity" + wave_type] + v_gouge_1)
        + central_params["h_grooves"] / (side1_params["velocity" + wave_type] + v_gouge_1)
        + central_params["h_grooves"] / (side1_params["velocity" + wave_type] + v_gouge_2)
        + side2_params["h_grooves"] / (side2_params["velocity" + wave_type] + v_gouge_2)
        )

        pzt_time = 2 * (side1_params["pzt_layer_width"]/2) / side1_params["pzt_velocity" + wave_type]

        # Sum up total travel time
        return (
          steel_only_time+ thick_g1 / v_gouge_1+ thick_g2 / v_gouge_2+ groove_time + pzt_time
        )
    
    else:
        print("No gouge velocity passed. Computing steel-only travel time")
        return steel_only_time

# Function Definitions
def process_uw_file(
    infile_path: Path,
    uw_data_handler: UltrasonicDataHandler,
    mechanical_dataframe: MechanicalDataHandler,
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
) -> None:
    """
    Process a single UW data file (ultrasonic waveforms).
    The mechanical data is already stored in assembly_dict by an external function,
    but we apply the final downsampling in that function for a consistent indexing
    between ultrasonic waveforms and mechanical arrays.
    """

  # Derive output filenames using pathlib
    outdir_path_l2norm = params["outdir_path_l2norm"]
    outdir_path_image  = params["outdir_path_image"]
    start_time = tm.time()
    outfile_name = infile_path.name.split(".")[0]
    outfile_path = outdir_path_l2norm / outfile_name
 
    first_waveform = True
    last_good_result = None  # <-- NEW: keep last successful inversion result

    observed_waveform_data = uw_data_handler.waveform_data
    waveform_metadata = uw_data_handler.metadata

    # -----------------------------
    # Time-based pairing (robust to different sampling frequencies)
    # -----------------------------
    paired = build_time_paired_waveform_mech_table(
        observed_waveform_data=observed_waveform_data,
        waveform_metadata=waveform_metadata,
        mechanical_dataframe=mechanical_dataframe,
    )

    results_pkl = outfile_path.with_suffix(".pkl")

    with open(results_pkl, "ab") as f_results:

        # Iterate over waveforms using time-based mechanical pairing
        for row in paired.itertuples(index=False):

            # If merge_asof could not find a mech sample within tolerance, row will contain NaNs
            # (normal_stress_MPa is a reliable sentinel column in your mech table)
            if not np.isfinite(getattr(row, "Index", np.nan)):
                print(f"WARNING: No mechanical match within tolerance for UW idx={row.uw_idx} at t_us={row.uw_time_s}. Skipping.")
                continue

            observed_waveform = observed_waveform_data[int(row.uw_idx)]
            mech_data = row

            update_assembly_dict_with_mech_data(assembly_dict=assembly_dict, mech_data=mech_data)
            rec_n = assembly_dict["rec_n"]
            outfile_name_wave = outfile_name + f"_rec_n_{rec_n}"


            # --- Build search grids ---
            # Default: keep whatever params already have (e.g., your broad prior)
            # Refined: only if we have a valid last_good_result
            if (not first_waveform) and (last_good_result is not None):
                velo_mean = last_good_result.get("best_gouge_velocity", None)
                damp_mean = last_good_result.get("best_gouge_damping", None)

                # Refine velocity grid only if numeric and positive
                if isinstance(velo_mean, (int, float)) and np.isfinite(velo_mean) and (velo_mean > 0.0):
                    velo_min  = velo_mean - 0.02 * velo_mean
                    velo_max  = velo_mean + 0.02 * velo_mean
                    velo_step = 0.1 * velo_mean

                    # Protect against degenerate ranges
                    if (velo_max > velo_min) and (velo_step > 0.0):
                        params["velocity_initial_list"] = np.arange(velo_min, velo_max, velo_step)

                # Refine damping grid only if numeric and finite
                if isinstance(damp_mean, (int, float)) and np.isfinite(damp_mean):
                    damp_min  = damp_mean - 0.01 * damp_mean
                    damp_max  = damp_mean + 0.01 * damp_mean
                    damp_step = 0.1 * damp_mean

                    if (damp_max > damp_min) and (damp_step > 0.0):
                        params["damping_initial_list"] = np.arange(damp_min, damp_max, damp_step)
                    else:
                        params["damping_initial_list"] = [0.0]

            # --- Always run the waveform search so SNR gating prints every time ---
            try:
                result = global_search_waveform(
                    observed_waveform       = observed_waveform,
                    waveform_metadata       = waveform_metadata,
                    outfile_name            = outfile_name_wave,
                    stf_handler             = stf_handler,
                    params                  = params,
                    assembly_dict           = assembly_dict
                )
            except Exception as e:
                # Do not silently swallow: you want to know if you are not processing waveforms
                print(f"Exception while processing rec_n={rec_n}: {type(e).__name__}: {e}")
                continue
            finally:
                plt.close('all')

            first_waveform = False

            # Update last_good_result only if we actually got a solution
            # (Your SNR skip path returns best_gouge_velocity=None)
            if isinstance(result, dict) and result.get("best_gouge_velocity", None) is not None:
                last_good_result = result

            # --- Persist results as you already do ---
            pickle.dump(result, f_results, protocol=pickle.HIGHEST_PROTOCOL)
            f_results.flush()
 

    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---")

def global_search_waveform(
    observed_waveform: np.ndarray,
    waveform_metadata: Dict[str,Any],
    outfile_name: str,
    stf_handler : UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str,Any],
) -> Dict[str, Union[float, np.ndarray, None]]:

    # Unpack needed dictionaries entries
    observed_time          = waveform_metadata['time_ax_waveform']         # time axes for the samples of a single waveform acquisition

    frequency_cutoff       = params['frequency_cutoff']
    minimum_SNR            = params['minimum_SNR']
    plot_save_interval     = params['plot_save_interval']
    movie_save_interval    = params['movie_save_interval']
    l2norm_plot_interval   = params['l2norm_plot_interval']
    outdir_path_image      = params['outdir_path_image']

    rec_n = assembly_dict["rec_n"]
              
    # Evaluate SNR    
    waveform_snr= evaluate_snr(observed_time,observed_waveform, assembly_dict["steel_only_time_p_wave"])
    if  waveform_snr < minimum_SNR:
        normal_stress = assembly_dict["normal_stress"]
        shear_stress = assembly_dict["shear_stress"]
        print(f"SNR {waveform_snr:.2f} < {minimum_SNR}. Skipping waveform at rec n: {rec_n}, Normal Stress: {normal_stress:.2f}, Shear Stress: {shear_stress:.2f}.")
        return {
        'gouge_velocity_list' : None,
        'best_gouge_velocity' : None,
        'gouge_damping_list'  : None,
        'best_gouge_damping'  : None,
        'L2norm_waveform_list': None,
        'gouge_velocity_model': None,
        'gouge_damping_model':  None
        }

    # plt.plot(observed_time,observed_waveform)
    # plt.show()

    # Generate velocity array
    gouge_velocity_list = params["velocity_initial_list"]
    gouge_damping_list  = params["damping_initial_list"]

    # Misfit_interval is choosen according to STF len
    stf_time = stf_handler.metadata["time_ax_waveform"]
    stf_duration = stf_time[-1]-stf_time[0]  

    # Prepare arguments for multiprocessing
    num_processes = cpu_count()-1
    def _build_args(gouge_velocity: float, gouge_damping: float):

        guessed_arrival_time = compute_dds_travel_time(
            assembly_dict = assembly_dict,
            v_gouge_1     = gouge_velocity,
            v_gouge_2     = gouge_velocity
        )
        
        misfit_interval = np.where(
            (observed_time > guessed_arrival_time) & 
            (observed_time < guessed_arrival_time + stf_duration)
        )[0]
        
        assembly_dict_guessed = set_assembly_dict_guessed_with_gouge_velocity_and_damping(
            assembly_dict, (gouge_velocity, gouge_velocity), (gouge_damping, gouge_damping)
        )
        
        # Return all arguments in a single tuple
        return (
            observed_waveform,
            observed_time,
            stf_handler,
            misfit_interval,
            params,
            assembly_dict_guessed,
        )

    # args_list = [_build_args(gv) for gv in gouge_velocity_list]
    args_list = [_build_args(v, d) for v, d in product(gouge_velocity_list, gouge_damping_list)]
    # Multiprocessing over velocities and dampipng
    with Pool(processes=num_processes) as pool:
        results = pool.map(global_search_run, args_list)

    # Sort results by velocity
    results_sorted = sorted(results, key=lambda x: x[0])
    gouge_velocity_list  = np.array([res[0] for res in results_sorted])
    L2norm_waveform_list = np.array([res[1] for res in results_sorted])
    misfit_interval_list = np.array([res[2] for res in results_sorted])
    gouge_damping_list   = np.array([res[3] for res in results_sorted])

    # # Possibly plot L2 norm vs. velocity
    # save_l2norm_plot = (assembly_dict['idx_wave'] % l2norm_plot_interval == 0) 
    # if save_l2norm_plot:
    #     unique_vels = np.unique(gouge_velocity_list)
    #     unique_damps = np.unique(gouge_damping_list)
    #     misfit_grid = np.full((len(unique_vels), len(unique_damps)), np.nan)

    #     vel_to_row = {vel: idx for idx, vel in enumerate(unique_vels)}
    #     damp_to_col = {damp: idx for idx, damp in enumerate(unique_damps)}

    #     for (v, misfit, _, a) in results_sorted:
    #         row = vel_to_row[v]
    #         col = damp_to_col[a]
    #         misfit_grid[row, col] = misfit

    #     l2norm_plot_name = f"{outfile_name}_L2norm"
    #     l2norm_plot_path = outdir_path_image / l2norm_plot_name

    #     plotter = Plotter()
    #     plotter.misfit_map(
    #         misfit_grid=misfit_grid,
    #         unique_damps=unique_damps,
    #         unique_vels=unique_vels,
    #         outfile_path=l2norm_plot_path
    #     )


    #####################################################
    # RE RUN WITH THE BEST MISFIT PARAMETERS
    ###################################################
    min_idx = np.argmin(L2norm_waveform_list)
    best_gouge_velocity = gouge_velocity_list[min_idx]
    best_gouge_damping = gouge_damping_list[min_idx]

    misfit_interval = misfit_interval_list[min_idx]
    assembly_dict["gouge_velocity_1"] = best_gouge_velocity
    assembly_dict["gouge_velocity_2"] = best_gouge_velocity
    assembly_dict["gouge_damping_1"]  = best_gouge_damping
    assembly_dict["gouge_damping_2"]  = best_gouge_damping
    gouge_thickness = assembly_dict["gouge_thickness_1"] 
    print(f"Waveform at rec_n: {rec_n}: min misfit at velocity = {best_gouge_velocity:.4f} cm/μs, damping: {best_gouge_damping:.5f} arrival time: {observed_time[misfit_interval][0]}, thickness: {gouge_thickness:.4f}")

    # Check boundary
    if min_idx in (0, len(L2norm_waveform_list) - 1):
        print(f"Minimum misfit is on the boundary!")
        # params["velocity_range"] = 2*params["velocity_range"]

    # Determine if we save plots and/or movies
    save_plot  = (assembly_dict['idx_wave'] % plot_save_interval == 0) 
    save_movie = (assembly_dict['idx_wave'] % movie_save_interval == 0) 

    # Construct output paths
    if save_plot:
        plot_output_name = f"{outfile_name}"
        plot_output_path = outdir_path_image / plot_output_name
    else:
        plot_output_path = None

    if save_movie:

        movie_output_name = f"{outfile_name}.mp4"
        movie_output_path = outdir_path_image / movie_output_name
    else:
        movie_output_path = None
    
    minimum_velocity = params["min_velocity2simulate"] if params["min_velocity2simulate"] else min_assembly_velocity(assembly_dict)
    maximum_velocity = params["max_velocity2simulate"] if params["max_velocity2simulate"] else max_assembly_velocity(assembly_dict)
    maximum_damping  = max(assembly_dict["gouge_damping_1"],assembly_dict["gouge_damping_1"])

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing               = params["absorbing"],
        geometry_type           ="dds",
        observed_time           = observed_time,
        observed_waveform       = observed_waveform,
        frequency_cutoff        = frequency_cutoff,
        assembly_dict           = assembly_dict,
        stf_handler             = stf_handler,
        misfit_interval         = misfit_interval,
        minimum_velocity        = minimum_velocity,
        maximum_velocity        = maximum_velocity,  
        maximum_damping         = maximum_damping,
        normalize_waveform      = True,
        enable_plotting         = save_plot,
        make_movie              = False,
        plot_output_path        = plot_output_path,
        movie_output_path       = movie_output_path
    )

    ############################################################################
    # LOCAL INVERSION
    ############################################################################    
    # dc_max_start = 0
    dc_max_start = 0.1 * best_gouge_velocity
    dc_threshold = 0.01*dc_max_start

    # da_max_start = 0
    da_max_start = 0.1 * best_gouge_damping
    da_threshold = 0.01* da_max_start

    dw_max_start = 0
    # dw_max_start = np.amax(simulation.source_handler.time_function)
    dw_threshold = 0.01*dw_max_start

    ds_max_start = 0
    # ds_max_start = np.amax(simulation.source_handler.spatial_function)
    ds_threshold = 0.01*ds_max_start

    simulation.run_local_inversion(
        n_iterations         = params["n_iterations"],
        dc_max_start         = dc_max_start,
        dc_threshold         = dc_threshold,
        da_max_start         = da_max_start,
        da_threshold         = da_threshold,
        dw_max_start         = dw_max_start,
        dw_threshold         = dw_threshold,
        ds_max_start         = ds_max_start,
        ds_threshold         = ds_threshold,
        reduce_factor        = params["reduce_factor"],
        normalize_waveform   = True,
        enable_plotting      = save_plot,
        make_movie           = False,
        plot_output_path     = plot_output_path,
        movie_output_path    = movie_output_path
        )

    return (
            {
        'gouge_velocity_list' : gouge_velocity_list,
        'best_gouge_velocity' : best_gouge_velocity,
        'gouge_damping_list'  : gouge_damping_list,
        'best_gouge_damping'  : best_gouge_damping,
        'L2norm_waveform_list': L2norm_waveform_list,
        'gouge_velocity_model': simulation.velocity_model_handler.velocity_array,
        'gouge_damping_model':  simulation.velocity_model_handler.velocity_array
        }
    )


def global_search_run(args):
    """
    Function to process a single velocity value in multiprocessing.
    """
    (
        observed_waveform,
        observed_time,
        stf_handler,
        misfit_interval,
        params,
        assembly_dict_guessed,
    ) = args

    # Unpack parameters

    plot_output_name = f"first_guess_"
    plot_output_path = params["outdir_path_image"] / plot_output_name
    frequency_cutoff = params['frequency_cutoff']
    minimum_velocity = params["min_velocity2simulate"] if params["min_velocity2simulate"] else min_assembly_velocity(assembly_dict_guessed)
    maximum_velocity = params["max_velocity2simulate"] if params["max_velocity2simulate"] else max_assembly_velocity(assembly_dict_guessed)
    maximum_damping  = max(assembly_dict_guessed["gouge_damping_1"],assembly_dict_guessed["gouge_damping_1"])

    # Call DDS_UW_simulation with gouge_velocity_tuple
    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing               = params["absorbing"],
        geometry_type           ="dds",
        observed_time           = observed_time,
        observed_waveform       = observed_waveform,
        stf_handler             = stf_handler,
        frequency_cutoff        = frequency_cutoff,
        assembly_dict           = assembly_dict_guessed,
        misfit_interval         = misfit_interval,
        minimum_velocity        = minimum_velocity,
        maximum_velocity        = maximum_velocity, 
        maximum_damping         = maximum_damping, 
        normalize_waveform      = True,
        enable_plotting         = False,
        plot_output_path        = plot_output_path
    )
    
    L2norm_new = simulation.misfit
    gouge_velocity = assembly_dict_guessed["gouge_velocity_1"]
    gouge_damping  = assembly_dict_guessed["gouge_damping_1"]
    theo_arrival_time = observed_time[misfit_interval][0]
    print(f"\tVelocity: {1e4*gouge_velocity:.0f}, damping:{gouge_damping:.5f}, Theo arrival time: {theo_arrival_time:.2f} => Misfit: {L2norm_new:.1f}")

    return gouge_velocity, L2norm_new, misfit_interval, gouge_damping

###############################################################################################################
# Main Execution
if __name__ == "__main__":

    import os, matplotlib
    if os.getenv("DISPLAY", "") == "":          # inside screen / batch job
        matplotlib.use("Agg")                  # non-GUI backend

    # Initialize directory manager
    dir_manager = DirectoryManager()
    
    # Basic experiment info
    machine_name    = "Brava_2"
    experiment_name = "s0273suw03anhdol_5030"
    wave_type       = "_p"    # that "_" is ugly, but needed
    data_type_uw    = "uw_data/data_tsv_files" # + wave_type
    data_type_mech  = "mechanical_data"
    mech_file_name  = f"{experiment_name}_data_rp"
    outfolder_name  = "test_christmass"
    # Create output directories
    outdir_path_l2norm = dir_manager.make_data_analysis_folders(
        machine_name    = machine_name,
        experiment_name = experiment_name,
        data_types      = [outfolder_name]
    )

    outdir_path_image = dir_manager.make_data_analysis_folders(
        machine_name    = machine_name,
        experiment_name = experiment_name,
        data_types      = [outfolder_name + "_images_and_movie"]
    )

    # Basic simulation parameters 
    params = {
        "absorbing"                 : False,        # set absorbing bounday, merdoso silicone maledetto
        "num_waveform2process"      : 10,         # int, equespatially waveforms to sample for processing. Of "None", process all
        "maxtime2simulate"          : 40,           # [mus]
        "frequency_cutoff"          : 4,            # [MHz] low pass onserved data and simulate up to this frequency
        "minimum_SNR"               : 15,            # skip computation until time interval where signal should be is above SNR times surely-only-noise part 
        "velocity_initial_list"     : np.linspace(0.26, 0.3000, 10),  # [cm/mus] first guess of best velocity. There is a visual tool for it, if needed
        "min_velocity2simulate"     : 0.20,         # [cm/mus] if not passed, computed by assembly and gouge velocity range
        "max_velocity2simulate"     : 0.61,         # [cm/mus]
        "damping_initial_list"      : np.linspace(0.0000,0.0007, 5),
        "plot_save_interval"        : 1,
        "movie_save_interval"       : 1000,
        "l2norm_plot_interval"      : 1000,
        "outdir_path_l2norm"        : outdir_path_l2norm[0],
        "outdir_path_image"         : outdir_path_image[0],
        "n_iterations"              : 10,
        "reduce_factor"             : 10/9
    }

    # Load the Source Time Function (HDF5)
    stf_h5_path = (# FAKE FIX
        dir_manager.paths.analysis_root("on_bench", "STF_ss10_05")
        # / f"source_time_functions{wave_type}"
#################################################
# # FAKE FIX !!!!!!!!!!!!!!!!!!!!!!! WRONG SOURCE TIME FUNCTION
#########################################################
        / f"source_time_functions_s" 
        / "stf.h5"
    )

    # read the STF saved under /stf/width250_volt70/cycle_000000/source_waveform
    stf_handler = UltrasonicDataHandler.read_stf_h5(
        stf_h5_path,
        name="width250_volt70",   # "width250_volt70_local_inversion" when available
        cycle_index=0,
    )

    # keep the sign convention you had before; make it 1D if your solver expects 1D
    stf_handler.waveform_data = (stf_handler.waveform_data).squeeze(0)   # shape (L,)
    stf_handler.lowpass(cutoff_mhz=params['frequency_cutoff'])
    stf_handler.waveform_data = stf_handler.waveform_data[:86]
    stf_handler.metadata['time_ax_waveform'] =     stf_handler.metadata['time_ax_waveform'][:86]


    # Load Mechanical Data
    mech_folder = Path(dir_manager.base_dir) / f"experiments_{machine_name}" / experiment_name / data_type_mech
    mech_data, sync_data, sync_peaks = MechanicalDataHandler.find_and_load(mech_folder, mech_file_name)

    # Build a dictionary containing all the relevant assembly parameters
    blocks_json = Path(dir_manager.base_dir) / "metadata" / "blocks_metadata.json"
    side1_params, side2_params, central_params = BlockMetadataHandler.load_blocks_metadata(
    blocks_json, ("guglielmi_side1","guglielmi_side2","central_block_titanium")
    )

    assembly_dict = {
        "side1_params"        : side1_params,
        "side2_params"        : side2_params,
        "central_params"      : central_params,
        "wave_type"           : wave_type,
        "transmitter_position": side1_params["z_pzt2grove"],
        "receiver_position"   : side2_params["z_pzt2grove"],
    }
    assembly_dict["velocity" + wave_type] = side1_params["velocity" + wave_type]
    # Duct-taper to check mininimum possible arrival time. TO BE REMOVED!!!
    assembly_dict["steel_only_time_p_wave"] = compute_dds_travel_time(assembly_dict=assembly_dict,wave_type="_p")
    
    # Make UW path list
    infile_path_list_uw = sorted( dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))
 
    # Process each UW file
    for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):
        infile_name = infile_path.name.split(".")[0]

        if infile_name != "001_compaction":
            print(infile_name)
            continue

        mech_data_slice = mechdata_slice4uw_processing(mech_data,
                                                       sync_peaks, 
                                                       chosen_uw_file, 
                                                       num_waveform2process=params["num_waveform2process"])

        # Load & process ultrasonic data and metadata from TSV, with preprocessing
        uw_data_handler = (
            UltrasonicDataHandler.read_tsv(infile_path)
                .remove_mean()
                .downsample_waveforms(params["num_waveform2process"])     # keep N waveforms (evenly sampled)
                .truncate_time(params["maxtime2simulate"])                # limit to maxtime [µs]
                .zero_before(assembly_dict["steel_only_time_p_wave"] / 5) # zero early window [µs]
                .lowpass(params["frequency_cutoff"])                    # optional, cutoff in MHz
        )

        t_acq = uw_data_handler.metadata.get("time_ax_acquisition", None)
        print("mech time start/end =", mech_data_slice["time_s"].iloc[0], mech_data_slice["time_s"].iloc[-1])

        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        # import matplotlib.colors as colors
        # import numpy as np

        # time = uw_data_handler.metadata['time_ax_waveform']
        # data = uw_data_handler.waveform_data[500:,:]   # (n_waveforms, n_samples)

        # n_waveforms = data.shape[0]
        # norm = colors.Normalize(vmin=0, vmax=n_waveforms - 1)
        # cmap = cm.winter

        # fig, ax = plt.subplots(figsize=(8, 5))

        # for i in range(n_waveforms):
        #     ax.plot(
        #         time,
        #         data[i, :],
        #         color=cmap(norm(i)),
        #         alpha=0.2,
        #         linewidth=0.7
        #     )

        # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # sm.set_array([])

        # cbar = fig.colorbar(sm, ax=ax, label="Waveform index (time order)")
        # cbar.set_ticks([0, n_waveforms//2, n_waveforms-1])
        # cbar.set_ticklabels(["0", f"{n_waveforms//2}", f"{n_waveforms-1}"])

        # # fig.colorbar(sm, ax=ax, label="Waveform index (time order)")

        # ax.set_xlabel("Time")
        # ax.set_ylabel("Amplitude")
        # ax.set_title("Waveform Evolution Over Time")

        # plt.show()
        # sys.exit()

        process_uw_file(
            infile_path             = infile_path,
            uw_data_handler         = uw_data_handler,
            mechanical_dataframe    = mech_data_slice,
            stf_handler             = stf_handler,
            params                  = params,
            assembly_dict           = assembly_dict
        )
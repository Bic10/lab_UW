# lab_uw/global_optimization_velocity_homogeneus.py

import sys
from pathlib import Path
import pickle
import pandas as pd
import time as tm
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Any, Dict

from lab_uw.data_io import UltrasonicDataHandler, BlockMetadataHandler, MechanicalDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.simulation_setup import *
from lab_uw.forward_modeling import *
from lab_uw.plotting import Plotter
from lab_uw.forward_modeling import ForwardModeler

def min_assembly_velocity(assembly_dict: Dict[str, Any]):
    return min(assembly_dict["gouge_velocity_1"], 
            assembly_dict["gouge_velocity_1"], 
            assembly_dict["side1_params"]["velocity" + wave_type],
            assembly_dict["side1_params"]["pzt_velocity" + wave_type],
            assembly_dict["side1_params"]["pla_velocity" + wave_type])
    
def max_assembly_velocity(assembly_dict: Dict[str, Any]):
    return min(assembly_dict["gouge_velocity_1"], 
            assembly_dict["gouge_velocity_1"], 
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
    assembly_dict["idx_processed_waveform"] = mech_data.Index

    # define assembly sample_dimensions all together. It is usefull for handling forward modeling 
    assembly_dict["sample_dimensions"]  = [
                                            side1_params["z_pzt2grove"],
                                            assembly_dict["gouge_thickness_1"] ,
                                            central_params["z"],
                                            assembly_dict["gouge_thickness_2"],
                                            side2_params["z_pzt2grove"],
                                        ]
        
def update_assembly_dict_with_gouge_velocity(
    assembly_dict: Dict[str, Any],
    gouge_tuple: Union[Tuple[float],Tuple[np.ndarray]],
) -> None:
    assembly_dict["gouge_velocity_1"], assembly_dict["gouge_velocity_1"] = gouge_tuple

def mechdata_slice4uw_processing(mech_data, sync_peaks, chosen_uw_file, num_waveform2porcess=None):      
    from math import ceil  
    start_sync      = sync_peaks[2 * chosen_uw_file]
    end_sync        = sync_peaks[2 * chosen_uw_file + 1]
    mech_data_slice = mech_data.iloc[start_sync : end_sync].copy()
    if num_waveform2porcess:
        downsampling    = max(1, ceil(len(mech_data_slice)/ num_waveform2porcess)) 

    return mech_data_slice.iloc[:: downsampling].reset_index(drop=True)

def compute_dds_travel_time(
    assembly_dict: Dict[str, Any],
    v_gouge_1: float = None,
    v_gouge_2: float = None
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
    wave_type       = assembly_dict["wave_type"]

    # Compute the steel-only portion
    steel_only_time = (
        (side1_params["z_pzt2grove"] - side1_params["h_grooves"]) / side1_params["velocity" + wave_type]
      + (side2_params["z_pzt2grove"] - side2_params["h_grooves"]) / side2_params["velocity" + wave_type]
      + (central_params["z"] - 2 * central_params["h_grooves"]) / central_params["velocity" + wave_type]
    )

    if v_gouge_1 and v_gouge_2:
        # Gouge thickness contributions
        thick_g1 = assembly_dict["gouge_thickness_1"]  # in cm
        thick_g2 = assembly_dict["gouge_thickness_2"]  # in cm

        # Groove mixing contributions
        groove_time = (
          side1_params["h_grooves"] / (side1_params["velocity" + wave_type] + v_gouge_1)
        + side2_params["h_grooves"] / (side2_params["velocity" + wave_type] + v_gouge_2)
        + central_params["h_grooves"] / (side1_params["velocity" + wave_type] + v_gouge_1)
        + central_params["h_grooves"] / (side1_params["velocity" + wave_type] + v_gouge_2)
        )

        # Sum up total travel time
        return (
          steel_only_time+ thick_g1 / v_gouge_1+ thick_g2 / v_gouge_2+ groove_time
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
    outfile_name = infile_path.name.split(".")[0]  
    outfile_path = outdir_path_l2norm / outfile_name

    start_time = tm.time()
 
    # 3) Prepare for the loop
    velocity_ranges       = []
    L2norm_all_waveforms  = []
    estimated_velocities  = []
    previous_best_velocity = params["velocity_initial_cm/mus"]

    observed_waveform_data = uw_data_handler.waveform_data
    waveform_metadata = uw_data_handler.metadata
    # 4) Iterate over waveforms and mechanical data in sync
    for observed_waveform, mech_data in zip(observed_waveform_data, mechanical_dataframe.itertuples()):

        update_assembly_dict_with_mech_data(assembly_dict=assembly_dict, mech_data=mech_data)

        result = process_waveform(
            observed_waveform       = observed_waveform,
            waveform_metadata       = waveform_metadata,
            outfile_name            = outfile_name,
            previous_best_velocity   = previous_best_velocity,
            stf_handler             = stf_handler,
            params                  = params,
            assembly_dict           = assembly_dict
        )

        if result["best_gouge_velocity"]:
            previous_best_velocity = result["best_gouge_velocity"]

        velocity_ranges.append(result["gouge_velocity_list_waveform"])
        L2norm_all_waveforms.append(result["L2norm_waveform"])
        estimated_velocities.append(result["best_gouge_velocity"])

    # Save results
    results_pkl = outfile_path.with_suffix(".pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump({
            "L2norm_all_waveforms": L2norm_all_waveforms,
            "velocity_ranges": velocity_ranges,
            "estimated_velocities": estimated_velocities
        }, f)

    # Plot velocity & stress vs. ec_disp
    print(mechanical_dataframe)
    plotter = Plotter()
    plot_name_ec_disp = f"{outfile_name}_velocity_stress_vs_ec_disp"
    plot_path_ec_disp = outdir_path_image / plot_name_ec_disp
    plotter.plot_velocity_and_stresses(
        x_values        = mechanical_dataframe["ec_disp_mm"],
        velocities      = np.array(estimated_velocities),
        normal_stress   = mechanical_dataframe["normal_stress_MPa"].values,
        shear_stress    = mechanical_dataframe["shear_stress_MPa"].values,
        x_label         = "ec_disp_mm",
        velocity_label  = "Gouge Velocity (cm/µs)",
        stress_labels   = ("Normal Stress (MPa)", "Shear Stress (MPa)"),
        title           = "Gouge Velocity and Stress vs ec_disp_mm",
        outfile_path    = plot_path_ec_disp
    )

    # Plot velocity & stress vs. time_s
    plot_name_time = f"{outfile_name}_velocity_stress_vs_time"
    plot_path_time = outdir_path_image / plot_name_time
    plotter.plot_velocity_and_stresses(
        x_values        = mechanical_dataframe["time_s"],
        velocities      = np.array(estimated_velocities),
        normal_stress   = mechanical_dataframe["normal_stress_MPa"].values,
        shear_stress    = mechanical_dataframe["shear_stress_MPa"].values,
        x_label         = "time_s",
        velocity_label  = "Gouge Velocity (cm/µs)",
        stress_labels   = ("Normal Stress (MPa)", "Shear Stress (MPa)"),
        title           = "Gouge Velocity and Stress vs time_s",
        outfile_path    = plot_path_time
    )

    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---")

def process_waveform(
    observed_waveform: np.ndarray,
    waveform_metadata: Dict[str,Any],
    outfile_name: str,
    previous_best_velocity: Optional[float],
    stf_handler : UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str,Any],
) -> Dict[str, Union[float, np.ndarray, None]]:

    # Unpack needed dictionaries entries
    waveform_time = waveform_metadata['time_ax_waveform']         # time axes for the samples of a single waveform acquisition

    frequency_cutoff       = params['frequency_cutoff_MHz']
    minimum_SNR            = params['minimum_SNR']
    velocity_step          = params['velocity_step_cm/mus']
    c_range                = params['velocity_range_cm/mus']
    plot_save_interval     = params['plot_save_interval']
    movie_save_interval    = params['movie_save_interval']
    l2norm_plot_interval   = params['l2norm_plot_interval']
    outdir_path_image      = params['outdir_path_image']

    acquisition_time       = assembly_dict["acquisition_time"] # contain the time, referred to the start of the experiment, when the waveforms are acquired
    acq_time_label         = str(round(acquisition_time,3)).replace(".",",")
    idx_processed_waveform = assembly_dict["idx_processed_waveform"]
        
    # Search around previous velocity
    c_range_waveform  = c_range
    cmin_waveform     = previous_best_velocity - c_range_waveform
    cmax_waveform     = previous_best_velocity + c_range_waveform
    velocity_step_waveform   = velocity_step

    max_travel_time = compute_dds_travel_time(
        assembly_dict = assembly_dict,
        v_gouge_1     = cmin_waveform,
        v_gouge_2     = cmin_waveform
        )

    min_travel_time = compute_dds_travel_time(
        assembly_dict = assembly_dict,
        v_gouge_1     = cmax_waveform,
        v_gouge_2     = cmax_waveform
        )

    # Evaluate SNR
    sure_noise_interval = np.where(waveform_time < min_travel_time)
    good_data_interval  = np.where(waveform_time > min_travel_time)
    max_signal          = np.amax(observed_waveform[good_data_interval]) if good_data_interval[0].size else 1
    max_noise           = np.amax(observed_waveform[sure_noise_interval]) if sure_noise_interval[0].size else 1

    if max_signal / max_noise < minimum_SNR:
        print(f"SNR {max_signal / max_noise:.2f} < {minimum_SNR}. Skipping waveform at {acq_time_label}.")
        return {
            'gouge_velocity_list_waveform': None,
            'L2norm_waveform'             : None,
            'best_gouge_velocity'         : None,
        }
    
    stf_time = stf_handler.metadata["time_ax_waveform"]
    stf_duration = stf_time[-1]-stf_time[0]   
    misfit_interval = np.where((waveform_time > min_travel_time) & (waveform_time < max_travel_time + stf_duration))[0]

    # Generate velocity array
    gouge_velocity_list_waveform = np.arange(cmin_waveform, cmax_waveform, velocity_step_waveform)
    print(f"Velocity range = [{cmin_waveform:.4f}, {cmax_waveform:.4f}] with step={velocity_step_waveform:.4f}")

    # Prepare arguments for multiprocessing
    num_processes = cpu_count()

    def _build_args(gouge_velocity: float):
        update_assembly_dict_with_gouge_velocity(assembly_dict,(gouge_velocity, gouge_velocity))
        return (
            observed_waveform,
            waveform_time,
            stf_handler,
            misfit_interval,
            params,
            assembly_dict
        )

    args_list = [_build_args(gv) for gv in gouge_velocity_list_waveform]

    # Multiprocessing over velocities
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_velocity, args_list)

    # results is a list of (gouge_velocity_scalar, L2norm) sorted by scalar? We sort by velocity
    results_sorted = sorted(results, key=lambda x: x[0])
    gouge_velocity_list_waveform = np.array([res[0] for res in results_sorted])
    L2norm_waveform = np.array([res[1] for res in results_sorted])

    # Find min misfit
    min_idx = np.argmin(L2norm_waveform)
    best_gouge_velocity = gouge_velocity_list_waveform[min_idx]
    print(f"Waveform at {acq_time_label}: min misfit at velocity = {best_gouge_velocity:.4f} cm/μs")

    # Check boundary
    if min_idx in (0, len(L2norm_waveform) - 1):
        print(f"Minimum misfit is on the boundary, doubling next search range.")
        params["velocity_range_cm/mus"] = 2*params["velocity_range_cm/mus"]

    # Determine if we save plots and/or movies
    save_plot  = (idx_processed_waveform % plot_save_interval == 0) 
    save_movie = (idx_processed_waveform+1 % movie_save_interval == 0) 

    # Construct output paths
    if save_plot:
        plot_output_name = f"{outfile_name}_waveform_{acq_time_label}_vel_{1e4*best_gouge_velocity:.0f}"
        plot_output_path = outdir_path_image / plot_output_name
    else:
        plot_output_path = None

    if save_movie:
        movie_output_name = f"{outfile_name}_waveform_{acq_time_label}_vel_{1e4*best_gouge_velocity:.0f}.mp4"
        movie_output_path = outdir_path_image / movie_output_name
    else:
        movie_output_path = None
    
    minimum_velocity = params["min_velocity2simulate"] if params["min_velocity2simulate"] else min_assembly_velocity(assembly_dict)
    maximum_velocity = params["max_velocity2simulate"] if params["max_velocity2simulate"] else max_assembly_velocity(assembly_dict)

    synthetic_waveform,*_ = ForwardModeler().forward_simulation(
        geometry_type           ="dds",
        observed_time           = waveform_time,
        observed_waveform       = observed_waveform,
        frequency_cutoff        = frequency_cutoff,
        assembly_dict           = assembly_dict,
        stf_handler             = stf_handler,
        misfit_interval         = misfit_interval,
        minimum_velocity        = minimum_velocity,
        maximum_velocity        = maximum_velocity,  
        normalize_waveform      = True,
        enable_plotting         = save_plot,
        make_movie              = save_movie,
        plot_output_path        = plot_output_path,
        movie_output_path       = movie_output_path
    )

    # Possibly plot L2 norm vs. velocity
    save_l2norm_plot = (idx_processed_waveform % l2norm_plot_interval == 0) 
    if save_l2norm_plot:
        plotter = Plotter()
        l2norm_plot_name = f"{outfile_name}_L2norm_waveform_{acq_time_label}"
        l2norm_plot_path = outdir_path_image / l2norm_plot_name
        plotter.plot_l2_norm_vs_velocity(
            velocity            = gouge_velocity_list_waveform,
            L2norm              = L2norm_waveform,
            acquisition_time    = acq_time_label,
            outfile_path        = l2norm_plot_path
        )

    return {
        'gouge_velocity_list_waveform': gouge_velocity_list_waveform,
        'L2norm_waveform'             : L2norm_waveform,
        'best_gouge_velocity'         : best_gouge_velocity,
    }

def process_velocity(args):
    """
    Function to process a single velocity value in multiprocessing.
    """
    (
        observed_waveform,
        waveform_time,
        stf_handler,
        misfit_interval,
        params,
        assembly_dict
    ) = args

    # Unpack parameters
    frequency_cutoff_MHz = params['frequency_cutoff_MHz']

    minimum_velocity = params["min_velocity2simulate"] if params["min_velocity2simulate"] else min_assembly_velocity(assembly_dict)
    maximum_velocity = params["max_velocity2simulate"] if params["max_velocity2simulate"] else max_assembly_velocity(assembly_dict)

    # Call DDS_UW_simulation with gouge_velocity_tuple
    synthetic_waveform,*_ = ForwardModeler().forward_simulation(
        geometry_type           ="dds",
        observed_time           = waveform_time,
        observed_waveform       = observed_waveform,
        stf_handler             = stf_handler,
        frequency_cutoff        = frequency_cutoff_MHz,
        assembly_dict           = assembly_dict,
        misfit_interval         = misfit_interval,
        minimum_velocity        = minimum_velocity,
        maximum_velocity        = maximum_velocity, 
        normalize_waveform      = True,
        enable_plotting         = False
    )
    
    L2norm_new = compute_misfit(
        observed_waveform=observed_waveform,
        synthetic_waveform=synthetic_waveform,
        misfit_interval=misfit_interval
    )

    # Use the first element of the tuple for sorting and returning
    gouge_velocity_scalar = assembly_dict["gouge_velocity_1"]
    print(f"\tVelocity: {1e4*gouge_velocity_scalar:.0f} => Misfit: {L2norm_new:.0f}")

    return gouge_velocity_scalar, L2norm_new

###############################################################################################################
# Main Execution
if __name__ == "__main__":

    # Initialize directory manager
    dir_manager = DirectoryManager()
    
    # Basic experiment info
    machine_name    = "Brava_2"
    experiment_name = "s0216suw04anh_50"
    wave_type       = "_p"    # that "_" is ugly, but needed
    data_type_uw    = "uw_data/data_tsv_files" + wave_type
    data_type_mech  = "mechanical_data"
    mech_file_name  = f"{experiment_name}_data_rp"

    # Create output directories
    outdir_path_l2norm = dir_manager.make_data_analysis_folders(
        machine_name    = machine_name,
        experiment_name = experiment_name,
        data_types      = ["global_optimization_velocity" + wave_type]
    )

    outdir_path_image = dir_manager.make_data_analysis_folders(
        machine_name    = machine_name,
        experiment_name = experiment_name,
        data_types      = ["global_optimization_velocity_images_and_movie" + wave_type]
    )

    # Basic simulation parameters 
    params = {
        "num_waveform2porcess"      : 10,
        "maxtime2simulate_mus"      : 40,
        "frequency_cutoff_MHz"      : 4,
        "minimum_SNR"               : 3,
        "velocity_step_cm/mus"      : 0.002,
        "velocity_range_cm/mus"     : 0.01,
        "velocity_initial_cm/mus"   : 0.07,
        "min_velocity2simulate"     : 0.30,  # cm/mus
        "max_velocity2simulate"     : 0.60,  # cm/mus
        "plot_save_interval"        : 1,
        "movie_save_interval"       : 10,
        "l2norm_plot_interval"      : 1,
        "outdir_path_l2norm"        : outdir_path_l2norm[0],
        "outdir_path_image"         : outdir_path_image[0]
    }

    # Load the Source Time Function
    stf_handler = UltrasonicDataHandler.load_stf(
        dir_manager         = dir_manager,
        machine_name_stf    = "on_bench",
        experiment_name_stf = "STF",
        data_type_stf       = "data_analysis/source_time_functions" + wave_type,
        stf_chosen          = "width500_volt200_p2p",
        frequency_cutoff_MHz= params["frequency_cutoff_MHz"]
    )

    # Load Mechanical Data
    mech_data, sync_data, sync_peaks = MechanicalDataHandler.locate_and_load_data(
        dir_manager     = dir_manager,
        machine_name    = machine_name,
        experiment_name = experiment_name,
        data_type_mech  = data_type_mech,
        mech_file_name  = mech_file_name
    )

    # Build a dictionary containing all the relevant assembly parameters
    side1_params, side2_params, central_params = BlockMetadataHandler.load_blocks_metadata(
        dir_manager           = dir_manager,
        blocks_metadata_name  = "blocks_metadata.json",
        block_keys            = ("mauro_side1","mauro_side2","central_block1")
    )

    assembly_dict = {
        "side1_params"        : side1_params,
        "side2_params"        : side2_params,
        "central_params"      : central_params,
        "wave_type"           : wave_type,
        "transmitter_position": side1_params["z_pzt2grove"],
        "receiver_position"   : side2_params["z_pzt2grove"],
    }
    steel_only_time = compute_dds_travel_time(assembly_dict=assembly_dict)

    # Make UW path list
    infile_path_list_uw = sorted( dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))
 
    # Process each UW file
    for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):

        mech_data_slice = mechdata_slice4uw_processing(mech_data,
                                                       sync_peaks, 
                                                       chosen_uw_file, 
                                                       num_waveform2porcess=params["num_waveform2porcess"])

        # Load & process ultrasonic data and metadata from TSV, with preprocessing
        uw_data_handler = UltrasonicDataHandler.load_and_process_uw(
            infile_path                 = infile_path,
            zero_out_time               = steel_only_time,
            frequency_cutoff_MHz        = params["frequency_cutoff_MHz"],
            maxtime2simulate            = params["maxtime2simulate_mus"],
            number_of_waveforms2process = params["num_waveform2porcess"],
            time_ax_acquisition_start   = mech_data_slice["time_s"].values[0]
            )

        process_uw_file(
            infile_path             = infile_path,
            uw_data_handler         = uw_data_handler,
            mechanical_dataframe    = mech_data_slice,
            stf_handler             = stf_handler,
            params                  = params,
            assembly_dict           = assembly_dict
        )
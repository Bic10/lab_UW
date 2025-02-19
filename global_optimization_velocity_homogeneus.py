# lab_uw/global_optimization_velocity_homogeneus.py

# Libraries
import sys
from pathlib import Path
import pickle
import pandas as pd
import time as tm
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List

from lab_uw.data_io import UltrasonicDataHandler, BlockMetadataHandler, MechanicalDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.simulation_setup import *
from lab_uw.forward_modeling import *
from lab_uw.plotting import Plotter
from lab_uw.forward_modeling import ForwardModeler
from lab_uw.utils import pick_arrival_times

def update_assembly_dict_with_mech_data(
    assembly_dict: Dict[str, Any],
    mech_data: pd.DataFrame,
) -> None:
    """
    Update assembly_dict with per-waveform mechanical data entry
    """

    # thickness in mm -> convert to cm
    assembly_dict["thickness_gouge_1"] = mech_data["rgt_lt_mm"] / 10.0
    # for now, layers are assumed to have same thickness
    assembly_dict["thickness_gouge_2"] = mech_data["rgt_lt_mm"]
    assembly_dict["normal_stress"]     = mech_data["normal_stress_MPa"]
    assembly_dict["shear_stress"]      = mech_data["shear_stress_MPa"]
    assembly_dict["ec_disp_mm"]        = mech_data["ec_disp_mm"]
    assembly_dict["time_s"]            = mech_data["time_s"]

def mechdata_slice4uw_processing(mech_data, sync_peaks, chosen_uw_file, params):        
    start_sync      = sync_peaks[2 * chosen_uw_file]
    end_sync        = sync_peaks[2 * chosen_uw_file + 1]
    mech_data_slice = mech_data.iloc[start_sync : end_sync].copy()
    downsampling    = max(1, round(len(mech_data_slice)/ params["num_waveform2porcess"])) 
    mech_data_slice = mech_data_slice.iloc[:: downsampling].copy()  

    return mech_data_slice
def compute_dds_travel_time(assembly_travel_time,
                            side1_params,
                            side2_params,
                            central_params,
                            thickness_gouge_1,
                            thickness_gouge_2,
                            v_gouge_1,
                            v_gouge_2):
    
    return  (assembly_travel_time
            + thickness_gouge_1 / v_gouge_1
            + thickness_gouge_2 / v_gouge_2
            + side1_params["h_grooves"]/(side1_params["velocity"+wave_type]+v_gouge_1)
            + side2_params["h_grooves"]/(side2_params["velocity"+wave_type]+v_gouge_2)
            + central_params["h_grooves"]/(side1_params["velocity"+wave_type]+v_gouge_1)
            + central_params["h_grooves"]/(side1_params["velocity"+wave_type]+v_gouge_2)
            )

# Function Definitions
def process_uw_file(
    infile_path: Path,
    arrival_time_interval: List[Any],
    uw_data_handler: UltrasonicDataHandler,
    mechanical_data_handler: MechanicalDataHandler,
    stf_waveform: np.ndarray,
    stf_time: np.ndarray,
    stf_duration: float,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
) -> None:
    """
    Process a single UW data file (ultrasonic waveforms).
    The mechanical data is already stored in assembly_dict by an external function,
    but we apply the final downsampling in that function for a consistent indexing
    between ultrasonic waveforms and mechanical arrays.
    """
    print(f"PROCESSING UW DATA IN {infile_path}:")

  # Derive output filenames using pathlib
    outdir_path_l2norm = Path(params["outdir_path_l2norm"])
    outdir_path_image  = Path(params["outdir_path_image"])
    outfile_name = infile_path.name.split(".")[0]  
    outfile_path = outdir_path_l2norm / outfile_name

    start_time = tm.time()
 
    # 3) Prepare for the loop
    velocity_ranges       = []
    L2norm_all_waveforms  = []
    estimated_velocities  = []
    normal_stress_values  = []
    shear_stress_values   = []
    ec_disp_values        = []
    time_s_values         = []
    previous_min_velocity = None

    observed_waveform_data = uw_data_handler.waveform_data
    observed_waveform_time = uw_data_handler.metadata
    # 4) Iterate over waveforms and mechanical data in sync
    for observed_waveform, (idx_waveform, mech_data) in zip(observed_waveform_data, mechanical_data_handler.iterrows()):
        
        update_assembly_dict_with_mech_data(assembly_dict=assembly_dict, mech_data=mech_data)

        result = process_waveform(
            arrival_time_interval,
            observed_waveform=observed_waveform,
            observed_time=observed_waveform_time,
            idx_waveform=idx_waveform,
            outfile_name=outfile_name,
            previous_min_velocity=previous_min_velocity,
            stf_waveform=stf_waveform,
            stf_time=stf_time,
            stf_duration=stf_duration,
            params=params,
            assembly_dict=assembly_dict
        )
        previous_min_velocity = result["previous_min_velocity"]
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
    plotter = Plotter()
    plot_name_ec_disp = f"{outfile_name}_velocity_stress_vs_ec_disp"
    plot_path_ec_disp = outdir_path_image / plot_name_ec_disp
    plotter.plot_velocity_and_stresses(
        x_values=np.array(ec_disp_values),
        velocities=np.array(estimated_velocities),
        normal_stress=np.array(normal_stress_values),
        shear_stress=np.array(shear_stress_values),
        x_label="ec_disp_mm",
        velocity_label="Gouge Velocity (cm/µs)",
        stress_labels=("Normal Stress (MPa)", "Shear Stress (MPa)"),
        title="Gouge Velocity and Stress vs ec_disp_mm",
        outfile_path=plot_path_ec_disp
    )

    # Plot velocity & stress vs. time_s
    plot_name_time = f"{outfile_name}_velocity_stress_vs_time"
    plot_path_time = outdir_path_image / plot_name_time
    plotter.plot_velocity_and_stresses(
        x_values=np.array(time_s_values),
        velocities=np.array(estimated_velocities),
        normal_stress=np.array(normal_stress_values),
        shear_stress=np.array(shear_stress_values),
        x_label="time_s",
        velocity_label="Gouge Velocity (cm/µs)",
        stress_labels=("Normal Stress (MPa)", "Shear Stress (MPa)"),
        title="Gouge Velocity and Stress vs time_s",
        outfile_path=plot_path_time
    )

    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---")

def process_waveform(
    arrival_time_interval: List[float],
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    idx_waveform: int,
    outfile_name: str,
    previous_min_velocity: Optional[float],
    stf_waveform: np.ndarray,
    stf_time: np.ndarray,
    stf_duration: float,
    params: Dict[str, Any],
    assembly_dict: Dict[str,Any],
) -> Dict[str, Union[float, np.ndarray, None]]:

    # Unpack additional parameters
    frequency_cutoff       = params['frequency_cutoff_MHz']
    minimum_SNR            = params['minimum_SNR']
    c_step                 = params['c_step_cm/mus']
    c_range                = params['c_range_cm/mus']
    range_scaling_factor   = params['range_scaling_factor']
    plot_save_interval     = params['plot_save_interval']
    movie_save_interval    = params['movie_save_interval']
    l2norm_plot_interval   = params['l2norm_plot_interval']
    outdir_path_image_list = params['outdir_path_image']

    side1_params           = assembly_dict['side1_params']
    side2_params           = assembly_dict['side2_params']
    central_params         = assembly_dict['central_params']
    assembly_travel_time   = assembly_dict['assembly_travel_time']
    wave_type              = assembly_dict["wave_type"]

    # Convert outdir_path_image_list[0] to a Path
    outdir_path_image = Path(outdir_path_image_list) if isinstance(outdir_path_image_list, str) else outdir_path_image_list
    if isinstance(outdir_path_image, list):
        outdir_path_image = Path(outdir_path_image[0])

    # ------------------------------------------
    # 1) INITIAL VELOCITY ESTIMATION
    # ------------------------------------------
    if previous_min_velocity is None:
        # Attempt velocity from manual picks
        try:
            from lab_uw.utils import solve_quadratic_equation
            estimated_velocities = []
            for picked_time in arrival_time_interval:      
                # To get an estimation of the velocity from the travel times, we have a
                # Quadratic eq: A*vel^2 + B*vel + C = 0
                Delta_t = picked_time - assembly_dict['assembly_travel_time']
                L_g = assembly_dict["thickness_gouge_1"] + assembly_dict["thickness_gouge_2"]
                L_h = side1_params["h_grooves"] + 2 * central_params["h_grooves"] + side2_params["h_grooves"]
                A = 0.5 * Delta_t
                ##### For now assume the velocity is the same for all the blcoks. So just pick one
                B = 0.5 * Delta_t * side1_params["velocity"+wave_type] - 0.5 * L_g - L_h
                C = -0.5 * L_g * side1_params["velocity"+wave_type] 
                solutions = solve_quadratic_equation(A, B, C, real_only=True, positive_only=True)
                
                if solutions:
                    estimated_velocities.extend(solutions)
                    for sol in solutions:
                        print(f"Estimated velocity: {sol}")
                else:
                    print("No real solution for cmin_waveform from manual picks.")

            # Then you do:
            if estimated_velocities:
                cmin_waveform = min(estimated_velocities)
                cmax_waveform = max(estimated_velocities)
                print(f"Manual picks -> cmin={cmin_waveform:.4f}, cmax={cmax_waveform:.4f}")
            else:
                raise ValueError("Manual picks gave no valid solutions.")

        except:
            # Fallback if no manual picks are valid: empirical estimate of granular material velocity from literature
            # Problem: this should be material-dependent
            # cmin_waveform = 0.035 * (normal_stress**0.25)
            # cmax_waveform = 0.055 * (normal_stress**0.25)

            if wave_type == '_s':
                cmin_waveform = 0.1
                cmax_waveform = 0.25

            if wave_type == '_p':
                cmin_waveform = 0.2
                cmax_waveform = 0.4

            print(f"No manual velocity estimates found. Using fallback cmin={cmin_waveform:.4f}, cmax={cmax_waveform:.4f}")

        c_step_waveform = c_step

        # Evaluate SNR in a smaller time window          
        max_travel_time = compute_dds_travel_time(
            assembly_travel_time = assembly_travel_time,
            side1_params         = side1_params,
            side2_params         = side2_params,
            central_params       = central_params,
            thickness_gouge_1    = thickness_gouge_1,
            thickness_gouge_2    = thickness_gouge_2,
            v_gouge_1            = cmin_waveform,
            v_gouge_2            = cmin_waveform
            )

        min_travel_time = compute_dds_travel_time(
            assembly_travel_time = assembly_travel_time,
            side1_params         = side1_params,
            side2_params         = side2_params,
            central_params       = central_params,
            thickness_gouge_1    = thickness_gouge_1,
            thickness_gouge_2    = thickness_gouge_2,
            v_gouge_1            = cmax_waveform,
            v_gouge_2            =cmax_waveform
            )
        
        # Evaluate SNR
        sure_noise_interval = np.where(observed_time < min_travel_time)
        good_data_interval = np.where(observed_time > min_travel_time)
        max_signal = np.amax(observed_waveform[good_data_interval]) if good_data_interval[0].size else 1
        max_noise = np.amax(observed_waveform[sure_noise_interval]) if sure_noise_interval[0].size else 1

        if max_signal / max_noise < minimum_SNR:
            print(f"SNR {max_signal / max_noise:.2f} < {minimum_SNR}. Skipping waveform {idx_waveform}.")
            return {
                'previous_min_velocity'       : None,
                'gouge_velocity_list_waveform': None,
                'L2norm_waveform'             : None,
                'best_gouge_velocity'         : None,
                'range_factor'                : None
            }

        is_first_waveform = True

    else:
        # For subsequent waveforms, search around previous velocity
        is_first_waveform = False
        c_range_waveform  = range_scaling_factor * c_range
        cmin_waveform     = previous_min_velocity - c_range_waveform
        cmax_waveform     = previous_min_velocity + c_range_waveform
        c_step_waveform   = c_step

        max_travel_time = compute_dds_travel_time(
            assembly_travel_time = assembly_travel_time,
            side1_params         = side1_params,
            side2_params         = side2_params,
            central_params       = central_params,
            thickness_gouge_1    = thickness_gouge_1,
            thickness_gouge_2    = thickness_gouge_2,
            v_gouge_1            = cmin_waveform,
            v_gouge_2            = cmin_waveform
            )

        min_travel_time = compute_dds_travel_time(
            assembly_travel_time = assembly_travel_time,
            side1_params         = side1_params,
            side2_params         = side2_params,
            central_params       = central_params,
            thickness_gouge_1    = thickness_gouge_1,
            thickness_gouge_2    = thickness_gouge_2,
            v_gouge_1            = cmax_waveform,
            v_gouge_2            = cmax_waveform
            )
        
    misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time + stf_duration))[0]
    
    # Generate velocity array
    gouge_velocity_list_waveform = np.arange(cmin_waveform, cmax_waveform, c_step_waveform)
    print(f"Velocity range = [{cmin_waveform:.4f}, {cmax_waveform:.4f}] with step={c_step_waveform:.4f}")

    # Prepare arguments for multiprocessing
    num_processes = cpu_count()

    def _build_args(gouge_velocity: float):
        return (
            (gouge_velocity, gouge_velocity),  # same velocity for both layers
            observed_waveform,
            thickness_gouge_1,
            thickness_gouge_2,
            misfit_interval,
            observed_time,
            stf_time,
            stf_waveform,
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
    print(f"Waveform {idx_waveform}: min misfit at velocity = {best_gouge_velocity:.4f} cm/μs")

    # Check boundary
    if min_idx in (0, len(L2norm_waveform) - 1):
        print(f"Minimum misfit is on the boundary, doubling next search range.")
        range_scaling_factor *= range_scaling_factor

    previous_min_velocity = best_gouge_velocity

    # Determine if we save plots and/or movies
    save_plot  = (idx_waveform % plot_save_interval == 0) 
    save_movie = (idx_waveform+1 % movie_save_interval == 0) 

    # Construct output paths
    outdir_path_image = Path(outdir_path_image)
    if save_plot:
        plot_output_name = f"{outfile_name}_waveform_{overall_index}_vel_{1e4*best_gouge_velocity:.0f}"
        plot_output_path = outdir_path_image / plot_output_name
    else:
        plot_output_path = None

    if save_movie:
        movie_output_name = f"{outfile_name}_waveform_{overall_index}_vel_{1e4*best_gouge_velocity:.0f}.mp4"
        movie_output_path = outdir_path_image / movie_output_name
    else:
        movie_output_path = None
    
    synthetic_waveform, _,_,_,_,_,_,_,_,_,_,_ = ForwardModeler().dds_forward_simulation(
        observed_time           = observed_time,
        observed_waveform       = observed_waveform,
        stf_time                = stf_time,
        stf_waveform            = stf_waveform,
        frequency_cutoff        = frequency_cutoff,
        assembly_dict           = assembly_dict,
        gouge_velocity          = (best_gouge_velocity, best_gouge_velocity),
        gouge_thickness         = (thickness_gouge_1,thickness_gouge_2),
        misfit_interval         = misfit_interval,
        fixed_minimum_velocity  = best_gouge_velocity,
        normalize_waveform      = True,
        enable_plotting         = save_plot,
        make_movie              = save_movie,
        plot_output_path        = str(plot_output_path) if plot_output_path else None,
        movie_output_path       = str(movie_output_path) if movie_output_path else None
    )

    # Possibly plot L2 norm vs. velocity
    save_l2norm_plot = (idx_waveform % l2norm_plot_interval == 0) or is_first_waveform
    if save_l2norm_plot:
        plotter = Plotter()
        l2norm_plot_name = f"{outfile_name}_L2norm_waveform_{overall_index}"
        l2norm_plot_path = outdir_path_image / l2norm_plot_name
        plotter.plot_l2_norm_vs_velocity(
            gouge_velocity      = gouge_velocity_list_waveform,
            L2norm              = L2norm_waveform,
            overall_index       = overall_index,
            outfile_path        = l2norm_plot_path
        )

    return {
        'previous_min_velocity'       : previous_min_velocity,
        'gouge_velocity_list_waveform': gouge_velocity_list_waveform,
        'L2norm_waveform'             : L2norm_waveform,
        'best_gouge_velocity'         : best_gouge_velocity,
        'range_factor'                : range_scaling_factor
    }

def process_velocity(args):
    """
    Function to process a single velocity value in multiprocessing.
    """
    (
        gouge_velocity_tuple,  
        observed_waveform,
        thickness_gouge_1,
        thickness_gouge_2,
        misfit_interval,
        observed_time,
        stf_time,
        stf_waveform,
        params,
        assembly_dict
    ) = args

    # Unpack parameters
    frequency_cutoff_MHz = params['frequency_cutoff_MHz']

    # Call DDS_UW_simulation with gouge_velocity_tuple
    synthetic_waveform, _,_,_,_,_,_,_,_,_,_,_ = ForwardModeler().dds_forward_simulation(
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_time=stf_time,
        stf_waveform=stf_waveform,
        frequency_cutoff=frequency_cutoff_MHz,
        assembly_dict = assembly_dict,
        gouge_velocity=gouge_velocity_tuple,  # Pass the tuple
        gouge_thickness=(thickness_gouge_1,thickness_gouge_2),
        misfit_interval=misfit_interval,
        fixed_minimum_velocity=min(gouge_velocity_tuple),
        normalize_waveform=True,
        enable_plotting=False
    )

    L2norm_new = compute_misfit(
        observed_waveform=observed_waveform,
        synthetic_waveform=synthetic_waveform,
        misfit_interval=misfit_interval
    )

    # Use the first element of the tuple for sorting and returning
    gouge_velocity_scalar = gouge_velocity_tuple[0]
    print(f"\tVelocity: {gouge_velocity_scalar} => Misfit: {L2norm_new}")

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
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=["global_optimization_velocity" + wave_type]
    )
    outdir_path_image = dir_manager.make_data_analysis_folders(
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=["global_optimization_velocity_images_and_movie" + wave_type]
    )
    print(f"The misfits calculated will be saved at path:\n{outdir_path_l2norm[0]}")

    # Basic simulation parameters 
    params = {
        "maxtime2simulate_mus"      : 40,
        "frequency_cutoff_MHz"      : 4,
        "minimum_SNR"               : 5,
        "c_step_cm/mus"             : 0.00110,
        "c_range_cm/mus"            : 0.01,
        "range_scaling_factor"      : 1,
        "plot_save_interval"        : 1,
        "movie_save_interval"       : 1,
        "l2norm_plot_interval"      : 1,
        "num_waveform2porcess"      : 10,
        "outdir_path_l2norm"        : outdir_path_l2norm[0],
        "outdir_path_image"         : outdir_path_image[0]
    }

    # Load the Source Time Function
    stf_waveform, stf_time, stf_duration = UltrasonicDataHandler.load_stf(
        dir_manager=dir_manager,
        machine_name_stf="on_bench",
        experiment_name_stf="STF",
        data_type_stf="data_analysis/source_time_functions" + wave_type,
        stf_chosen="width500_volt200_p2p",
        frequency_cutoff_MHz=params["frequency_cutoff_MHz"]
    )

    # Load Mechanical Data
    mech_data, sync_data, sync_peaks = MechanicalDataHandler.locate_and_load_data(
        dir_manager=dir_manager,
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_type_mech=data_type_mech,
        mech_file_name=mech_file_name
    )

    # Build a dictionary containing all the relevant assembly parameters
    side1_params, side2_params, central_params = BlockMetadataHandler.load_blocks_metadata(
        dir_manager=dir_manager,
        blocks_metadata_name="blocks_metadata.json",
        block_keys=("mauro_side1","mauro_side2","central_block1")
    )

    assembly_dict = {
        "side1_params"        : side1_params,
        "side2_params"        : side2_params,
        "central_params"      : central_params,
        "wave_type"           : wave_type,
        "transmitter_position": side1_params["z_pzt2grove"],
        "receiver_position"   : side2_params["z_pzt2grove"],
        # compute steel-only lower-bound
        "assembly_travel_time": (
              (side1_params["z_pzt2grove"] - side1_params["h_grooves"]) / side1_params["velocity" + wave_type]
            + (side2_params["z_pzt2grove"] - side2_params["h_grooves"]) / side2_params["velocity" + wave_type]
            + (central_params["z"] - 2*central_params["h_grooves"]) / central_params["velocity" + wave_type]
            )
    }

    # Make UW path list
    infile_path_list_uw = sorted( dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))

    # Prepare manual pick arrival times for first guess velocities
    arrival_times_list = pick_arrival_times(
        dir_manager=dir_manager,
        machine_name=machine_name,
        experiment_name=experiment_name,
        infile_path_list_uw=infile_path_list_uw,
        start_time= assembly_dict['assembly_travel_time']
        )
 
    # Process each UW file
    for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):
        arrival_time_interval = arrival_times_list[chosen_uw_file]

        mech_data_slice = mechdata_slice4uw_processing(mech_data,sync_peaks, chosen_uw_file, params)

        # Load & process ultrasonic data and metadata from TSV, with preprocessing
        uw_data_handler = (
            UltrasonicDataHandler.load_and_process_uw(
            infile_path=infile_path,
            zero_out_time=assembly_dict["assembly_travel_time"],
            frequency_cutoff_MHz=params["frequency_cutoff_MHz"],
            maxtime2simulate=params["maxtime2simulate_mus"],
            number_of_waveforms2process=params["num_waveform2porcess"],
            time_ax_acquisition_start = mech_data_slice["time_s"].values[0]
            )
        )

        process_uw_file(
            infile_path             = infile_path,
            arrival_time_interval   = arrival_time_interval,
            uw_data_handler         = uw_data_handler,
            mechanical_data_handler = mech_data_slice,
            stf_waveform            = stf_waveform,
            stf_time                = stf_time,
            stf_duration            = stf_duration,
            params                  = params,
            assembly_dict           = assembly_dict
        )
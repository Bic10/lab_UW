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

from lab_uw.data_io import UltrasonicDataHandler, MechanicalDataHandler, BlockMetadataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.signal_processing import SignalProcessor
from lab_uw.simulation_setup import *
from lab_uw.forward_modeling import *
from lab_uw.plotting import InteractivePlotter, Plotter
from lab_uw.forward_modeling import ForwardModeler

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
def load_and_process_stf(
    dir_manager: DirectoryManager,
    machine_name_stf: str,
    experiment_name_stf: str,
    data_type_stf: str,
    stf_choosen: str,
    frequency_cutoff_MHz: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads and processes a source time function (stf) from a file.

    Parameters
    ----------
    dir_manager : DirectoryManager
        DirectoryManager to locate the stf file path.
    machine_name_stf : str
        Machine name used for acquiring the stf.
    experiment_name_stf : str
        Experiment name for the stf data.
    data_type_stf : str
        Subfolder name indicating where stf data is stored.
    stf_choosen : str
        Stem of the stf file (without extension) to be loaded.
    frequency_cutoff_MHz : float
        Frequency cutoff in MHz for lowpass filtering of the stf.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (stf_waveform, stf_time, stf_duration)
        stf_waveform : Filtered stf waveform, zeroed at start.
        stf_time : Corresponding time axis of the waveform.
        stf_duration : Duration of the stf (stf_time[-1] - stf_time[0]).

    Raises
    ------
    FileNotFoundError
        If no stf file matching stf_choosen is found.
    """
    # Locate the stf file
    infile_path_stf_list = dir_manager.make_infile_path_list(
        machine_name=machine_name_stf,
        experiment_name=experiment_name_stf,
        data_type=data_type_stf
    )

    chosen_stf_path = None
    for infile_stf in infile_path_stf_list:
        if infile_stf.stem == stf_choosen:
            chosen_stf_path = infile_stf
            break
    if chosen_stf_path is None:
        raise FileNotFoundError(
            f"No stf file named '{stf_choosen}' found in {data_type_stf} "
            f"for experiment '{experiment_name_stf}'."
        )

    # Load stf data (JSON or TSV) using UltrasonicDataHandler
    stf_handler = UltrasonicDataHandler()
    stf_waveform_raw, stf_metadata = stf_handler.load_waveform_json(chosen_stf_path)

    stf_time = np.array(stf_metadata["time_ax_waveform"])
    signal_processor = SignalProcessor()

    # Apply lowpass filter
    stf_waveform_filt, _ = signal_processor.signal2noise_separation_lowpass(
        waveform_data=stf_waveform_raw,
        metadata=stf_metadata,
        freq_cut=frequency_cutoff_MHz
    )

    # Shift waveform to start at zero amplitude
    stf_waveform = stf_waveform_filt - stf_waveform_filt[0]
    stf_duration = stf_time[-1] - stf_time[0]

    return stf_waveform, stf_time, stf_duration

def load_mechanical_data(
    dir_manager: DirectoryManager,
    machine_name: str,
    experiment_name: str,
    data_type_mech: str,
    mech_file_name: str
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Locates and loads mechanical data, returning the DataFrame plus sync_data and sync_peaks.

    Parameters
    ----------
    dir_manager : DirectoryManager
        DirectoryManager instance for building file paths.
    machine_name : str
        Machine name used for mechanical data experiment.
    experiment_name : str
        Experiment name for mechanical data.
    data_type_mech : str
        Subfolder or data type under which mechanical data is stored.
    mech_file_name : str
        File name (including extension) for the mechanical data file.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]
        - mech_data : The loaded mechanical DataFrame.
        - sync_data : Array of synchronization data, or None if not found.
        - sync_peaks : Indices of synchronization peaks, or None if not found.

    Raises
    ------
    FileNotFoundError
        If the file named mech_file_name is not found among mechanical data files.
    """
    infile_path_list_mech = dir_manager.make_infile_path_list(
        machine_name, experiment_name, data_type=data_type_mech
    )
    mech_data_path = None
    for infile_path in infile_path_list_mech:
        if infile_path.name == mech_file_name:
            mech_data_path = infile_path
            break
    if mech_data_path is None:
        raise FileNotFoundError(f"{mech_file_name} not found in mechanical data.")

    # Create a MechanicalDataHandler from the CSV file
    mech_handler = MechanicalDataHandler.make_mechanical_data(mech_data_path)
    mech_data = mech_handler.mech_data

    # Attempt to find synchronization data
    sync_data, sync_peaks = mech_handler.find_sync_values()

    return mech_data, sync_data, sync_peaks

def pick_arrival_times(
    dir_manager, machine_name, experiment_name, infile_path_list_uw, start_time=0
):
    """
    Prepare manual pick arrival times by processing UW files.

    Parameters:
        dir_manager (DirectoryManager): Manages directory paths.
        machine_name (str): Name of the machine used for the experiment.
        experiment_name (str): Name of the experiment.
        infile_path_list_uw (list[Path]): List of paths to ultrasonic waveform files.
        start_time (float, optional): Starting time for processing. Defaults to 0s.

    Returns:
        list: A list of manual pick arrival time intervals.
    """
    # Prepare output directory
    experiment_path = dir_manager.base_dir / f"experiments_{machine_name}" / experiment_name
    picked_travel_times_dir = experiment_path / 'data_analysis' / 'picked_travel_times'
    picked_travel_times_dir.mkdir(parents=True, exist_ok=True)

    arrival_times_list = []
    for infile_path_uw in infile_path_list_uw:
        stem = Path(infile_path_uw.stem).stem  # Get file stem
        new_file_name = f"{stem}.pkl"
        infile_path_travel_times = picked_travel_times_dir / new_file_name

        try:
            with open(infile_path_travel_times, 'rb') as f:
                arrival_times_list.append(pickle.load(f))
        except FileNotFoundError:
            waveform_choosed = 0
            
            # Instantiate UltrasonicDataHandler using the Path object
            ultrasonic_handler = UltrasonicDataHandler.make_UW_data(infile_path_uw)
            observed_waveform_data, metadata = ultrasonic_handler.waveform_data, ultrasonic_handler.metadata
            observed_waveform = observed_waveform_data[waveform_choosed]
            observed_time = metadata['time_ax_waveform']

            picked_times = InteractivePlotter().manual_pick_arrival_times(
                observed_time=observed_time,
                observed_waveform=observed_waveform,
                start_time=start_time,
                outfile_path=infile_path_travel_times
            )
            arrival_times_list.append(picked_times)

    return arrival_times_list

def process_uw_file(
    infile_path: Path,
    chosen_uw_file: int,
    arrival_time_interval: List[Any],
    sync_peaks: np.ndarray,
    mech_data: Any,
    stf_waveform: np.ndarray,
    stf_time: np.ndarray,
    stf_duration: float,
    params: Dict[str, Any],
    assembly_dict: Dict[str,Any],
) -> None:
    """
    Process a single UW data file.

    Parameters
    ----------
    infile_path : Path
        The path to the ultrasonic waveforms file.
    chosen_uw_file : int
        Index of the current UW file in the experiment.
    arrival_time_interval : list
        List of manually picked arrival times for some waveforms.
    sync_peaks : np.ndarray
        Array of synchronization indices for mechanical data.
    mech_data : Any
        Mechanical DataFrame or structure containing mechanical data (stress, displacement).
    stf_waveform : np.ndarray
        Source time function waveform (filtered, zeroed).
    stf_time : np.ndarray
        Corresponding time axis of the STF waveform.
    stf_duration : float
        Duration of the STF waveform.
    params : Dict[str, Any]
        Dictionary of parameters needed for processing.
    assembly_dict: Dict[Dict],        
        Dictionary of the Dictionaries containing material and dimensions of the various parts of the experiment

    Returns
    -------
    None
        Saves results to disk and optionally creates plots of velocity vs. displacement/stress.
    """
    print(f"PROCESSING UW DATA IN {infile_path}:")

    # Unpack parameters
    frequency_cutoff_MHz = params['frequency_cutoff_MHz']
    number_of_waveforms2process = params["number_of_waveforms2process"]
    outdir_path_l2norm = params['outdir_path_l2norm']
    outdir_path_image = params['outdir_path_image']

    # Initialize results
    velocity_ranges = []
    L2norm_all_waveforms = []
    estimated_velocities = []

    # Initialize lists for mechanical data
    normal_stress_values = []
    shear_stress_values = []
    ec_disp_mm_values = []
    time_s_values = []

    # Derive output filenames using pathlib
    outfile_name = infile_path.name.split(".")[0]  
    outfile_path = outdir_path_l2norm / outfile_name

    start_time = tm.time()

    # Load UW data
    ultrasonic_handler = UltrasonicDataHandler.make_UW_data(infile_path)
    observed_waveform_data, metadata = ultrasonic_handler.waveform_data, ultrasonic_handler.metadata
    observed_time = metadata['time_ax_waveform']

    # Preprocessing: remove mean, zero out first N samples
    observed_waveform_data = observed_waveform_data - np.mean(observed_waveform_data)
    initial_time_removed = np.searchsorted(observed_time,assembly_dict['assembly_travel_time'])
    observed_waveform_data[:, :initial_time_removed] = 0
    idx = 800
    observed_waveform = observed_waveform_data[:idx]
    observed_time = metadata['time_ax_waveform'][idx]
    # Lowpass filtering
    signal_processor = SignalProcessor()
    observed_waveform_data, _ = signal_processor.signal2noise_separation_lowpass(
        waveform_data=observed_waveform_data,
        metadata=metadata,
        freq_cut=frequency_cutoff_MHz
    )

    # Possibly reduce the number of samples
    total_time_to_simulate = int(metadata['number_of_samples'])
    observed_waveform_data = observed_waveform_data[:, :total_time_to_simulate]
    observed_time = observed_time[:total_time_to_simulate]

    # Downsampling waveforms
    downsampling = max(1, round(metadata['number_of_waveforms'] / number_of_waveforms2process))
    print(f"Number of waveforms: {metadata['number_of_waveforms']}, wanting {number_of_waveforms2process}, downsampling factor: {downsampling}")

    # Extract thickness & stress from mechanical data
    try:
        start_sync = sync_peaks[2 * chosen_uw_file]
        end_sync = sync_peaks[2 * chosen_uw_file + 1]
    except (TypeError, IndexError):
        # fallback if sync_peaks is partial
        start_sync = sync_peaks[2 * chosen_uw_file]
        end_sync = metadata['number_of_waveforms']

    # Convert mm -> cm
    thickness_gouge_1_list = mech_data.rgt_lt_mm[start_sync:end_sync].values / 10.0
    thickness_gouge_2_list = thickness_gouge_1_list  # same thickness for both layers?
    normal_stress_list = mech_data.normal_stress_MPa[start_sync:end_sync].values
    shear_stress_list = mech_data.shear_stress_MPa[start_sync:end_sync].values
    ec_disp_mm_list = mech_data.ec_disp_mm[start_sync:end_sync].values
    time_s_list = mech_data.time_s[start_sync:end_sync].values

    previous_min_velocity = None

    # Loop through waveforms
    for idx_waveform, (
            thick_g1, thick_g2, normal_stress, shear_stress, ec_disp_mm, time_s
        ) in enumerate(
            zip(
                thickness_gouge_1_list[::downsampling],
                thickness_gouge_2_list[::downsampling],
                normal_stress_list[::downsampling],
                shear_stress_list[::downsampling],
                ec_disp_mm_list[::downsampling],
                time_s_list[::downsampling]
            )
        ):

        idx_data = idx_waveform * downsampling

        # Adjust thickness (In some of the reduced data are surely wrong
        # thick_g1 *= 2.0
        # thick_g2 *= 2.0

        try:
            observed_waveform = observed_waveform_data[idx_data]
        except IndexError:
            # Out of waveforms, break
            break

        overall_index = start_sync + idx_data

        normal_stress_values.append(normal_stress)
        shear_stress_values.append(shear_stress)
        ec_disp_mm_values.append(ec_disp_mm)
        time_s_values.append(time_s)

        # Process the waveform (assuming process_waveform is defined/imported)
        result = process_waveform(
            arrival_time_interval,
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            idx_waveform=idx_waveform,
            overall_index=overall_index,
            outfile_name=outfile_name,
            previous_min_velocity=previous_min_velocity,
            thickness_gouge_1=thick_g1,
            thickness_gouge_2=thick_g2,
            normal_stress=normal_stress,
            shear_stress=shear_stress,
            stf_waveform=stf_waveform,
            stf_time=stf_time,
            stf_duration=stf_duration,
            params=params,
            assembly_dict=assembly_dict
        )

        previous_min_velocity = result['previous_min_velocity']
        velocity_ranges.append(result['gouge_velocity_list_waveform'])
        L2norm_all_waveforms.append(result['L2norm_waveform'])
        estimated_velocities.append(result['best_gouge_velocity'])

    # Save results to a pickle
    results_pkl = outfile_path.with_suffix(".pkl")  # e.g. path/to/l2norm/filename.pkl
    with open(results_pkl, 'wb') as f:
        pickle.dump({
            'L2norm_all_waveforms': L2norm_all_waveforms,
            'velocity_ranges': velocity_ranges,
            'estimated_velocities': estimated_velocities
        }, f)

    # 1) Velocity and stress vs ec_disp_mm
    plotter = Plotter()
    plot_name_ec_disp = f"{outfile_name}_velocity_stress_vs_ec_disp"
    plot_path_ec_disp = Path(outdir_path_image) / plot_name_ec_disp  # outdir_path_image is in params

    plotter.plot_velocity_and_stresses(
        x_values=np.array(ec_disp_mm_values),
        velocities=np.array(estimated_velocities),
        normal_stress=np.array(normal_stress_values),
        shear_stress=np.array(shear_stress_values),
        x_label='ec_disp_mm',
        velocity_label='Gouge Velocity (cm/μs)',
        stress_labels=('Normal Stress (MPa)', 'Shear Stress (MPa)'),
        title='Gouge Velocity and Stress vs ec_disp_mm',
        outfile_path=plot_path_ec_disp
    )

    # 2) Velocity and stress vs time_s
    plot_name_time = f"{outfile_name}_velocity_stress_vs_time"
    plot_path_time = Path(outdir_path_image) / plot_name_time

    plotter.plot_velocity_and_stresses(
        x_values=np.array(time_s_values),
        velocities=np.array(estimated_velocities),
        normal_stress=np.array(normal_stress_values),
        shear_stress=np.array(shear_stress_values),
        x_label='time_s',
        velocity_label='Gouge Velocity (cm/μs)',
        stress_labels=('Normal Stress (MPa)', 'Shear Stress (MPa)'),
        title='Gouge Velocity and Stress vs time_s',
        outfile_path=plot_path_time
    )

    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---")

def process_waveform(
    arrival_time_interval: List[float],
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    idx_waveform: int,
    overall_index: int,
    outfile_name: str,
    previous_min_velocity: Optional[float],
    thickness_gouge_1: float,
    thickness_gouge_2: float,
    normal_stress: float,
    shear_stress: float,
    stf_waveform: np.ndarray,
    stf_time: np.ndarray,
    stf_duration: float,
    params: Dict[str, Any],
    assembly_dict: Dict[str,Any],
) -> Dict[str, Union[float, np.ndarray, None]]:
    """
    Process a single waveform by scanning possible gouge velocities, computing misfit,
    and simulating a synthetic waveform at the best velocity.

    Parameters
    ----------
    arrival_time_interval : list of float
        List of manually picked arrival times.
    observed_waveform : np.ndarray
        The actual observed waveform for this iteration (1D array).
    observed_time : np.ndarray
        Time axis for the observed waveform.
    idx_waveform : int
        Index of this waveform in the entire experiment's waveforms.
    overall_index : int
        Overall index in the mechanical data synchronization.
    outfile_name : str
        Stem for output filenames.
    previous_min_velocity : float or None
        Best velocity from the previous waveform, or None if this is the first.
    thickness_gouge_1 : float
        Thickness of gouge layer 1 [cm].
    thickness_gouge_2 : float
        Thickness of gouge layer 2 [cm].
    normal_stress : float
        Normal stress [MPa].
    shear_stress : float
        Shear stress [MPa].
    stf_waveform : np.ndarray
        Source time function waveform (filtered, zero-started).
    stf_time : np.ndarray
        Corresponding time axis of the STF.
    stf_duration : float
        Duration of the STF waveform.
    assembly_travel_time : float
        Time needed for wave to travel through the steel assembly, excluding gouge.
    c_step : float
        Step size for scanning gouge velocities [cm/μs].
    c_range : float
        Half-range for scanning velocities (in subsequent waveforms).
    frequency_cutoff_MHz : float
        Frequency cutoff for simulating or filtering waveforms.
    assembly_dict : dict
        Dictionary with the assembly metadata, like dimensions, blocks velocity, pzt position       
    params : dict
        Dictionary containing relevant parameters.

    Returns
    -------
    dict
        A dictionary with:
            'previous_min_velocity': Updated best velocity for the next iteration,
            'gouge_velocity_list_waveform': The array of tested velocities,
            'L2norm_waveform': The computed L2 norms for each velocity,
            'best_gouge_velocity': The best-fit velocity,
            'range_factor': Possibly updated range scaling factor if min was at boundary.
    """
    # Unpack additional parameters
    frequency_cutoff = params['frequency_cutoff_MHz']
    minimum_SNR = params['minimum_SNR']
    c_step = params['c_step_cm/mus']
    c_range = params['c_range_cm/mus']
    range_scaling_factor = params['range_scaling_factor']
    plot_save_interval = params['plot_save_interval']
    movie_save_interval = params['movie_save_interval']
    l2norm_plot_interval = params['l2norm_plot_interval']
    outdir_path_image_list = params['outdir_path_image']

    side1_params = assembly_dict['side1_params']
    side2_params = assembly_dict['side2_params']
    central_params = assembly_dict['central_params']
    assembly_travel_time = assembly_dict['assembly_travel_time']
    wave_type = assembly_dict["wave_type"]

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
                L_g = thickness_gouge_1 + thickness_gouge_2
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
        max_travel_time = compute_dds_travel_time(assembly_travel_time= assembly_travel_time,
                                    side1_params=side1_params,
                                    side2_params=side2_params,
                                    central_params=central_params,
                                    thickness_gouge_1=thickness_gouge_1,
                                    thickness_gouge_2=thickness_gouge_2,
                                    v_gouge_1=cmin_waveform,
                                    v_gouge_2=cmin_waveform)

        min_travel_time = compute_dds_travel_time(assembly_travel_time= assembly_travel_time,
                                    side1_params=side1_params,
                                    side2_params=side2_params,
                                    central_params=central_params,
                                    thickness_gouge_1=thickness_gouge_1,
                                    thickness_gouge_2=thickness_gouge_2,
                                    v_gouge_1=cmax_waveform,
                                    v_gouge_2=cmax_waveform)
        
        ()

        # Evaluate SNR
        sure_noise_interval = np.where(observed_time < min_travel_time)
        good_data_interval = np.where(observed_time > min_travel_time)
        max_signal = np.amax(observed_waveform[good_data_interval]) if good_data_interval[0].size else 1
        max_noise = np.amax(observed_waveform[sure_noise_interval]) if sure_noise_interval[0].size else 1

        if max_signal / max_noise < minimum_SNR:
            print(f"SNR {max_signal / max_noise:.2f} < {minimum_SNR}. Skipping waveform {idx_waveform}.")
            return {
                'previous_min_velocity': None,
                'gouge_velocity_list_waveform': None,
                'L2norm_waveform': None,
                'best_gouge_velocity': None,
                'range_factor': None
            }

        is_first_waveform = True

    else:
        # For subsequent waveforms, search around previous velocity
        is_first_waveform = False
        c_range_waveform = range_scaling_factor * c_range
        cmin_waveform = previous_min_velocity - c_range_waveform
        cmax_waveform = previous_min_velocity + c_range_waveform
        c_step_waveform = c_step

        max_travel_time = compute_dds_travel_time(assembly_travel_time= assembly_travel_time,
                                    side1_params=side1_params,
                                    side2_params=side2_params,
                                    central_params=central_params,
                                    thickness_gouge_1=thickness_gouge_1,
                                    thickness_gouge_2=thickness_gouge_2,
                                    v_gouge_1=cmin_waveform,
                                    v_gouge_2=cmin_waveform)

        min_travel_time = compute_dds_travel_time(assembly_travel_time= assembly_travel_time,
                                    side1_params=side1_params,
                                    side2_params=side2_params,
                                    central_params=central_params,
                                    thickness_gouge_1=thickness_gouge_1,
                                    thickness_gouge_2=thickness_gouge_2,
                                    v_gouge_1=cmax_waveform,
                                    v_gouge_2=cmax_waveform)
        
    misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time + stf_duration))[0]
    
    # sys.exit(f"{min_travel_time},{max_travel_time}")
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
    save_plot = (idx_waveform % plot_save_interval == 0) 
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
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_time=stf_time,
        stf_waveform=stf_waveform,
        frequency_cutoff=frequency_cutoff,
        assembly_dict=assembly_dict,
        gouge_velocity=(best_gouge_velocity, best_gouge_velocity),
        gouge_thickness=(thickness_gouge_1,thickness_gouge_2),
        misfit_interval=misfit_interval,
        fixed_minimum_velocity=best_gouge_velocity,
        normalize_waveform=True,
        enable_plotting=save_plot,
        make_movie=save_movie,
        plot_output_path=str(plot_output_path) if plot_output_path else None,
        movie_output_path=str(movie_output_path) if movie_output_path else None
    )

    # Possibly plot L2 norm vs. velocity
    save_l2norm_plot = (idx_waveform % l2norm_plot_interval == 0) or is_first_waveform
    if save_l2norm_plot:
        plotter = Plotter()
        l2norm_plot_name = f"{outfile_name}_L2norm_waveform_{overall_index}"
        l2norm_plot_path = outdir_path_image / l2norm_plot_name
        plotter.plot_l2_norm_vs_velocity(
            gouge_velocity=gouge_velocity_list_waveform,
            L2norm=L2norm_waveform,
            overall_index=overall_index,
            outfile_path=l2norm_plot_path
        )

    return {
        'previous_min_velocity': previous_min_velocity,
        'gouge_velocity_list_waveform': gouge_velocity_list_waveform,
        'L2norm_waveform': L2norm_waveform,
        'best_gouge_velocity': best_gouge_velocity,
        'range_factor': range_scaling_factor
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
    machine_name = "Brava_2"
    experiment_name = "s0216suw04anh_50"
    wave_type = "_p"    # that "_" is ugly, but needed
    data_type_uw = "uw_data/data_tsv_files" + wave_type
    data_type_mech = "mechanical_data"
    mech_file_name = f"{experiment_name}_data_rp"

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
        "frequency_cutoff_MHz": 4,
        "minimum_SNR": 5,
        "c_step_cm/mus":  50 * (1e2 / 1e6),
        "c_range_cm/mus":  100 * (1e2 / 1e6),
        "range_scaling_factor": 1,
        "plot_save_interval": 1,
        "movie_save_interval": 1,
        "l2norm_plot_interval": 1,
        "number_of_waveforms2process": 10,
        "outdir_path_l2norm": outdir_path_l2norm[0],
        "outdir_path_image": outdir_path_image[0]
    }

    # Load the source time function
    stf_waveform, stf_time, stf_duration = load_and_process_stf(
        dir_manager=dir_manager,
        machine_name_stf="on_bench",
        experiment_name_stf="STF",
        data_type_stf="data_analysis/source_time_functions" + wave_type,
        stf_choosen="width250_volt200_p2p",
        frequency_cutoff_MHz= params['frequency_cutoff_MHz']
    )

    # Load mechanical data
    mech_data, sync_data, sync_peaks = load_mechanical_data(
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
    assembly_dict= {"side1_params": side1_params, "side2_params":side2_params, "central_params":central_params}

    # fixed travel time in the assembly, without gouge and excluding grouves. It is the lower bound for signal detection
    assembly_dict['assembly_travel_time'] = ((side1_params["z_pzt2grove"]-side1_params["h_grooves"])/side1_params["velocity"+ wave_type]
                                       +(side2_params["z_pzt2grove"]-side2_params["h_grooves"])/side2_params["velocity" + wave_type]
                                       +(central_params["z"]-2*central_params['h_grooves'])/central_params["velocity" + wave_type]
                                        )
    assembly_dict["wave_type"] = wave_type
    assembly_dict["transmitter_position"] = side1_params["z_pzt2grove"]
    assembly_dict["receiver_position"] = side2_params["z_pzt2grove"]

    # Make UW path list
    infile_path_list_uw = sorted(
        dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw)
    )

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
        process_uw_file(
            infile_path=infile_path,
            chosen_uw_file=chosen_uw_file,
            arrival_time_interval=arrival_time_interval,
            sync_peaks=sync_peaks,
            mech_data=mech_data,
            stf_waveform=stf_waveform,
            stf_time=stf_time,
            stf_duration=stf_duration,
            params=params,
            assembly_dict = assembly_dict
        )
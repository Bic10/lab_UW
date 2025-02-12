# lab_uw/source_receiver_simulation_parameters.py

import sys
from pathlib import Path
import pickle
import time as tm
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple, Union, Optional

from lab_uw.data_io import UltrasonicDataHandler, BlockMetadataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.signal_processing import SignalProcessor
from lab_uw.forward_modeling import ForwardModeler, compute_misfit
from lab_uw.plotting import Plotter

###############################################
# 1) LOAD AND PROCESS STF
###############################################
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
    """
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

    stf_handler = UltrasonicDataHandler()
    stf_waveform_raw, stf_metadata = stf_handler.load_waveform_json(chosen_stf_path)

    stf_time = np.array(stf_metadata["time_ax_waveform"])
    signal_processor = SignalProcessor()

    stf_waveform_filt, _ = signal_processor.signal2noise_separation_lowpass(
        waveform_data=stf_waveform_raw,
        metadata=stf_metadata,
        freq_cut=frequency_cutoff_MHz
    )

    # Shift waveform so the first sample is zero
    stf_waveform = stf_waveform_filt - stf_waveform_filt[0]
    stf_duration = stf_time[-1] - stf_time[0]

    return stf_waveform, stf_time, stf_duration

###############################################
# 2) PROCESS UW FILE
###############################################
def process_uw_file(
    infile_path: Path,
    stf_waveform: np.ndarray,
    stf_time: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    montecarlo: Dict[str, Any]
) -> None:
    """
    Process a single UW data file, performing a Monte Carlo set of forward simulations
    with random draws from distributions specified in 'montecarlo'.
    """
    print(f"PROCESSING UW DATA IN {infile_path}:")

    # Unpack parameters
    maxtime2simulate = params["maxtime2simulate_mus"]
    frequency_cutoff_MHz = params["frequency_cutoff_MHz"]
    number_of_waveforms2process = params["number_of_waveforms2process"]
    outdir_path_l2norm = params["outdir_path_l2norm"]
    outdir_path_image = params["outdir_path_image"]

    # Derive output filenames
    outfile_name = infile_path.stem.split(".")[0] # it is just a stupid problem: the files are saved with double extension
    outfile_path = outdir_path_l2norm / outfile_name

    start_time = tm.time()

    # Load UW data
    ultrasonic_handler = UltrasonicDataHandler.make_UW_data(infile_path)
    observed_waveform_data, metadata = ultrasonic_handler.waveform_data, ultrasonic_handler.metadata
    observed_time = metadata["time_ax_waveform"]

    # Remove mean from the entire dataset
    observed_waveform_data = observed_waveform_data - np.mean(observed_waveform_data)

    # Possibly reduce the number of samples (time-limiting)
    if maxtime2simulate:
        idx_maxtime = np.searchsorted(observed_time, maxtime2simulate)
        observed_waveform_data = observed_waveform_data[:, :idx_maxtime]
        observed_time = observed_time[:idx_maxtime]
        print("Truncating at maxtime2simulate:", maxtime2simulate)

    # Downsampling waveforms
    downsampling = max(1, round(metadata["number_of_waveforms"] / number_of_waveforms2process))
    print(
        f"Number of waveforms: {metadata['number_of_waveforms']}, "
        f"wanting {number_of_waveforms2process}, downsampling factor: {downsampling}"
    )

    # We only want 1 "mean" waveform for analysis
    observed_waveform = np.mean(observed_waveform_data, axis=0)

    # Lowpass filtering
    signal_processor = SignalProcessor()
    observed_waveform, _ = signal_processor.signal2noise_separation_lowpass(
        waveform_data=observed_waveform,
        metadata=metadata,
        freq_cut=frequency_cutoff_MHz
    )

    # Run the Monte Carlo approach on this single "mean" waveform
    result = process_waveform(
        observed_waveform=observed_waveform,
        observed_time=observed_time,
        outfile_name=outfile_name,
        stf_waveform=stf_waveform,
        stf_time=stf_time,
        params=params,
        assembly_dict=assembly_dict,
        montecarlo=montecarlo
    )

    # Save results to a pickle
    # We'll store the entire 'result' dict for completeness
    results_pkl = outfile_path.with_suffix(".pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump({
            "montecarlo_result": result,
            "metadata": metadata,
            "params": params
        }, f)

    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---\n")

###############################################
# 3) PROCESS A SINGLE WAVEFORM WITH MONTE CARLO
###############################################
def process_waveform(
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    outfile_name: str,
    stf_waveform: np.ndarray,
    stf_time: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    montecarlo: Dict[str, Any]
) -> Dict[str, Union[float, np.ndarray, None]]:
    """
    Perform multiple (num_iterations) forward simulations, each with random draws
    for steel_velocity, pzt_velocity, geometry spreads, etc. Then pick the one
    with minimal L2 misfit, and plot/return the best parameters.
    """

    frequency_cutoff = params["frequency_cutoff_MHz"]
    outdir_path_image_list = params["outdir_path_image"]
    wave_type = assembly_dict["wave_type"]

    # Convert image path to Path object
    outdir_path_image = (
        Path(outdir_path_image_list)
        if isinstance(outdir_path_image_list, str)
        else outdir_path_image_list
    )
    if isinstance(outdir_path_image, list):
        outdir_path_image = Path(outdir_path_image[0])

    # We'll measure misfit only where observed_time > 0
    misfit_interval = np.where(observed_time > 0)[0]

    #### MONTE CARLO: read parameters from 'montecarlo' ####
    num_iteration = montecarlo["num_iterations"]

    steel_mean  = montecarlo["steel_velocity_mean"]
    steel_std   = montecarlo["steel_velocity_std"]
    pzt_mean    = montecarlo["pzt_velocity_mean"]
    pzt_std     = montecarlo["pzt_velocity_std"]

    # Sizing for transmitter and receiver "spreading" or "position" parameters:
    spread_tx_mean = montecarlo["spreading_factor_transmitter_mean"]
    spread_tx_std  = montecarlo["spreading_factor_transmitter_std"]
    spread_rx_mean = montecarlo["spreading_factor_receiver_mean"]
    spread_rx_std  = montecarlo["spreading_factor_receiver_std"]

    # Uniform ranges for positions and radius factors
    pos_edge_low  = montecarlo["position2edge_low"]
    pos_edge_high = montecarlo["position2edge_high"]
    radius_low    = montecarlo["radius_factor_low"]
    radius_high   = montecarlo["radius_factor_high"]

    #### Build argument list for each iteration ####
    args_list = []
    for iteration in range(num_iteration):
        steel_velocity2simulate = np.random.normal(loc=steel_mean, scale=steel_std)
        pzt_velocity2simulate   = np.random.normal(loc=pzt_mean,   scale=pzt_std)

        spreading_factor_tx = np.random.normal(loc=spread_tx_mean, scale=spread_tx_std)
        spreading_factor_rx = np.random.normal(loc=spread_rx_mean, scale=spread_rx_std)

        position2edge_tx   = np.random.uniform(low=pos_edge_low, high=pos_edge_high)
        position2edge_rx   = np.random.uniform(low=pos_edge_low, high=pos_edge_high)

        radius_factor_tx   = np.random.uniform(low=radius_low, high=radius_high)
        radius_factor_rx   = np.random.uniform(low=radius_low, high=radius_high)

        args_list.append((
            spreading_factor_tx,
            spreading_factor_rx,
            position2edge_tx,
            position2edge_rx,
            radius_factor_tx,
            radius_factor_rx,
            steel_velocity2simulate,
            pzt_velocity2simulate,
            observed_waveform,
            misfit_interval,
            observed_time,
            stf_time,
            stf_waveform,
            params,
            assembly_dict,
            montecarlo
        ))



    #### Parallel simulations
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_velocity, args_list)

    # results is a list of tuples:
    # [ (pzt_vel, steel_vel, L2misfit, spread_tx, spread_rx, pos_tx, pos_rx, rad_tx, rad_rx), ... ]

    # Sort by L2 misfit => index 2 in each tuple
    results_sorted = sorted(results, key=lambda x: x[2])  # x[2] is L2norm

    # Extract arrays for plotting/analyzing
    pzt_velocity_list = np.array([res[0] for res in results_sorted])
    steel_velocity_list = np.array([res[1] for res in results_sorted])
    L2norm_waveform = np.array([res[2] for res in results_sorted])

    # Best solution
    min_idx = 0  # After sorting, the best is at index 0
    best_pzt_velocity          = results_sorted[min_idx][0]
    best_steel_velocity        = results_sorted[min_idx][1]
    best_L2                    = results_sorted[min_idx][2]
    best_spread_tx             = results_sorted[min_idx][3]
    best_spread_rx             = results_sorted[min_idx][4]
    best_position2edge_tx      = results_sorted[min_idx][5]
    best_position2edge_rx      = results_sorted[min_idx][6]
    best_radius_factor_tx      = results_sorted[min_idx][7]
    best_radius_factor_rx      = results_sorted[min_idx][8]

    # Print the best results
    print(f"\n{outfile_name}:")
    print(f"  Best Steel Velocity   = {best_steel_velocity:.4f} cm/μs")
    print(f"  Best PZT Velocity     = {best_pzt_velocity:.4f} cm/μs")
    print(f"  Best L2 Misfit        = {best_L2:.4e}")
    print("  Best Geometry:")
    print(f"    spread_tx           = {best_spread_tx:.3f}")
    print(f"    spread_rx           = {best_spread_rx:.3f}")
    print(f"    pos2edge_tx         = {best_position2edge_tx:.3f}")
    print(f"    pos2edge_rx         = {best_position2edge_rx:.3f}")
    print(f"    radius_factor_tx    = {best_radius_factor_tx:.3f}")
    print(f"    radius_factor_rx    = {best_radius_factor_rx:.3f}")

    # Now let's do a final forward simulation with these best parameters
    assembly_dict["velocity" + wave_type] = best_steel_velocity
    assembly_dict["pzt_velocity" + wave_type] = best_pzt_velocity

    # Also set in 'montecarlo' if your forward model needs them
    montecarlo["spreading_factor_transmitter"] = best_spread_tx
    montecarlo["spreading_factor_receiver"] = best_spread_rx
    montecarlo["position2edge_transmitter"] = best_position2edge_tx
    montecarlo["position2edge_receiver"] = best_position2edge_rx
    montecarlo["radius_factor_transmitter"] = best_radius_factor_tx
    montecarlo["radius_factor_receiver"] = best_radius_factor_rx

    plot_output_name = f"{outfile_name}_best_simulation"
    plot_output_path = outdir_path_image / plot_output_name
    synthetic_waveform, _,_,_,_,_,_,_,_,_,_ = ForwardModeler().block_forward_simulation(
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_time=stf_time,
        stf_waveform=stf_waveform,
        frequency_cutoff=params["frequency_cutoff_MHz"],
        assembly_dict=assembly_dict,
        montecarlo=montecarlo,
        misfit_interval=misfit_interval,
        minimum_velocity=params["min_velocity2simulate"],
        maximum_velocity=params["max_velocity2simulate"],
        normalize_waveform=True,
        enable_plotting=True,
        plot_output_path=plot_output_path
    )

    # Plot L2 norm vs. steel velocity
    # (If you also want PZT velocity on the plot, you could do a 2D scatter, but let's keep it simple.)
    plotter = Plotter()
    plot_output_name = f"{outfile_name}_L2norm_waveform"
    l2norm_plot_path = outdir_path_image / plot_output_name
    plotter.plot_l2_norm_vs_velocity(
        velocity=steel_velocity_list,
        L2norm=L2norm_waveform,
        overall_index="",
        outfile_path=l2norm_plot_path
    )

    # Return final info
    return {
        "velocity_list_waveform": steel_velocity_list,
        "L2norm_waveform": L2norm_waveform,
        "best_steel_velocity": best_steel_velocity,
        "best_pzt_velocity": best_pzt_velocity,
        "best_L2_misfit": best_L2,
        "best_spread_tx": best_spread_tx,
        "best_spread_rx": best_spread_rx,
        "best_position2edge_tx": best_position2edge_tx,
        "best_position2edge_rx": best_position2edge_rx,
        "best_radius_factor_tx": best_radius_factor_tx,
        "best_radius_factor_rx": best_radius_factor_rx
    }

###############################################
# 4) PROCESS ONE VELOCITY DRAW
###############################################
def process_velocity(args):
    """
    Function to process a single set of random-draw parameters in the Monte Carlo.

    Returns a tuple with:
        (pzt_vel, steel_vel, L2misfit,
         spread_tx, spread_rx, pos_tx, pos_rx,
         radius_tx, radius_rx)
    so we can keep track of geometry parameters, too.
    """
    (
        spreading_factor_transmitter,
        spreading_factor_receiver,
        position2edge_transmitter,
        position2edge_receiver,
        radius_factor_transmitter,
        radius_factor_receiver,
        steel_velocity2simulate,
        pzt_velocity2simulate,
        observed_waveform,
        misfit_interval,
        observed_time,
        stf_time,
        stf_waveform,
        params,
        assembly_dict,
        montecarlo
    ) = args

    wave_type = assembly_dict["wave_type"]

    # Overwrite the assembly_dict for this iteration
    assembly_dict["velocity" + wave_type] = steel_velocity2simulate
    assembly_dict["pzt_velocity" + wave_type] = pzt_velocity2simulate

    # Also store in 'montecarlo' if the ForwardModeler needs them
    montecarlo["spreading_factor_transmitter"] = spreading_factor_transmitter
    montecarlo["spreading_factor_receiver"] = spreading_factor_receiver
    montecarlo["position2edge_transmitter"] = position2edge_transmitter
    montecarlo["position2edge_receiver"] = position2edge_receiver
    montecarlo["radius_factor_transmitter"] = radius_factor_transmitter
    montecarlo["radius_factor_receiver"] = radius_factor_receiver

    # Run forward simulation for this draw
    synthetic_waveform, _,_,_,_,_,_,_,_,_,_ = ForwardModeler().block_forward_simulation(
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_time=stf_time,
        stf_waveform=stf_waveform,
        frequency_cutoff=params["frequency_cutoff_MHz"],
        assembly_dict=assembly_dict,
        montecarlo=montecarlo,
        misfit_interval=misfit_interval,
        minimum_velocity=params["min_velocity2simulate"],
        maximum_velocity=params["max_velocity2simulate"],
        normalize_waveform=True,
        enable_plotting=False
    )

    # Calculate misfit
    L2norm_new = compute_misfit(
        observed_waveform=observed_waveform,
        synthetic_waveform=synthetic_waveform,
        misfit_interval=misfit_interval
    )

    print((f"\tPZT={pzt_velocity2simulate:.4f}, Steel={steel_velocity2simulate:.4f}, txspread={spreading_factor_transmitter:.2f}, rxspread={spreading_factor_receiver:.2f}, tx2edge={position2edge_transmitter:.2f}, rx2edge={position2edge_receiver:.2f}, tx rad={radius_factor_transmitter:.2f}, rx rad={radius_factor_receiver:.4f}  => L2={L2norm_new:.4e}"))

    return (
        pzt_velocity2simulate,
        steel_velocity2simulate,
        L2norm_new,
        spreading_factor_transmitter,
        spreading_factor_receiver,
        position2edge_transmitter,
        position2edge_receiver,
        radius_factor_transmitter,
        radius_factor_receiver
    )


###############################################
# 5) MAIN
###############################################
if __name__ == "__main__":

    dir_manager = DirectoryManager()

    # Basic experiment info
    machine_name = "on_bench"
    experiment_name = "STF"
    wave_type = "_p"  # e.g., compressional wave

    # Data location
    data_type_uw = f"uw_data/data_tsv_files{wave_type}"

    # Create output directories
    outdir_path_l2norm = dir_manager.make_data_analysis_folders(
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=[f"source_receiver_simulation_parameters{wave_type}"]
    )
    outdir_path_image = dir_manager.make_data_analysis_folders(
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=[f"source_receiver_simulation_parameters_images_and_movie{wave_type}"]
    )
    print(f"Misfits will be saved at:\n{outdir_path_l2norm[0]}")

    # Build dictionary with assembly metadata
    block = BlockMetadataHandler.load_blocks_metadata(
        dir_manager=dir_manager,
        blocks_metadata_name="blocks_metadata.json",
        block_keys=("on_bench_STF2",)
    )
    assembly_dict = block[0]
    assembly_dict["wave_type"] = wave_type
    assembly_dict["transmitter_position"] = 0
    assembly_dict["receiver_position"] = assembly_dict["z"]

    # Basic simulation parameters
    params = {
        "maxtime2simulate_mus": 40,
        "frequency_cutoff_MHz": 6,
        "minimum_SNR": 5,
        "min_velocity2simulate": 3000 * (1e2 / 1e6),  # cm/mus
        "max_velocity2simulate": 6000 * (1e2 / 1e6),  # cm/mus
        "plot_save_interval": 1,
        "movie_save_interval": 50,
        "l2norm_plot_interval": 1,
        "number_of_waveforms2process": 10,
        "outdir_path_l2norm": outdir_path_l2norm[0],
        "outdir_path_image": outdir_path_image[0]
    }

    #### MONTE CARLO PARAMETERS DEFINED HERE ####
    # You can set the means/stdevs for velocity, geometry, etc.
    montecarlo = {
        "num_iterations": 1000,  # how many random draws to try

        # Velocities
        "steel_velocity_mean": assembly_dict["velocity" + wave_type],
        "steel_velocity_std": 50 * (1e2 / 1e6),              
        "pzt_velocity_mean": assembly_dict["pzt_velocity" + wave_type], 
        "pzt_velocity_std": 500* (1e2 / 1e6),

        # Spreading (Gaussian) means / std for pzt source/receiver spatial distribution
        "spreading_factor_transmitter_mean": 1.0,
        "spreading_factor_transmitter_std": 0.005,
        "spreading_factor_receiver_mean": 1.0,
        "spreading_factor_receiver_std": 0.005,

        # Uniform range for positions relative to edges pzt-steel
        "position2edge_low": 0.9,
        "position2edge_high": 1.1,

        # Uniform range for radius factors: 
        # how many nodes to use to approximate the tx/rx positions in case they do not correspond precisely to one node
        "radius_factor_low": 1.9,
        "radius_factor_high": 2.0
    }

    # Make UW path list
    infile_path_list_uw = sorted(
        dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw)
    )

    # Process each UW file
    for infile_path in infile_path_list_uw:
        # A) Load the source time function
        infile_name = infile_path.name.split(".")[0]
        stf_waveform, stf_time, _ = load_and_process_stf(
            dir_manager=dir_manager,
            machine_name_stf="on_bench",
            experiment_name_stf="STF",
            data_type_stf=f"data_analysis/source_time_functions{wave_type}",
            stf_choosen=infile_name,
            frequency_cutoff_MHz=params["frequency_cutoff_MHz"]
        )

        # B) Run the main simulation routine
        process_uw_file(
            infile_path=infile_path,
            stf_waveform=stf_waveform,
            stf_time=stf_time,
            params=params,
            assembly_dict=assembly_dict,
            montecarlo=montecarlo
        )

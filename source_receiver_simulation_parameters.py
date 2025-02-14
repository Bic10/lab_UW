# lab_uw/source_receiver_simulation_parameters.py

from pathlib import Path
import pickle
import time as tm
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Tuple, Union
import matplotlib.pyplot as plt
import corner

from lab_uw.data_io import UltrasonicDataHandler, BlockMetadataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.signal_processing import SignalProcessor
from lab_uw.forward_modeling import ForwardModeler, compute_misfit

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

def process_uw_file(
    infile_path: Path,
    stf_waveform: np.ndarray,
    stf_time: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    montecarlo: Dict[str, Any]
) -> None:
    """
    Process a single UW data file multiple times (e.g., 100) with the same
    Monte Carlo approach, storing each 'best' solution. Then aggregate
    those best solutions to visualize the distribution of parameters.
    """
    print(f"PROCESSING UW DATA IN {infile_path}:")

    # Unpack parameters
    maxtime2simulate            = params["maxtime2simulate_mus"]
    frequency_cutoff_MHz        = params["frequency_cutoff_MHz"]
    number_of_waveforms2process = params["number_of_waveforms2process"]
    outdir_path_l2norm          = params["outdir_path_l2norm"]
    outdir_path_image           = params["outdir_path_image"]

    start_time = tm.time()

    # 1) Load UW data
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

    # 2) We'll run the Monte Carlo approach multiple times
    n_repeats = 100

    # We will store the best parameters from each run in a list of dicts
    all_best_params = []

    for idx in range(n_repeats):
        # Derive output filenames
        outfile_name = infile_path.stem.split(".")[0] + "_" + str(idx) # it is just a stupid problem: the files are saved with double extension
        outfile_path = outdir_path_l2norm / outfile_name
        # Run the Monte Carlo for this run
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
        results_pkl = outfile_path.with_suffix(".pkl")
        with open(results_pkl, "wb") as f:
            pickle.dump(
                {"montecarlo_result": result, "metadata": metadata, "params": params},
                f
            )

        # Also collect the best parameters in memory
        best_dict = {
            "best_steel_velocity": result["best_steel_velocity"],
            "best_pzt_velocity": result["best_pzt_velocity"],
            "best_spread_tx": result["best_spread_tx"],
            "best_spread_rx": result["best_spread_rx"],
            "best_position2edge_tx": result["best_position2edge_tx"],
            "best_position2edge_rx": result["best_position2edge_rx"],
            "best_radius_factor_tx": result["best_radius_factor_tx"],
            "best_radius_factor_rx": result["best_radius_factor_rx"],
            "best_L2_misfit": result["best_L2_misfit"]
        }
        all_best_params.append(best_dict)

    # 3) Now we have a list of 100 best solutions. Let's aggregate them.
    #    We can do a box plot (or any other) to show distribution for each parameter.

    # Convert to numpy arrays for convenience
    steel_vals  = np.array([p["best_steel_velocity"] for p in all_best_params])
    pzt_vals    = np.array([p["best_pzt_velocity"]   for p in all_best_params])
    spread_tx   = np.array([p["best_spread_tx"]      for p in all_best_params])
    spread_rx   = np.array([p["best_spread_rx"]      for p in all_best_params])
    pos_tx      = np.array([p["best_position2edge_tx"] for p in all_best_params])
    pos_rx      = np.array([p["best_position2edge_rx"] for p in all_best_params])
    rad_tx      = np.array([p["best_radius_factor_tx"] for p in all_best_params])
    rad_rx      = np.array([p["best_radius_factor_rx"] for p in all_best_params])
    l2_vals     = np.array([p["best_L2_misfit"]       for p in all_best_params])

    # Combine them in a 2D array for boxplot convenience
    # We'll omit L2 from the boxplot or put it last
    param_matrix = np.column_stack([
        steel_vals, pzt_vals, spread_tx, spread_rx, pos_tx, pos_rx, rad_tx, rad_rx
    ])
    param_labels = [
        "Steel Vel", "PZT Vel", "Spread Tx", "Spread Rx",
        "Pos2Edge Tx", "Pos2Edge Rx", "Radius Tx", "Radius Rx"
    ]

    # 4) Create a figure with a box plot
    fig, ax = plt.subplots(figsize=(10,6))
    box = ax.boxplot(param_matrix, patch_artist=True, labels=param_labels, showmeans=True)

    # Style the boxplot
    for patch in box['boxes']:
        patch.set(facecolor="lightblue", alpha=0.5)
    for median in box['medians']:
        median.set(color="red", linewidth=2)
    for mean_line in box['means']:
        mean_line.set(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=5)

    ax.set_title(f"Distribution of Best Parameters\n{infile_path.stem} ({n_repeats} runs)")
    ax.set_ylabel("Parameter Value")

    # We can annotate the means or standard dev if desired
    # e.g. compute them manually:
    means = param_matrix.mean(axis=0)
    stds  = param_matrix.std(axis=0)
    # etc.

    # Optionally, we can do a separate subplot or figure for L2 distribution
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(l2_vals, bins=10, color="lightgreen", edgecolor="k", alpha=0.7)
    ax2.set_title(f"L2 Misfit Distribution\n{infile_path.stem} ({n_repeats} runs)")
    ax2.set_xlabel("L2 Misfit")
    ax2.set_ylabel("Count")

    # Save these figures
    boxplot_path = outdir_path_image / f"{infile_path.stem}_aggregated_boxplot.png"
    fig.savefig(boxplot_path, dpi=150)
    plt.close(fig)

    hist_path = outdir_path_image / f"{infile_path.stem}_l2_distribution.png"
    fig2.savefig(hist_path, dpi=150)
    plt.close(fig2)

    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---\n")

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
    with minimal L2 misfit. Additionally, create scatter plots of L2 vs. each parameter.
    """

    wave_type = assembly_dict["wave_type"]
    outdir_path_image_list = params["outdir_path_image"]

    # Convert output path to Path object
    outdir_path_image = (
        Path(outdir_path_image_list)
        if isinstance(outdir_path_image_list, str)
        else outdir_path_image_list
    )
    if isinstance(outdir_path_image, list):
        outdir_path_image = Path(outdir_path_image[0])

    ###### For compatibility with global optimization
    misfit_interval = np.where(observed_time > 0)[0]

    # Monte Carlo parameters
    num_iteration = montecarlo["num_iterations"]
    steel_low     = montecarlo["steel_velocity_low"]
    steel_high    = montecarlo["steel_velocity_high"]
    pzt_low       = montecarlo["pzt_velocity_low"]
    pzt_high      = montecarlo["pzt_velocity_high"]
    spread_low    = montecarlo["spreading_factor_low"]
    spread_high   = montecarlo["spreading_factor_high"]
    pos_edge_low  = montecarlo["position2edge_low"]
    pos_edge_high = montecarlo["position2edge_high"]
    radius_low    = montecarlo["radius_factor_low"]
    radius_high   = montecarlo["radius_factor_high"]

    # Build argument list
    args_list = []
    for iteration in range(num_iteration):
        steel_velocity2simulate = np.random.uniform(low=steel_low, high=steel_high)
        pzt_velocity2simulate   = np.random.uniform(low=pzt_low, high=pzt_high)
        spreading_factor_tx     = np.random.uniform(low=spread_low, high=spread_high)
        spreading_factor_rx     = np.random.uniform(low=spread_low, high=spread_high)
        position2edge_tx        = np.random.uniform(low=pos_edge_low, high=pos_edge_high)
        position2edge_rx        = np.random.uniform(low=pos_edge_low, high=pos_edge_high)
        radius_factor_tx        = np.random.uniform(low=radius_low, high=radius_high)
        radius_factor_rx        = np.random.uniform(low=radius_low, high=radius_high)

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

    # Run in parallel
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_velocity, args_list)

    # Sort by L2 misfit => index 2
    results_sorted = sorted(results, key=lambda x: x[2])

    # Extract into arrays
    pzt_array       = np.array([r[0] for r in results_sorted])
    steel_array     = np.array([r[1] for r in results_sorted])
    L2_array        = np.array([r[2] for r in results_sorted])
    spread_tx_array = np.array([r[3] for r in results_sorted])
    spread_rx_array = np.array([r[4] for r in results_sorted])
    pos_tx_array    = np.array([r[5] for r in results_sorted])
    pos_rx_array    = np.array([r[6] for r in results_sorted])
    rad_tx_array    = np.array([r[7] for r in results_sorted])
    rad_rx_array    = np.array([r[8] for r in results_sorted])

    # Best solution is the first in sorted list
    best_pzt_velocity          = results_sorted[0][0]
    best_steel_velocity        = results_sorted[0][1]
    best_L2                    = results_sorted[0][2]
    best_spread_tx             = results_sorted[0][3]
    best_spread_rx             = results_sorted[0][4]
    best_position2edge_tx      = results_sorted[0][5]
    best_position2edge_rx      = results_sorted[0][6]
    best_radius_factor_tx      = results_sorted[0][7]
    best_radius_factor_rx      = results_sorted[0][8]

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

    # Re-run forward simulation for best parameters (and optionally plot)
    assembly_dict["velocity" + wave_type]       = best_steel_velocity
    assembly_dict["pzt_velocity" + wave_type]   = best_pzt_velocity
    montecarlo["spreading_factor_transmitter"]  = best_spread_tx
    montecarlo["spreading_factor_receiver"]     = best_spread_rx
    montecarlo["position2edge_transmitter"]     = best_position2edge_tx
    montecarlo["position2edge_receiver"]        = best_position2edge_rx
    montecarlo["radius_factor_transmitter"]     = best_radius_factor_tx
    montecarlo["radius_factor_receiver"]        = best_radius_factor_rx

    plot_output_name = f"{outfile_name}_best_simulation"
    plot_output_path = outdir_path_image / plot_output_name
    synthetic_waveform, *_ = ForwardModeler().block_forward_simulation(
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

    ############################################################################
    # SCATTER PLOTS: L2 vs. each parameter
    ############################################################################
    param_labels = [
        ("Steel Velocity (cm/µs)", steel_array),
        ("PZT Velocity (cm/µs)",   pzt_array),
        ("Spread Tx",              spread_tx_array),
        ("Spread Rx",              spread_rx_array),
        ("Pos2Edge Tx",            pos_tx_array),
        ("Pos2Edge Rx",            pos_rx_array),
        ("Radius Factor Tx",       rad_tx_array),
        ("Radius Factor Rx",       rad_rx_array),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(14, 7), tight_layout=True)
    axs = axs.flatten()  # so we can index them 0..7
    for i, (label, param_data) in enumerate(param_labels):
        ax = axs[i]
        ax.scatter(param_data, L2_array, s=30, c="blue", alpha=0.7, edgecolors="k")
        ax.set_xlabel(label)
        ax.set_ylabel("L2 Misfit")
        # highlight best param with a red star
        best_val = param_data[0]
        best_l2  = L2_array[0]
        ax.scatter(best_val, best_l2, s=100, c="red", marker="*")

    scatter_plot_path = outdir_path_image / f"{outfile_name}_param_vs_L2.png"
    fig.suptitle(f"L2 vs. Parameters\n{outfile_name}")
    fig.savefig(scatter_plot_path, dpi=150)
    plt.close(fig)

    ############################################################################
    # CORNER PLOT: pairwise relationships among parameters
    # If you want to see how parameters vary in relation to each other and L2,
    # you can try a corner plot approach. We'll store all parameters + L2 in one array.
    # This can be quite large if many parameters, but let's demonstrate:
    # We'll use a small library or do manual pairwise scatter.
    ############################################################################
    # Example: manual pairwise scatter for 2-3 parameters can blow up quickly.
    # If you want a "corner" approach, check out e.g. `corner.py` library
    # (https://github.com/dfm/corner.py).
    # For demonstration, let's just do a quick pairwise for 3 parameters: steel, pzt, L2:

    data_for_corner = np.vstack([steel_array, pzt_array, L2_array]).T
    corner.corner(
        data_for_corner,
        labels=["Steel Velocity", "PZT Velocity", "L2"],
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )
    corner_plot_path = outdir_path_image / f"{outfile_name}_corner_plot.png"
    plt.savefig(corner_plot_path, dpi=150)
    plt.close()

    # Return final info
    return {
        "velocity_list_waveform": steel_array,
        "L2norm_waveform": L2_array,
        "best_steel_velocity": best_steel_velocity,
        "best_pzt_velocity": best_pzt_velocity,
        "best_L2_misfit": best_L2,
        "best_spread_tx": best_spread_tx,
        "best_spread_rx": best_spread_rx,
        "best_position2edge_tx": best_position2edge_tx,
        "best_position2edge_rx": best_position2edge_rx,
        "best_radius_factor_tx": best_radius_factor_tx,
        "best_radius_factor_rx": best_radius_factor_rx,
    }

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
    montecarlo["spreading_factor_transmitter"]  = spreading_factor_transmitter
    montecarlo["spreading_factor_receiver"]     = spreading_factor_receiver
    montecarlo["position2edge_transmitter"]     = position2edge_transmitter
    montecarlo["position2edge_receiver"]        = position2edge_receiver
    montecarlo["radius_factor_transmitter"]     = radius_factor_transmitter
    montecarlo["radius_factor_receiver"]        = radius_factor_receiver

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

    print((f"\tPZT={pzt_velocity2simulate:.3f}, Steel={steel_velocity2simulate:.3f}, txspread={spreading_factor_transmitter:.2f}, rxspread={spreading_factor_receiver:.2f}, tx2edge={position2edge_transmitter:.2f}, rx2edge={position2edge_receiver:.2f}, tx rad={radius_factor_transmitter:.2f}, rx rad={radius_factor_receiver:.4f}  => L2={L2norm_new:.4e}"))

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

################### MAIN ############################
if __name__ == "__main__":

    dir_manager = DirectoryManager()

    # Basic experiment info
    machine_name = "on_bench"
    experiment_name = "STF"
    wave_type = "_p"  # e.g., compressional wave
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
        "maxtime2simulate_mus"      : 12,
        "frequency_cutoff_MHz"      : 6,
        "minimum_SNR"               : 5,
        "min_velocity2simulate"     : 4000 * (1e2 / 1e6),  # cm/mus
        "max_velocity2simulate"     : 6000 * (1e2 / 1e6),  # cm/mus
        "plot_save_interval"        : 1,
        "movie_save_interval"       : 50,
        "l2norm_plot_interval"      : 1,
        "number_of_waveforms2process": 10,
        "outdir_path_l2norm": outdir_path_l2norm[0],
        "outdir_path_image": outdir_path_image[0]
    }

    #### MONTE CARLO PARAMETERS DEFINED HERE ####
    montecarlo = {
        "num_iterations": 1000,  # how many random draws to try
        "steel_velocity_low": assembly_dict["velocity" + wave_type]- 0.01,
        "steel_velocity_high": assembly_dict["velocity" + wave_type]+ 0.01,              
        "pzt_velocity_low": params["min_velocity2simulate"], 
        "pzt_velocity_high": params["max_velocity2simulate"],
        "spreading_factor_low": 0.1,
        "spreading_factor_high": 3.0,
        # Uniform range for positions relative to edges pzt-steel
        "position2edge_low": 0.3,
        "position2edge_high": 1.7,
        # how many nodes to use to approximate the tx/rx positions in case they do not correspond precisely to one node
        "radius_factor_low": 0.1,
        "radius_factor_high": 2.0
    }

    # Make UW path list
    infile_path_list_uw = sorted(
        dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw)
    )

    # Process each UW file
    for infile_path in infile_path_list_uw:
        # Load the source time function
        infile_name = infile_path.name.split(".")[0]
        stf_waveform, stf_time, _ = load_and_process_stf(
            dir_manager=dir_manager,
            machine_name_stf="on_bench",
            experiment_name_stf="STF",
            data_type_stf=f"data_analysis/source_time_functions{wave_type}",
            stf_choosen=infile_name,
            frequency_cutoff_MHz=params["frequency_cutoff_MHz"]
        )
        # Run the main simulation routine
        process_uw_file(
            infile_path=infile_path,
            stf_waveform=stf_waveform,
            stf_time=stf_time,
            params=params,
            assembly_dict=assembly_dict,
            montecarlo=montecarlo
        )
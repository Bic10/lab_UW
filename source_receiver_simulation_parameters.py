# lab_uw/source_receiver_simulation_parameters.py

from pathlib import Path
import time as tm
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Union
import matplotlib.pyplot as plt

from lab_uw.data_io import UltrasonicDataHandler, BlockMetadataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.forward_modeling import UltrasonicModeler, compute_misfit
from lab_uw.plotting import Plotter

def process_uw_file(
    infile_path: Path,
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    global_search_space: Dict[str, Any]
) -> None:
    """
    Process a single UW data file multiple times (e.g., 100) with the same
    Monte Carlo approach, storing each 'best' solution. Then aggregate
    those best solutions to visualize the distribution of parameters.
    """
    print(f"PROCESSING UW DATA IN {infile_path.name}:")

    # Unpack parameters
    outdir_path_l2norm          = params["outdir_path_l2norm"]
    outdir_path_image           = params["outdir_path_image"]

    # Load and preprocess uw data
    uw_data_handler = UltrasonicDataHandler.load_and_process_uw(
        infile_path=infile_path,
        frequency_cutoff=params["frequency_cutoff"],
        maxtime2simulate=params["maxtime2simulate_mus"],
        number_of_waveforms2process=params["number_of_waveforms2process"]
    )

    observed_waveform_data = uw_data_handler.waveform_data
    metadata = uw_data_handler.metadata
    observed_time = metadata["time_ax_waveform"]

    # We only want 1 "mean" waveform for analysis
    observed_waveform = np.mean(observed_waveform_data, axis=0)

    # 2) We'll run the Monte Carlo approach multiple times
    n_repeats = 25

    # We will store the best parameters from each run in a list of dicts
    all_best_params = []

    for idx_waveform in range(n_repeats):
        # Derive output filenames
        outfile_name = infile_path.stem.split(".")[0] + "_" + str(idx_waveform) # it is just a stupid problem: the files are saved with double extension
        outfile_path = outdir_path_l2norm / outfile_name
        # Run the Monte Carlo for this run

        start_time = tm.time()

        result = process_waveform(
            idx_waveform        = idx_waveform,
            observed_waveform   = observed_waveform,
            observed_time       = observed_time,
            stf_handler         = stf_handler,
            params              = params,
            assembly_dict       = assembly_dict,
            global_search_space = global_search_space,
            outfile_name        = outfile_name,
        )

        print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---\n")

        # Save results to a pickle
        results_pkl = outfile_path.with_suffix(".pkl")
        with open(results_pkl, "wb") as f:
            pickle.dump(
                {"global_search_space_result": result, "params": params},
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

    plotter = Plotter()
    # Boxplot of best parameters
    boxplot_title = f"Distribution of Best Parameters\n{infile_path.stem} ({n_repeats} runs)"
    boxplot_outpath = outdir_path_image / f"{infile_path.stem}_aggregated_boxplot.png"
    plotter.plot_boxplot_parameters(param_matrix=param_matrix, 
                                    param_labels=param_labels, 
                                    title=boxplot_title,
                                    ylabel=f"Parameter values", 
                                    outfile_path=boxplot_outpath)

    # histogram of l2 misfit
    hist_title = f"L2 Misfit Distribution\n{infile_path.stem} ({n_repeats} runs)"
    hist_outpath = outdir_path_image / f"{infile_path.stem}_l2_distribution.png"
    plotter.plot_histogram_l2_distribution(data=l2_vals, 
                                           title=hist_title, 
                                           bins=20, 
                                           outfile_path=hist_outpath)

def process_waveform(
    idx_waveform: int,
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    outfile_name: str,
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    global_search_space: Dict[str, Any]
) -> Dict[str, Union[float, np.ndarray, None]]:
    """
    Perform multiple (num_iterations) forward simulations, each with random draws
    for steel_velocity, pzt_velocity, geometry spreads, etc. Then pick the one
    with minimal L2 misfit. Additionally, create scatter plots of L2 vs. each parameter.
    """

    wave_type = assembly_dict["wave_type"]
    # Replace the above block with something like:
    if isinstance(params["outdir_path_image"], list):
        outdir_path_image = Path(params["outdir_path_image"][0])
    else:
        outdir_path_image = Path(params["outdir_path_image"])

    ###### For compatibility with global optimization
    misfit_interval = np.where(observed_time >= 0)[0]

    # Monte Carlo parameters
    num_iteration = 1 if idx_waveform == 0 else global_search_space["num_iterations"] 

    steel_low     = global_search_space["steel_velocity_low"]
    steel_high    = global_search_space["steel_velocity_high"]
    pzt_low       = global_search_space["pzt_velocity_low"]
    pzt_high      = global_search_space["pzt_velocity_high"]
    spread_low    = global_search_space["spreading_factor_low"]
    spread_high   = global_search_space["spreading_factor_high"]
    pos_edge_low  = global_search_space["position2edge_low"]
    pos_edge_high = global_search_space["position2edge_high"]
    radius_low    = global_search_space["radius_factor_low"]
    radius_high   = global_search_space["radius_factor_high"]

    # Build argument list
    args_list = []
    for iteration in range(num_iteration):
        if idx_waveform == 0:
            steel_velocity2simulate   = 0.3205 
            pzt_velocity2simulate     = 0.2164 
            spreading_factor_tx  = 0.045
            spreading_factor_rx  = 0.616
            position2edge_tx     = -0.782
            position2edge_rx     = -0.830
            radius_factor_tx     = 0.308
            radius_factor_rx     = 1.710

        else:
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
            stf_handler,
            params,
            assembly_dict,
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

    montecarlo = {}
    montecarlo["spreading_factor_transmitter"]  = best_spread_tx
    montecarlo["spreading_factor_receiver"]     = best_spread_rx
    montecarlo["position2edge_transmitter"]     = best_position2edge_tx
    montecarlo["position2edge_receiver"]        = best_position2edge_rx
    montecarlo["radius_factor_transmitter"]     = best_radius_factor_tx
    montecarlo["radius_factor_receiver"]        = best_radius_factor_rx

    plot_output_name = f"{outfile_name}_best_simulation"
    movie_output_name = f"{outfile_name}_movie"
    plot_output_path = outdir_path_image / plot_output_name
    movie_output_path = outdir_path_image / movie_output_name

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        geometry_type="block",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler = stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_dict,
        montecarlo=montecarlo,
        misfit_interval=misfit_interval,
        minimum_velocity=params["min_velocity2simulate"],
        maximum_velocity=params["max_velocity2simulate"],
        normalize_waveform= True,
        enable_plotting=True,
        plot_output_path=plot_output_path,
        movie_output_path=movie_output_path
    )

    ############################################################################
    # SCATTER PLOTS: L2 vs. each parameter
    ############################################################################
    plotter = Plotter()
    scatter_title = f"L2 vs. Parameters\n{outfile_name}"
    scatter_outpath = outdir_path_image / f"{outfile_name}_param_vs_L2.png"
    param_list = [
        ("Steel Velocity (cm/µs)", steel_array),
        ("PZT Velocity (cm/µs)",   pzt_array),
        ("Spread Tx",              spread_tx_array),
        ("Spread Rx",              spread_rx_array),
        ("Pos2Edge Tx",            pos_tx_array),
        ("Pos2Edge Rx",            pos_rx_array),
        ("Radius Factor Tx",       rad_tx_array),
        ("Radius Factor Rx",       rad_rx_array),
    ]

    plotter.plot_scatter_l2_vs_parameters(
        param_list=param_list,
        l2_values=L2_array,
        title=scatter_title,
        best_index=0,  # if the best is the first index in your sorted arrays
        outfile_path=scatter_outpath
    )

    ############################################################################
    # LOCAL INVERSION
    ############################################################################    
    dc_max_start = 0

    # dc_max_start = 0.1*(np.amax(params["max_velocity2simulate"])- assembly_dict["velocity" + wave_type])
    dc_threshold = 0.01*dc_max_start

    dw_max_start = np.amax(simulation.source_handler.time_function)
    ds_max_start = 0
    # ds_max_start = np.amax(simulation.source_handler.spatial_function)

    simulation.run_local_inversion(observed_time=observed_time,
                                   observed_waveform=observed_waveform,
                                   misfit_interval=misfit_interval,
                                   n_iterations=50,
                                   dc_max_start=dc_max_start,
                                   dc_threshold=dc_threshold,
                                   reduce_factor=10/9,
                                   dw_max_start=dw_max_start,
                                   ds_max_start=ds_max_start,
                                   minimum_velocity=params["min_velocity2simulate"],
                                   maximum_velocity=params["max_velocity2simulate"],
                                   normalize_waveform = True,
                                   enable_plotting=True,
                                   plot_output_path=plot_output_path
                                   )
    
    stf_handler.waveform_data = np.interp(stf_handler.metadata["time_ax_waveform"], 
                                          simulation.sim_time_handler.simulation_time, 
                                          simulation.source_handler.time_function)

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
        stf_handler,
        params,
        assembly_dict,
    ) = args

    wave_type = assembly_dict["wave_type"]

    # Overwrite the assembly_dict for this iteration
    assembly_dict["velocity" + wave_type] = steel_velocity2simulate
    assembly_dict["pzt_velocity" + wave_type] = pzt_velocity2simulate

    # Also store in 'montecarlo' if the UltrasonicModeler needs them
    montecarlo = {}
    montecarlo["spreading_factor_transmitter"]  = spreading_factor_transmitter
    montecarlo["spreading_factor_receiver"]     = spreading_factor_receiver
    montecarlo["position2edge_transmitter"]     = position2edge_transmitter
    montecarlo["position2edge_receiver"]        = position2edge_receiver
    montecarlo["radius_factor_transmitter"]     = radius_factor_transmitter
    montecarlo["radius_factor_receiver"]        = radius_factor_receiver

    # Run forward simulation for this draw
    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        geometry_type="block",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler = stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_dict,
        montecarlo=montecarlo,
        misfit_interval=misfit_interval,
        minimum_velocity=params["min_velocity2simulate"],
        maximum_velocity=params["max_velocity2simulate"],
        normalize_waveform=True,
        enable_plotting=False
    )

    synthetic_waveform = simulation.synthetic_waveform

    # Calculate misfit
    L2norm_new = compute_misfit(
        observed_waveform=observed_waveform,
        synthetic_waveform=synthetic_waveform,
        misfit_interval=misfit_interval
    )
    
    print((f"\tL2={L2norm_new:.4e}\tSteel={steel_velocity2simulate:.3f}, PZT={pzt_velocity2simulate:.3f}, txspread={spreading_factor_transmitter:.3f}, rxspread={spreading_factor_receiver:.3f}, tx2edge={position2edge_transmitter:.3f}, rx2edge={position2edge_receiver:.3f}, txrad={radius_factor_transmitter:.4f}, rxrad={radius_factor_receiver:.4f}"))

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
    experiment_name = "STF_ss10_05"
    wave_type = "_s"  # e.g., compressional wave
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
        data_types=[f"source_receiver_simulation_parameters{wave_type}_images_and_movie_2025-03-30_only_STF_no_smooth_si_geospreading_si_norm"]
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
    assembly_dict["sample_dimensions"] = [assembly_dict["z"]] 

    # Basic simulation parameters
    params = {
        "maxtime2simulate_mus"      : 60,
        "frequency_cutoff"          : 6,     # MHz
        "minimum_SNR"               : 5,
        "min_velocity2simulate"     : 0.2,  # cm/mus
        "max_velocity2simulate"     : 0.4,  # cm/mus
        "plot_save_interval"        : 1,
        "movie_save_interval"       : 1,
        "l2norm_plot_interval"      : 1,
        "number_of_waveforms2process": 10,
        "outdir_path_l2norm": outdir_path_l2norm[0],
        "outdir_path_image": outdir_path_image[0]
    }

    #### MONTE CARLO PARAMETERS DEFINED HERE ####
    global_search_space = {
        "num_iterations": 500,  # how many random draws to try
        "steel_velocity_low": assembly_dict["velocity" + wave_type]- 0.01,
        "steel_velocity_high": assembly_dict["velocity" + wave_type]+ 0.015,              
        "pzt_velocity_low": assembly_dict["pzt_velocity" + wave_type], 
        "pzt_velocity_high": assembly_dict["velocity" + wave_type]+ 0.015,
        "spreading_factor_low" : 0.00001,
        "spreading_factor_high": 1,
        # Uniform range for positions relative to edges pzt-steel
        "position2edge_low" : -0.9,
        "position2edge_high": 0,
        # how many nodes to use to approximate the tx/rx positions in case they do not correspond precisely to one node
        "radius_factor_low": 0.002,
        "radius_factor_high": 2,
    }

    # Make UW path list
    infile_path_list_uw = sorted(
        dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw)
    )

    stf_chosen = "width250_volt70"

    # Process each UW file
    for infile_path in infile_path_list_uw:
        # Load the source time function
        infile_name = infile_path.name.split(".")[0]
        if infile_name != stf_chosen:
            continue
        # Load the Source Time Function
        stf_handler = UltrasonicDataHandler.load_stf(
            dir_manager=dir_manager,
            machine_name_stf=machine_name,
            experiment_name_stf=experiment_name,
            data_type_stf="data_analysis/source_time_functions" + wave_type,
            stf_chosen=stf_chosen,
            frequency_cutoff=params["frequency_cutoff"]
        )

        # Run the main simulation routine
        process_uw_file(
            infile_path=infile_path,
            stf_handler=stf_handler,
            params=params,
            assembly_dict=assembly_dict,
            global_search_space=global_search_space
        )
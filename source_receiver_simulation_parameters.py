# lab_uw/source_receiver_simulation_parameters.py

import sys
from pathlib import Path
import time as tm
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Union
import matplotlib.pyplot as plt

from lab_uw.data_io import UltrasonicDataHandler, BlockMetadataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.forward_modeling import UltrasonicModeler
from lab_uw.plotting import Plotter

from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

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

    # Load and preprocess uw data
    uw_data_handler = UltrasonicDataHandler.load_and_process_uw(
        infile_path=infile_path,
        frequency_cutoff=params["frequency_cutoff"],
        maxtime2simulate=params["maxtime2simulate"],
        number_of_waveforms2process=params["number_of_waveforms2process"]
    )

    observed_waveform_data = uw_data_handler.waveform_data
    metadata = uw_data_handler.metadata
    observed_time = metadata["time_ax_waveform"]
    # observed_waveform_data = butter_bandpass_filter(observed_waveform_data, 0.25, params["frequency_cutoff"], metadata["sampling_rate"])

    # We only want 1 "mean" waveform for analysis
    observed_waveform = np.mean(observed_waveform_data, axis=0)

    # 2) We'll run the Monte Carlo approach multiple times
    n_repeats = 10

    # We will store the best parameters from each run in a list of dicts
    # all_best_params = []

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
    # num_iteration = 1 if idx_waveform == 0 else global_search_space["num_iterations"] 
    num_iteration = global_search_space["num_iterations"] 

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

    min_multiplier = global_search_space["min_multiplier"]
    max_multiplier = global_search_space["max_multiplier"]
    multiplier_STF_array = np.geomspace(min_multiplier,max_multiplier, num_iteration)

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

        multiplier_stf          = multiplier_STF_array[iteration]

        args_list.append((
            spreading_factor_tx,
            spreading_factor_rx,
            position2edge_tx,
            position2edge_rx,
            radius_factor_tx,
            radius_factor_rx,
            steel_velocity2simulate,
            pzt_velocity2simulate,
            multiplier_stf,
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
    multiplier_array= np.array([r[9] for r in results_sorted])

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
    best_multiplier            = results_sorted[0][9]

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
    print(f"    multiplier_STF      = {best_multiplier:.3f}")

    # # SAVE BEST PZT VELOCITY AND STEEL IN THE BLOCK METADATA FILE
    # handler.update_block_params(
    #     "mauro_desolda_side1",
    #     {"z": 2.5, "velocity_s": 0.3333, "some_new_param": 999}
    # )

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
    montecarlo["multiplier_STF"]                = best_multiplier

    # Determine if we save plots and/or movies
    save_plot  = (idx_waveform % params["plot_save_interval"] == 0) 
    save_movie = (idx_waveform % params["movie_save_interval"] == 0) 

    # Construct output paths
    if save_plot:
        plot_output_name = f"{outfile_name}_best_simulation"
        plot_output_path = outdir_path_image / plot_output_name
    else:
        plot_output_path = None

    if save_movie:
        movie_output_name = f"{outfile_name}_movie.mp4"
        movie_output_path = outdir_path_image / movie_output_name
    else:
        movie_output_path = None

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        geometry_type       = "block",
        observed_time       = observed_time,
        observed_waveform   = observed_waveform,
        stf_handler         = stf_handler,
        frequency_cutoff    = params["frequency_cutoff"],
        assembly_dict       = assembly_dict,
        montecarlo          = montecarlo,
        misfit_interval     = misfit_interval,
        minimum_velocity    = params["min_velocity2simulate"],
        maximum_velocity    = params["max_velocity2simulate"],
        normalize_waveform  = False,
        enable_plotting     = save_plot,
        make_movie          = False,
        plot_output_path    = plot_output_path,
        movie_output_path   = movie_output_path
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
        # ("Radius Factor Tx",       rad_tx_array),
        # ("Radius Factor Rx",       rad_rx_array),
        # ("multiplier_STF",           multiplier_array)
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
    dw_threshold = 0.01*dw_max_start

    ds_max_start = 0
    # ds_max_start = np.amax(simulation.source_handler.spatial_function)
    ds_threshold = 0.01*ds_max_start

    simulation.run_local_inversion(
                                   n_iterations=params["n_iterations"],
                                   dc_max_start=dc_max_start,
                                   dc_threshold=dc_threshold,
                                   dw_max_start=dw_max_start,
                                   dw_threshold=dw_threshold,
                                   ds_max_start=ds_max_start,
                                   ds_threshold=ds_threshold,
                                   reduce_factor=params["reduce_factor"],
                                   normalize_waveform = False,
                                   enable_plotting=True,
                                   plot_output_path=plot_output_path
                                   )
    
    stf_handler.waveform_data = np.interp(stf_handler.metadata["time_ax_waveform"], 
                                          simulation.sim_time_handler.simulation_time, 
                                          simulation.source_handler.time_function)

    if params["save_local_inversion_STF"]:
        stf_from_inverison_outfile_name = stf_handler.infile.name + params["saved_STF_file_name"]
        stf_from_inverison_outfile_path = stf_handler.infile.parent / stf_from_inverison_outfile_name


        stf_handler.waveform_data = butter_bandpass_filter(stf_handler.waveform_data, 0.25, 12.4, 25)    

        stf_handler.save_waveform_json(data = stf_handler.waveform_data, 
                                        metadata = stf_handler.metadata, 
                                        outfile_path = stf_from_inverison_outfile_path)

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
        "best_multiplier"      : best_multiplier
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
        multiplier_STF,
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
    montecarlo["multiplier_STF"]                = multiplier_STF

    # Run forward simulation for this draw
    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing           = params["absorbing"],
        geometry_type       = "block",
        observed_time       = observed_time,
        observed_waveform   = observed_waveform,
        stf_handler         = stf_handler,
        frequency_cutoff    = params["frequency_cutoff"],
        assembly_dict       = assembly_dict,
        montecarlo          = montecarlo,
        misfit_interval     = misfit_interval,
        minimum_velocity    = params["min_velocity2simulate"],
        maximum_velocity    = params["max_velocity2simulate"],
        normalize_waveform  = False,
        enable_plotting     = False
    )

    # Calculate misfit
    L2norm_new = simulation.misfit
    
    # print((f"\tL2={L2norm_new:.4e}\tSteel={steel_velocity2simulate:.3f}, PZT={pzt_velocity2simulate:.3f}, txspread={spreading_factor_transmitter:.3f}, rxspread={spreading_factor_receiver:.3f}, tx2edge={position2edge_transmitter:.3f}, rx2edge={position2edge_receiver:.3f}, txrad={radius_factor_transmitter:.4f}, rxrad={radius_factor_receiver:.4f}"))

    return (
        pzt_velocity2simulate,
        steel_velocity2simulate,
        L2norm_new,
        spreading_factor_transmitter,
        spreading_factor_receiver,
        position2edge_transmitter,
        position2edge_receiver,
        radius_factor_transmitter,
        radius_factor_receiver,
        multiplier_STF
    )

################### MAIN ############################
if __name__ == "__main__":

    dir_manager = DirectoryManager()

    # Basic experiment info
    machine_name    = "on_bench"
    experiment_name = "STF_ss10_05"
    wave_type       = "_s"  # e.g., compressional wave
    data_type_uw    = f"uw_data/data_tsv_files{wave_type}"
    outfolder_name  = f"simulation_parameters{wave_type}_2025-04-14_only_STF_30s_stf_bandpass_from_original_multiplier" 

    # Create output directories
    outdir_path_l2norm = dir_manager.make_data_analysis_folders(
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=[outfolder_name]
    )
    outdir_path_image = dir_manager.make_data_analysis_folders(
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=["images_and_movie_" + outfolder_name]
    )

    # Basic simulation parameters
    params = {
        "absorbing"                 : False,
        "save_local_inversion_STF"  : True,
        "saved_STF_file_name"       : "_local_inversion",   # this string will be added to the "stf_chosen" file name, so to not overdrive the original data
        "maxtime2simulate"          : 30,   # mus
        "frequency_cutoff"          : 6,     # MHz
        "minimum_SNR"               : 5,
        "min_velocity2simulate"     : 0.2,  # cm/mus
        "max_velocity2simulate"     : 0.4,  # cm/mus
        "plot_save_interval"        : 1,
        "movie_save_interval"       : 1000,
        "l2norm_plot_interval"      : 1,
        "number_of_waveforms2process": 10,
        "outdir_path_l2norm"        : outdir_path_l2norm[0],
        "outdir_path_image"         : outdir_path_image[0],
        "n_iterations"              : 40,
        "reduce_factor"             : 10/9
    }

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

    #### MONTE CARLO PARAMETERS DEFINED HERE ####
    global_search_space = {
        "num_iterations"       : 1000,  # how many random draws to try
        "steel_velocity_low"   : assembly_dict["velocity" + wave_type] - 0.005,
        "steel_velocity_high"  : assembly_dict["velocity" + wave_type] + 0.005,              
        "pzt_velocity_low"     : params["min_velocity2simulate"], # assembly_dict["pzt_velocity" + wave_type], 
        "pzt_velocity_high"    : params["max_velocity2simulate"], # assembly_dict["pzt_velocity" + wave_type],
        "spreading_factor_low" : 0.1,
        "spreading_factor_high": 1.0,
        # Uniform range for positions relative to edges pzt-steel
        "position2edge_low"    : -0.8,
        "position2edge_high"   : 0.,
        # how many nodes to use to approximate the tx/rx positions in case they do not correspond precisely to one node
        "radius_factor_low"    : 1.0,
        "radius_factor_high"   : 1.0,
        "min_multiplier"       : 1.,
        "max_multiplier"       : 1.
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
            # stf_chosen= "width250_volt70_local_inversion_bigboss",
            stf_chosen= "width250_volt70_multiplier",

            # stf_chosen=stf_chosen,
            frequency_cutoff= 12.5  # LEAVE THE NIQUIST, BUT FIX IT! SOMEHOW THE LOWPASS IS WRONG, IT DISTORTS THE WAVE
        )

        # freq, amplitude, phase = stf_handler.compute_amplitude_phase_spectrum()

        # Run the main simulation routine
        process_uw_file(
            infile_path=infile_path,
            stf_handler=stf_handler,
            params=params,
            assembly_dict=assembly_dict,
            global_search_space=global_search_space
        )
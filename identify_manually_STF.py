import sys
import numpy as np
from pathlib import Path

from lab_uw.directory_manager import DirectoryManager
from lab_uw.data_io import UltrasonicDataHandler, BlockMetadataHandler
from lab_uw.plotting import InteractivePlotter, Plotter

def pick_direct_arrival(
    observed_time: np.ndarray,
    waveform: np.ndarray
) -> tuple[float, float]:
    """
    Lets the user pick two times: start and end of the direct arrival.

    Parameters
    ----------
    observed_time : np.ndarray
        1D array of time samples.
    waveform : np.ndarray
        Single 1D waveform.

    Returns
    -------
    (t_start_direct, t_end_direct) : (float, float)
        The two picked times in ascending order.
    """
    picks = InteractivePlotter().manual_pick_arrival_times(
        observed_time=observed_time,
        observed_waveform=waveform,
        start_time=0.0,  # Just a reference
        outfile_path=None
    )
    if len(picks) < 2:
        sys.exit("Not enough picks made. Please re-run and pick at least 2 times.")
    picks_sorted = sorted(picks)
    return picks_sorted[0], picks_sorted[1]

def compute_reflections_arrival_time(
    t_start_direct: float,
    direct_arrival_span: float,
    observed_time: np.ndarray,
    max_time: float
) -> list[float]:
    """
    Identifies reflection arrivals at times 3x, 5x, 7x, ... of the direct arrival.

    Parameters
    ----------
    t_start_direct : float
        The start time of the direct arrival (or onset).
    direct_arrival_span : float
        The one-way time of the direct wave.
    observed_time : np.ndarray
        The entire time axis; we use observed_time[-1] to avoid going beyond data.
    max_time : float
        Usually observed_time[-1].

    Returns
    -------
    list of float
        Reflection arrival start times: 3x, 5x, etc.
    """
    arrivals = [t_start_direct]
    # second arrival ~ 3 x direct
    tA = 3 * t_start_direct
    while tA + direct_arrival_span <= max_time:
        arrivals.append(tA)
        tA += 2 * t_start_direct  # next reflection is +2 * direct
    return arrivals

import numpy as np

import numpy as np

def compute_reflections_correlation(
    direct_wave_data: np.ndarray,
    idx_Dstart: int,
    idx_Dend: int,
    arrivals: list[float],
    observed_time: np.ndarray,
    waveform: np.ndarray
) -> list[dict]:
    """
    Computes reflection snippets for each reflection arrival, then builds
    a cross-correlation matrix among the direct wave (index 0) and
    all reflection wave snippets (indices 1..N).

    Parameters
    ----------
    direct_wave_data : np.ndarray
        The snippet of the direct wave (1D).
    idx_Dstart : int
        The start index of the direct arrival in 'observed_time'.
    idx_Dend : int
        The end index of the direct arrival in 'observed_time'.
    arrivals : list[float]
        The arrival times (index 0 is direct or reference, so we skip it for reflection).
        Typically, arrivals[0] is the direct arrival time, but we treat that as "reference."
    observed_time : np.ndarray
        The full time axis for the entire waveform.
    waveform : np.ndarray
        The entire 1D waveform data.

    Returns
    -------
    reflection_info_list : list of dict
        Each dict has:
          {
            'arrival_time': float,
            'reflection_time': np.ndarray,  # snippet time array
            'reflection_data': np.ndarray,  # snippet wave array
            'corr_coeff': float             # correlation direct wave-reflection

          }

    """

    reflection_info_list = []
    direct_len = idx_Dend - idx_Dstart

    # List-of-1D wave snippets; first row => direct wave snippet
    wave_snippets = []
    wave_snippets.append(direct_wave_data)  # index 0 => direct wave

    # Build reflection snippets
    for i, arr_time in enumerate(arrivals):
        if i == 0:
            # The arrivals[0] is the direct arrival time -> skip
            continue

        ref_start_idx = np.searchsorted(observed_time, arr_time)
        ref_end_idx   = ref_start_idx + direct_len

        if ref_end_idx > len(observed_time):
            # reflection snippet out of bounds; skip
            break

        reflection_time = observed_time[ref_start_idx:ref_end_idx]
        reflection_data = waveform[ref_start_idx:ref_end_idx]

        # compute correlation
        if len(direct_wave_data) == len(reflection_data):
            cmat = np.corrcoef(direct_wave_data, reflection_data)
            corr_coeff = cmat[0, 1]
        else:
            corr_coeff = float('nan')

        reflection_info_list.append({
            'arrival_time': arr_time,
            'reflection_time': reflection_time,
            'reflection_data': reflection_data,
            'corr_coeff': corr_coeff
        })

    return reflection_info_list


def iterate_direct_arrival_times(
    observed_time: np.ndarray,
    waveform: np.ndarray,
    t_start_nominal: float,
    direct_arrival_span: float,
    search_margin: float,
    step_size: float,
    max_time: float
) -> dict:
    
    """
    Search around a nominal direct-arrival start time to maximize the sum of reflection correlations.

    Parameters
    ----------
    observed_time : np.ndarray
        Full time axis of the waveform.
    waveform : np.ndarray
        1D array of the waveform data.
    t_start_nominal : float
        The user's initially picked direct-arrival start time.
    direct_arrival_span : float
        The length (in time) of the direct arrival snippet, e.g. t_end_direct - t_start_direct.
    search_margin : float
        The +/- margin around t_start_nominal to search, in the same time units as observed_time.
    step_size : float
        The increment for scanning candidate start times. E.g. 0.1 us.
    max_time : float
        Usually observed_time[-1], to ensure we stay in range.

    Returns
    -------
    dict
        {
          "best_t_start"    : float,
          "best_corr_norm"   : float,
          "reflection_info" : list[dict],  # from the compute_reflections_correlation for the best t_start
          "arrivals"        : list[float], # reflection arrival times for the best t_start
        }
    """

    from numpy import inf
    best_t_start  = t_start_nominal
    best_corr_score = -inf
    best_reflection_info_list = []  

    # Prepare candidate times
    t_min = max(0, t_start_nominal - search_margin)
    t_max = min(max_time, t_start_nominal + search_margin)
    candidate_times = np.arange(t_min, t_max, step_size)

    if len(candidate_times) == 0:
        candidate_times = [t_start_nominal]

    for candidate_t_start in candidate_times:
        candidate_t_end = candidate_t_start + direct_arrival_span
        if candidate_t_end > max_time:
            continue

        # direct wave snippet
        idxD_start = np.searchsorted(observed_time, candidate_t_start)
        idxD_end   = np.searchsorted(observed_time, candidate_t_end)
        direct_data = waveform[idxD_start:idxD_end]
        direct_time = observed_time[idxD_start:idxD_end]

        arrivals = compute_reflections_arrival_time(
            t_start_direct=candidate_t_start,
            direct_arrival_span=direct_arrival_span,
            observed_time=observed_time,
            max_time=max_time
        )

        reflection_info_list = compute_reflections_correlation(
            direct_wave_data=direct_data,
            idx_Dstart=idxD_start,
            idx_Dend=idxD_end,
            arrivals=arrivals,
            observed_time=observed_time,
            waveform=waveform
        )
        # Evaluate correlation
        # reflection_info is a list of dict: each has 'corr_coeff'
        corr_score = 0.0
        for ref in reflection_info_list:
            if not np.isnan(ref['corr_coeff']):
                corr_score += ref['corr_coeff']
        corr_score = corr_score/len(reflection_info_list)

        # 5) keep track of best
        if corr_score > best_corr_score:
            best_corr_score  = corr_score
            best_t_start = candidate_t_start
            best_reflection_info_list = reflection_info_list

    return {
        "best_t_start": best_t_start,
        "best_corr_score": best_corr_score,
        "best_reflection_info_list": best_reflection_info_list,
    }


#################################################################################################
def main():
    base_dir = "/home/michele/Desktop/Dottorato/active_source_implementation"
    machine_name = "on_bench"
    experiment_name = "STF"
    data_type = "uw_data/data_tsv_files"
    wave_tipe = "_p"
    block_metadata_filename = "blocks_metadata.json"
    block_id = "on_bench_STF2"
    # Guess or define a margin, step_size
    search_margin = 1    # [mus] +/- interval search around the user pick
    step_size = 0.01     # [mus] increments step. At best should be the sampling frequency

    dir_manager = DirectoryManager(base_dir=base_dir)
    # Load block thickness
    block_params, = BlockMetadataHandler.load_blocks_metadata(
        dir_manager=dir_manager,
        blocks_metadata_name=block_metadata_filename,
        block_keys=(block_id,)
    )
    infile_path_list = sorted(dir_manager.make_infile_path_list(
        machine_name, experiment_name, data_type=data_type + wave_tipe
    ))

    # Use only one file for manual picking, then the same guess will be used for all the others
    file_manual_pick = np.random.choice(len(infile_path_list))

    for infile_path in infile_path_list:
        # Load the single file and get the wave/time
        ultrasonic_handler = UltrasonicDataHandler.make_UW_data(infile_path)
        data, metadata = ultrasonic_handler.waveform_data, ultrasonic_handler.metadata
        observed_time = metadata['time_ax_waveform']
        # Average all waveforms in that file
        waveform = np.mean(data, axis=0)

        # Pick direct arrival times
        if file_manual_pick:
            t_start_direct_picked, t_end_direct_picked = pick_direct_arrival(observed_time, waveform)
            direct_arrival_span = t_end_direct_picked - t_start_direct_picked
            print(f"Picked direct arrival: start={t_start_direct_picked:.2f}, end={t_end_direct_picked:.2f}")
            file_manual_pick = None

        results = iterate_direct_arrival_times(
            observed_time=observed_time,
            waveform=waveform,
            t_start_nominal=t_start_direct_picked,
            direct_arrival_span=direct_arrival_span,
            search_margin=search_margin,
            step_size=step_size,
            max_time=observed_time[-1]
        )

        best_t_start = results["best_t_start"]
        best_reflection_info_list = results["best_reflection_info_list"]
        best_corr_score = results["best_corr_score"]

        print(f"Best t_start = {best_t_start:.3f} %\\mu%s")
        print(f"Best correlation score =\n{best_corr_score}")

        #  best snippet => direct wave snippet
        final_idx_Dstart = np.searchsorted(observed_time, best_t_start)
        final_idx_Dend   = np.searchsorted(observed_time, best_t_start+direct_arrival_span)
        direct_wave_data = waveform[final_idx_Dstart:final_idx_Dend]
        direct_wave_time = observed_time[final_idx_Dstart:final_idx_Dend]

        # SAVE STFs AND THEIR METADATA 
        data_types_output = ["source_time_functions"+wave_tipe,"stf_images_reflection_windows"+wave_tipe,"stf_images_direct-reflections_correlation"+wave_tipe]
        outdir_path_data = dir_manager.make_data_analysis_folders(machine_name = machine_name, experiment_name=experiment_name, data_types=data_types_output)

        infile_path = Path(infile_path)  # Convert to Path object
        while infile_path.suffix:
            infile_path = infile_path.with_suffix('')
        infile_name = infile_path.stem
        outfile_name = infile_name 

        stf_data = direct_wave_data
        # HEREDITATE METADATA AND UPDATE IT
        stf_metadata = metadata.copy()         # copy is needed, otherwise both will point to the same memory!!!
        stf_metadata['number_of_samples'] = len(stf_data)
        stf_metadata['number_of_waveforms'] = 1
        stf_metadata['time_ax_waveform'] = metadata['time_ax_waveform'][final_idx_Dstart:final_idx_Dend]
        ultrasonic_handler.save_waveform_json(data = stf_data, 
                                            metadata = stf_metadata, 
                                            outfile_path = Path(outdir_path_data[0]) / outfile_name)

        # Plot final reflection windows
        plotter = Plotter()
        try: 
            plotter.plot_reflection_windows(
                observed_time=observed_time,
                waveform=waveform,
                t_start_direct=best_t_start,
                t_end_direct=best_t_start + direct_arrival_span,
                reflection_info_list=best_reflection_info_list,
                idx_Dstart=final_idx_Dstart,
                idx_Dend=final_idx_Dend,
                title="Reflection windows for best t_start",
                outfile_path= Path(outdir_path_data[1]) / outfile_name
            )

        except:
            print(observed_time.size)
            print(waveform.size)

        # Also overlay the final correlations:
        plotter.plot_direct_and_reflections(
            direct_wave_time=direct_wave_time,
            direct_wave_data=direct_wave_data,
            reflection_info_list=best_reflection_info_list,
            outfile_path= Path(outdir_path_data[2]) / outfile_name 
        )

        # Compute velocity
        block_thickness_cm = block_params["z"]
        velocity_estimate = block_thickness_cm / best_t_start
        print(f"Estimated velocity = {velocity_estimate:.4f} cm/us (assuming t_start is one-way travel)")

if __name__ == "__main__":
    main()

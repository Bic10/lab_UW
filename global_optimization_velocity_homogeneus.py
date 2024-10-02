import glob
import time as tm
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import pickle

from file_io import *
from signal_processing import *
from synthetic_data import *
from LAB_UW_forward_modeling import *

################################################################################################################################

# Function Definitions
def find_mechanical_data(file_path_list, pattern):
    """
    Find a specific file in a list of file paths using a pattern.
    """
    for file_path in file_path_list:
        if glob.fnmatch.fnmatch(file_path, pattern):
            print("MECHANICAL DATA CHOSEN:", file_path)
            return file_path
    return None  # No file found in the list

def find_sync_values(mech_data_path):
    """
    Find synchronization peak values within a mechanical data file.
    """
    mech_data = pd.read_csv(mech_data_path, sep=',', skiprows=[1])
    sync_data = mech_data.sync

    # Find synchronization peaks in the synchronization data
    sync_peaks, _ = find_peaks(sync_data, prominence=4.2, height=4)
    return mech_data, sync_data, sync_peaks

##########################################################################################################################################

# General Parameters and Constants
frequency_cutoff = 2  # [MHz] maximum frequency of the data we want to reproduce

# Constants throughout the entire experiment
side_block_1 = 2           # [cm] width of first side block
side_block_2 = 2           # [cm] width of second side block
central_block = 4.8        # [cm] width of central block
pzt_layer_width = 0.1      # [cm] piezoelectric transducer width
pmma_layer_width = 0.1      # [cm] pmmate supporting the PZT
steel_velocity = 3374 * (1e2 / 1e6)  # [cm/μs] steel shear wave velocity
pzt_velocity = 2000 * (1e2 / 1e6)    # [cm/μs] PZT shear wave velocity
pmma_velocity = 0.4 * 0.1392          # [cm/μs] pmmate supporting the PZT

# Initial guessed velocity model of the sample
cmin = 600 * (1e2 / 1e6)        
cmax = 1000 * (1e2 / 1e6) 
c_step = 10 * (1e2 / 1e6)
initial_gouge_velocity_list = np.arange(cmin, cmax, c_step)  # Initial velocity range to test

# Fixed travel time through constant sample dimensions
pzt_depth = 1        # [cm] Position of the PZT with respect to the external side of the block
transmitter_position = pzt_depth    # [cm] Position of the transmitter from the beginning of the sample

fixed_travel_time = 2 * (side_block_1 - transmitter_position + central_block) / steel_velocity  # travel time of direct wave into the blocks

## GET OBSERVED DATA

# Load and process the pulse waveform: it is going to be our time source function
machine_name_pulse = "on_bench"
experiment_name_pulse = "glued_pzt"
data_type_pulse = "data_analysis/wavelets_from_glued_pzt"
infile_path_list_pulse = make_infile_path_list(machine_name=machine_name_pulse, experiment_name=experiment_name_pulse, data_type=data_type_pulse)

# Assuming only one pulse is used, we can load it outside the loop
pulse_path = sorted(infile_path_list_pulse)[0]
pulse_waveform, pulse_metadata = load_waveform_json(pulse_path)
pulse_time = pulse_metadata['time_ax_waveform']
pulse_waveform, _ = signal2noise_separation_lowpass(pulse_waveform, pulse_metadata, freq_cut=frequency_cutoff)
pulse_waveform = pulse_waveform - pulse_waveform[0]  # Make the pulse start from zero

dt_pulse = pulse_time[1] - pulse_time[0]
pulse_duration = pulse_time[-1] - pulse_time[0]

#########################################################################################

# DATA FOLDERS
machine_name = "Brava_2"
experiment_name = "s0108"
data_type_uw = 'data_tsv_files'
data_type_mech = 'mechanical_data'
sync_file_pattern = '*s*_data_rp'  # Pattern to find specific experiment in mechanical data

# LOAD MECHANICAL DATA (Only once)
infile_path_list_mech = make_infile_path_list(machine_name, experiment_name, data_type=data_type_mech)
mech_data_path = find_mechanical_data(infile_path_list_mech, sync_file_pattern)
mech_data, sync_data, sync_peaks = find_sync_values(mech_data_path)

# Manually picked sync peaks for plotting purposes (uncomment if needed)
# if experiment_name == "s0108":
#     steps_carrara = [5582, 8698, 15050, 17990, 22000, 23180, 36229, 39391, 87940, 89744,
#                      126306, 128395, 134000, 135574, 169100, 172600, 220980, 223000,
#                      259432, 261425, 266429, 268647, 279733, 282787, 331437, 333778,
#                      369610, 374824]
#     sync_peaks = steps_carrara

# elif experiment_name == "s0103":
#     steps_mont = [4833, 8929, 15166, 18100, 22188, 23495, 36297, 39000, 87352, 89959,
#                   154601, 156625, 162000, 165000, 168705, 170490, 182000, 184900,
#                   233364, 235558, 411811, 462252]
#     sync_peaks = steps_mont

# MAKE UW PATH LIST (Only once)
infile_path_list_uw = sorted(make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))

# Create output directories (Moved Outside the Loop)
outdir_path_l2norm = make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name, data_types=["global_optimization_velocity"])
print(f"The misfits calculated will be saved at path:\n\t {outdir_path_l2norm[0]}")

# Initialize variables to store results
velocity_ranges = []          # To store the velocity range used for each waveform
L2norm_all_waveforms = []     # To store the misfit values for each waveform
estimated_velocities = []     # To store the estimated velocity for each waveform

# Main Loop Over UW Files
for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):

    # CHOOSE OUTFILE_PATH
    outfile_name = os.path.basename(infile_path).split('.')[0]
    outfile_path = os.path.join(outdir_path_l2norm[0], outfile_name)

    print('PROCESSING UW DATA IN %s: ' % infile_path)
    start_time = tm.time()

    # LOAD UW DATA
    observed_waveform_data, metadata = make_UW_data(infile_path)
    observed_time = metadata['time_ax_waveform']

    # REMOVE EVERYTHING BEFORE initial_time_removed
    initial_time_removed = 0  # [μs]
    observed_time = observed_time[observed_time >= initial_time_removed]
    observed_waveform_data = observed_waveform_data[:, observed_time >= initial_time_removed]

    # FREQUENCY LOW PASS (Frequency cutoff is constant)
    observed_waveform_data, _ = signal2noise_separation_lowpass(observed_waveform_data, metadata, freq_cut=frequency_cutoff)

    # DOWNSAMPLING THE WAVEFORMS
    number_of_waveforms_wanted = 100
    # Subsampling around the step
    data_OBS = observed_waveform_data[:sync_peaks[2 * chosen_uw_file + 1] - sync_peaks[2 * chosen_uw_file]]
    metadata['number_of_waveforms'] = len(data_OBS)
    downsampling = max(1, round(metadata['number_of_waveforms'] / number_of_waveforms_wanted))
    print(f"Number of waveforms in the selected subset: {metadata['number_of_waveforms']}")
    print(f"Number of waveforms wanted: {number_of_waveforms_wanted}")
    print(f"Downsampling waveforms by a factor: {downsampling}")

    # EXTRACT LAYER THICKNESS FROM MECHANICAL DATA
    thickness_gouge_1_list = mech_data.rgt_lt_mm[sync_peaks[2 * chosen_uw_file]: sync_peaks[2 * chosen_uw_file + 1]].values / 10  # Convert mm to cm
    thickness_gouge_2_list = thickness_gouge_1_list  # Assuming both layers have the same thickness

    # Initialize previous_min_velocity for the first waveform
    previous_min_velocity = None

    # Process each waveform sequentially
    for idx_waveform, (thickness_gouge_1, thickness_gouge_2) in enumerate(zip(thickness_gouge_1_list[::downsampling], thickness_gouge_2_list[::downsampling])):
        idx = idx_waveform * downsampling
        observed_waveform = data_OBS[idx] - np.mean(data_OBS[idx])

        # Define the velocity range for this waveform
        if previous_min_velocity is None:
            # First waveform, use initial velocity range
            cmin_waveform = cmin
            cmax_waveform = cmax
            c_step_waveform = c_step
        else:
            # For subsequent waveforms, center around previous min velocity
            c_range = 50 * (1e2 / 1e6)  # [cm/μs], adjust as needed
            cmin_waveform = previous_min_velocity - c_range
            cmax_waveform = previous_min_velocity + c_range
            c_step_waveform = c_step / 2  # Use a smaller step size for better resolution

            # Ensure cmin_waveform and cmax_waveform are within physical limits
            cmin_waveform = max(cmin_waveform, cmin)
            cmax_waveform = min(cmax_waveform, cmax)

        # DEFINE EVALUATION INTERVAL FOR L2 NORM OF THE RESIDUALS
        max_travel_time = fixed_travel_time + thickness_gouge_1 / cmin_waveform + thickness_gouge_2 / cmin_waveform
        min_travel_time = fixed_travel_time + thickness_gouge_1 / cmax_waveform + thickness_gouge_2 / cmax_waveform
        misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time + pulse_duration))

        # Create velocity list for this waveform
        gouge_velocity_list_waveform = np.arange(cmin_waveform, cmax_waveform, c_step_waveform)
        velocity_ranges.append(gouge_velocity_list_waveform)

        # Prepare arguments for multiprocessing over velocities
        args_list = []
        for gouge_velocity in gouge_velocity_list_waveform:
            args = (
                gouge_velocity,
                observed_waveform,
                thickness_gouge_1,
                thickness_gouge_2,
                misfit_interval,
                observed_time,
                pulse_time,
                pulse_waveform,
                transmitter_position,
            )
            args_list.append(args)

        # Define multiprocessing function
        def process_velocity(args):
            (
                gouge_velocity,
                observed_waveform,
                thickness_gouge_1,
                thickness_gouge_2,
                misfit_interval,
                observed_time,
                pulse_time,
                pulse_waveform,
                transmitter_position,
            ) = args

            sample_dimensions = [side_block_1, thickness_gouge_1, central_block, thickness_gouge_2, side_block_2]
            receiver_position = sum(sample_dimensions) - pzt_depth  # [cm] Receiver is in the side_block_2

            synthetic_waveform = DDS_UW_simulation(
                observed_time=observed_time,
                observed_waveform=observed_waveform,
                pulse_time=pulse_time,
                pulse_waveform=pulse_waveform,
                misfit_interval=misfit_interval,
                sample_dimensions=sample_dimensions,
                frequency_cutoff=frequency_cutoff,
                transmitter_position=transmitter_position,
                receiver_position=receiver_position,
                pzt_layer_width=pzt_layer_width,
                pmma_layer_width=pmma_layer_width,
                steel_velocity=steel_velocity,
                gouge_velocity=gouge_velocity,
                pzt_velocity=pzt_velocity,
                pmma_velocity=pmma_velocity,
                normalize_waveform=True,
                enable_plotting=False)

            L2norm_new = compute_misfit(
                observed_waveform=observed_waveform,
                synthetic_waveform=synthetic_waveform,
                misfit_interval=misfit_interval)

            return gouge_velocity, L2norm_new

        # Set up multiprocessing over velocities
        num_processes = cpu_count()
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_velocity, args_list)

        # Collect results
        results.sort(key=lambda x: x[0])  # Sort by gouge_velocity
        gouge_velocity_list_waveform = np.array([result[0] for result in results])
        L2norm_waveform = np.array([result[1] for result in results])
        L2norm_all_waveforms.append(L2norm_waveform)

        # Find the gouge_velocity with minimum misfit
        min_idx = np.argmin(L2norm_waveform)
        previous_min_velocity = gouge_velocity_list_waveform[min_idx]
        estimated_velocities.append(previous_min_velocity)
        print(f"Waveform {idx_waveform}: Minimum misfit at gouge_velocity = {previous_min_velocity:.4f} cm/μs")

    # After processing all waveforms in the current UW file, save the results
    # Save the data using pickle
    with open(outfile_path + '_results.pkl', 'wb') as f:
        pickle.dump({
            'L2norm_all_waveforms': L2norm_all_waveforms,
            'velocity_ranges': velocity_ranges,
            'estimated_velocities': estimated_velocities
        }, f)

    print("--- %s seconds for processing %s ---" % (tm.time() - start_time, outfile_name))

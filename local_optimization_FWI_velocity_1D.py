# local_optimization_FWI_velocity_1D.py

import glob
import os
import time as tm
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from file_io import *
from signal_processing import *
from synthetic_data import *
from LAB_UW_forward_modeling import *
from plotting import plot_velocity_and_stresses, plot_l2_norm_vs_velocity

###############################################################################################################
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

def find_sync_values(mech_data):
    """
    Find synchronization peak values within a mechanical data file.
    """
    try:
        sync_data = mech_data.sync
        # Find synchronization peaks in the synchronization data
        sync_peaks, _ = find_peaks(sync_data, prominence=4.2, height=4)
        return sync_data, sync_peaks

    except:
        print("There is no synchronization data in the reduced file. Please insert it manually!")
        return None, None

def load_and_process_pulse_waveform(frequency_cutoff):
    """
    Load and process the pulse waveform to be used as the time source function.
    """
    machine_name_pulse = "on_bench"
    experiment_name_pulse = "glued_pzt"
    data_type_pulse = "data_analysis/wavelets_from_PIS1_PIS2_glued_250ns"
    infile_path_list_pulse = make_infile_path_list(machine_name=machine_name_pulse, experiment_name=experiment_name_pulse, data_type=data_type_pulse)
    
    # Assuming only one pulse is used
    pulse_path = sorted(infile_path_list_pulse)[0]
    pulse_waveform, pulse_metadata = load_waveform_json(pulse_path)
    pulse_time = pulse_metadata['time_ax_waveform']
    pulse_waveform, _ = signal2noise_separation_lowpass(pulse_waveform, pulse_metadata, freq_cut=frequency_cutoff)
    pulse_waveform = (pulse_waveform - pulse_waveform[0])  # Make the pulse start from zero
    
    dt_pulse = pulse_time[1] - pulse_time[0]
    pulse_duration = pulse_time[-1] - pulse_time[0]
    return pulse_waveform, pulse_time, pulse_duration

def process_uw_file(infile_path, chosen_uw_file, sync_peaks, mech_data, pulse_waveform, pulse_time, pulse_duration, frequency_cutoff, outdir_path_l2norm, params):
    """
    Process a single UW data file.
    """
    # Unpack parameters
    minimum_SNR = params['minimum_SNR']
    fixed_travel_time = params['fixed_travel_time']
    c_step = params['c_step']
    c_range = params['c_range']
    h_groove_side = params['h_groove_side']
    h_groove_central = params['h_groove_central']
    steel_velocity = params['steel_velocity']
    side_block_1 = params['side_block_1']
    central_block = params['central_block']
    side_block_2 = params['side_block_2']
    pzt_depth = params['pzt_depth']
    transmitter_position = params['transmitter_position']
    pzt_layer_width = params['pzt_layer_width']
    pmma_layer_width = params['pmma_layer_width']
    pzt_velocity = params['pzt_velocity']
    pmma_velocity = params['pmma_velocity']
    outdir_path_image = params['outdir_path_image']

    # Initialize variables to store results
    estimated_models = []     # To store the estimated velocity models for each waveform
    normal_stress_values = []
    shear_stress_values = []
    ec_disp_mm_values = []
    time_s_values = []

    # CHOOSE OUTFILE_PATH
    outfile_name = os.path.basename(infile_path).split('.')[0]
    outfile_path = os.path.join(outdir_path_l2norm[0], outfile_name)

    print('PROCESSING UW DATA IN %s: ' % infile_path)
    start_time = tm.time()

    # LOAD UW DATA
    observed_waveform_data, metadata = make_UW_data(infile_path)
    observed_time = metadata['time_ax_waveform']

    # Preprocessing steps:
    observed_waveform_data = observed_waveform_data - np.mean(observed_waveform_data)
    initial_time_removed = 300  # number of samples
    observed_waveform_data[:, :initial_time_removed] = 0
    # Frequency low-pass filter
    observed_waveform_data, _ = signal2noise_separation_lowpass(observed_waveform_data, metadata, freq_cut=frequency_cutoff)

    # Simulate a smaller piece of data. Since we are going to evaluate misfit in a smaller interval
    total_time_to_simulate = int(0.66 * metadata['number_of_samples'])  
    observed_waveform_data = observed_waveform_data[:, :total_time_to_simulate]
    observed_time = observed_time[:total_time_to_simulate]
    # Downsampling the waveforms
    number_of_waveforms_wanted = metadata['number_of_waveforms']

    downsampling = max(1, round(metadata['number_of_waveforms'] / number_of_waveforms_wanted))
    print(f"Number of waveforms in the selected subset: {metadata['number_of_waveforms']}")
    print(f"Number of waveforms wanted: {number_of_waveforms_wanted}")
    print(f"Downsampling waveforms by a factor: {downsampling}")

    # Extract layer thickness from mechanical data
    try: 
        thickness_gouge_1_list = mech_data.rgt_lt_mm[sync_peaks[2 * chosen_uw_file]: sync_peaks[2 * chosen_uw_file + 1]].values / 10  # Convert mm to cm
        thickness_gouge_2_list = thickness_gouge_1_list  # Assuming both layers have the same thickness
        normal_stress_list = mech_data.normal_stress_MPa[sync_peaks[2 * chosen_uw_file]: sync_peaks[2 * chosen_uw_file + 1]].values 
        shear_stress_list = mech_data.shear_stress_MPa[sync_peaks[2 * chosen_uw_file]: sync_peaks[2 * chosen_uw_file + 1]].values 

        # Extract ec_disp_mm and time_s
        ec_disp_mm_list = mech_data.ec_disp_mm[sync_peaks[2 * chosen_uw_file]: sync_peaks[2 * chosen_uw_file + 1]].values
        time_s_list = mech_data.time_s[sync_peaks[2 * chosen_uw_file]: sync_peaks[2 * chosen_uw_file + 1]].values

    except (TypeError, IndexError):
        thickness_gouge_1_list = mech_data.rgt_lt_mm[sync_peaks[2 * chosen_uw_file]: metadata['number_of_waveforms']].values / 10  # Convert mm to cm
        thickness_gouge_2_list = thickness_gouge_1_list  # Assuming both layers have the same thickness
        normal_stress_list = mech_data.normal_stress_MPa[sync_peaks[2 * chosen_uw_file]: metadata['number_of_waveforms']].values
        shear_stress_list = mech_data.shear_stress_MPa[sync_peaks[2 * chosen_uw_file]: metadata['number_of_waveforms']].values

        # Extract ec_disp_mm and time_s
        ec_disp_mm_list = mech_data.ec_disp_mm[sync_peaks[2 * chosen_uw_file]: metadata['number_of_waveforms']].values
        time_s_list = mech_data.time_s[sync_peaks[2 * chosen_uw_file]: metadata['number_of_waveforms']].values

    # Initialize previous velocity model
    previous_velocity_model = None

    # Load initial velocity from global optimization results
    initial_velocity_directory = make_data_analysis_folders(machine_name=params['machine_name'], experiment_name=params['experiment_name'], data_types=["global_optimization_velocity"])
    initial_velocity_file_name = os.path.basename(infile_path).split('.')[0]
    initial_velocity_file_path = os.path.join(initial_velocity_directory[0], initial_velocity_file_name)

    with open(initial_velocity_file_path + '_results.pkl', 'rb') as f:
        data = pickle.load(f)

    # Access the data from the loaded dictionary
    estimated_velocities = data['estimated_velocities']

    # Process each waveform sequentially
    for idx_waveform, (thickness_gouge_1, thickness_gouge_2, normal_stress, shear_stress, ec_disp_mm, time_s) in enumerate(zip(thickness_gouge_1_list[::downsampling], thickness_gouge_2_list[::downsampling], normal_stress_list[::downsampling], shear_stress_list[::downsampling], ec_disp_mm_list[::downsampling], time_s_list[::downsampling])):
        initial_velocity = estimated_velocities[idx_waveform]
        idx = idx_waveform * downsampling
        thickness_gouge_1 *= 2                 # IT IS A SILLY PROBLEM FOR THE CURRENT COMPUTATION OF LAYER THICKNESS
        thickness_gouge_2 *= 2
        print(f"Layer thickness: {thickness_gouge_1}\tNormal_stress: {normal_stress}\tShear_stress: {shear_stress}")

        try:
            observed_waveform = observed_waveform_data[idx] 
        except IndexError:
            break      # Exit the loop when there are no more waveforms

        # Compute the overall index for synchronization
        overall_index = sync_peaks[2 * chosen_uw_file] + idx

        # Append stress and displacement values
        normal_stress_values.append(normal_stress)
        shear_stress_values.append(shear_stress)
        ec_disp_mm_values.append(ec_disp_mm)
        time_s_values.append(time_s)

        # Process the waveform
        updated_velocity_model = process_waveform(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            idx_waveform=idx_waveform,
            overall_index=overall_index,
            outfile_name=outfile_name,
            initial_velocity_model=initial_velocity if previous_velocity_model is None else previous_velocity_model,
            thickness_gouge_1=thickness_gouge_1,
            thickness_gouge_2=thickness_gouge_2,
            pulse_waveform=pulse_waveform,
            pulse_time=pulse_time,
            pulse_duration=pulse_duration,
            fixed_travel_time=fixed_travel_time,
            params=params
        )

        # Update previous_velocity_model
        previous_velocity_model = updated_velocity_model

        # Append the updated velocity model to the list
        estimated_models.append(updated_velocity_model)

    # After processing all waveforms, save the estimated models
    with open(outfile_path + '_estimated_models.pkl', 'wb') as f:
        pickle.dump({'estimated_models': estimated_models}, f)

    print("--- %s seconds for processing %s ---" % (tm.time() - start_time, outfile_name))

def process_waveform(
    observed_waveform,
    observed_time,
    idx_waveform,
    overall_index,
    outfile_name,
    initial_velocity_model,
    thickness_gouge_1,
    thickness_gouge_2,
    pulse_waveform,
    pulse_time,
    pulse_duration,
    fixed_travel_time,
    params
):
    """
    Process a single waveform using FWI to update the 1D velocity model.
    """
    # Unpack parameters
    frequency_cutoff = params['frequency_cutoff']
    h_groove_side = params['h_groove_side']
    h_groove_central = params['h_groove_central']
    steel_velocity = params['steel_velocity']
    side_block_1 = params['side_block_1']
    central_block = params['central_block']
    side_block_2 = params['side_block_2']
    pzt_depth = params['pzt_depth']
    transmitter_position = params['transmitter_position']
    pzt_layer_width = params['pzt_layer_width']
    pmma_layer_width = params['pmma_layer_width']
    pzt_velocity = params['pzt_velocity']
    pmma_velocity = params['pmma_velocity']
    outdir_path_image = params['outdir_path_image'][0]
    fixed_minimum_velocity = params['fixed_minimum_velocity']  # Use the fixed value

    # Preprocess observed waveform
    observed_waveform = observed_waveform - np.mean(observed_waveform)

    # Unpack tuple of velocity model
    # Check if initial_velocity_model is iterable (tuple or list)
    if isinstance(initial_velocity_model, (tuple, list)):
        # If it's a tuple or list, unpack normally
        (gouge_1_velocity_initial, gouge_2_velocity_initial) = initial_velocity_model
    else:
        # If it's a scalar, use the same value for both gouge velocities
        gouge_1_velocity_initial = gouge_2_velocity_initial = initial_velocity_model

    # Calculate travel time interval where evaluate the misfit
    gouge_1_velocity_average = np.mean(gouge_1_velocity_initial)  
    gouge_2_velocity_average = np.mean(gouge_2_velocity_initial)  

    min_travel_time = fixed_travel_time  \
    + thickness_gouge_1 / gouge_1_velocity_average \
    + thickness_gouge_2 / gouge_2_velocity_average \
    + 2 * (h_groove_side + h_groove_central) / (steel_velocity + gouge_1_velocity_average) \
    + 2 * (h_groove_side + h_groove_central) / (steel_velocity + gouge_1_velocity_average) 
    
    max_travel_time = min_travel_time + 2*pulse_duration
    misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time))

    # Check Signal-to-Noise Ratio: evaluate only waveform above min_SNR
    sure_noise_interval = np.where(observed_time < min_travel_time)
    good_data_interval = np.where(observed_time > min_travel_time)
    max_signal = np.amax(observed_waveform[good_data_interval])
    max_noise = np.amax(observed_waveform[sure_noise_interval])
    if max_signal / max_noise < params['minimum_SNR']:
        print(f"Signal to Noise ratio for waveform {idx_waveform} = {max_signal / max_noise}. Skipped computation")
        return initial_velocity_model  # Return initial model if SNR is too low

    sample_dimensions = [side_block_1, thickness_gouge_1, central_block, thickness_gouge_2, side_block_2]
    receiver_position = pzt_depth  # [cm] Receiver is in side_block_2

    # Call DDS_UW_simulation with gradient descent
    synthetic_waveform, updated_gouge_velocities = DDS_UW_simulation(
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        pulse_time=pulse_time,
        pulse_waveform=pulse_waveform,
        sample_dimensions=sample_dimensions,
        h_groove_side=h_groove_side,
        h_groove_central=h_groove_central,
        frequency_cutoff=frequency_cutoff,
        transmitter_position=transmitter_position,
        receiver_position=receiver_position,
        pzt_layer_width=pzt_layer_width,
        pmma_layer_width=pmma_layer_width,
        steel_velocity=steel_velocity,
        gouge_velocity=(gouge_1_velocity_initial, gouge_2_velocity_initial),
        pzt_velocity=pzt_velocity,
        pmma_velocity=pmma_velocity,
        misfit_interval=misfit_interval,
        fixed_minimum_velocity=fixed_minimum_velocity,
        iterative_gradient_descent=True,
        normalize_waveform=True,
        enable_plotting=True,
        make_movie=False,
        movie_output_path="simulation.mp4"
    )

    # Combine updated gouge velocities
    updated_velocity_model = updated_gouge_velocities

    # Save the updated velocity model
    model_output_path = os.path.join(outdir_path_image, f'{outfile_name}_waveform_{overall_index}_velocity_model.pkl')
    with open(model_output_path , 'wb') as f:
        pickle.dump({
            '1D_velocity_gouge_tuple': updated_velocity_model
                    }, f)
        
    return updated_velocity_model


###############################################################################################################
# Main Execution

if __name__ == "__main__":
    # General Parameters and Constants
    frequency_cutoff = 2  # [MHz] maximum frequency of the data we want to reproduce
    minimum_SNR = 5       # The minimum signal-to-noise ratio accepted to start computation

    # Constants throughout the entire experiment
    side_block_1 = 2.93                 # [cm] width of first side block, with grooves
    side_block_2 = 2.93                 # [cm] width of second side block with grooves
    h_groove_side = 0.059               # [cm] height of side block grooves
    central_block = 4.88                # [cm] width of central block, with grooves
    h_groove_central = 0.096            # [cm] height of central block grooves
    pzt_layer_width = 0.1               # [cm] piezoelectric transducer width
    pmma_layer_width = 0.0              # [cm] PMMA supporting the PZT (not present in this case)
    steel_velocity = 3374 * (1e2 / 1e6) # [cm/μs] steel shear wave velocity
    pzt_velocity = 2000 * (1e2 / 1e6)   # [cm/μs] PZT shear wave velocity
    pmma_velocity = 1590 * (1e2 / 1e6)  # [cm/μs] PMMA velocity

    # Set fixed_minimum_velocity once for the entire simulation
    fixed_minimum_velocity = 700 * (1e2 / 1e6)  # Example: Set a physically correct small value

    # Initial guessed velocity model of the sample: literature range for gouge at atmospheric pressure. Not used in the current implementation
    c_step = 3 * (1e2 / 1e6)
    c_range = 100 * (1e2 / 1e6)

    # Fixed travel time through constant sample dimensions
    pzt2grove = 1.71        # [cm] distance between the PZT and the top of the grooves
    pzt_depth = side_block_1 - pzt2grove  # [cm] Position of the PZT with respect to the external side of the block
    transmitter_position = pzt_depth      # [cm] Position of the transmitter from the beginning of the sample

    fixed_travel_time = (2 * (side_block_1 - transmitter_position) + central_block - 2 * h_groove_side - 2 * h_groove_central) / steel_velocity  # travel time of direct wave into the blocks

    # GET OBSERVED DATA
    pulse_waveform, pulse_time, pulse_duration = load_and_process_pulse_waveform(frequency_cutoff)

    # DATA FOLDERS
    machine_name = "Brava_2"
    experiment_name = "s0176"
    data_type_uw = 'data_tsv_files_sv1_sv2_only'
    data_type_mech = 'mechanical_data'
    sync_file_pattern = '*s*_data_rp'  # Pattern to find specific experiment in mechanical data

    # LOAD MECHANICAL DATA (Only once)
    infile_path_list_mech = make_infile_path_list(machine_name, experiment_name, data_type=data_type_mech)
    mech_data_path = find_mechanical_data(infile_path_list_mech, sync_file_pattern)
    mech_data = pd.read_csv(mech_data_path, sep='\t', skiprows=[1])
    sync_data, sync_peaks = find_sync_values(mech_data)

    if sync_peaks is None:
        # Add synchronization manually if not found
        sync_peaks = [2389, None, 5273, None, 523455, None, 801825, None, 1056935, None, 127865, None, 1396245]

    # MAKE UW PATH LIST (Only once)
    infile_path_list_uw = sorted(make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))

    # Create output directories
    outdir_path_l2norm = make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name, data_types=["local_optimization_velocity"])
    outdir_path_image = make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name, data_types=["local_optimization_velocity_images_and_movie"])

    print(f"The inversion results will be saved at path:\n\t {outdir_path_l2norm[0]}")

    # Parameters dictionary to pass around
    params = {
        'minimum_SNR': minimum_SNR,
        'fixed_travel_time': fixed_travel_time,
        'c_step': c_step,
        'c_range': c_range,
        'h_groove_side': h_groove_side,
        'h_groove_central': h_groove_central,
        'steel_velocity': steel_velocity,
        'side_block_1': side_block_1,
        'central_block': central_block,
        'side_block_2': side_block_2,
        'pzt_depth': pzt_depth,
        'transmitter_position': transmitter_position,
        'pzt_layer_width': pzt_layer_width,
        'pmma_layer_width': pmma_layer_width,
        'pzt_velocity': pzt_velocity,
        'pmma_velocity': pmma_velocity,
        'plot_save_interval': 5,  # Save plots every 5 waveforms
        'movie_save_interval': 100,
        'outdir_path_image': outdir_path_image,
        'frequency_cutoff': frequency_cutoff,
        'machine_name': machine_name,
        'experiment_name': experiment_name,
        'fixed_minimum_velocity': fixed_minimum_velocity,  # Add fixed_minimum_velocity to the params
        'normal_stress_values': []  # To be populated during processing
    }

    # Main Loop Over UW Files
    for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):
        if chosen_uw_file == 0:
            continue
        process_uw_file(
            infile_path=infile_path,
            chosen_uw_file=chosen_uw_file,
            sync_peaks=sync_peaks,
            mech_data=mech_data,
            pulse_waveform=pulse_waveform,
            pulse_time=pulse_time,
            pulse_duration=pulse_duration,
            frequency_cutoff=frequency_cutoff,
            outdir_path_l2norm=outdir_path_l2norm,
            params=params  # Pass params, which now includes fixed_minimum_velocity
        )

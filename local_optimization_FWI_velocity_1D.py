import glob
import time as tm
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

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
pmma_layer_width = 0.1     # [cm] PMMA layer width
steel_velocity = 3374 * (1e2 / 1e6)  # [cm/μs] steel shear wave velocity
pzt_velocity = 2000 * (1e2 / 1e6)    # [cm/μs] PZT shear wave velocity
pmma_velocity = 0.4 * 0.1392         # [cm/μs] PMMA layer velocity

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

# Manually picked sync peaks for plotting purposes
if experiment_name == "s0108":
    steps_carrara = [5582, 8698, 15050, 17990, 22000, 23180, 36229, 39391, 87940, 89744,
                     126306, 128395, 134000, 135574, 169100, 172600, 220980, 223000,
                     259432, 261425, 266429, 268647, 279733, 282787, 331437, 333778,
                     369610, 374824]
    sync_peaks = steps_carrara

elif experiment_name == "s0103":
    steps_mont = [4833, 8929, 15166, 18100, 22188, 23495, 36297, 39000, 87352, 89959,
                  154601, 156625, 162000, 165000, 168705, 170490, 182000, 184900,
                  233364, 235558, 411811, 462252]
    sync_peaks = steps_mont

# MAKE UW PATH LIST (Only once)
infile_path_list_uw = sorted(make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))

# Create output directories
outdir_path_l2norm = make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name, data_types=["FWI_results"])
print(f"The inversion results will be saved at path:\n\t {outdir_path_l2norm[0]}")

# Initialize variables to store results
estimated_models = []     # To store the estimated velocity models for each waveform

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

    # Homogeneous velocity model stored using global_optimization_velocity_homogeneous.py
    initial_velocity_guess = 800 * (1e2 / 1e6)  # [cm/μs], initial guess for gouge velocity
    c_range = 100 * (1e2 / 1e6)  # [cm/μs], adjust as needed
    cmin_waveform = initial_velocity_guess - c_range
    cmax_waveform = initial_velocity_guess + c_range

    # REMOVE EVERYTHING BEFORE initial_time_removed
    initial_time_removed = 0  # [μs]
    observed_time = observed_time[observed_time >= initial_time_removed]
    observed_waveform_data = observed_waveform_data[:, observed_time >= initial_time_removed]

    # FREQUENCY LOW PASS (Frequency cutoff is constant)
    observed_waveform_data, _ = signal2noise_separation_lowpass(observed_waveform_data, metadata, freq_cut=frequency_cutoff)

    # DOWNSAMPLING THE WAVEFORMS
    number_of_waveforms_wanted = 20
    observed_waveform_data = observed_waveform_data[:sync_peaks[2 * chosen_uw_file + 1] - sync_peaks[2 * chosen_uw_file]]  # Subsampling around the step
    metadata['number_of_waveforms'] = len(observed_waveform_data)
    downsampling = max(1, round(metadata['number_of_waveforms'] / number_of_waveforms_wanted))
    print(f"Number of waveforms in the selected subset: {metadata['number_of_waveforms']}")
    print(f"Number of waveforms wanted: {number_of_waveforms_wanted}")
    print(f"Downsampling waveforms by a factor: {downsampling}")

    # EXTRACT LAYER THICKNESS FROM MECHANICAL DATA
    thickness_gouge_1_list = mech_data.rgt_lt_mm[sync_peaks[2 * chosen_uw_file]: sync_peaks[2 * chosen_uw_file + 1]].values / 10  # Convert mm to cm
    thickness_gouge_2_list = thickness_gouge_1_list  # Assuming both layers have the same thickness

    # Initialize previous velocity model
    velocity_model_previous = None

    # Process each waveform sequentially
    for idx_waveform, (thickness_gouge_1, thickness_gouge_2) in enumerate(zip(thickness_gouge_1_list[::downsampling], thickness_gouge_2_list[::downsampling])):
        idx = idx_waveform * downsampling
        observed_waveform = observed_waveform_data[idx] - np.mean(observed_waveform_data[idx])

        # Adjust receiver position based on current sample dimensions
        sample_dimensions = [side_block_1, thickness_gouge_1, central_block, thickness_gouge_2, side_block_2]
        receiver_position = sum(sample_dimensions) - pzt_depth

        # DEFINE EVALUATION INTERVAL FOR L2 NORM OF THE RESIDUALS
        max_travel_time = fixed_travel_time + thickness_gouge_1 / cmin_waveform + thickness_gouge_2 / cmin_waveform
        min_travel_time = fixed_travel_time + thickness_gouge_1 / cmax_waveform + thickness_gouge_2 / cmax_waveform
        misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time + pulse_duration))

        # Initialize the velocity model
        if idx_waveform == 0:
            # First waveform: use homogeneous velocity model stored using global_optimization_velocity_homogeneous.py
            gouge_velocity_initial = initial_velocity_guess
            # Build initial velocity model
            spatial_axis, delta_x, num_x = compute_grid(sample_dimensions, gouge_velocity_initial, frequency_cutoff)
            velocity_model, idx_gouge_1, idx_gouge_2, idx_pzt_1, idx_pzt_2 = build_velocity_model(
                spatial_axis,
                sample_dimensions,
                transmitter_position,
                receiver_position,
                pzt_layer_width,
                pmma_layer_width,
                steel_velocity,
                gouge_velocity_initial,
                pzt_velocity,
                pmma_velocity,
                plotting=False)
        else:
            # Subsequent waveforms: use updated velocity model from previous waveform
            # Adjust sample_dimensions if thicknesses have changed
            spatial_axis, delta_x, num_x = compute_grid(sample_dimensions, gouge_velocity_initial, frequency_cutoff)
            # Rebuild indices for gouge layers
            velocity_model, idx_gouge_1, idx_gouge_2, idx_pzt_1, idx_pzt_2 = build_velocity_model(
                spatial_axis,
                sample_dimensions,
                transmitter_position,
                receiver_position,
                pzt_layer_width,
                pmma_layer_width,
                steel_velocity,
                gouge_velocity_initial,
                pzt_velocity,
                pmma_velocity,
                plotting=False)
            # Copy over the updated velocities in gouge layers
            velocity_model[idx_gouge_1] = velocity_model_previous[idx_gouge_1]
            velocity_model[idx_gouge_2] = velocity_model_previous[idx_gouge_2]

        # Initialize time variables
        simulation_time, delta_t, num_t = prepare_time_variables(observed_time, delta_x, steel_velocity)
        # Interpolate source time function
        interpolated_pulse = interpolate_source(pulse_time, pulse_waveform, delta_t)
        src_time_function = np.zeros(len(simulation_time))
        src_time_function[:np.size(interpolated_pulse)] = interpolated_pulse

        # Generate source and receiver spatial functions
        isx = np.argmin(np.abs(spatial_axis - transmitter_position))
        irx = np.argmin(np.abs(spatial_axis - receiver_position))
        sigma = pzt_layer_width / 100  # Must study the spatial distribution
        src_spatial_function = synthetic_source_spatial_function(spatial_axis, isx, sigma=sigma, plotting=False)
        rec_spatial_function = synthetic_source_spatial_function(spatial_axis, irx, sigma=sigma, plotting=False)

        # Number of FWI iterations: NEED A BETTER STOPPING CRITERION
        n_iterations = 10

        for iteration in range(n_iterations):
            print(f"Waveform {idx_waveform}, Iteration {iteration+1}/{n_iterations}")

            # Forward modeling
            wavefield_forward = pseudospectral_1D(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial=src_spatial_function,
                source_time=src_time_function,
                velocity_model=velocity_model,
                compute_derivative=False,
                reverse_time=False
            )

            # Synthetic data at receiver
            waveform_recorded = np.sum(wavefield_forward * rec_spatial_function, axis=1)

            # Interpolate synthetic data to observed data time axis
            synthetic_waveform = np.interp(observed_time, simulation_time, waveform_recorded)

            # Compute residuals
            residual = np.zeros(observed_waveform.shape)
            residual[misfit_interval] = observed_waveform[misfit_interval] - synthetic_waveform[misfit_interval]

            # Prepare adjoint source: residuals at receiver location, reverse time
            residual_interp = np.interp(simulation_time, observed_time, residual)
            adj_src_time_function = residual_interp[::-1]  # Reverse time
            adj_src_spatial_function = rec_spatial_function

            # Adjoint modeling
            wavefield_adjoint = pseudospectral_1D(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial=adj_src_spatial_function,
                source_time=adj_src_time_function,
                velocity_model=velocity_model,
                compute_derivative=False,
                reverse_time=True
            )

            # Compute spatial derivatives of forward and adjoint wavefields
            du_dx = np.gradient(wavefield_forward, axis=1) / delta_x
            dlam_dx = np.gradient(wavefield_adjoint, axis=1) / delta_x

            # Compute gradient
            gradient = np.zeros(num_x)
            for ix in range(num_x):
                gradient[ix] = -2 * velocity_model[ix] * np.sum(du_dx[:, ix] * dlam_dx[:, ix]) * delta_t

            # Apply gradient only to gouge layers
            gradient_update = np.zeros(num_x)
            gradient_update[idx_gouge_1] = gradient[idx_gouge_1]
            gradient_update[idx_gouge_2] = gradient[idx_gouge_2]

            # Update velocity model
            step_length = 10 * (1e2 / 1e6)  # Adjust as needed
            velocity_model -= step_length * gradient_update

            # Ensure velocities in steel blocks and PZT layers remain constant
            velocity_model[idx_pzt_1] = pzt_velocity
            velocity_model[idx_pzt_2] = pzt_velocity
            non_gouge_indices = ~np.isin(np.arange(num_x), np.concatenate((idx_gouge_1, idx_gouge_2, idx_pzt_1, idx_pzt_2)))
            velocity_model[non_gouge_indices] = steel_velocity

            # Optionally, plot the velocity model
            plt.figure()
            plt.plot(spatial_axis, velocity_model)
            plt.title(f"Velocity Model at Iteration {iteration+1}")
            plt.xlabel("Position (cm)")
            plt.ylabel("Velocity (cm/μs)")
            plt.show()

            # Optionally, compute and print misfit
            misfit = 0.5 * np.sum(residual ** 2)
            print(f"Misfit: {misfit}")

        # Save updated velocity model for use in next waveform
        velocity_model_previous = velocity_model.copy()

        # Save the estimated velocity model
        estimated_models.append(velocity_model)

        # Optionally, save the velocity model to a file
        model_output_path = outfile_path + f'_waveform_{idx_waveform}_velocity_model.npy'
        np.save(model_output_path, velocity_model)

    print("--- %s seconds for processing %s ---" % (tm.time() - start_time, outfile_name))

# After processing all waveforms, you can save the estimated models
with open(outfile_path + '_estimated_models.pkl', 'wb') as f:
    pickle.dump({'estimated_models': estimated_models}, f)

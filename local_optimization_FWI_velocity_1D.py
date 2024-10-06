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
h_groove = 0.1                      # [cm] height of blocks grooves
side_block_1_total = 2.1            # [cm] width of first side block, including groove
side_block_2_total = 2.1            # [cm] width of second side block, including groove
central_block_total = 5.0           # [cm] width of central block, including grooves on both sides
pzt_layer_width = 0.1               # [cm] piezoelectric transducer width
pmma_layer_width = 0.1              # [cm] PMMA supporting the PZT
steel_velocity = 3374 * (1e2 / 1e6) # [cm/μs] steel shear wave velocity
pzt_velocity = 2000 * (1e2 / 1e6)   # [cm/μs] PZT shear wave velocity
pmma_velocity = 0.4 * 0.1392        # [cm/μs] PMMA supporting the PZT

# Fixed travel time through constant sample dimensions
pzt_depth = 1.0  # [cm] Depth from the external edge of the side block to the PZT layer

# Positions of transmitter and receiver (measured from the external edge of Side Block 1)
x_transmitter = pzt_depth

# Note: The receiver position will be updated in the loop as sample dimensions change
# For the initial fixed travel time calculation, we'll use initial gouge thicknesses
initial_thickness_gouge = 0.5  # [cm], example initial thickness

# Compute initial receiver position (will be updated in the loop)
x_receiver_initial = (
    side_block_1_total
    + initial_thickness_gouge
    + central_block_total
    + initial_thickness_gouge
    + side_block_2_total
    - pzt_depth
)

# Fixed travel time through constant sample dimensions (excluding variable gouge thicknesses)
fixed_travel_time = (
    (2 * (side_block_1_total - h_groove - pzt_depth))
    + (central_block_total - 2 * h_groove)
) / steel_velocity

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

# MAKE UW PATH LIST (Only once)
infile_path_list_uw = sorted(make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))

# Create output directories
outdir_path_l2norm = make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name, data_types=["FWI_results"])
print(f"The inversion results will be saved at path:\n\t {outdir_path_l2norm[0]}")

# Initialize variables to store results
estimated_models = []     # To store the estimated velocity models for each waveform
idx_dict_previous = None  # To store indices from the previous waveform
spatial_axis_previous = None  # To store spatial axis from the previous waveform
velocity_model_previous = None  # To store velocity model from the previous waveform

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
    initial_velocity_guess = 750 * (1e2 / 1e6)  # [cm/μs], initial guess for gouge velocity
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

        # Adjust sample dimensions based on current gouge thicknesses
        sample_dimensions = [side_block_1_total, thickness_gouge_1, central_block_total, thickness_gouge_2, side_block_2_total]

        # Compute receiver position based on current sample dimensions
        x_receiver = (
            side_block_1_total
            + thickness_gouge_1
            + central_block_total
            + thickness_gouge_2
            + side_block_2_total
            - pzt_depth
        )

        # DEFINE EVALUATION INTERVAL FOR L2 NORM OF THE RESIDUALS
        max_travel_time = fixed_travel_time + (thickness_gouge_1 + thickness_gouge_2) / cmin_waveform
        min_travel_time = fixed_travel_time + (thickness_gouge_1 + thickness_gouge_2) / cmax_waveform
        misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time + pulse_duration))

        # Compute the start and end positions for the grid to exclude air regions
        # Start at the beginning of PMMA layer on Side Block 1
        x_start = x_transmitter - pmma_layer_width - pzt_layer_width
        # End at the end of PMMA layer on Side Block 2
        x_end = x_receiver + pzt_layer_width + pmma_layer_width

        # Initialize the velocity model
        if idx_waveform == 0:
            # First waveform: use homogeneous velocity model
            gouge_velocity_initial = initial_velocity_guess
            # Build initial velocity model
            spatial_axis, delta_x, num_x = compute_grid(sample_dimensions, gouge_velocity_initial, frequency_cutoff, x_start, x_end)
            velocity_model, idx_dict = build_velocity_model(
                x=spatial_axis,
                sample_dimensions=sample_dimensions,
                h_groove=h_groove,
                x_start=x_start,
                x_transmitter=x_transmitter,
                x_receiver=x_receiver,
                pzt_layer_width=pzt_layer_width,
                pmma_layer_width=pmma_layer_width,
                steel_velocity=steel_velocity,
                gouge_velocity=gouge_velocity_initial,
                pzt_velocity=pzt_velocity,
                pmma_velocity=pmma_velocity,
                plotting=False)
            # Save indices and spatial axis for future use
            idx_dict_previous = idx_dict.copy()
            spatial_axis_previous = spatial_axis.copy()
        else:
            # Subsequent waveforms: use updated velocity model from previous waveform
            # Adjust sample_dimensions if thicknesses have changed
            spatial_axis, delta_x, num_x = compute_grid(sample_dimensions, gouge_velocity_initial, frequency_cutoff, x_start, x_end)

            # Build initial velocity model with initial guess in gouge regions
            velocity_model, idx_dict = build_velocity_model(
                x=spatial_axis,
                sample_dimensions=sample_dimensions,
                h_groove=h_groove,
                x_start=x_start,
                x_transmitter=x_transmitter,
                x_receiver=x_receiver,
                pzt_layer_width=pzt_layer_width,
                pmma_layer_width=pmma_layer_width,
                steel_velocity=steel_velocity,
                gouge_velocity=gouge_velocity_initial,  # Initial guess
                pzt_velocity=pzt_velocity,
                pmma_velocity=pmma_velocity,
                plotting=False)

            # Interpolate previous velocities onto current gouge regions
            # Gouge Layer 1
            x_prev_gouge_1 = spatial_axis_previous[idx_dict_previous['gouge_1']]
            v_prev_gouge_1 = velocity_model_previous[idx_dict_previous['gouge_1']]
            x_current_gouge_1 = spatial_axis[idx_dict['gouge_1']]
            v_current_gouge_1 = np.interp(x_current_gouge_1, x_prev_gouge_1, v_prev_gouge_1)
            # Gouge Layer 2
            x_prev_gouge_2 = spatial_axis_previous[idx_dict_previous['gouge_2']]
            v_prev_gouge_2 = velocity_model_previous[idx_dict_previous['gouge_2']]
            x_current_gouge_2 = spatial_axis[idx_dict['gouge_2']]
            v_current_gouge_2 = np.interp(x_current_gouge_2, x_prev_gouge_2, v_prev_gouge_2)

            # Update the velocity model with the interpolated velocities
            velocity_model[idx_dict['gouge_1']] = v_current_gouge_1
            velocity_model[idx_dict['gouge_2']] = v_current_gouge_2

            # Save indices and spatial axis for future use
            idx_dict_previous = idx_dict.copy()
            spatial_axis_previous = spatial_axis.copy()

        # Initialize time variables
        simulation_time, delta_t, num_t = prepare_time_variables(observed_time, delta_x, steel_velocity)
        # Interpolate source time function
        interpolated_pulse = interpolate_source(pulse_time, pulse_waveform, delta_t)
        src_time_function = np.zeros(len(simulation_time))
        src_time_function[:np.size(interpolated_pulse)] = interpolated_pulse

        # Create source and receiver spatial functions
        # Adjust positions for the grid starting at x_start
        transmitter_position_in_grid = x_transmitter - x_start
        receiver_position_in_grid = x_receiver - x_start

        src_spatial_function = arbitrary_position_filter(
            spatial_axis=spatial_axis,
            dx=delta_x,
            position=transmitter_position_in_grid,
            radius=10
        )
        rec_spatial_function = arbitrary_position_filter(
            spatial_axis=spatial_axis,
            dx=delta_x,
            position=receiver_position_in_grid,
            radius=10
        )

        # Number of FWI iterations: NEED A BETTER STOPPING CRITERION
        n_iterations = 10

        for iteration in range(n_iterations):
            print(f"Waveform {idx_waveform}, Iteration {iteration+1}/{n_iterations}")

            # Forward modeling with derivative computation
            wavefield_forward, derivative_wavefield_forward = pseudospectral_1D(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial=src_spatial_function,
                source_time=src_time_function,
                velocity_model=velocity_model,
                compute_derivative=True,
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
            adj_src_time_function = 2 * residual_interp[::-1]  # Reverse time
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
            )

            # Compute gradient using vectorized operations
            # Ensure that wavefield arrays are of shape (num_t, num_x)

            # Compute the element-wise product over all time steps and spatial points
            print(f"velocity model: {velocity_model.shape}\nadjoint: {wavefield_adjoint.shape}\nDerivative: {derivative_wavefield_forward.shape}")
            product = (2 / (velocity_model ** 3)) * wavefield_adjoint * derivative_wavefield_forward  # Shape: (num_t, num_x)

            # Integrate over time (sum over the time axis)
            gradient = np.sum(product, axis=0) * delta_t  # Shape: (num_x,)

            # Apply gradient only to gouge layers
            gradient_update = np.zeros(num_x)
            gradient_update[idx_dict['gouge_1']] = gradient[idx_dict['gouge_1']]
            gradient_update[idx_dict['gouge_2']] = gradient[idx_dict['gouge_2']]

            # Update velocity model
            step_length = 10 * (1e2 / 1e6)  # Adjust as needed
            velocity_model -= step_length * gradient_update

            # Reassign velocities in all regions except the gouge layers
            # Set velocities in PZT layers
            velocity_model[idx_dict['pzt_1']] = pzt_velocity
            velocity_model[idx_dict['pzt_2']] = pzt_velocity

            # Set velocities in PMMA layers
            velocity_model[idx_dict['pmma_1']] = pmma_velocity
            velocity_model[idx_dict['pmma_2']] = pmma_velocity

            # Set velocities in steel regions
            steel_indices = np.concatenate((
                idx_dict['side_block_1'],
                idx_dict['central_block'],
                idx_dict['side_block_2']
            ))
            velocity_model[steel_indices] = steel_velocity

            # Set velocities in grooves
            groove_velocity = 0.5 * (gouge_velocity_initial + steel_velocity)
            velocity_model[idx_dict['groove_sb1']] = groove_velocity
            velocity_model[idx_dict['groove_sb2']] = groove_velocity
            velocity_model[idx_dict['groove_cb1']] = groove_velocity
            velocity_model[idx_dict['groove_cb2']] = groove_velocity

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

        # After inversion iterations, before moving to the next waveform
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

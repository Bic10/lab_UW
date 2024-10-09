import glob
import os
import time as tm
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks

from file_io import *
from signal_processing import *
from synthetic_data import *
from LAB_UW_forward_modeling import *

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
    range_scaling_factor = params['range_scaling_factor']
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
    velocity_ranges = []          # To store the velocity range used for each waveform
    L2norm_all_waveforms = []     # To store the misfit values for each waveform
    estimated_velocities = []     # To store the estimated velocity for each waveform

    # Initialize lists to store stress and displacement data
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

    # simulate a smaller piece of data. Since we are going to evaluate misfit in a smaller interval
    total_time_to_simulate = int(0.66*metadata['number_of_samples'])  
    print(f"total time to simulate {total_time_to_simulate}")
    observed_waveform_data = observed_waveform_data[:,:total_time_to_simulate ]
    observed_time = observed_time[:total_time_to_simulate]
    print(observed_waveform_data.shape)
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

    # Initialize previous_min_velocity for the first waveform
    previous_min_velocity = None

    # Process each waveform sequentially
    for idx_waveform, (thickness_gouge_1, thickness_gouge_2, normal_stress, shear_stress, ec_disp_mm, time_s) in enumerate(zip(thickness_gouge_1_list[::downsampling], thickness_gouge_2_list[::downsampling], normal_stress_list[::downsampling], shear_stress_list[::downsampling], ec_disp_mm_list[::downsampling], time_s_list[::downsampling])):
        idx = idx_waveform * downsampling
        thickness_gouge_1 *= 2                 # IT IS A SILLY PROBLEM FOR THE CURRENT COMPUTATION OF LAYER THICKNESS
        thickness_gouge_2 *= 2
        print(f"Layer thickness: {thickness_gouge_1}\tNormal_stress: {normal_stress}\tShear_stress: {shear_stress}")

        try:
            observed_waveform = observed_waveform_data[idx] 
        except IndexError:
            break      # exit the loop when there is not waveforms anymore

        # Compute the overall index for synchronization
        overall_index = sync_peaks[2 * chosen_uw_file] + idx

        # Append stress and displacement values
        normal_stress_values.append(normal_stress)
        shear_stress_values.append(shear_stress)
        ec_disp_mm_values.append(ec_disp_mm)
        time_s_values.append(time_s)

        # Process the waveform
        result = process_waveform(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            idx_waveform=idx_waveform,
            overall_index=overall_index,
            outfile_name=outfile_name,
            previous_min_velocity=previous_min_velocity,
            thickness_gouge_1=thickness_gouge_1,
            thickness_gouge_2=thickness_gouge_2,
            normal_stress=normal_stress,
            shear_stress=shear_stress,
            pulse_waveform=pulse_waveform,
            pulse_time=pulse_time,
            pulse_duration=pulse_duration,
            fixed_travel_time=fixed_travel_time,
            c_step=c_step,
            c_range=c_range,
            frequency_cutoff=frequency_cutoff,
            params=params
        )

        # Update previous_min_velocity based on result
        previous_min_velocity = result['previous_min_velocity']

        # Append results to lists
        velocity_ranges.append(result['gouge_velocity_list_waveform'])
        L2norm_all_waveforms.append(result['L2norm_waveform'])
        estimated_velocities.append(result['best_gouge_velocity'])

    # After processing all waveforms, save the results
    with open(outfile_path + '_results.pkl', 'wb') as f:
        pickle.dump({
            'L2norm_all_waveforms': L2norm_all_waveforms,
            'velocity_ranges': velocity_ranges,
            'estimated_velocities': estimated_velocities
        }, f)

    # Define plot path
    plot_name = f"{outfile_name}_velocity_stress_vs_ec_disp"
    plot_path = os.path.join(outdir_path_image[0], plot_name)

    # Call the plotting function
    plot_velocity_and_stresses(
        x_values=ec_disp_mm_values,
        velocities=estimated_velocities,
        normal_stress=normal_stress_values,
        shear_stress=shear_stress_values,
        x_label='ec_disp_mm',
        velocity_label='Gouge Velocity (cm/μs)',
        stress_labels=('Normal Stress (MPa)', 'Shear Stress (MPa)'),
        title='Gouge Velocity and Stress vs ec_disp_mm',
        outfile_path=plot_path,
        )

    # Define plot path
    plot_name = f"{outfile_name}_velocity_stress_vs_time"
    plot_path = os.path.join(outdir_path_image[0], plot_name)

    # Call the plotting function
    plot_velocity_and_stresses(
        x_values=time_s_values,
        velocities=estimated_velocities,
        normal_stress=normal_stress_values,
        shear_stress=shear_stress_values,
        x_label='time_s',
        velocity_label='Gouge Velocity (cm/μs)',
        stress_labels=('Normal Stress (MPa)', 'Shear Stress (MPa)'),
        title='Gouge Velocity and Stress vs time_s',
        outfile_path=plot_path,
        )


    print("--- %s seconds for processing %s ---" % (tm.time() - start_time, outfile_name))

def process_waveform(observed_waveform, 
                     observed_time, 
                     idx_waveform, 
                     overall_index, 
                     outfile_name, 
                     previous_min_velocity, 
                     thickness_gouge_1, 
                     thickness_gouge_2, 
                     normal_stress, 
                     shear_stress, 
                     pulse_waveform, 
                     pulse_time, 
                     pulse_duration, 
                     fixed_travel_time, 
                     c_step, 
                     c_range, 
                     frequency_cutoff, 
                     params):
    """
    Process a single waveform.
    """
    # Unpack parameters
    minimum_SNR = params['minimum_SNR']
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
    plot_save_interval = params['plot_save_interval']
    movie_save_interval = params['movie_save_interval']
    l2norm_plot_interval = params['l2norm_plot_interval']
    outdir_path_image = params['outdir_path_image'][0]  # Since it's a list
    range_scaling_factor = params['range_scaling_factor']

    if previous_min_velocity is None:
        # # Initialize a list to store picked arrival times
        # picked_times = []

        # # Function to handle mouse clicks
        # def onclick(event):
        #     if event.button == 1:  # Left click to pick a point
        #         picked_time = event.xdata
        #         picked_times.append(picked_time)
        #         print(f"Picked time: {picked_time} seconds")

        #         # Mark the picked time on the plot
        #         plt.axvline(x=picked_time, color='r', linestyle='--')
        #         plt.draw()

        # # Plot the waveform data
        # fig, ax = plt.subplots()
        # ax.plot(observed_time, observed_waveform, label='Waveform')
        # ax.set_xlabel('Time (seconds)')
        # ax.set_ylabel('Amplitude')
        # ax.set_title('Pick Arrival Times by Clicking')
        # ax.legend()

        # # Connect the click event to the onclick function
        # cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # # Show the plot and allow picking
        # plt.show()

        # print(f"Picked arrival time: {picked_times}")

        # estimated_velocities = []

        # for picked_time in picked_times:
        #     # Derived parameters
        #     Delta_t = picked_time - fixed_travel_time
        #     L_g = thickness_gouge_1 + thickness_gouge_2
        #     L_h = 2 * h_groove_side + 2 * h_groove_central

        #     # Coefficients for the quadratic equation
        #     A = Delta_t * 0.5
        #     B = Delta_t * 0.5 * steel_velocity - L_g * 0.5 - L_h
        #     C = -L_g * 0.5 * steel_velocity

        #     # Solve the quadratic equation
        #     discriminant = B**2 - 4 * A * C

        #     if discriminant >= 0:
        #         # Two possible solutions for cmin_waveform
        #         estimated_velocity_1 = (-B + np.sqrt(discriminant)) / (2 * A)
        #         estimated_velocity_2 = (-B - np.sqrt(discriminant)) / (2 * A)
        #         estimated_velocities.append(estimated_velocity_1)
        #     else:
        #         print("No real solution exists for cmin_waveform.")

        # if not estimated_velocities:
        #     print("No valid initial velocity estimates. Skipping waveform.")
        #     return {'previous_min_velocity': previous_min_velocity}

        # cmin_waveform = min(estimated_velocities)
        # cmax_waveform = max(estimated_velocities)

        # Estimate initial velocity: must be moved outside
        cmin_waveform = 0.035 * (normal_stress**0.25)
        cmax_waveform = 0.055 * (normal_stress**0.25)
        c_step_waveform = c_step  # Use the defined step size

        # Define evaluation interval for L2 norm of the residuals
        max_travel_time = fixed_travel_time + thickness_gouge_1 / cmin_waveform + thickness_gouge_2 / cmin_waveform + 2 * (2 * h_groove_side + 2 * h_groove_central) / (steel_velocity + cmin_waveform)
        min_travel_time = fixed_travel_time + thickness_gouge_1 / cmax_waveform + thickness_gouge_2 / cmax_waveform + 2 * (2 * h_groove_side + 2 * h_groove_central) / (steel_velocity + cmax_waveform)
        misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time + pulse_duration))

        sure_noise_interval = np.where(observed_time < min_travel_time)
        good_data_interval = np.where(observed_time > min_travel_time)
        max_signal = np.amax(observed_waveform[good_data_interval])
        max_noise = np.amax(observed_waveform[sure_noise_interval])
        if max_signal / max_noise < minimum_SNR:
            print(f"Signal to Noise ratio for waveform {idx_waveform} = {max_signal / max_noise}. Skipped computation")
            return {'previous_min_velocity': None,
                    'gouge_velocity_list_waveform': None,
                    'L2norm_waveform': None,
                    'best_gouge_velocity': None,
                    'range_scaling_factor': None
                    }
        
        else: 
            is_the_first_evaluated_waveform = True

        print(f"Evaluating shear wave velocity in the interval: {cmin_waveform:.4f}-{cmax_waveform:.4f}")

    else:
        is_the_first_evaluated_waveform = False
        # For subsequent waveforms, center around previous min velocity
        c_range_waveform = range_scaling_factor * c_range # [cm/μs], adjust as needed
        cmin_waveform = previous_min_velocity - c_range_waveform
        cmax_waveform = previous_min_velocity + c_range_waveform
        c_step_waveform = c_step

        # Define evaluation interval for L2 norm of the residuals
        max_travel_time = fixed_travel_time + thickness_gouge_1 / cmin_waveform + thickness_gouge_2 / cmin_waveform + 2 * (2 * h_groove_side + 2 * h_groove_central) / (steel_velocity + cmin_waveform)
        min_travel_time = fixed_travel_time + thickness_gouge_1 / cmax_waveform + thickness_gouge_2 / cmax_waveform + 2 * (2 * h_groove_side + 2 * h_groove_central) / (steel_velocity + cmax_waveform)
        misfit_interval = np.where((observed_time > min_travel_time) & (observed_time < max_travel_time + pulse_duration))

    # Create velocity list for this waveform
    gouge_velocity_list_waveform = np.arange(cmin_waveform, cmax_waveform, c_step_waveform)

    # Prepare arguments for multiprocessing over velocities
    args_list = []
    for gouge_velocity in gouge_velocity_list_waveform:
        gouge_velocity_tuple = (gouge_velocity, gouge_velocity)  # Use the same velocity for both layers
        args = (
            gouge_velocity_tuple,
            observed_waveform,
            thickness_gouge_1,
            thickness_gouge_2,
            misfit_interval,
            observed_time,
            pulse_time,
            pulse_waveform,
            transmitter_position,
            params
        )
        args_list.append(args)

    # Set up multiprocessing over velocities
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_velocity, args_list)

    # Collect results
    results.sort(key=lambda x: x[0])  # Sort by gouge_velocity_scalar
    gouge_velocity_list_waveform = np.array([result[0] for result in results])
    L2norm_waveform = np.array([result[1] for result in results])

    # Find the gouge_velocity with minimum misfit
    min_idx = np.argmin(L2norm_waveform)
    if min_idx in (0,-1):
        print(f"Mimimum of the misfit at the border of the velocity evaluaated: doubling the next interval")
        range_scaling_factor = 2

    previous_min_velocity = gouge_velocity_list_waveform[min_idx]
    print(f"Waveform {idx_waveform}: Minimum misfit at gouge_velocity = {previous_min_velocity:.4f} cm/μs")

    # After finding the best-fit velocity, generate and plot the synthetic waveform
    best_gouge_velocity = previous_min_velocity

    # Use the same sample_dimensions and positions as before
    sample_dimensions = [side_block_1, thickness_gouge_1, central_block, thickness_gouge_2, side_block_2]
    receiver_position = pzt_depth  # [cm] Receiver is in the side_block_2

    # Decide whether to save plots and movies
    save_plot = idx_waveform % plot_save_interval == 0
    save_movie = idx_waveform % movie_save_interval == 0

    if save_plot or is_the_first_evaluated_waveform:
        plot_output_name = f"{outfile_name}_waveform_{overall_index}_velocity_{best_gouge_velocity:.4f}"
        plot_output_path = os.path.join(outdir_path_image, plot_output_name)
    else:
        plot_output_path = None

    if save_movie:
        movie_output_name = f"{outfile_name}_waveform_{overall_index}_velocity_{best_gouge_velocity:.4f}.mp4"
        movie_output_path = os.path.join(outdir_path_image, movie_output_name)
    else:
        movie_output_path = None

    # Call DDS_UW_simulation with gouge_velocity as a tuple
    synthetic_waveform = DDS_UW_simulation(
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
        gouge_velocity=(best_gouge_velocity, best_gouge_velocity),  # Pass as tuple
        pzt_velocity=pzt_velocity,
        pmma_velocity=pmma_velocity,
        misfit_interval=misfit_interval,
        normalize_waveform=True,
        enable_plotting=save_plot,
        make_movie=save_movie,
        plot_output_path=plot_output_path,
        movie_output_path=movie_output_path
    )

    save_l2norm_plot =idx_waveform % l2norm_plot_interval == 0
    if save_l2norm_plot or is_the_first_evaluated_waveform:               # I want to save the first one alway, cause is the one with the bigger range
        l2norm_plot_name = f"{outfile_name}_L2norm_waveform_{overall_index}"
        l2norm_plot_path = os.path.join(outdir_path_image, l2norm_plot_name)
        plot_l2_norm_vs_velocity(
            gouge_velocity = gouge_velocity_list_waveform,
            L2norm = L2norm_waveform,
            overall_index= overall_index,
            outfile_path =l2norm_plot_path
        ) 
    
    # Return results
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
        gouge_velocity_tuple,  # Now a tuple
        observed_waveform,
        thickness_gouge_1,
        thickness_gouge_2,
        misfit_interval,
        observed_time,
        pulse_time,
        pulse_waveform,
        transmitter_position,
        params
    ) = args

    # Unpack parameters
    h_groove_side = params['h_groove_side']
    h_groove_central = params['h_groove_central']
    steel_velocity = params['steel_velocity']
    side_block_1 = params['side_block_1']
    central_block = params['central_block']
    side_block_2 = params['side_block_2']
    pzt_depth = params['pzt_depth']
    pzt_layer_width = params['pzt_layer_width']
    pmma_layer_width = params['pmma_layer_width']
    pzt_velocity = params['pzt_velocity']
    pmma_velocity = params['pmma_velocity']
    frequency_cutoff = params['frequency_cutoff']

    sample_dimensions = [side_block_1, thickness_gouge_1, central_block, thickness_gouge_2, side_block_2]
    receiver_position = pzt_depth  # [cm] Receiver is in the side_block_2

    # Call DDS_UW_simulation with gouge_velocity_tuple
    synthetic_waveform = DDS_UW_simulation(
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
        gouge_velocity=gouge_velocity_tuple,  # Pass the tuple
        pzt_velocity=pzt_velocity,
        pmma_velocity=pmma_velocity,
        misfit_interval=misfit_interval,
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

    return gouge_velocity_scalar, L2norm_new

###############################################################################################################
# Main Execution

if __name__ == "__main__":
    # General Parameters and Constants
    frequency_cutoff = 2  # [MHz] maximum frequency of the data we want to reproduce
    minimum_SNR = 5       # the minimum signal to noise ratio accepted to start computation

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

    # Initial guessed velocity model of the sample: literature range for gouge at atmospheric pressure
    c_step = 3 * (1e2 / 1e6)
    c_range = 100 * (1e2 / 1e6)
    range_scaling_factor = 1         # initial c_range is c_range*range_scaling_factor

    # Fixed travel time through constant sample dimensions
    pzt2grove = 1.71        # [cm] distance between the PZT and the top of the grooves
    pzt_depth = side_block_1 - pzt2grove  # [cm] Position of the PZT with respect to the external side of the block
    transmitter_position = pzt_depth      # [cm] Position of the transmitter from the beginning of the sample

    fixed_travel_time = (2 * (side_block_1 - transmitter_position) + central_block - 2 * h_groove_side - 2 * h_groove_central) / steel_velocity  # travel time of direct wave into the blocks
    print(f"Fixed travel time: {fixed_travel_time}")

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
    outdir_path_l2norm = make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name, data_types=["global_optimization_velocity"])
    outdir_path_image = make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name, data_types=["global_optimization_velocity_images_and_movie"])

    print(f"The misfits calculated will be saved at path:\n\t {outdir_path_l2norm[0]}")

    # Parameters dictionary to pass around
    params = {
        'minimum_SNR': minimum_SNR,
        'fixed_travel_time': fixed_travel_time,
        'c_step': c_step,
        'c_range': c_range,
        'range_scaling_factor': range_scaling_factor,
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
        'plot_save_interval': 5,  # Save plots every 50 waveforms
        'movie_save_interval': 100,
        'l2norm_plot_interval': 5,  # Save L2 norm plots every 50 waveforms
        'outdir_path_image': outdir_path_image,
        'frequency_cutoff': frequency_cutoff
    }

    # Main Loop Over UW Files
    for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):
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
            params=params
        )

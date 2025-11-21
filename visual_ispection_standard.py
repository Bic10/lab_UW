# lab_uw/visual_ispection_standard.py

import sys
import time as tm
import scipy.signal
from lab_uw.data_io.data_io import UltrasonicDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.signal_processing import SignalProcessor
from lab_uw.plotting import Plotter
import scipy

###### INPUT #######
machine_name = "Brava_2"
experiment_name = "s0244suwanh3_30"
data_type = "uw_data/data_tsv_files"
base_dir = "/home/michele/Desktop/Dottorato/active_source_implementation"

### The plots we want to see for a quick inspection of the experiment ###
image_types = (
    "stacked_waveforms",
    "amplitude_maps",
    "amplitude_spectrum_maps",
    "example_of_filtered_spectrum",
    "example_of_filtered_waveform",
)
remove_initial_samples = 0  # number of samples to be removed at the beginning, to get rid of the noise burst.
highlight_start = 0
highlight_end = 0
xlim_plot = 40
ticks_steps_waveforms = 5  # [microseconds] plot ticks
step_wf_to_plot = 500  # get one waveform each step_wf_to_plot
freq_cut = 6  # [Hz] lowpass frequency threshold

# Create instances of Plotter, DataHandler, and DirectoryManager
plotter = Plotter()
directory_manager = DirectoryManager(base_dir=base_dir)
signal_processor = SignalProcessor()

# Generate file paths and output directories
infile_path_list = directory_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type)
outdir_path_images = directory_manager.make_images_folders(machine_name, experiment_name, image_types)

for infile_path in sorted(infile_path_list):
    try:
        print(f'PROCESSING UW DATA IN {infile_path}: ')

        # Load data and metadata using DataHandler
        data_handler = UltrasonicDataHandler.load_UW_data(infile_path)
        # channels_dict = UltrasonicDataHandler.load_multi_channel_UW_data(infile_path)

        # for channel_name, data_handler in channels_dict.items():
        data, metadata = data_handler.waveform_data, data_handler.metadata
        data, metadata = signal_processor.remove_starting_noise(data, metadata, remove_initial_samples)

        ## Detrend data. Detrending along acquisition time axis (axis=0) highligh changing in the shape of the waveform during the experiment
        # data= scipy.signal.detrend(data=data, axis=-1)
        # data = scipy.signal.detrend(data=data, axis=0, type='constant')
        # data = scipy.signal.detrend(data=data, axis=0, type='linear')
        
        # Detrend and taper (following McNamara's approach)
        taper_percent = 0.1  # 10% taper
        window_scipy = scipy.signal.windows.tukey(metadata['number_of_samples'], alpha=2*taper_percent)
        data = data * window_scipy

        # Prepare output file paths
        outfile_name = infile_path
        while outfile_name.suffix:
            outfile_name = outfile_name.with_suffix("")
        outfile_name = outfile_name.name # + channel_name

        outfile_path_stacked_waveforms = outdir_path_images[0] / outfile_name
        outfile_path_amp_map = outdir_path_images[1] / outfile_name
        outfile_path_spectrum_map = outdir_path_images[2] / outfile_name

        ## PLOT DATA ###
        start_time = tm.time()

        # Plot stacked waveforms
        plotter.uw_all_plot(
            data=data,
            metadata=metadata,
            step_wf_to_plot=step_wf_to_plot,
            highlight_start=highlight_start,
            highlight_end=highlight_end,
            xlim_plot=xlim_plot,
            ticks_steps_waveforms=ticks_steps_waveforms,
            outfile_path=str(outfile_path_stacked_waveforms),
        )

        # Plot amplitude map
        plotter.amplitude_map(
            data=data,
            metadata=metadata,
            outfile_path=str(outfile_path_amp_map),
        )

        # Filter data with lowpass filter and plot amplitude spectrum map
        filtered_data, noise_reconstructed = signal_processor.signal2noise_separation_lowpass(
            waveform_data=data,
            metadata=metadata,
            freq_cut=freq_cut,
            plotting=True,
            outfile_path=str(outfile_path_spectrum_map)
        )


        # Plot examples of filtered spectrum and original vs filtered waveforms
        wave_chosed_list = range(
            0, metadata['number_of_waveforms'], step_wf_to_plot)
        
        for wave_chosed in wave_chosed_list:
            single_waveform = data[wave_chosed, :]
            outfile_path_example_filtered_spectrum = outdir_path_images[3] / f"{outfile_name}_{wave_chosed}"
            outfile_path_example_original_vs_filtered = outdir_path_images[4] / f"{outfile_name}_{wave_chosed}"

            print(f"wave_chosed: {wave_chosed}")
            print(f"Filtered spectrum path: {outfile_path_example_filtered_spectrum}")
            print(f"Original vs filtered path: {outfile_path_example_original_vs_filtered}")


            filtered_single_waveform, filtered_noise_reconstructed = signal_processor.signal2noise_separation_lowpass(
                waveform_data=single_waveform,
                metadata=metadata,
                freq_cut=freq_cut,
                plotting=True,
                outfile_path= outfile_path_example_filtered_spectrum
            )

            plotter.signal_vs_filtered_signal_plot(
                single_waveform=single_waveform,
                single_waveform_filtered=filtered_single_waveform,
                metadata=metadata,
                freq_cut=freq_cut,
                outfile_path= outfile_path_example_original_vs_filtered,
            )

        print(f"--- {tm.time() - start_time} seconds for processing {outfile_name} ---")

    except Exception as e:
        print(f"An error occurred while processing {infile_path}: {e}")
        pass

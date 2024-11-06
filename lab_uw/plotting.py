# src/plotting.py

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# Define global variables for font type, fontsize, figure size, and color palette
FONT_TYPE = "Ubuntu"
FONT_SIZE = 30
FIGURE_SIZE = (16, 8)
FORMAT = ".png"

# Define the color palette for the plots
colors = {
    'reseda_green': '#788054',
    'dutch_white': '#E0D6B4',
    'khaki': '#CABB9E',
    'platinum': '#E7E5E2',
    'black_olive': '#322D1E',
    'sandybrown': 'sandybrown',
    'lightgrey': 'lightgrey',
    'lightsteelblue': 'lightsteelblue',
    'indianred': 'indianred',
    'teal': 'teal',
    'darkslategray': 'darkslategray'
}

# Define default settings for plots
default_settings = {
    'colors': colors,
    'fontsize_title': FONT_SIZE,
    'fontsize_labels': 0.7 * FONT_SIZE,
    'fontsize_ticks': 0.7 * FONT_SIZE,
    'line_width': 4.0,
    'figure_size': FIGURE_SIZE,
    'format': FORMAT,
}

def output_path_choice(fig: plt.Figure, outfile_path: str = None, format: str = FORMAT) -> None:
    '''
    Saving or showing option for plot functions
    - fig: The figure object to be saved or shown.
    - outfile_path: Path to save the plot.
    - format: File format for saving the plot.
    '''
    if outfile_path:
        outfile_name = os.path.basename(outfile_path)
        current_title = fig.axes[0].get_title()
        fig.axes[0].set_title(current_title + " " + outfile_name)
        fig.savefig(outfile_path + format, dpi=300)
        plt.close(fig)  # Close the figure to prevent it from being displayed
 #       print(f"Plot saved to {outfile_path}{format}")
    else:
        plt.show()

def uw_all_plot(data: np.ndarray, 
                metadata: dict, 
                step_wf_to_plot: int, 
                highlight_start: int, 
                highlight_end: int, 
                xlim_plot: float, 
                ticks_steps_waveforms: float, 
                outfile_path: str = None, 
                settings=default_settings) -> None:
    '''
    Plots stacked waveforms with optional highlighting.

    Args:
    - data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - step_wf_to_plot: Step size for plotting waveforms.
    - highlight_start: Index of the start point for highlighting.
    - highlight_end: Index of the end point for highlighting.
    - xlim_plot: Limit for the x-axis.
    - ticks_steps_waveforms: Step size for ticks on the waveform axis.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    '''

    time_ax_waveform = metadata["time_ax_waveform"]
    time_ticks_waveforms = np.arange(time_ax_waveform[0], time_ax_waveform[-1], ticks_steps_waveforms)
    
    data_to_plot = data[::step_wf_to_plot]
    ymax = 1.3 * np.amax(data_to_plot)
    ymin = 1.3 * np.amin(data_to_plot)

    fig, ax = plt.subplots(figsize=settings['figure_size'])

    ax.plot(time_ax_waveform, data_to_plot.T, color='black', linewidth=0.8, alpha=0.5)
    ax.plot(time_ax_waveform[highlight_start:highlight_end], data[highlight_start:highlight_end], color='red')
    ax.set_xlabel('Time [$\mu s$]', fontsize=settings['fontsize_labels'])
    ax.set_ylabel('Amplitude [.]', fontsize=settings['fontsize_labels'])
    ax.set_xticks(time_ticks_waveforms)
    ax.set_ylim([ymin, ymax])
    ax.set_xlim(time_ax_waveform[0], xlim_plot)
    ax.grid(alpha=0.1)
    ax.set_title("Stacked Waveforms", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def amplitude_map(data: np.ndarray, 
                  metadata: dict, 
                  amp_scale: float, 
                  outfile_path: str = None, 
                  settings=default_settings) -> None:
    '''
    Plots an amplitude map of waveform data.

    Args:
    - data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - amp_scale: Scale for amplitude.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    '''

    time_ax_waveform = metadata['time_ax_waveform']
    sampling_rate = metadata['sampling_rate']
    number_of_samples = metadata['number_of_samples']
    acquisition_frequency = metadata['acquisition_frequency']

    fig, ax = plt.subplots(figsize=settings['figure_size'])
    ax.set_title("Amplitude Map", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    cmap = plt.get_cmap('seismic')
    
    im = ax.imshow(data.T, aspect='auto', origin='lower', interpolation='none',
                   cmap=cmap, vmin=-amp_scale, vmax=amp_scale,
                   extent=[0, data.shape[0] * acquisition_frequency, 0, number_of_samples * sampling_rate])
    
    cbar = fig.colorbar(im, pad=0.04)
    cbar.set_label("Relative Amplitude", fontsize=settings['fontsize_labels'])

    ax.set_xlabel('Time [s]', fontsize=settings['fontsize_labels'])
    ax.set_ylabel('Waveform Time [$\mu s$]', fontsize=settings['fontsize_labels'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def amplitude_spectrum_map(signal_freqs: np.ndarray,
                           amp_spectrum: np.ndarray, 
                           metadata: dict, 
                           outfile_path: str = None, 
                           settings=default_settings) -> None:
    '''
    Plots an amplitude spectrum map of waveform data.

    Args:
    - signal_freqs: Frequencies of the signal.
    - amp_spectrum: Amplitude spectrum.
    - metadata: Metadata dictionary containing information about the data.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    '''
    
    wave_num, wave_len = amp_spectrum.shape
    spectrum_length = round(wave_len / 2)
    signal_freqs = signal_freqs[:spectrum_length]
    amp_spectrum = amp_spectrum[:, :spectrum_length]

    time_ax_acquisition = metadata['time_ax_acquisition']

    fig, ax = plt.subplots(figsize=settings['figure_size'])
    pcm = ax.pcolormesh(time_ax_acquisition, signal_freqs, amp_spectrum.T, cmap="gist_gray", norm=mcolors.LogNorm())
    ax.set_title('Amplitude Spectrum Map', fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    ax.set_xlabel("Time [s]", fontsize=settings['fontsize_labels'])
    ax.set_ylabel("Frequency [$MHz$]", fontsize=settings['fontsize_labels'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])
    cbar = fig.colorbar(pcm, pad=0.04)
    cbar.set_label("Spectral Amplitude", fontsize=settings['fontsize_labels'])

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def filtered_amp_and_phase_spectrum_plot(signal_freqs: np.ndarray, amp_spectrum: np.ndarray, phase_spectrum: np.ndarray, filtered_amp_spectrum: np.ndarray, lowpass_filter: np.ndarray, sampling_rate: float, outfile_path: str = None, settings=default_settings) -> None:
    '''
    Plots the filtered amplitude and phase spectrum.

    Args:
    - signal_freqs: Frequencies of the signal.
    - amp_spectrum: Amplitude spectrum.
    - phase_spectrum: Phase spectrum.
    - filtered_amp_spectrum: Filtered amplitude spectrum.
    - lowpass_filter: Low-pass filter.
    - sampling_rate: Sampling rate of the signal.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    '''

    spectrum_length = round(len(amp_spectrum) / 2)
    amp_spectrum = amp_spectrum[:spectrum_length]
    phase_spectrum = phase_spectrum[:spectrum_length]
    filtered_amp_spectrum = filtered_amp_spectrum[:spectrum_length]
    signal_freqs = signal_freqs[:spectrum_length]
    lowpass_filter = lowpass_filter[:spectrum_length]

    max_freq = signal_freqs[np.argmax(amp_spectrum)]

    fig, ax = plt.subplots(2, 1, figsize=settings['figure_size'])

    ax[0].semilogy(signal_freqs, amp_spectrum, label="Amplitude Spectrum")
    ax[0].semilogy(signal_freqs, lowpass_filter * np.amax(amp_spectrum), label="Filter Shape")
    ax[0].semilogy(signal_freqs, filtered_amp_spectrum, label="Filtered Amplitude Spectrum")
    ax[0].vlines(max_freq, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r", "--", label=f"Max Spectrum = {max_freq:.2f} MHz")
    ax[0].vlines(2, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r", "-", label="Sensor Resonance Frequency = 2 MHz")
    ax[0].legend(fontsize=settings['fontsize_ticks'])
    ax[0].set_xlim([0, np.amax(signal_freqs)])
    ax[0].set_ylabel("Amplitude [a.u.]", fontsize=settings['fontsize_labels'])
    ax[0].set_xlabel("Frequency [MHz]", fontsize=settings['fontsize_labels'])
    ax[0].set_title("Amplitude Spectrum of a Waveform", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    ax[0].tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])

    ax[1].plot(signal_freqs, phase_spectrum)
    ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
    ax[1].set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax[1].set_ylabel("Phase [rad]", fontsize=settings['fontsize_labels'])
    ax[1].set_xlabel("Frequency [MHz]", fontsize=settings['fontsize_labels'])
    ax[1].set_title("Phase Spectrum of a Waveform", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    ax[1].tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def signal_vs_filtered_signal_plot(single_waveform: np.ndarray, 
                                   single_waveform_filtered: np.ndarray, 
                                   metadata: dict, freq_cut: float, 
                                   outfile_path: str = None, 
                                   settings=default_settings) -> None:
    """
    Plot a comparison between a waveform and its filtered version.

    Args:
    - single_waveform: The original waveform data.
    - single_waveform_filtered: The filtered waveform data.
    - metadata: Metadata containing information about the waveform.
    - freq_cut: Cut-off frequency for filtering.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    """
    time_ax = metadata['time_ax_waveform']
    fig, ax = plt.subplots(figsize=settings['figure_size'])
    ax.plot(time_ax, single_waveform, color="lightgray", label="Original Waveform", linewidth=settings['line_width'])
    ax.plot(time_ax, single_waveform_filtered, color="black", label="Filtered Waveform", linewidth=settings['line_width'])
    ax.set_xlabel('Time [$\mu s$]', fontsize=settings['fontsize_labels'])
    ax.set_ylabel('Amplitude [a.u.]', fontsize=settings['fontsize_labels'])
    ax.set_title(f"Effect of Lowpass Filtering at {freq_cut:.2f} MHz", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    ax.legend(fontsize=settings['fontsize_ticks'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def wavelet_selection_plot(time: np.ndarray, 
                           waveform: np.ndarray, 
                           ratio: np.ndarray, 
                           index_max_list: list, 
                           index_min_before_list: list, 
                           index_min_after_list: list, 
                           outfile_path: str = None, 
                           settings=default_settings) -> None:
    """
    Plot waveform data along with STA/LTA analysis results.

    Args:
    - time: Time array.
    - waveform: Waveform data.
    - ratio: STA/LTA ratio.
    - index_max_list: List of indices for maximum values.
    - index_min_before_list: List of indices for minimum values before maximum.
    - index_min_after_list: List of indices for minimum values after maximum.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=settings['figure_size'])
    ax.plot(time, waveform, label='Recorded Waveform', alpha=0.5, linewidth=settings['line_width'])
    norm = (np.amax(waveform) / np.amax(ratio))
    ax.plot(time, norm * ratio, label='STA/LTA on Waveform')
    ax.plot(time[index_max_list], norm * ratio[index_max_list], "r.", label='Maxima')
    ax.plot(time[index_min_before_list], norm * ratio[index_min_before_list], "g.", label='Minima Before')
    ax.plot(time[index_min_after_list], norm * ratio[index_min_after_list], "k.", label='Minima After')
    ax.set_xlabel('Time [$\mu$s]', fontsize=settings['fontsize_labels'])
    ax.set_ylabel('Amplitude [a.u.]', fontsize=settings['fontsize_labels'])
    ax.legend(loc="lower left", fontsize=settings['fontsize_ticks'])
    ax.set_title("Wavelet Selection Using STA/LTA", fontsize=settings['fontsize_title'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def plot_simulation_waveform(t: np.ndarray, 
                             sp_simulated: np.ndarray, 
                             sp_recorded: np.ndarray, 
                             misfit_interval: np.ndarray, 
                             outfile_path: str = None, 
                             settings=default_settings):
    """
    Plot the simulated waveform against the recorded waveform.

    Args:
    - t: Time array.
    - sp_simulated: Simulated waveform.
    - sp_recorded: Recorded waveform.
    - misfit_interval: Indices of the misfit interval.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=settings['figure_size'])
    ax.plot(t, sp_recorded, label="Recorded Waveform", color=settings['colors']['black_olive'], linewidth=settings['line_width'])
    ax.plot(t, sp_simulated, label="Simulated Waveform", color=settings['colors']['indianred'], linewidth=settings['line_width'])
    
    # Convert misfit_interval indices to time values
    if len(misfit_interval[0]) > 0:
        misfit_start_time = t[misfit_interval[0][0]]
        misfit_end_time = t[misfit_interval[0][-1]]
        
        # Add shaded region for misfit interval
        ax.axvspan(misfit_start_time, misfit_end_time, color=settings['colors']['dutch_white'], alpha=0.5)
        ax.text(misfit_start_time, min(sp_recorded), 'Misfit Evaluation Interval', ha='left', fontsize=settings['fontsize_labels'], color='darkslategray')
    else:
        print("Misfit interval is empty; cannot shade region.")
    
    ax.set_title("Ultrasonic Wave Simulation", fontsize=settings['fontsize_title'])
    ax.set_xlabel("Time [$\mu s$]", fontsize=settings['fontsize_labels'])
    ax.set_ylabel("Amplitude [a.u.]", fontsize=settings['fontsize_labels'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])
    ax.legend(fontsize=settings['fontsize_ticks'])
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def make_movie_from_simulation(
    outfile_path: str,
    x: np.ndarray,
    t: np.ndarray,
    sp_field: np.ndarray,
    sp_recorded: np.ndarray,
    sample_dimensions: tuple,
    idx_dict: dict,
    settings = default_settings
):
    """
    Create an animation of the wavefield simulation.

    Args:
    - outfile_path: Path to save the movie file.
    - x: Spatial axis.
    - t: Time array.
    - sp_field: Simulated wavefield (2D array).
    - sp_recorded: Recorded waveform.
    - sample_dimensions: Dimensions of the sample.
    - idx_dict: Dictionary of indices for different layers.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    """

    movie_sampling = 10  # Downsampling of the snapshot to speed up movie

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=settings['figure_size'], gridspec_kw={'width_ratios': [10, 1]})
    ylim = 1.3 * np.amax(np.abs(sp_field))

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([-ylim, ylim])
    ax.set_title("Ultrasonic Wavefield in DDS Experiment", fontsize=settings['fontsize_title'], color='darkslategray')
    ax.set_xlabel("Sample Length [cm]", fontsize=settings['fontsize_labels'], color='darkslategray')
    ax.set_ylabel('Relative Shear Wave Amplitude', fontsize=settings['fontsize_labels'], color='darkslategray')
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])

    # Shading layers based on indices in idx_dict
    layers = [
        {'name': 'Gouge Layer 1', 'idx': idx_dict['gouge_1'], 'color': settings['colors']['sandybrown']},
        {'name': 'Gouge Layer 2', 'idx': idx_dict['gouge_2'], 'color': settings['colors']['sandybrown']},
        {'name': 'PZT Layer 1', 'idx': idx_dict['pzt_1'], 'color': settings['colors']['indianred']},
        {'name': 'PZT Layer 2', 'idx': idx_dict['pzt_2'], 'color': settings['colors']['indianred']},
        {'name': 'Grooves', 'idx': np.concatenate([idx_dict['groove_sb1'], idx_dict['groove_cb1'], idx_dict['groove_cb2'], idx_dict['groove_sb2']]), 'color': settings['colors']['lightgrey']},
        {'name': 'Steel Blocks', 'idx': np.concatenate([idx_dict['side_block_1'], idx_dict['central_block'], idx_dict['side_block_2']]), 'color': settings['colors']['lightsteelblue']},
    ]

    for layer in layers:
        ax.axvspan(x[layer['idx'][0]], x[layer['idx'][-1]], color=layer['color'], alpha=0.3, label=layer['name'])

    # Plot transmitter and receiver positions
    x_tr = x[idx_dict['pzt_1'][-1]]
    y_tr = 0
    pzt_width = x[idx_dict['pzt_1'][-1]] - x[idx_dict['pzt_1'][0]]
    pzt_height = 4 * pzt_width
    ax.add_patch(Rectangle((x_tr -  pzt_width, y_tr - pzt_height / 2), pzt_width, pzt_height, color=settings['colors']['teal']))
    ax.text(x_tr - pzt_width / 2, y_tr - pzt_height, 'Transmitter', ha='center', fontsize=settings['fontsize_labels'], color=settings['colors']['darkslategray'])

    x_rc = x[idx_dict['pzt_2'][0]]
    y_rc = 0
    ax.add_patch(Rectangle((x_rc, y_rc - pzt_height / 2), pzt_width, pzt_height, color=settings['colors']['teal']))
    ax.text(x_rc + pzt_width / 2, y_rc - pzt_height, 'Receiver', ha='center', fontsize=settings['fontsize_labels'], color=settings['colors']['darkslategray'])

    # Configure ax2 for the recorded signal
    ax2.set_ylim([t[0], t[-1]])
    ax2.set_xlim([-1, 1])  # Set x-limits to small range around zero
    ax2.set_title("Recorded Signal", fontsize=settings['fontsize_title'], color='darkslategray')
    ax2.axis('off')
    ax2.invert_yaxis()

    fig.tight_layout()

    # Initialize lines for animation
    line_wavefield, = ax.plot([], [], color=settings['colors']['darkslategray'], lw=1.5)
    line_wavefield.set_linewidth(3.0)
    line_recorded, = ax2.plot([], [], color=settings['colors']['darkslategray'], lw=1.5)
    line_recorded.set_linewidth(3.0)

    # Prepare data for animation
    sp_movie = sp_field[::movie_sampling]
    sp_recorded_movie = sp_recorded[::movie_sampling] / np.amax(np.abs(sp_recorded))
    t_recorded_movie = t[::movie_sampling]
    num_frames = len(sp_movie)

    def update_frame(frame):
        line_wavefield.set_data(x, sp_movie[frame])
        line_recorded.set_data(sp_recorded_movie[:frame], t_recorded_movie[:frame])
        return line_wavefield, line_recorded

    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=True, interval=20)
    ani.save(outfile_path, fps=30, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)

def plot_velocity_model(
    x: np.ndarray,
    c: np.ndarray,
    layer_starts: np.ndarray,
    pzt_layer_width: float,
    pmma_layer_width: float,
    outfile_path: str = None,
    settings = default_settings
) -> None:
    """
    Plot the velocity model with layers and smoothing.

    Args:
        x (np.ndarray): Spatial axis.
        c (np.ndarray): Velocity model array.
        layer_starts (np.ndarray): Cumulative positions along the sample.
        pzt_layer_width (float): Width of the PZT layer.
        pmma_layer_width (float): Width of the PMMA layer.
        outfile_path (str, optional): Path to save the plot. Defaults to None.
        settings (dict, optional): Plot settings. Defaults to default_settings.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=settings['figure_size'])
    ax.plot(x, c, label='Velocity Model', color='black', linewidth=settings['line_width'])

    layers = [
        {'name': 'PMMA Layer 1', 'start': layer_starts[0], 'end': layer_starts[1], 'color': settings['colors']['platinum']},
        {'name': 'PZT Layer 1', 'start': layer_starts[1], 'end': layer_starts[2], 'color': settings['colors']['indianred']},
        {'name': 'Side Block 1', 'start': layer_starts[2], 'end': layer_starts[3], 'color': settings['colors']['lightsteelblue']},
        {'name': 'Groove SB1', 'start': layer_starts[3], 'end': layer_starts[4], 'color': settings['colors']['lightgrey']},
        {'name': 'Gouge Layer 1', 'start': layer_starts[4], 'end': layer_starts[5], 'color': settings['colors']['sandybrown']},
        {'name': 'Groove CB1', 'start': layer_starts[5], 'end': layer_starts[6], 'color': settings['colors']['lightgrey']},
        {'name': 'Central Block', 'start': layer_starts[6], 'end': layer_starts[7], 'color': settings['colors']['lightsteelblue']},
        {'name': 'Groove CB2', 'start': layer_starts[7], 'end': layer_starts[8], 'color': settings['colors']['lightgrey']},
        {'name': 'Gouge Layer 2', 'start': layer_starts[8], 'end': layer_starts[9], 'color': settings['colors']['sandybrown']},
        {'name': 'Groove SB2', 'start': layer_starts[9], 'end': layer_starts[10], 'color': settings['colors']['lightgrey']},
        {'name': 'Side Block 2', 'start': layer_starts[10], 'end': layer_starts[11], 'color': settings['colors']['lightsteelblue']},
        {'name': 'PZT Layer 2', 'start': layer_starts[11], 'end': layer_starts[12], 'color': settings['colors']['indianred']},
        {'name': 'PMMA Layer 2', 'start': layer_starts[12], 'end': layer_starts[13], 'color': settings['colors']['platinum']},
    ]

    labels_used = set()

    for layer in layers:
        label = layer['name'] if layer['name'] not in labels_used else None
        labels_used.add(layer['name'])
        ax.axvspan(layer['start'], layer['end'], color=layer['color'], alpha=0.3, label=label)

    # Plot transmitter and receiver positions
    transmitter_pos = pzt_layer_width + pmma_layer_width
    receiver_pos = x[-1] - pzt_layer_width - pmma_layer_width
    ax.axvline(transmitter_pos, color="red", linestyle='-', label='Transmitter')
    ax.axvline(receiver_pos, color="green", linestyle='-', label='Receiver')

    ax.set_title("Velocity Model with Layers and Smoothing", fontsize=settings['fontsize_title'])
    ax.set_xlabel("Position (cm)", fontsize=settings['fontsize_labels'])
    ax.set_ylabel("Velocity (cm/μs)", fontsize=settings['fontsize_labels'])
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=settings['fontsize_ticks'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def plot_synthetic_spatial_function(
    x: np.ndarray,
    spatial_function: np.ndarray,
    outfile_path: str = None,
    settings = default_settings
) -> None:
    """
    Plot the synthetic spatial function.

    Args:
        x (np.ndarray): Spatial axis.
        spatial_function (np.ndarray): The synthetic spatial function values.
        outfile_path (str, optional): Path to save the plot. Defaults to None.
        settings (dict, optional): Plot settings. Defaults to default_settings.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=settings['figure_size'])
    ax.plot(x, spatial_function, linewidth=settings['line_width'])
    ax.set_title("Synthetic Spatial Function", fontsize=settings['fontsize_title'])
    ax.set_xlabel("Position (cm)", fontsize=settings['fontsize_labels'])
    ax.set_ylabel("Amplitude", fontsize=settings['fontsize_labels'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def plot_velocity_and_stresses(
    x_values: np.ndarray,
    velocities: np.ndarray,
    normal_stress: np.ndarray,
    shear_stress: np.ndarray,
    x_label: str,
    velocity_label: str,
    stress_labels: tuple[str, str],
    title: str,
    outfile_path: str = None,
    settings= default_settings
) -> None:
    """
    Plot estimated velocities and two stresses vs a common x-axis variable.

    Args:
        x_values (np.ndarray): The x-axis values (e.g., ec_disp_mm or time_s).
        velocities (np.ndarray): Estimated velocities.
        normal_stress (np.ndarray): Normal stress values.
        shear_stress (np.ndarray): Shear stress values.
        x_label (str): Label for the x-axis.
        velocity_label (str): Label for the velocity y-axis.
        stress_labels (Tuple[str, str]): Labels for the stress y-axes.
        title (str): Title of the plot.
        outfile_path (str, optional): Path to save the plot. Defaults to None.
        settings (dict, optional): Plot settings. Defaults to default_settings.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=settings['figure_size'])

    # Plot velocities on the left y-axis
    color1 = 'tab:blue'
    ax.set_xlabel(x_label, fontsize=settings['fontsize_labels'])
    ax.set_ylabel(velocity_label, color=color1, fontsize=settings['fontsize_labels'])
    ax.plot(x_values, velocities, color=color1, label=velocity_label, linewidth=settings['line_width'])
    ax.tick_params(axis='y', labelcolor=color1, labelsize=settings['fontsize_ticks'])
    ax.tick_params(axis='x', labelsize=settings['fontsize_ticks'])

    # Create a second y-axis for normal stress
    ax2 = ax.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(stress_labels[0], color=color2, fontsize=settings['fontsize_labels'])
    ax2.plot(x_values, normal_stress, color=color2, linestyle='--', label=stress_labels[0], linewidth=settings['line_width'])
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=settings['fontsize_ticks'])

    # Adjust the position of ax2 to make room for a third y-axis
    ax2.spines['right'].set_position(('axes', 1.0))

    # Create a third y-axis for shear stress
    ax3 = ax.twinx()
    color3 = 'tab:green'
    ax3.set_ylabel(stress_labels[1], color=color3, fontsize=settings['fontsize_labels'])
    ax3.plot(x_values, shear_stress, color=color3, linestyle='-', label=stress_labels[1], linewidth=settings['line_width'])
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=settings['fontsize_ticks'])

    # Offset the third y-axis
    ax3.spines['right'].set_position(('axes', 1.15))

    # Add grid, legend, and title
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left', fontsize=settings['fontsize_ticks'])

    ax.set_title(title, fontsize=settings['fontsize_title'])
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])

def plot_l2_norm_vs_velocity(
    gouge_velocity: np.ndarray,
    L2norm: np.ndarray,
    overall_index: int,
    outfile_path: str = None,
    settings: dict = default_settings
) -> None:
    """
    Plot and save the L2 norm vs Gouge velocity plot at specified intervals.

    Args:
        gouge_velocity (np.ndarray): Array of gouge velocity values.
        L2norm (np.ndarray): Array of L2 norm values corresponding to the velocity.
        overall_index (int): Index of the waveform for plot title and file name.
        outfile_name (str): Base name for the output plot file.
        outdir_path_image (str): Directory to save the plot file.
        idx_waveform (int): Current waveform index for interval checking.
        l2norm_plot_interval (int): Interval to save plots (e.g., every Nth waveform).
        settings (dict, optional): Plot settings. Defaults to default_settings.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=settings['figure_size'])
    
    # Plotting the L2 norm vs gouge velocity
    ax.plot(gouge_velocity, L2norm, linewidth=settings['line_width'])
    ax.set_xlabel('Gouge Velocity (cm/μs)', fontsize=settings['fontsize_labels'])
    ax.set_ylabel('L2 Norm of Residuals', fontsize=settings['fontsize_labels'])
    ax.set_title(f'L2 Norm vs Gouge Velocity for Waveform {overall_index}', fontsize=settings['fontsize_title'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])
    ax.grid(alpha=0.3)
    
    # Save the plot using the custom function and settings
    output_path_choice(fig=fig, outfile_path=outfile_path, format=settings['format'])
        

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
FORMAT = ".eps"

# Define the color palette for the plots
colors = {
    'reseda_green': '#788054',
    'dutch_white': '#E0D6B4',
    'khaki': '#CABB9E',
    'platinum': '#E7E5E2',
    'black_olive': '#322D1E'
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

def output_path_choice(plot: plt.Figure, outfile_path: str = None, format: str = FORMAT) -> None:
    '''
    Saving or showing option for plot functions
    - plot: The plot object to be saved or shown.
    - outfile_path: Path to save the plot.
    - format: File format for saving the plot.
    '''
    if outfile_path:
        outfile_name = os.path.basename(outfile_path)
        current_title = plt.gca().get_title()
        plt.title(current_title + " " + outfile_name)
        plt.savefig(outfile_path + format, dpi=300)
        plt.close()  # Close the figure to prevent it from being displayed
        print(f"Plot saved to {outfile_path}")
    else:
        plt.show()

def uw_all_plot(data: np.ndarray, metadata: dict, step_wf_to_plot: int, highlight_start: int, highlight_end: int, xlim_plot: float, ticks_steps_waveforms: float, outfile_path: str = None, settings=default_settings) -> None:
    '''
    Plots stacked waveforms with optional highlighting.

    Args:
    - data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - step_wf_to_plot: Step size for plotting waveforms.
    - highlight_start: Index of the start point for highlighting.
    - highlight_end: Index of the end point for highlighting.
    - xlim_plot: Limits for the x-axis.
    - ticks_steps_waveforms: Step size for ticks on the waveform axis.
    - outfile_path: Path to save the plot.
    - settings: Dictionary with plot settings.

    Returns:
    - None
    '''

    time_ax_waveform = metadata["time_ax_waveform"]
    time_ticks_waveforms = np.arange(time_ax_waveform[0], time_ax_waveform[-1], ticks_steps_waveforms)
    
    data = data[::step_wf_to_plot]
    ymax = 1.3 * np.amax(data)
    ymin = 1.3 * np.amin(data)

    plt.figure(figsize=settings['figure_size'])

    plt.plot(time_ax_waveform, data.T, color='black', linewidth=0.8, alpha=0.5)
    plt.plot(time_ax_waveform[highlight_start:highlight_end], data[highlight_start:highlight_end], color='red')
    plt.xlabel('Time [$\mu s$]', fontsize=settings['fontsize_labels'])
    plt.ylabel('Amplitude [.]', fontsize=settings['fontsize_labels'])
    plt.xticks(time_ticks_waveforms)
    plt.ylim([ymin, ymax])
    plt.xlim(time_ax_waveform[0], xlim_plot)
    plt.grid(alpha=0.1)
    plt.title("Stacked Waveforms", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)

    output_path_choice(plot=plt, outfile_path=outfile_path, format=settings['format'])

def amplitude_map(data: np.ndarray, metadata: dict, amp_scale: float, outfile_path: str = None, settings=default_settings) -> None:
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
    start_time, end_time = time_ax_waveform[0], time_ax_waveform[-1]
    sampling_rate = metadata['sampling_rate']
    number_of_samples = metadata['number_of_samples']
    acquisition_frequency = metadata['acquisition_frequency']

    fig, ax1 = plt.subplots(ncols=1, figsize=settings['figure_size'])
    ax1.set_title("Amplitude Map", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    cmap = plt.get_cmap('seismic')
    
    im = ax1.imshow(data.T, aspect='auto', origin='lower', interpolation='none',
                    cmap=cmap, vmin=-amp_scale, vmax=amp_scale,
                    extent=[0, data.shape[0] * acquisition_frequency, 0, number_of_samples * sampling_rate])
    
    cbar = fig.colorbar(im, pad=0.04)
    cbar.set_label("Relative Amplitude", fontsize=settings['fontsize_labels'])

    ax1.set_xlabel('Time [s]', fontsize=settings['fontsize_labels'])
    ax1.set_ylabel('Wf_time [$\mu s$]', fontsize=settings['fontsize_labels'])

    fig.tight_layout()

    output_path_choice(plot=plt, outfile_path=outfile_path, format=settings['format'])

def amplitude_spectrum_map(signal_freqs: np.ndarray, amp_spectrum: np.ndarray, metadata: dict, outfile_path: str = None, settings=default_settings) -> None:
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
    plt.pcolormesh(time_ax_acquisition, signal_freqs, amp_spectrum.T, cmap="gist_gray", norm=mcolors.LogNorm())
    plt.title('Amplitude Spectrum Map', fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    plt.xlabel("Time [s]", fontsize=settings['fontsize_labels'])
    plt.ylabel("Frequency [$MHz $]", fontsize=settings['fontsize_labels'])
    cbar = plt.colorbar(pad=0.04)
    cbar.set_label("Spectral Amplitude", fontsize=settings['fontsize_labels'])

    output_path_choice(plot=plt, outfile_path=outfile_path, format=settings['format'])

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

    f, ax = plt.subplots(2, 1, figsize=settings['figure_size'])

    ax[0].semilogy(signal_freqs, amp_spectrum, label="Amplitude Spectrum")
    ax[0].semilogy(signal_freqs, lowpass_filter * np.amax(amp_spectrum), label="Filter Shape")
    ax[0].semilogy(signal_freqs, filtered_amp_spectrum, label="Amplitude Spectrum Filtered")
    ax[0].vlines(max_freq, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r", "--", label=f"Maximum of the spectrum = {max_freq:.2f} MHz")
    ax[0].vlines(2, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r", "-", label=f"Resonance Frequency of the sensor = 2 MHz")
    ax[0].legend()
    ax[0].set_xlim([0, np.amax(signal_freqs)])
    ax[0].set_ylabel("Amplitude [.]")
    ax[0].set_xlabel("Frequency [MHz]")
    ax[0].set_title("Amplitude Spectrum of a waveform", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)

    ax[1].plot(signal_freqs, phase_spectrum)
    ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
    ax[1].set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax[1].set_ylabel("Phase [rad]")
    ax[1].set_xlabel("Frequency [MHz]")
    ax[1].set_title("Phase Spectrum of a waveform", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)

    plt.tight_layout()

    output_path_choice(plot=plt, outfile_path=outfile_path, format=settings['format'])

def signal_vs_filtered_signal_plot(single_waveform: np.ndarray, single_waveform_filtered: np.ndarray, metadata: dict, freq_cut: float, outfile_path: str = None, settings=default_settings) -> None:
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
    plt.plot(time_ax, single_waveform, color="lightgray", label="waveform")
    plt.plot(time_ax, single_waveform_filtered, color="black", label="filtered waveform")
    plt.xlabel('Time [$\mu s$]', fontsize=settings['fontsize_labels'])
    plt.ylabel('Amplitude [.]', fontsize=settings['fontsize_labels'])
    plt.title(f"Effect of lowpass filtering at {freq_cut:.2f} MHz", fontsize=settings['fontsize_title'], fontname=FONT_TYPE)
    plt.legend()

    output_path_choice(plot=plt, outfile_path=outfile_path, format=settings['format'])

def wavelet_selection_plot(time: np.ndarray, waveform: np.ndarray, ratio: np.ndarray, index_max_list: list, index_min_before_list: list, index_min_after_list: list, outfile_path: str = None, settings=default_settings) -> None:
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
    fig = plt.figure(figsize=settings['figure_size'])
    plt.plot(time, waveform, label='Waveform Recorded', alpha=0.5, linewidth=3.0)
    norm = (np.amax(waveform) / np.amax(ratio))
    plt.plot(time, norm * ratio, label='STA/LTA on Waveform')
    plt.plot(time[index_max_list], norm * ratio[index_max_list], "r.")
    plt.plot(time[index_min_before_list], norm * ratio[index_min_before_list], "g.")
    plt.plot(time[index_min_after_list], norm * ratio[index_min_after_list], "k.")
    plt.xticks(fontsize=settings['fontsize_ticks'])
    plt.yticks(fontsize=settings['fontsize_ticks'])
    plt.xlabel('Time [$\mu$s]', fontsize=settings['fontsize_labels'])
    plt.ylabel('Amplitude [a.u.]', fontsize=settings['fontsize_labels'])
    plt.legend(loc="lower left", fontsize=settings['fontsize_ticks'])
    plt.title("Identify the wavelet: measure in homogeneous space", fontsize=settings['fontsize_title'])

    output_path_choice(plot=plt, outfile_path=outfile_path, format=settings['format'])

def plot_simulation_waveform(t, sp_recorded, settings=default_settings):
    ax = plt.gca()
    ax.plot(t, sp_recorded, label="UW simulated", color=settings['colors']['platinum'], linewidth=settings['line_width'])
    ax.set_title("Ultrasonic Wave Simulation", fontsize=settings['fontsize_title'])
    ax.set_xlabel("Time", fontsize=settings['fontsize_labels'])
    ax.set_ylabel("Amplitude", fontsize=settings['fontsize_labels'])
    ax.tick_params(axis='both', which='major', labelsize=settings['fontsize_ticks'])
    plt.legend()
    plt.show()
    
def make_movie(outfile_path, x, t, sp_field, sp_recorded, side_block_1, gouge_1, central_block, gouge_2, side_block_2, idx_gouge_1, idx_gouge_2, idx_pzt_1, idx_pzt_2, settings=default_settings):
    movie_sampling = 10  # downsampling of the snapshot to speed up movie

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=settings['figure_size'], gridspec_kw={'width_ratios': [10, 1]})
    ylim = 1.3 * np.amax(sp_field)

    ax.set(xlim=[x[0], x[-1]], ylim=[-ylim, ylim])
    ax.set_title("Ultrasonic Wavefield In DDS Experiment", fontsize=settings['fontsize_title'], color='darkslategray')
    ax.set_xlabel("Sample length [cm]", fontsize=settings['fontsize_labels'], color='darkslategray')
    ax.set_ylabel('Relative Shear Wave Amplitude', fontsize=settings['fontsize_labels'], color='darkslategray')

    ax.axvspan(x[0], x[idx_gouge_1[0]], color='lightsteelblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_1[-1]], x[idx_gouge_2[0]], color='lightsteelblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_2[-1]], x[-1], color='lightsteelblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_1[0]], x[idx_gouge_1[-1]], color='aliceblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_2[0]], x[idx_gouge_2[-1]], color='aliceblue', alpha=0.5)

    xtr = x[idx_pzt_1[0]]
    ytr = 0
    pzt_width = x[idx_pzt_1[-1]] - x[idx_pzt_1[0]]
    pzt_height = 4 * pzt_width
    ax.add_patch(Rectangle((xtr - 1.5 * pzt_width, ytr - pzt_height / 2), pzt_width, pzt_height, color='teal'))
    ax.text(xtr - pzt_width / 2, ytr - pzt_height, 'Transmitter', ha='center', fontsize=settings['fontsize_labels'], color='darkslategray')

    xrc = x[idx_pzt_2[-1]]
    yrc = 0
    ax.add_patch(Rectangle((xrc + 0.5 * pzt_width, yrc - pzt_height / 2), pzt_width, pzt_height, color='teal'))
    ax.text(xrc, yrc - pzt_height, 'Receiver', ha='center', fontsize=settings['fontsize_labels'], color='darkslategray')

    ax2.set(ylim=[t[0], t[-1]], xlim=[-0.7 * ylim, 0.7 * ylim])
    ax2.set_title("Rec", fontsize=settings['fontsize_title'], color='darkslategray')
    ax2.set_axis_off()
    ax2.invert_yaxis()

    fig.tight_layout()

    def movie_uw(frame):
        line.set_xdata(x[idx_pzt_1[0] - 30:idx_pzt_2[-1] + 30])
        line.set_ydata(sp_movie[frame][idx_pzt_1[0] - 30:idx_pzt_2[-1] + 30])
        line1.set_xdata(sp_recorded_movie[:frame])
        line1.set_ydata(t_recorded_movie[:frame])
        return line,

    line, = ax.plot([], [], color='darkslategray', lw=1.5)
    line.set_linewidth(3.0)
    line1, = ax2.plot([], [], color='darkslategray', lw=1.5)
    line.set_linewidth(3.0)

    sp_movie = sp_field[::movie_sampling]
    sp_recorded_movie = sp_recorded[::movie_sampling] / np.amax(sp_recorded)
    t_recorded_movie = t[::movie_sampling]
    num_frames = len(sp_movie)

    ani = animation.FuncAnimation(fig, movie_uw, frames=num_frames, blit=True, interval=20)
    ani.save(outfile_path, fps=30, extra_args=['-vcodec', 'libx264'])

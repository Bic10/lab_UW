# src/plotting.py

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Define global variables for font type, fontsize, and figure size
FONT_TYPE = "Ubuntu"
FONT_SIZE = 30
FIGURE_SIZE = (16, 8)
FORMAT = ".eps"
# COLORPALETTE = ???           define it too globally

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
        pass
        # plt.show(plot)


def uw_all_plot(data: np.ndarray, metadata: dict, step_wf_to_plot: int,
                highlight_start: int, highlight_end: int, xlim_plot: float,
                ticks_steps_waveforms: float, outfile_path: str = None,  format: str = FORMAT)  -> None:
    '''
    Plots stacked waveforms with optional highlighting.

    Args:
    - outfile_path: Path to save the plot.
    - data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - step_wf_to_plot: Step size for plotting waveforms.
    - highlight_start: Index of the start point for highlighting.
    - highlight_end: Index of the end point for highlighting.
    - ylim_plot: Limits for the y-axis.
    - xlim_plot: Limits for the x-axis.
    - format: File format for saving the plot.
    - ticks_steps_waveforms: Step size for ticks on the waveform axis.

    Returns:
    - None
    '''

    time_ax_waveform = metadata["time_ax_waveform"]
    time_ticks_waveforms = np.arange(time_ax_waveform[0], time_ax_waveform[-1], ticks_steps_waveforms)
    
    data  = data[::step_wf_to_plot]
    ymax = 1.3*np.amax(data)
    ymin = 1.3*np.amin(data)

    plt.figure(figsize=FIGURE_SIZE)

    plt.plot(time_ax_waveform, data.T, 
            color='black', linewidth=0.8, alpha=0.5)
    plt.plot(time_ax_waveform[highlight_start:highlight_end], data[highlight_start:highlight_end], color='red')
    plt.xlabel('Time [$\mu s$]', fontsize=0.7*FONT_SIZE)
    plt.ylabel('Amplitude [.]', fontsize=0.7*FONT_SIZE)
    plt.xticks(time_ticks_waveforms)
    plt.ylim([ymin, ymax])
    plt.xlim(time_ax_waveform[0], xlim_plot)
    plt.grid(alpha=0.1)
    plt.title("Stacked Waveforms ", fontsize=FONT_SIZE, fontname=FONT_TYPE)

    output_path_choice(plot=plt, outfile_path=outfile_path, format = format)
    

def amplitude_map(data: np.ndarray, metadata: dict, amp_scale: float,
                   outfile_path: str = None, format: str = FORMAT) -> None:
    '''
    Plots an amplitude map of waveform data.

    Args:
    - outfile_path: Path to save the plot.
    - data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - amp_scale: Scale for amplitude.
    - format: File format for saving the plot.

    Returns:
    - None
    '''    

    time_ax_waveform = metadata['time_ax_waveform']
    start_time, end_time = time_ax_waveform[0], time_ax_waveform[-1]
    sampling_rate = metadata['sampling_rate']
    number_of_samples = metadata['number_of_samples']
    acquisition_frequency =  metadata['acquition_frequency']

    fig, ax1 = plt.subplots(ncols=1, figsize=FIGURE_SIZE)
    ax1 = plt.subplot()
    ax1.set_title("Amplitude Map ", fontsize=FONT_SIZE, fontname=FONT_TYPE)
    cmap = plt.get_cmap('seismic')
    
    im = ax1.imshow(data.T, aspect='auto', origin='lower', interpolation='none',
                    cmap=cmap, vmin=-amp_scale, vmax=amp_scale, 
                    extent=[0, data.shape[0] * acquisition_frequency, 0 , number_of_samples * sampling_rate])
    
    cbar = fig.colorbar(im, pad=0.04)
    cbar.set_label("Relative Amplitude", fontsize=0.5*FONT_SIZE)

    ax1.set_xlabel('Time [s]', fontsize=0.7*FONT_SIZE)
    ax1.set_ylabel('Wf_time [$\mu s$] ', fontsize=0.7*FONT_SIZE)
    ax1.set_xscale

    fig.tight_layout()
    
    output_path_choice(plot=plt, outfile_path=outfile_path, format = format)

    
def amplitude_spectrum_map(signal_freqs: np.ndarray, amp_spectrum: np.ndarray,
                           metadata: dict, outfile_path: str = None, format: str = FORMAT) -> None:
    '''
    Plots an amplitude map of waveform data.

    Args:
    - outfile_path: Path to save the plot.
    - signal_freqs: Frequencies of the signal.
    - amp_spectrum: Amplitude spectrum.
    - metadata: Metadata dictionary containing information about the data.
    - format: File format for saving the plot. Default is PNG.

    Returns:
    - None
    '''
    wave_num, wave_len = amp_spectrum.shape
    
    spectrum_length = round(wave_len / 2)
    signal_freqs = signal_freqs[:spectrum_length]
    amp_spectrum = amp_spectrum[:, :spectrum_length]

    time_ax_acquisition = metadata['time_ax_acquisition']
    plt.pcolormesh(time_ax_acquisition, signal_freqs, amp_spectrum.T, cmap="gist_gray", norm=mcolors.LogNorm())   
    plt.title('Amplitude Map', fontsize=FONT_SIZE, fontname=FONT_TYPE)
    plt.xlabel("Time [s]", fontsize=0.7*FONT_SIZE)
    plt.ylabel("Frequency [$MHz $]", fontsize=0.7*FONT_SIZE)
    cbar = plt.colorbar(pad=0.04)
    cbar.set_label("Spectral Amplitude", fontsize=0.5*FONT_SIZE)

    output_path_choice(plot=plt, outfile_path=outfile_path, format = format)


def filtered_amp_and_phase_spectrum_plot(signal_freqs: np.ndarray, amp_spectrum: np.ndarray,
                                          phase_spectrum: np.ndarray, filtered_amp_spectrum: np.ndarray,
                                          lowpass_filter: np.ndarray, sampling_rate: float, 
                                           outfile_path: str = None,  format : str = FORMAT) -> None:
    '''
    Plots the filtered amplitude and phase spectrum.

    Args:
    - outfile_path: Path to save the plot.
    - signal_freqs: Frequencies of the signal.
    - amp_spectrum: Amplitude spectrum.
    - phase_spectrum: Phase spectrum.
    - filtered_amp_spectrum: Filtered amplitude spectrum.
    - lowpass_filter: Low-pass filter.
    - sampling_rate: Sampling rate of the signal.

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

    f, ax = plt.subplots(2, 1, figsize=FIGURE_SIZE)

    ax[0].semilogy(signal_freqs, amp_spectrum, label="Amplitude Spectrum")
    ax[0].semilogy(signal_freqs, lowpass_filter * np.amax(amp_spectrum), label="Filter Shape")
    ax[0].semilogy(signal_freqs, filtered_amp_spectrum, label="Amplitude Spectrum Filtered")
    ax[0].vlines(max_freq, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r",
                 "--", label=f"Maximum of the spectrum = {max_freq:.2f} MHz")
    ax[0].vlines(2, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r",
                 "-", label=f"Resonance Frequency of the sensor = 2 MHz")
    ax[0].legend()
    ax[0].set_xlim([0, np.amax(signal_freqs)])
    ax[0].set_ylabel("Amplitude [.]")
    ax[0].set_xlabel("Frequency [MHz]")
    ax[0].set_title("Amplitude Spectrum of a waveform", fontsize=FONT_SIZE, fontname=FONT_TYPE)

    ax[1].plot(signal_freqs, phase_spectrum)

    ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
    ax[1].set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax[1].set_ylabel("Phase [rad]")
    ax[1].set_xlabel("Frequency [MHz]")
    ax[1].set_title("Phase Spectrum of a waveform", fontsize=FONT_SIZE, fontname=FONT_TYPE)

    plt.tight_layout()

    output_path_choice(plot=plt, outfile_path=outfile_path, format = format)


def signal_vs_filtered_signal_plot(single_waveform: np.ndarray, 
                                   single_waveform_filtered: np.ndarray, 
                                   metadata: dict[str, np.ndarray], 
                                   freq_cut: float, 
                                   outfile_path: str = None, format: str = FORMAT) -> None:
    """
    Plot a comparison between a waveform and its filtered version.

    Parameters:
        outfile_path (str): The path to save the plot.
        single_waveform (np.ndarray): The original waveform data.
        single_waveform_filtered (np.ndarray): The filtered waveform data.
        metadata (Dict[str, np.ndarray]): Metadata containing information about the waveform.
        format (str, optional): The format of the output file (default is ".eps").

    Returns:
        None
    """
    time_ax = metadata['time_ax_waveform']
    plt.plot(time_ax, single_waveform, color="lightgray", label="waveform")
    plt.plot(time_ax, single_waveform_filtered, color="black", label="filtered waveform")
    plt.xlabel('Time [$\mu s$]', fontsize=0.7*FONT_SIZE)
    plt.ylabel('Amplitude [.]', fontsize=0.7*FONT_SIZE)
    plt.title("Effect of lowpass filtering at %2.2f MHz" % freq_cut, fontsize=FONT_SIZE, fontname=FONT_TYPE)
    plt.legend()

    output_path_choice(plot=plt, outfile_path=outfile_path, format = format)


def wavelet_selection_plot(time: np.ndarray, 
                 waveform: np.ndarray, 
                 ratio: np.ndarray, 
                 index_max_list: list[int], 
                 index_min_before_list: list[int], 
                 index_min_after_list: list[int], 
                 outfile_path: str = None, 
                 format: str = FORMAT) -> None:
    """
    Plot waveform data along with STA/LTA analysis results.

    Args:
        time (np.ndarray): Time array.
        waveform (np.ndarray): Waveform data.
        ratio (np.ndarray): STA/LTA ratio.
        index_max_list (list[int]): List of indices for maximum values.
        index_min_before_list (list[int]): List of indices for minimum values before maximum.
        index_min_after_list (list[int]): List of indices for minimum values after maximum.
        outfile_path (str): Path to save the plot.
        format (str): Format of the saved plot.
    """
    # Plot the data vs sta_lta picking
    fig = plt.figure(figsize=FIGURE_SIZE)
    plt.plot(time, waveform, label='Waveform Recorded', alpha=0.5, linewidth=3.0)
    norm = (np.amax(waveform) / np.amax(ratio))
    plt.plot(time, norm * ratio, label='STA/LTA on Waveform')
    plt.plot(time[index_max_list], norm * ratio[index_max_list], "r.")
    plt.plot(time[index_min_before_list], norm * ratio[index_min_before_list], "g.")
    plt.plot(time[index_min_after_list], norm * ratio[index_min_after_list], "k.")
    plt.xticks(fontsize=0.7 * FONT_SIZE)
    plt.yticks(fontsize=0.7 * FONT_SIZE)
    plt.xlabel('Time [$\mu$s]', fontsize=0.7 * FONT_SIZE)
    plt.ylabel('Amplitude [a.u.]', fontsize=0.7 * FONT_SIZE)
    plt.legend(loc="lower left", fontsize=0.7 * FONT_SIZE)
    plt.title("Identify the wavelet: measure in homogeneous space", fontsize=FONT_SIZE)

    output_path_choice(plot=plt, outfile_path=outfile_path, format=format)

def make_movie(outfile_path, side_block_1: float, gouge_1: float, central_block: float, gouge_2: float, side_block_2: float) -> None:
    # MAKE A MOVIE FROM THE RECORDED FIELD
    movie_sampling = 10  # downsampling of the snapshot to speed up movie

    # Initialize figure and axes
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [10, 1]})    
 
    ylim = 1.3 * np.amax(sp_field)  # this is going to be around the higher value of pressure field

    # Central plot
    ax.set(xlim=[x[0], x[-1]], ylim=[-ylim, ylim])

    # set labels and font size
    ax.set_title("Utrasonic Wavefield In DDS Experiment", fontsize=20, color='darkslategray')
    ax.set_xlabel("Sample length [cm]", fontsize=15, color='darkslategray')
    ax.set_ylabel('Relative Shear Wave Amplitude', fontsize=15, color='darkslategray')

    # this lines are just to shadow in different colours steel and gouge
    idx_gouge_1 = np.where((x > side_block_1) & (x < side_block_1 + gouge_1))[0]
    idx_gouge_2 = np.where((x > side_block_1 + gouge_1 + central_block) & (x < side_block_1 + gouge_1 + central_block + gouge_2))[0]
    ax.axvspan(x[0], x[idx_gouge_1[0]], color='lightsteelblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_1[-1]], x[idx_gouge_2[0]], color='lightsteelblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_2[-1]], x[-1], color='lightsteelblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_1[0]], x[idx_gouge_1[-1]], color='aliceblue', alpha=0.5)
    ax.axvspan(x[idx_gouge_2[0]], x[idx_gouge_2[-1]], color='aliceblue', alpha=0.5)


    # # Add text in the middle of the first axvspan
    # ax.text((x[idx_gouge_1[0]] + x[idx_gouge_2[-1]]) / 2, -ylim / 2, 'Steel', ha='center', fontsize=15)

    # # Add text close to the end of the last axvspan
    # ax.text(x[idx_gouge_2[0]], -ylim / 2, 'Gouge', ha='center', fontsize=15, rotation='vertical')

    # Add the "transmitter"
    xtr = x[idx_pzt_1[0]]
    ytr = 0
    pzt_width= x[idx_pzt_1[-1]]-x[idx_pzt_1[0]] # Adjust the size of the square as needed
    pzt_hight = 4*pzt_width
    ax.add_patch(Rectangle((xtr - 1.5*pzt_width, ytr - pzt_hight/ 2), pzt_width, pzt_hight, color='teal'))
    ax.text(xtr - pzt_width/2, ytr - pzt_hight, 'Transmitter', ha='center', fontsize=15, color='darkslategray' )

    # Add the "receiver"
    xrc = x[idx_pzt_2[-1]]
    yrc = 0
    pzt_width= x[idx_pzt_1[-1]]-x[idx_pzt_1[0]] # Adjust the size of the square as needed
    pzt_hight = 4*pzt_width
    ax.add_patch(Rectangle((xrc + 0.5*pzt_width , yrc - pzt_hight/ 2), pzt_width, pzt_hight, color='teal'))
    ax.text(xrc, yrc - pzt_hight, 'Receiver', ha='center', fontsize=15, color='darkslategray')

    # Lateral plot
    ax2.set(ylim=[t[0], t[-1]], xlim=[-0.7*ylim, 0.7*ylim])


    ax2.set_title("Rec", fontsize=20, color='darkslategray')
    # ylabel = ax2.set_ylabel("Waveform Recorded", fontsize=10)
    ax2.set_axis_off()
    ax2.invert_yaxis()

    fig.tight_layout()

    def movie_uw(frame):
        # Plot x against sp__movie
        line.set_xdata(x[idx_pzt_1[0]-30:idx_pzt_2[-1]+30])
        line.set_ydata(sp_movie[frame][idx_pzt_1[0]-30:idx_pzt_2[-1]+30])
        line1.set_xdata(sp_recorded_movie[:frame])
        line1.set_ydata(t_recorded_movie[:frame])
        return line,

    # Initialize empty line object
    line, = ax.plot([], [], color='darkslategray', lw=1.5)
    line.set_linewidth(3.0)
    line1, = ax2.plot([], [],color='darkslategray', lw=1.5)
    line.set_linewidth(3.0)

    # data for central plot: the spatial field evolution
    sp_movie = sp_field[::movie_sampling]

    # the side plot

    sp_movie = sp_field[::movie_sampling]

    # the side plot: waveform recording
    sp_recorded_movie = sp_recorded[::movie_sampling]/np.amax(sp_recorded)
    t_recorded_movie = t[::movie_sampling]
    # Define the number of frames
    num_frames = len(sp_movie)

    # Create animation
    ani = animation.FuncAnimation(fig, movie_uw, frames=num_frames, blit=True, interval=20)

    # Save the animation as an MP4 file
    ani.save(outfile_path, fps=30, extra_args=['-vcodec', 'libx264'])


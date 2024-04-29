#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import re

def remove_starting_noise(data: np.ndarray, metadata: dict, remove_initial_samples: int = 0) -> tuple[np.ndarray, dict]:
    '''
    Removes initial samples from the data and updates metadata accordingly.

    Args:
    - data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - remove_initial_samples: Number of initial samples to remove.

    Returns:
    - Tuple of modified data array and updated metadata dictionary.
    '''
    data = data[:, remove_initial_samples:]
    metadata['number_of_samples'] = metadata['number_of_samples'] - remove_initial_samples
    metadata['time_ax_waveform'] = metadata['time_ax_waveform'][remove_initial_samples:]

    return data, metadata

def remove_empty_lines(infile: iter, number_of_samples: int, encoding: str = 'iso8859') -> list[list[str]]:
    '''
    Removes empty lines from input file.

    Args:
    - infile: Iterable containing lines from the input file.
    - number_of_samples: Number of samples per line.
    - encoding: Encoding type of the input file.

    Returns:
    - List of lines with sufficient number of samples.
    '''
    waveform_list = []             
    for line in infile:
        line = line.decode(encoding)
        line = line.split("\t")
        if len(line) < number_of_samples:
            continue
        waveform_list.append(line)
    return waveform_list


def make_UW_data(infile_path: str) -> tuple[np.ndarray, dict]:
    '''
    Reads data from a binary file and extracts data and metadata.

    Args:
    - infile_path: Path to the input binary file.

    Returns:
    - Tuple containing data array and metadata dictionary.
    '''
    with open(infile_path, "rb") as infile:
        acquisition_info, time_info = extract_metadata_from_tsv(infile, encoding='iso8859')

        number_of_samples = time_info[2]
        sampling_rate = time_info[3]
        time_ax_waveform = np.arange(time_info[0], time_info[1], sampling_rate)

        acquisition_frequency = acquisition_info[2]
        time_ax_acquisition = np.arange(acquisition_info[0], acquisition_info[1], acquisition_frequency)

        waveform_list = remove_empty_lines(infile, number_of_samples, encoding='iso8859')
        corrected_number_of_waveforms = len(waveform_list)
        time_ax_acquisition = time_ax_acquisition[:corrected_number_of_waveforms]

        metadata = {"number_of_samples": number_of_samples,
                    'sampling_rate': sampling_rate,
                    "time_ax_waveform": time_ax_waveform,
                    'number_of_waveforms': corrected_number_of_waveforms,
                    "acquition_frequency": acquisition_frequency,
                    'time_ax_acquisition': time_ax_acquisition}

        data = np.array(waveform_list).astype(float)

    return data, metadata


def extract_metadata_from_tsv(infile: iter, encoding: str = 'iso8859') -> tuple[list[float], list[float]]:
    '''
    Extracts metadata from a TSV file.    AT THE MOMENT WE ONLY MANAGE TO USE THIRD AND FORTH LINES.
    IMPLEMENT LATER THE REST

    Args:
    - infile: Iterable containing lines from the input file.
    - encoding: Encoding type of the input file.

    Returns:
    - Tuple containing acquisition information and time information extracted from the TSV file.


    '''
    # encoding = 'iso8859' #just to read greek letters: there is a "mu" in the euroscan file  

    general = next(infile).decode(encoding)
    amplitude_scale = next(infile).decode(encoding)
    time_scale = next(infile).decode(encoding)
    acquisition_scale = next(infile).decode(encoding)
    acquisition_info = re.findall(r"\d+\.*\d*", acquisition_scale)
    acquisition_info = [float(entry) for entry in acquisition_info]
    time_info = re.findall(r"\d+\.*\d*",time_scale)   # find all float in the string "time_scale" using regular expression:
                                            # the first argument of findall must be a "regular expression". It start with an r so
                                            # is an r-string, that means "\" can be read as a normal character
                                            # The regular expression "\d+\.*\d*" means "match all the digits that are adiacent as
                                            # a single entry and, if they are followed by a dot and some other digits, they are still
                                            # the same numbers". So it gets as a single number both 1, 10, 1.4, 10.42 etc.
    time_info = [float(entry) for entry in time_info]           # just convert the needed info in float number

    return acquisition_info, time_info


def make_infile_path_list(machine_name: str, experiment_name: str, data_type:str) -> list[str]:
    '''
    Generates a list of input file paths based on the machine name and experiment name.

    Args:
    - machine_name: Name of the machine.
    - experiment_name: Name of the experiment.

    Returns:
    - List of input file paths.
    '''
    code_path = os.getcwd()
    parent_folder = os.path.abspath(os.path.join(code_path, os.pardir))
    indir_path = os.path.join(parent_folder,"experiments_"+ machine_name, experiment_name, data_type)

    infile_path_list = []   
    for infile_name in os.listdir(indir_path):
        file_path = os.path.join(indir_path, infile_name)
        infile_path_list.append(file_path)

    return infile_path_list

def make_data_analysis_folders(machine_name: str, experiment_name: str, data_types: list[str]) -> list[str]:
    '''
    Creates folders for storing elaborated data on the based on the machine name, experiment name, and data types.

    Args:
    - machine_name: Name of the machine.
    - experiment_name: Name of the experiment.
    - data_types: List of data types.

    Returns:
    - List of folder paths for each data type.
    '''
    folder_name = "data_analysis"

    code_path = os.getcwd()
    parent_folder = os.path.abspath(os.path.join(code_path, os.pardir))
    folder_path = os.path.join(parent_folder,"experiments_"+ machine_name, experiment_name, folder_name)

    outdir_path_datas = []
    for im in data_types:
        outdir_path_data = os.path.join(folder_path, im)
        outdir_path_datas.append(outdir_path_data)

        if not os.path.exists(outdir_path_data):
            os.makedirs(outdir_path_data)  
    
    return outdir_path_datas    

def make_images_folders(machine_name: str, experiment_name: str, image_types: list[str]) -> list[str]:
    '''
    Creates folders for storing images based on the machine name, experiment name, and image types.

    Args:
    - machine_name: Name of the machine.
    - experiment_name: Name of the experiment.
    - image_types: List of image types.

    Returns:
    - List of folder paths for each image type.
    '''
    folder_name = "images"

    code_path = os.getcwd()
    parent_folder = os.path.abspath(os.path.join(code_path, os.pardir))
    folder_path = os.path.join(parent_folder,"experiments_"+ machine_name, experiment_name, folder_name)

    outdir_path_images = []
    for im in image_types:
        outdir_path_image = os.path.join(folder_path, im)
        outdir_path_images.append(outdir_path_image)

        if not os.path.exists(outdir_path_image):
            os.makedirs(outdir_path_image)  
    
    return outdir_path_images



def output_path_choice(plot, outfile_path=None, formato="jpg"):
    '''
    Saving or showing option for plot functions
    - plot: The plot object to be saved or shown.
    - outfile_path: Path to save the plot.
    - formato: File format for saving the plot.
    '''
    if outfile_path:
        outfile_name = os.path.basename(outfile_path)
        current_title = plt.gca().get_title()
        plt.title(current_title + " " + outfile_name)
        plt.savefig(outfile_path + formato, dpi=300)
        plt.close()  # Close the figure to prevent it from being displayed
        print(f"Plot saved to {outfile_path}")
    else:
        pass
        # plt.show(plot)


def uw_all_plot(data: np.ndarray, metadata: dict, step_wf_to_plot: int,
                highlight_start: int, highlight_end: int, xlim_plot: float,
                ticks_steps_waveforms: float, outfile_path: str = None,  formato: str = ".png")  -> None:
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
    - formato: File format for saving the plot.
    - ticks_steps_waveforms: Step size for ticks on the waveform axis.

    Returns:
    - None
    '''

    time_ax_waveform = metadata["time_ax_waveform"]
    time_ticks_waveforms = np.arange(time_ax_waveform[0], time_ax_waveform[-1], ticks_steps_waveforms)
    
    data  = data[::step_wf_to_plot]
    ymax = 1.3*np.amax(data)
    ymin = 1.3*np.amin(data)

    for idx,waveform in  enumerate(data):
        plt.figure(figsize = (13,4))
        # plt.plot(time_ax_waveform, data.T, 
        #         color = 'black', linewidth = 0.8, alpha = 0.5)
        plt.plot(time_ax_waveform, waveform, 
                color = 'black', linewidth = 0.8, alpha = 0.5)
        plt.plot(time_ax_waveform[highlight_start:highlight_end], waveform[highlight_start:highlight_end], color = 'red')
        plt.xlabel('Time [$\mu s$]', fontsize = 12)
        plt.ylabel('Amplitude [.]', fontsize = 12)
        plt.xticks(time_ticks_waveforms)
        plt.ylim([ymin,ymax])
        plt.xlim(time_ax_waveform[0], xlim_plot)
        plt.grid(alpha = 0.1)
        plt.title("Stacked Waveforms ", fontsize = 20)

        output_path_choice(plot = plt, outfile_path=outfile_path+str(idx), formato=formato)
    

def amplitude_map(data: np.ndarray, metadata: dict, amp_scale: float,
                  formato: str, outfile_path: str = None) -> None:
    '''
    Plots an amplitude map of waveform data.

    Args:
    - outfile_path: Path to save the plot.
    - data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - amp_scale: Scale for amplitude.
    - formato: File format for saving the plot.

    Returns:
    - None
    '''    

    time_ax_waveform = metadata['time_ax_waveform']
    start_time, end_time = time_ax_waveform[0], time_ax_waveform[-1]
    sampling_rate = metadata['sampling_rate']
    number_of_samples = metadata['number_of_samples']
    acquisition_frequency =  metadata['acquition_frequency']

    fig,ax1 = plt.subplots(ncols=1,figsize=(15,8))
    ax1 = plt.subplot()
    ax1.set_title("Amplitude Map " , fontsize = 12)
    cmap = plt.get_cmap('seismic')
    
    im = ax1.imshow(data.T, aspect='auto',origin='lower',interpolation='none',
                    cmap = cmap, vmin = -amp_scale, vmax = amp_scale, 
                    extent = [0,data.shape[0]*acquisition_frequency, 0 , number_of_samples*sampling_rate])
    
    cbar= fig.colorbar(im,pad=0.04)
    cbar.set_label("Relative Amplitude", fontsize = 14)

    ax1.set_xlabel('Time [s]', fontsize = 12)
    ax1.set_ylabel('Wf_time [$\mu s$] ', fontsize = 12)
    ax1.set_xscale

    fig.tight_layout()
    
    output_path_choice(plot = plt, outfile_path=outfile_path, formato=formato)

    

## SPECTRAL ANALYSIS


def lowpass_mask(data_length: int, winlen: int) -> np.ndarray:
    '''
    Builds a low pass filter mask.

    Args:
    - data_length: Length of the data.
    - winlen: Length of the hanning window.

    Returns:
    - Low pass filter mask as a numpy array.
    '''
    mask = np.hanning(winlen)
    mask_start = mask[round(winlen/2):]
    mask_end = mask[:round(winlen/2)]

    lowpass_filter = np.zeros(data_length)
    lowpass_filter[0:len(mask_start)] = mask_start
    lowpass_filter[data_length-len(mask_end):] = mask_end

    return lowpass_filter


def amplitude_spectrum_map(signal_freqs: np.ndarray, amp_spectrum: np.ndarray,
                           metadata: dict, outfile_path: str = None, formato: str = ".png") -> None:
    '''
    Plots an amplitude map of waveform data.

    Args:
    - outfile_path: Path to save the plot.
    - signal_freqs: Frequencies of the signal.
    - amp_spectrum: Amplitude spectrum.
    - metadata: Metadata dictionary containing information about the data.
    - formato: File format for saving the plot. Default is PNG.

    Returns:
    - None
    '''
    wave_num, wave_len = amp_spectrum.shape
    
    spectrum_length = round(wave_len/2)
    signal_freqs = signal_freqs[:spectrum_length]
    amp_spectrum = amp_spectrum[:, :spectrum_length]

    time_ax_acquisition = metadata['time_ax_acquisition']
    plt.pcolormesh(time_ax_acquisition, signal_freqs, amp_spectrum.T, cmap="gist_gray", norm=mcolors.LogNorm())   
    plt.title('Amplitude Map', fontsize=12)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [$MHz $]")
    cbar = plt.colorbar(pad=0.04)
    cbar.set_label("Spectral Amplitude", fontsize=14)

    output_path_choice(plot = plt, outfile_path=outfile_path, formato=formato)




def signal2noise_separation_lowpass(waveform_data: np.ndarray, metadata: dict, freq_cut: float = 5, outfile_path: str = None, formato:str = ".png") -> tuple[np.ndarray, np.ndarray]:
    '''
    Separates signal and noise using a low-pass filter.

    Args:
    - outfile_path: Path to save the plot.
    - waveform_data: Numpy array representing waveform data.
    - metadata: Metadata dictionary containing information about the data.
    - freq_cut: Frequency cutoff for the low-pass filter. Default is 5.

    Returns:
    - Tuple containing the filtered signal and the reconstructed noise.
    '''
    time_ax_waveform = metadata['time_ax_waveform']
    sampling_rate = metadata['sampling_rate']

    try:
        wave_num, wave_len = waveform_data.shape
    except ValueError:
        wave_num = 1
        wave_len = len(waveform_data)

    amp_spectrum = np.abs(np.fft.fft(waveform_data))
    phase_spectrum = np.angle(np.fft.fft(waveform_data))
    signal_freqs = np.fft.fftfreq(wave_len, sampling_rate)

    winlen = int((wave_len * freq_cut) / np.amax(signal_freqs))
    lowpass_filter = lowpass_mask(wave_len, winlen)

    filtered_amp_spectrum = lowpass_filter * amp_spectrum
    filtered_fourier = filtered_amp_spectrum * np.exp(1j * phase_spectrum)
    filtered_signal = np.fft.ifft(filtered_fourier).real

    noise_spectrum = amp_spectrum - filtered_amp_spectrum
    noise_fourier = noise_spectrum * np.exp(1j * phase_spectrum)
    noise_reconstructed = np.fft.ifft(noise_fourier).real

    if wave_num == 1:
        filtered_amp_and_phase_spectrum_plot(signal_freqs, amp_spectrum, phase_spectrum,
                                               filtered_amp_spectrum, lowpass_filter, sampling_rate, 
                                               outfile_path=outfile_path, formato=formato)
    else:
        amplitude_spectrum_map(signal_freqs, filtered_amp_spectrum[:, :winlen], metadata, outfile_path=outfile_path,
                                formato=formato)

    return filtered_signal, noise_reconstructed


def filtered_amp_and_phase_spectrum_plot(signal_freqs: np.ndarray, amp_spectrum: np.ndarray,
                                          phase_spectrum: np.ndarray, filtered_amp_spectrum: np.ndarray,
                                          lowpass_filter: np.ndarray, sampling_rate: float, 
                                           outfile_path: str = None,  formato : str = ".png") -> None:
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

    f, ax = plt.subplots(2, 1, figsize=(13, 8))

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
    ax[0].set_title("Amplitude Spectrum of a waveform")

    ax[1].plot(signal_freqs, phase_spectrum)

    ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
    ax[1].set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    ax[1].set_ylabel("Phase [rad]")
    ax[1].set_xlabel("Frequency [MHz]")
    ax[1].set_title("Phase Spectrum of a waveform")

    plt.tight_layout()

    output_path_choice(plot = plt, outfile_path=outfile_path, formato=formato)





def signal_vs_filtered_signal_plot(single_waveform: np.ndarray, 
                                   single_waveform_filtered: np.ndarray, metadata: dict[str, np.ndarray], 
                                   freq_cut: float, formato: str = ".png", outfile_path: str = None) -> None:
    """
    Plot a comparison between a waveform and its filtered version.

    Parameters:
        outfile_path (str): The path to save the plot.
        single_waveform (np.ndarray): The original waveform data.
        single_waveform_filtered (np.ndarray): The filtered waveform data.
        metadata (Dict[str, np.ndarray]): Metadata containing information about the waveform.
        formato (str, optional): The format of the output file (default is ".png").

    Returns:
        None
    """
    time_ax = metadata['time_ax_waveform']
    plt.plot(time_ax, single_waveform, color="lightgray", label="waveform")
    plt.plot(time_ax, single_waveform_filtered, color="black", label="filtered waveform")
    plt.xlabel('Time [$\mu s$]')
    plt.ylabel('Amplitude [.]')
    plt.title("Effect of lowpass filtering at %2.2f MHz" % freq_cut)
    plt.legend()

    output_path_choice(plot = plt, outfile_path=outfile_path, formato=formato)


    
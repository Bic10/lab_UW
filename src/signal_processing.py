# src/signal_processing.py

import numpy as np
from math import ceil
from plotting import *

# WAVEFORMS PREPROCESSING
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

# SPECTRAL ANALYSIS
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

def signal2noise_separation_lowpass(waveform_data: np.ndarray, metadata: dict, freq_cut: float = 5, outfile_path: str = None, format:str = ".png") -> tuple[np.ndarray, np.ndarray]:
    '''
    !!! IT MUST BE IMPROVED: DINSENTAGLED THE PLOTTING FUNCTIONALITY FROM THE LAWASS ONE
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

    # if wave_num == 1:
    #     filtered_amp_and_phase_spectrum_plot(signal_freqs, amp_spectrum, phase_spectrum,
    #                                            filtered_amp_spectrum, lowpass_filter, sampling_rate, 
    #                                            outfile_path=outfile_path)
    # else:
    #     amplitude_spectrum_map(signal_freqs, filtered_amp_spectrum[:, :winlen], metadata, outfile_path=outfile_path)

    return filtered_signal, noise_reconstructed

# WAVELETS IDENTIFICATION
def sta_lta(waveform, sta_window, lta_window, energy=True):
    """
    Implements the STA/LTA algorithm on a 1D signal.

    Args:
    - signal: 1D numpy array representing the input signal.
    - sta_window: Length of the short-term average window (in samples).
    - lta_window: Length of the long-term average window (in samples).

    Returns:
    - A numpy array representing the STA/LTA ratio.
    """

    if energy:
        waveform = waveform**2      # do the sta/lta on the energy of the signal

    # Calculate STA (Short-Term Average)
    sta = np.convolve(np.abs(waveform), np.ones(sta_window), mode='same') / sta_window

    # Calculate LTA (Long-Term Average)
    lta = np.convolve(np.abs(waveform), np.ones(lta_window), mode='same') / lta_window

    # Calculate STA/LTA ratio
    sta_lta_ratio = sta / lta

    return sta_lta_ratio


def select_wavelets_given_known_numbers_of_them(waveform, chunk_n=5, offset=0, tolerance=0.1):
    """
    Find the relative argmax along with the minimums before and after the maximum within defined waveform chunks.

    Parameters:
    - waveform (array_like): The input waveform.
    - chunk_n (int, optional): Number of chunks to divide the waveform into. Default is 5.
    - offset (int, optional): Offset value to start indexing the waveform. Default is 0.
    - tolerance (float, optional): Tolerance value to consider when searching for minimums around the maximum.
                                   Default is 0.1.

    Returns:
    - index_max_list (list): List of indices of maximum values within each chunk.
    - index_min_before_list (list): List of indices of minimum values before each maximum within each chunk.
    - index_min_after_list (list): List of indices of minimum values after each maximum within each chunk.
    """

    chunk_len = ceil(len(waveform) / chunk_n)
    index_max_list = []
    index_min_before_list = []
    index_min_after_list = []

    for chunk in range(chunk_n):
        start = offset + chunk_len * chunk
        end = start + chunk_len
        chunk_max = np.amax(waveform[start:end])
        idx_chunk_max = np.argmax(waveform[start:end])

        # For each chunk, these are the absolute minimum before and after the maximum.
        chunk_min_before = np.amin(waveform[start:start + idx_chunk_max])
        idx_chunk_min_before = np.argmin(waveform[start:start + idx_chunk_max])

        chunk_min_after = np.amin(waveform[start + idx_chunk_max:end])
        idx_chunk_min_after = idx_chunk_max + np.argmin(waveform[start + idx_chunk_max:end])

        # Iteratively find minimums around the maximum considering a tolerance value.
        search_interval = idx_chunk_max - idx_chunk_min_before
        mask_wave = chunk_max * np.ones(len(waveform[start:end]))

        for idx in range(2, search_interval, 5):
            mask_wave[idx_chunk_max - idx:idx_chunk_max] = waveform[start + idx_chunk_max - idx:start + idx_chunk_max]
            difference = np.amin(mask_wave) - chunk_min_before
            if difference < tolerance:
                idx_chunk_min_before = np.argmin(mask_wave)
                break

        search_interval = idx_chunk_min_after - idx_chunk_max
        mask_wave = chunk_max * np.ones(len(waveform[start:end]))

        for idx in range(2, search_interval, 5):
            mask_wave[idx_chunk_max:idx_chunk_max + idx] = waveform[start + idx_chunk_max:start + idx_chunk_max + idx]
            difference = np.amin(mask_wave) - chunk_min_after
            if difference < tolerance:
                idx_chunk_min_after = np.argmin(mask_wave)
                break

        index_max = idx_chunk_max + start
        index_min_before = idx_chunk_min_before + start
        index_min_after = idx_chunk_min_after + start

        index_max_list.append(index_max)
        index_min_before_list.append(index_min_before)
        index_min_after_list.append(index_min_after)

    return index_max_list, index_min_before_list, index_min_after_list


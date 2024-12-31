# lab_uw/signal_processing.py

import numpy as np
from typing import Tuple, List, Optional
from math import ceil

from lab_uw.plotting import Plotter

class SignalProcessor:
    """
    A collection of signal processing functions for waveform data.
    """

    @staticmethod
    def fourier_derivative_2nd(f: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute the second spatial derivative using the Fourier transform.

        Args:
            f (np.ndarray): Input function.
            dx (float): Grid spacing.

        Returns:
            np.ndarray: Second spatial derivative.
        """
        nx = f.size
        kmax = np.pi / dx
        dk = kmax / (nx / 2)
        k = np.arange(nx)
        k[:nx // 2] *= dk
        k[nx // 2:] = (k[:nx // 2] - kmax)

        ff = np.fft.fft(f)
        ff = (1j * k) ** 2 * ff
        df_num = np.real(np.fft.ifft(ff))
        return df_num

    @staticmethod
    def remove_starting_noise(
        data: np.ndarray,
        metadata: dict,
        remove_initial_samples: int = 0
    ) -> Tuple[np.ndarray, dict]:
        '''
        Removes initial samples from the data and updates metadata accordingly.

        Args:
            data (np.ndarray): 2D array of shape (n_waveforms, n_samples).
            metadata (dict): Metadata dictionary containing 'number_of_samples' and 'time_ax_waveform'.
            remove_initial_samples (int): Number of initial samples to remove.

        Returns:
            Tuple[np.ndarray, dict]: The modified data array and updated metadata dictionary.

        Raises:
            TypeError: If data is not a numpy array.
            ValueError: If data is not 2D or remove_initial_samples exceeds data length.
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array.")
        if data.ndim != 2:
            raise ValueError("data must be a 2D numpy array of shape (n_waveforms, n_samples).")

        n_samples = data.shape[1]
        if remove_initial_samples > n_samples:
            raise ValueError("remove_initial_samples cannot exceed the number of samples in data.")

        data = data[:, remove_initial_samples:]

        metadata = metadata.copy()
        metadata['number_of_samples'] -= remove_initial_samples
        metadata['time_ax_waveform'] = metadata['time_ax_waveform'][remove_initial_samples:]

        return data, metadata

    @staticmethod
    def lowpass_mask(freqs: np.ndarray, 
                     freq_cut: float, 
                     transition_width: float = 1) -> np.ndarray:
        '''
        Builds a smooth low-pass filter mask using a Hanning window in the frequency domain.

        Args:
            freqs (np.ndarray): Array of frequency bins from FFT.
            freq_cut (float): Cutoff frequency in Hz.
            transition_width (float): Width of the transition region around the cutoff frequency in Hz.

        Returns:
            np.ndarray: Low-pass filter mask with smooth transition.
        '''
        if not isinstance(freqs, np.ndarray):
            raise TypeError("freqs must be a numpy array.")
        if freq_cut <= 0:
            raise ValueError("freq_cut must be a positive number.")
        if transition_width <= 0:
            raise ValueError("transition_width must be a positive number.")
        
        # Initialize the filter as zeros
        lowpass_filter = np.zeros_like(freqs)
        
        # Frequencies below the cutoff minus half of the transition width have full pass
        lowpass_filter[np.abs(freqs) < (freq_cut - transition_width / 2)] = 1.0

        # Apply a Hanning window in the transition region
        transition_idx = (np.abs(freqs) >= (freq_cut - transition_width / 2)) & (np.abs(freqs) <= (freq_cut + transition_width / 2))
        transition_size = np.sum(transition_idx)
        
        if transition_size > 1:  # Ensure there are enough points for the window
            hanning_window = np.hanning(transition_size)
            half_hanning_window = hanning_window[int(transition_size/2):]
            flipped_half_hanning_window = np.flip(half_hanning_window)
            specular_hanning_window_for_stupid_fft_frequency_implementation = np.concatenate([half_hanning_window,flipped_half_hanning_window])
            # For negative frequencies, reverse the first half of the window
            lowpass_filter[transition_idx] = specular_hanning_window_for_stupid_fft_frequency_implementation
        
        return lowpass_filter

    @staticmethod
    def signal2noise_separation_lowpass(
        waveform_data: np.ndarray,
        metadata: dict,
        freq_cut: float = 5.0,
        transition_width: float = 1,
        plotting = False,
        outfile_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Separates signal and noise using a low-pass filter.

        Args:
            waveform_data (np.ndarray): Waveform data array of shape (n_waveforms, n_samples) or (n_samples,).
            metadata (dict): Metadata containing 'sampling_rate' and 'time_ax_waveform'.
            freq_cut (float): Frequency cutoff for the low-pass filter in Hz.
            transition_width (float): Width of the transition region around the cutoff frequency in Hz.
            plot_debug (bool): If True, plots the filter and spectra for debugging.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered signal and reconstructed noise arrays.

        Raises:
            TypeError: If waveform_data is not a numpy array.
            ValueError: If waveform_data is not 1D or 2D numpy array.
            KeyError: If required keys are missing in metadata.
        '''
        if not isinstance(waveform_data, np.ndarray):
            raise TypeError("waveform_data must be a numpy array.")
        if waveform_data.ndim not in [1, 2]:
            raise ValueError("waveform_data must be a 1D or 2D numpy array.")

        required_keys = ['sampling_rate', 'time_ax_waveform']
        for key in required_keys:
            if key not in metadata:
                raise KeyError(f"Missing '{key}' in metadata.")

        sampling_rate = metadata['sampling_rate']
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive number.")

        # Ensure waveform_data is 2D
        if waveform_data.ndim == 1:
            waveform_data = waveform_data[np.newaxis, :]

        n_waveforms, n_samples = waveform_data.shape

        # Compute frequency axis
        freqs = np.fft.fftfreq(n_samples, d=sampling_rate)

        # Create low-pass filter mask using the updated lowpass_mask function
        lowpass_filter = SignalProcessor.lowpass_mask(freqs, freq_cut, transition_width=transition_width)

        # Apply the mask to the FFT of the data
        fft_data = np.fft.fft(waveform_data, axis=1)
        filtered_fft = fft_data * lowpass_filter

        # Inverse FFT to get the filtered signal
        filtered_signal = np.fft.ifft(filtered_fft, axis=1).real

        # Compute the noise by subtracting the filtered signal from the original data
        noise = waveform_data - filtered_signal

        # Squeeze the output if input was 1D
        filtered_signal = filtered_signal.squeeze()
        noise = noise.squeeze()

        if plotting:
            plotter = Plotter()

            if n_waveforms == 1:
                amp_spectrum = np.abs(fft_data.squeeze())
                phase_spectrum = np.angle(fft_data.squeeze())
                filtered_amp_spectrum = np.abs(filtered_fft.squeeze())
                plotter.filtered_amp_and_phase_spectrum_plot(signal_freqs=freqs,
                                                             amp_spectrum=amp_spectrum,
                                                             phase_spectrum=phase_spectrum,
                                                             filtered_amp_spectrum=filtered_amp_spectrum,
                                                             lowpass_filter=lowpass_filter,
                                                             freq_cut=freq_cut,
                                                             outfile_path=outfile_path)
            
            else:
                filtered_amp_spectrum = np.abs(filtered_fft)
                plotter.amplitude_spectrum_map(signal_freqs=freqs,
                                            amp_spectrum=filtered_amp_spectrum,
                                            metadata=metadata,
                                            freq_cut = freq_cut,
                                            outfile_path=outfile_path)
                
                plotter.amplitude_spectrum_distribution(signal_freqs=freqs,
                                            amp_spectrum=filtered_amp_spectrum,
                                            metadata=metadata,
                                            freq_cut = freq_cut,
                                            outfile_path=outfile_path + "_distribution")

        return filtered_signal, noise

    @staticmethod
    def sta_lta(
        waveform: np.ndarray,
        sta_window: int,
        lta_window: int,
        energy: bool = True
    ) -> np.ndarray:
        """
        Implements the STA/LTA algorithm on a 1D signal.

    Args:
    - signal: 1D numpy array representing the input signal.
    - sta_window: Length of the short-term average window (in samples).
    - lta_window: Length of the long-term average window (in samples).

        Returns:
            np.ndarray: A numpy array representing the STA/LTA ratio.

        Raises:
            TypeError: If waveform is not a numpy array.
            ValueError: If waveform is not 1D or if window sizes are invalid.
        """
        if not isinstance(waveform, np.ndarray):
            raise TypeError("waveform must be a numpy array.")
        if waveform.ndim != 1:
            raise ValueError("waveform must be a 1D numpy array.")
        if not isinstance(sta_window, int) or sta_window <= 0:
            raise ValueError("sta_window must be a positive integer.")
        if not isinstance(lta_window, int) or lta_window <= 0:
            raise ValueError("lta_window must be a positive integer.")
        if sta_window > lta_window:
            raise ValueError("sta_window should be less than or equal to lta_window.")

        if energy:
            waveform = waveform.copy() ** 2  # Avoid modifying the original waveform

        # Calculate STA (Short-Term Average)
        sta = np.convolve(np.abs(waveform), np.ones(sta_window), mode='same') / sta_window

        # Calculate LTA (Long-Term Average)
        lta = np.convolve(np.abs(waveform), np.ones(lta_window), mode='same') / lta_window

        # Avoid division by zero
        epsilon = 1e-10
        sta_lta_ratio = sta / (lta + epsilon)

        return sta_lta_ratio

    @staticmethod
    def select_wavelets_given_known_numbers_of_them(
        waveform: np.ndarray,
        chunk_n: int = 5,
        offset: int = 0,
        tolerance: float = 0.1
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Finds the indices of maximum values and corresponding minima before and after each maximum within defined waveform chunks.

        Args:
            waveform (np.ndarray): 1D array representing the input waveform.
            chunk_n (int, optional): Number of chunks to divide the waveform into. Default is 5.
            offset (int, optional): Offset value to start indexing the waveform. Default is 0.
            tolerance (float, optional): Tolerance for refining minima around the maximum. Default is 0.1.

        Returns:
            Tuple[List[int], List[int], List[int]]: Lists of indices for maxima, minima before, and minima after each maximum.

        Raises:
            TypeError: If waveform is not a numpy array.
            ValueError: If waveform is not 1D.
        """
        if not isinstance(waveform, np.ndarray):
            raise TypeError("waveform must be a numpy array.")
        if waveform.ndim != 1:
            raise ValueError("waveform must be a 1D numpy array.")
        if not isinstance(chunk_n, int) or chunk_n <= 0:
            raise ValueError("chunk_n must be a positive integer.")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        if not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("tolerance must be a non-negative number.")

        chunk_len = ceil(len(waveform) / chunk_n)
        index_max_list = []
        index_min_before_list = []
        index_min_after_list = []

        for chunk in range(chunk_n):
            start = offset + chunk_len * chunk
            end = min(start + chunk_len, len(waveform))
            chunk_waveform = waveform[start:end]

            if len(chunk_waveform) == 0:
                continue

            # Find the maximum in the chunk
            idx_chunk_max = np.argmax(chunk_waveform)
            index_max = idx_chunk_max + start

            # Find minima before the maximum
            if idx_chunk_max > 0:
                idx_chunk_min_before = np.argmin(chunk_waveform[:idx_chunk_max])
                index_min_before = idx_chunk_min_before + start
            else:
                index_min_before = index_max

            # Find minima after the maximum
            if idx_chunk_max < len(chunk_waveform) - 1:
                idx_chunk_min_after = idx_chunk_max + np.argmin(chunk_waveform[idx_chunk_max + 1:]) + 1
                index_min_after = idx_chunk_min_after + start
            else:
                index_min_after = index_max

            index_max_list.append(index_max)
            index_min_before_list.append(index_min_before)
            index_min_after_list.append(index_min_after)

        return index_max_list, index_min_before_list, index_min_after_list

# lab_uw/plotting.py

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from tkinter import Button
import pickle

class Plotter:
    """
    A class containing methods for plotting various aspects of waveform data and simulations.
    """

    # Define default settings for plots
    FONT_TYPE = "Ubuntu"
    FONT_SIZE = 30
    FIGURE_SIZE = (16, 8)
    FORMAT = ".png"

    # Define the color palette for the plots
    COLORS = {
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

    DEFAULT_SETTINGS = {
        'colors': COLORS,
        'fontsize_title': FONT_SIZE,
        'fontsize_subplot_title': int(0.7 * FONT_SIZE),
        'fontsize_labels': int(0.7 * FONT_SIZE),
        'fontsize_ticks': int(0.5 * FONT_SIZE),
        'line_width': 1.0,
        'figure_size': FIGURE_SIZE,
        'format': FORMAT,
    }

    def __init__(self, settings: Optional[Dict] = None):
        if settings is None:
            self.settings = self.DEFAULT_SETTINGS.copy()
        else:
            self.settings = {**self.DEFAULT_SETTINGS, **settings}

    def output_path_choice(
        self,
        fig: plt.Figure,
        outfile_path: Optional[Union[str, Path]] = None,
        format: Optional[str] = None
    ) -> None:
        """
        Save or display the figure based on the provided outfile_path.
        """
        if format is None:
            format = self.settings['format']

        if outfile_path:
            outfile_path = Path(outfile_path)
            if outfile_path.suffix != format:
                outfile_path = outfile_path.with_suffix(format)
            fig.savefig(outfile_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

    def uw_all_plot(self,
                    data: np.ndarray,
                    metadata: Dict,
                    step_wf_to_plot: int,
                    highlight_start: int,
                    highlight_end: int,
                    xlim_plot: float,
                    ticks_steps_waveforms: float,
                    outfile_path: Optional[str] = None) -> None:
        """
        Example method for plotting stacked waveforms with optional highlighting.
        """
        if data.ndim != 2:
            raise ValueError("data must be a 2D numpy array.")
        if 'time_ax_waveform' not in metadata:
            raise KeyError("metadata must contain 'time_ax_waveform'.")

        time_ax_waveform = metadata["time_ax_waveform"]
        if highlight_start < 0 or highlight_end > data.shape[0]:
            raise ValueError("highlight_start and highlight_end must be within the range of data indices.")

        time_ticks_waveforms = np.arange(time_ax_waveform[0], time_ax_waveform[-1], ticks_steps_waveforms)
        data_to_plot = data[::step_wf_to_plot]
        ymax = 1.3 * np.amax(data_to_plot)
        ymin = 1.3 * np.amin(data_to_plot)

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        ax.plot(time_ax_waveform, data_to_plot.T, color='black', linewidth=0.8, alpha=0.5)
        ax.plot(time_ax_waveform, data[highlight_start:highlight_end].T, color='red')

        ax.set_xlabel('Time [$\\mu s$]', fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel('Amplitude [a.u.]', fontsize=self.settings['fontsize_labels'])
        ax.set_xticks(time_ticks_waveforms)
        ax.set_ylim([ymin, ymax])
        ax.set_xlim(time_ax_waveform[0], xlim_plot)
        ax.grid(alpha=0.1)
        ax.set_title("Stacked Waveforms", fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def amplitude_map(self,
                      data: np.ndarray,
                      metadata: Dict,
                      outfile_path: Optional[str] = None) -> None:
        """
        Plots an amplitude map of waveform data.

        Args:
            data (np.ndarray): 2D array representing waveform data (n_waveforms, n_samples).
            metadata (Dict): Metadata dictionary containing necessary keys.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            KeyError: If required keys are missing in metadata.
        """
        required_keys = ['time_ax_waveform', 'sampling_rate', 'number_of_samples', 'acquisition_frequency']
        for key in required_keys:
            if key not in metadata:
                raise KeyError(f"Missing '{key}' in metadata.")

        time_ax_waveform = metadata['time_ax_waveform']
        first_sample_time = time_ax_waveform[0]
        last_sample_time = time_ax_waveform[-1]
        time_ax_acquisition = metadata['time_ax_acquisition']
        first_waveform_time = time_ax_acquisition[0]
        last_waveform_time = time_ax_acquisition[-1]

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        cmap = plt.get_cmap('seismic')

        extent = [first_waveform_time, last_waveform_time, first_sample_time, last_sample_time]

        amp_scale_limit = max(-np.amin(data),np.amax(data))
        im = ax.imshow(data.T, aspect='auto', origin='lower', interpolation='none',
                       cmap=cmap, vmin=-amp_scale_limit, vmax=amp_scale_limit, extent=extent)

        cbar = fig.colorbar(im, pad=0.04)
        cbar.set_label("Relative Amplitude", fontsize=self.settings['fontsize_labels'])

        ax.set_title("Amplitude Map", fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax.set_xlabel('Experiment Time [s]', fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel('Waveform Travel Time [$\\mu s$]', fontsize=self.settings['fontsize_labels'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)


    def amplitude_spectrum_map(
        self,
        signal_freqs: np.ndarray,
        amp_spectrum: np.ndarray,
        metadata: Dict,
        freq_cut: float,
        outfile_path: Optional[str] = None
    ) -> None:
        """
        Plots an amplitude spectrum map of waveform data.

        Parameters
        ----------
        signal_freqs : np.ndarray
            Frequencies of the signal (size n_samples).
        amp_spectrum : np.ndarray
            Amplitude spectrum of shape (n_waveforms, n_samples).
        metadata : dict
            Metadata dictionary containing 'time_ax_acquisition'.
        freq_cut : float
            Maximum frequency to display on the y-axis (in MHz).
        outfile_path : str, optional
            Path to save the plot.
        
        Raises
        ------
        KeyError
            If 'time_ax_acquisition' is missing in metadata.
        ValueError
            If dimensions of amp_spectrum and signal_freqs do not match or if amp_spectrum is not 2D.
        """
        if 'time_ax_acquisition' not in metadata:
            raise KeyError("metadata must contain 'time_ax_acquisition'.")

        if amp_spectrum.ndim != 2:
            raise ValueError("amp_spectrum must be a 2D numpy array.")

        time_ax_acquisition = metadata['time_ax_acquisition']
        wave_num, wave_len = amp_spectrum.shape

        # Only plot half of the spectrum if it's symmetrical (e.g., for real signals)
        spectrum_length = wave_len // 2
        signal_freqs = signal_freqs[:spectrum_length]
        amp_spectrum = amp_spectrum[:, :spectrum_length]

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Using LogNorm to highlight wide dynamic range
        pcm = ax.pcolormesh(
            time_ax_acquisition,
            signal_freqs,
            amp_spectrum.T,
            cmap="plasma",
            norm=mcolors.LogNorm(vmin=1e-3, vmax=amp_spectrum.max())
        )
        ax.set_ylim([0, freq_cut])
        ax.set_title('Amplitude Spectrum Map', fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax.set_xlabel("Time [s]", fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel("Frequency [MHz]", fontsize=self.settings['fontsize_labels'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        cbar = fig.colorbar(pcm, pad=0.04)
        cbar.set_label("Spectral Amplitude", fontsize=self.settings['fontsize_labels'])

        fig.tight_layout()

        # Use your existing output path choice method
        self.output_path_choice(fig=fig, outfile_path=outfile_path)


    def amplitude_spectrum_distribution(
        self,
        signal_freqs: np.ndarray,
        amp_spectrum: np.ndarray,
        metadata: Dict,
        freq_cut: float,
        outfile_path: Optional[str] = None
    ) -> None:
        """
        Provides alternative visualizations to see how the amplitude spectrum
        changes across waveforms in the experiment.

        Parameters
        ----------
        signal_freqs : np.ndarray
            Frequencies of the signal (n_samples).
        amp_spectrum : np.ndarray
            Amplitude spectrum of shape (n_waveforms, n_samples).
        metadata : Dict
            Metadata dictionary containing 'time_ax_acquisition'.
        freq_cut : float
            Maximum frequency to display on the y-axis (in MHz).
        outfile_path : str, optional
            Path to save the resulting figure. If None, the figure is shown.

        Notes
        -----
        - This function creates three subplots:
        1) Line plot (with alpha-blended lines),
        2) 2D histogram using a log color scale,
        3) The same 2D histogram using a linear color scale.
        - For large datasets, the alpha-blended line plot may be slow. The 2D
        histograms can reveal hidden structures more efficiently.
        """
        if 'time_ax_acquisition' not in metadata:
            raise KeyError("metadata must contain 'time_ax_acquisition'.")

        # Check dimensions
        if amp_spectrum.ndim != 2:
            raise ValueError("amp_spectrum must be a 2D numpy array.")
        wave_num, wave_len = amp_spectrum.shape

        # Only use half the spectrum if you have symmetrical data
        spectrum_length = wave_len // 2
        signal_freqs = signal_freqs[:spectrum_length]
        amp_spectrum = amp_spectrum[:, :spectrum_length]

        time_ax_acquisition = metadata['time_ax_acquisition']
        if len(time_ax_acquisition) != wave_num:
            raise ValueError("time_ax_acquisition length does not match amp_spectrum waveforms.")

        # We only plot up to freq_cut
        freq_mask = signal_freqs <= freq_cut
        freq_subset = signal_freqs[freq_mask]
        amp_spectrum_subset = amp_spectrum[:, freq_mask]

        # Create the figure and subplots
        fig, axes = plt.subplots(nrows=3, figsize=self.settings['figure_size'], constrained_layout=True)

        # -- 1) Line Plot with alpha --
        # Each waveform: X axis = frequencies, Y axis = amplitude
        # We'll transpose so lines go freq -> amplitude across wave_num series.
        # This can be slow for large wave_num, so alpha is used to help reveal structure.
        axes[0].plot(freq_subset, amp_spectrum_subset.T, color="C0", alpha=0.05)
        axes[0].set_title("Line Plot with Alpha", fontsize=self.settings['fontsize_title'])
        axes[0].set_xlabel("Frequency [MHz]", fontsize=self.settings['fontsize_labels'])
        axes[0].set_ylabel("Amplitude [a.u.]", fontsize=self.settings['fontsize_labels'])
        axes[0].tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        axes[0].set_xlim([0, freq_cut])

        # -- 2) 2D Histogram with log color scale --
        # Flatten time and freq so we can call np.histogram2d(x, y).
        # 'x' ~ freq, 'y' ~ amplitude, but let's do:
        # x -> freq, y -> amplitude values or log(amplitude).
        # Or, we can do a quick partial "interpolation" approach as in the snippet you shared.
        # For demonstration, let's just flatten.
        freq_flat = np.broadcast_to(freq_subset, amp_spectrum_subset.shape).ravel()
        amp_flat = amp_spectrum_subset.ravel()

        # Build 2D histogram
        # You can tune bins here for freq and amplitude scale
        # Implement fridman-diaconis should be more robust
        num_freq_bins = 200
        num_amp_bins = 200

        # We do amplitude in log scale to help spread out the dynamic range
        amp_flat_positive = amp_flat[amp_flat > 0]  # must be positive for log scale
        freq_flat_positive = freq_flat[amp_flat > 0]

        # Convert amplitude to dB scale or keep it linear - let's keep linear for demonstration
        # but must handle 0 or negative values carefully for log norm
        freq_bins = np.linspace(0, freq_cut, num_freq_bins)
        # for amplitude, pick a range that captures the data well:
        amp_min, amp_max = amp_flat_positive.min(), amp_flat_positive.max()
        amp_bins = np.logspace(np.log10(amp_min), np.log10(amp_max), num_amp_bins)

        h, xedges, yedges = np.histogram2d(
            freq_flat_positive, amp_flat_positive, bins=[freq_bins, amp_bins]
        )

        pcm = axes[1].pcolormesh(
            xedges, yedges, h.T,
            cmap="plasma",
            norm=mcolors.LogNorm(vmin=1, vmax=h.max()),
            rasterized=True
        )
        axes[1].set_title("2D Histogram (Log Colorscale)", fontsize=self.settings['fontsize_title'])
        axes[1].set_xlabel("Frequency [MHz]", fontsize=self.settings['fontsize_labels'])
        axes[1].set_ylabel("Amplitude [a.u.]", fontsize=self.settings['fontsize_labels'])
        axes[1].tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        axes[1].set_xlim([0, freq_cut])
        axes[1].set_ylim([amp_min, amp_max])
        cbar = fig.colorbar(pcm, ax=axes[1], pad=0.01)
        cbar.set_label("# Points (log scale)", fontsize=self.settings['fontsize_labels'])

        # -- 3) Same 2D histogram, linear color scale --
        pcm2 = axes[2].pcolormesh(
            xedges, yedges, h.T,
            cmap="plasma",
            vmax=h.max(),
            rasterized=True
        )
        axes[2].set_title("2D Histogram (Linear Colorscale)", fontsize=self.settings['fontsize_title'])
        axes[2].set_xlabel("Frequency [MHz]", fontsize=self.settings['fontsize_labels'])
        axes[2].set_ylabel("Amplitude [a.u.]", fontsize=self.settings['fontsize_labels'])
        axes[2].tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        axes[2].set_xlim([0, freq_cut])
        axes[2].set_ylim([amp_min, amp_max])
        cbar2 = fig.colorbar(pcm2, ax=axes[2], pad=0.01)
        cbar2.set_label("# Points (linear scale)", fontsize=self.settings['fontsize_labels'])

        # Tweak layout or call fig.tight_layout() if constrained_layout is off
        # Save or show
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def filtered_amp_and_phase_spectrum_plot(self,
                                             signal_freqs: np.ndarray,
                                             amp_spectrum: np.ndarray,
                                             phase_spectrum: np.ndarray,
                                             filtered_amp_spectrum: np.ndarray,
                                             lowpass_filter: np.ndarray,
                                             freq_cut: float,
                                             outfile_path: Optional[str] = None) -> None:
        """
        Plots the filtered amplitude and phase spectrum.

        Args:
            signal_freqs (np.ndarray): Frequencies of the signal.
            amp_spectrum (np.ndarray): Amplitude spectrum.
            phase_spectrum (np.ndarray): Phase spectrum.
            filtered_amp_spectrum (np.ndarray): Filtered amplitude spectrum.
            lowpass_filter (np.ndarray): Low-pass filter mask.
            freq_cut (float): Cut-off frequency.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If input arrays are not 1D or lengths do not match.
        """
        arrays = [signal_freqs, amp_spectrum, phase_spectrum, filtered_amp_spectrum, lowpass_filter]
        if not all(arr.ndim == 1 for arr in arrays):
            raise ValueError("All input arrays must be 1D numpy arrays.")
        if not all(len(arr) == len(signal_freqs) for arr in arrays):
            raise ValueError("All input arrays must have the same length.")

        spectrum_length = len(signal_freqs) // 2
        amp_spectrum = amp_spectrum[:spectrum_length]
        phase_spectrum = phase_spectrum[:spectrum_length]
        filtered_amp_spectrum = filtered_amp_spectrum[:spectrum_length]
        signal_freqs = signal_freqs[:spectrum_length]
        lowpass_filter = lowpass_filter[:spectrum_length]

        max_freq = signal_freqs[np.argmax(amp_spectrum)]

        fig, ax = plt.subplots(2, 1, figsize=self.settings['figure_size'])

        ax[0].semilogy(signal_freqs, amp_spectrum, label="Amplitude Spectrum")
        ax[0].semilogy(signal_freqs, lowpass_filter * np.amax(amp_spectrum), label="Filter Shape")
        ax[0].semilogy(signal_freqs, filtered_amp_spectrum, label="Filtered Amplitude Spectrum")
        ax[0].vlines(max_freq, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r", "--",
                     label=f"Max Spectrum = {max_freq:.2f} MHz")
        ax[0].vlines(freq_cut, np.amin(filtered_amp_spectrum), np.amax(filtered_amp_spectrum), "r", "-",
                     label=f"Cut-off Frequency = {freq_cut} MHz")
        ax[0].legend(fontsize=self.settings['fontsize_ticks'])
        ax[0].set_xlim([0, np.amax(signal_freqs)])
        ax[0].set_ylabel("Amplitude [a.u.]", fontsize=self.settings['fontsize_labels'])
        ax[0].set_xlabel("Frequency [MHz]", fontsize=self.settings['fontsize_labels'])
        ax[0].set_title("Amplitude Spectrum of a Waveform", fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax[0].tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        ax[1].plot(signal_freqs, phase_spectrum)
        ax[1].set_xlim([0, np.amax(signal_freqs)])
        ax[1].set_yticks(np.linspace(-np.pi, np.pi, 5))
        ax[1].set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax[1].set_ylabel("Phase [rad]", fontsize=self.settings['fontsize_labels'])
        ax[1].set_xlabel("Frequency [MHz]", fontsize=self.settings['fontsize_labels'])
        ax[1].set_title("Phase Spectrum of a Waveform", fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax[1].tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def signal_vs_filtered_signal_plot(self,
                                       single_waveform: np.ndarray,
                                       single_waveform_filtered: np.ndarray,
                                       metadata: Dict,
                                       freq_cut: float,
                                       outfile_path: Optional[str] = None) -> None:
        """
        Plot a comparison between a waveform and its filtered version.

        Args:
            single_waveform (np.ndarray): The original waveform data.
            single_waveform_filtered (np.ndarray): The filtered waveform data.
            metadata (Dict): Metadata containing 'time_ax_waveform'.
            freq_cut (float): Cut-off frequency for filtering.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If waveforms are not 1D arrays or lengths do not match.
            KeyError: If 'time_ax_waveform' is missing in metadata.
        """
        if 'time_ax_waveform' not in metadata:
            raise KeyError("metadata must contain 'time_ax_waveform'.")
        if single_waveform.ndim != 1 or single_waveform_filtered.ndim != 1:
            raise ValueError("single_waveform and single_waveform_filtered must be 1D numpy arrays.")
        if len(single_waveform) != len(single_waveform_filtered):
            raise ValueError("single_waveform and single_waveform_filtered must have the same length.")

        time_ax = metadata['time_ax_waveform']
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        ax.plot(time_ax, single_waveform, color="lightgray", label="Original Waveform", linewidth=self.settings['line_width'])
        ax.plot(time_ax, single_waveform_filtered, color="black", label="Filtered Waveform", linewidth=self.settings['line_width'])
        ax.set_xlabel('Time [$\\mu s$]', fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel('Amplitude [a.u.]', fontsize=self.settings['fontsize_labels'])
        ax.set_title(f"Effect of Lowpass Filtering at {freq_cut:.2f} MHz", fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax.legend(fontsize=self.settings['fontsize_ticks'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        ax.grid(alpha=0.3)

        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def wavelet_selection_plot(self,
                               time: np.ndarray,
                               waveform: np.ndarray,
                               ratio: np.ndarray,
                               index_max_list: List[int],
                               index_min_before_list: List[int],
                               index_min_after_list: List[int],
                               outfile_path: Optional[str] = None) -> None:
        """
        Plot waveform data along with STA/LTA analysis results.

        Args:
            time (np.ndarray): Time array.
            waveform (np.ndarray): Waveform data.
            ratio (np.ndarray): STA/LTA ratio.
            index_max_list (List[int]): List of indices for maximum values.
            index_min_before_list (List[int]): List of indices for minimum values before maximum.
            index_min_after_list (List[int]): List of indices for minimum values after maximum.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If input arrays are not 1D or lengths do not match.
        """
        if not all(arr.ndim == 1 for arr in [time, waveform, ratio]):
            raise ValueError("time, waveform, and ratio must be 1D numpy arrays.")
        if not (len(time) == len(waveform) == len(ratio)):
            raise ValueError("time, waveform, and ratio must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        ax.plot(time, waveform, label='Recorded Waveform', alpha=0.5, linewidth=self.settings['line_width'])
        norm = (np.amax(waveform) / np.amax(ratio))
        ax.plot(time, norm * ratio, label='STA/LTA on Waveform')
        ax.plot(time[index_max_list], norm * ratio[index_max_list], "r.", label='Maxima')
        ax.plot(time[index_min_before_list], norm * ratio[index_min_before_list], "g.", label='Minima Before')
        ax.plot(time[index_min_after_list], norm * ratio[index_min_after_list], "k.", label='Minima After')
        ax.set_xlabel('Time [$\\mu$s]', fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel('Amplitude [a.u.]', fontsize=self.settings['fontsize_labels'])
        ax.legend(loc="lower left", fontsize=self.settings['fontsize_ticks'])
        ax.set_title("Wavelet Selection Using STA/LTA", fontsize=self.settings['fontsize_title'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        ax.grid(alpha=0.3)

        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_simulation_waveform(self,
                                 t: np.ndarray,
                                 sp_simulated: np.ndarray,
                                 sp_recorded: np.ndarray,
                                 misfit_interval: np.ndarray,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the simulated waveform against the recorded waveform.

        Args:
            t (np.ndarray): Time array.
            sp_simulated (np.ndarray): Simulated waveform.
            sp_recorded (np.ndarray): Recorded waveform.
            misfit_interval (np.ndarray): Indices of the misfit interval.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If input arrays are not 1D or lengths do not match.
        """
        if not all(arr.ndim == 1 for arr in [t, sp_simulated, sp_recorded]):
            raise ValueError("t, sp_simulated, and sp_recorded must be 1D numpy arrays.")
        if not (len(t) == len(sp_simulated) == len(sp_recorded)):
            raise ValueError("t, sp_simulated, and sp_recorded must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        # Check misfit_interval
        if misfit_interval.size > 0:
            misfit_start_time = t[misfit_interval[0]]
            misfit_end_time = t[misfit_interval[-1]]

            # Add shaded region for misfit interval
            ax.axvspan(t[0],misfit_start_time, color=self.settings['colors']['lightsteelblue'])
            ax.axvspan(misfit_start_time, misfit_end_time, color=self.settings['colors']['sandybrown'], alpha=0.5)
            ax.axvspan(misfit_end_time,t[-1], color=self.settings['colors']['lightsteelblue'])            
            ax.text(misfit_start_time, min(sp_recorded), 'Misfit Evaluation Interval', ha='left', fontsize=self.settings['fontsize_labels'], color=self.settings['colors']['darkslategray'])
        else:
            print("Misfit interval is empty; cannot shade region.")

        COLORS = {
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
        ax.plot(t, sp_recorded, label="Recorded Waveform", color=self.settings['colors']['platinum'], linewidth=2*self.settings['line_width'])
        ax.plot(t, sp_simulated, label="Simulated Waveform", color=self.settings['colors']['indianred'], linewidth=2*self.settings['line_width'],alpha=0.25)

        ax.set_title("Ultrasonic Wave Simulation", fontsize=self.settings['fontsize_title'])
        ax.set_xlabel("Time [$\\mu s$]", fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel("Amplitude [a.u.]", fontsize=self.settings['fontsize_labels'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        ax.legend(fontsize=self.settings['fontsize_ticks'])
        ax.grid(alpha=0.3)
        ax.set_xlim(left=t[200],right=t[-1])
        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def make_movie_from_simulation(self,
                                    outfile_path: str,
                                    x: np.ndarray,
                                    t: np.ndarray,
                                    sp_field: np.ndarray,
                                    sp_recorded: np.ndarray,
                                    sample_dimensions: Tuple[float, float],
                                    idx_dict: Dict[str, np.ndarray]) -> None:
            """
            Create an animation of the wavefield simulation.

            Args:
                outfile_path (str): Path to save the movie file.
                x (np.ndarray): Spatial axis.
                t (np.ndarray): Time array.
                sp_field (np.ndarray): Simulated wavefield (2D array).
                sp_recorded (np.ndarray): Recorded waveform.
                sample_dimensions (Tuple[float, float]): Dimensions of the sample.
                idx_dict (Dict[str, np.ndarray]): Dictionary of indices for different layers.

            Raises:
                ValueError: If input arrays have incorrect dimensions or lengths.
            """
            # Input validation
            if x.ndim != 1 or t.ndim != 1:
                raise ValueError("x and t must be 1D numpy arrays.")
            if sp_field.ndim != 2:
                raise ValueError("sp_field must be a 2D numpy array.")
            if sp_field.shape != (len(t), len(x)):
                raise ValueError("sp_field shape must be (len(t), len(x)).")
            if sp_recorded.ndim != 1:
                raise ValueError("sp_recorded must be a 1D numpy array.")
            if len(sp_recorded) != len(t):
                raise ValueError("sp_recorded must have the same length as t.")

            movie_sampling = 10  # Downsampling of the snapshot to speed up movie

            fig, (ax, ax2) = plt.subplots(1, 2, figsize=self.settings['figure_size'], gridspec_kw={'width_ratios': [10, 1]})
            ylim = 1.3 * np.amax(np.abs(sp_field))

            ax.set_xlim([x[0], x[-1]])
            ax.set_ylim([-ylim, ylim])
            ax.set_title("Ultrasonic Wavefield in DDS Experiment", fontsize=self.settings['fontsize_title'], color=self.settings['colors']['darkslategray'])
            ax.set_xlabel("Sample Length [cm]", fontsize=self.settings['fontsize_labels'], color=self.settings['colors']['darkslategray'])
            ax.set_ylabel('Relative Shear Wave Amplitude', fontsize=self.settings['fontsize_labels'], color=self.settings['colors']['darkslategray'])
            ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

            # Shading layers based on indices in idx_dict
            # work around for the new added possibility to plot the experiment of an homogeneus block, the one to find STF
            if len(sample_dimensions) == 1:
                layers = [
                      {'name': 'PZT Layer 1', 'idx': idx_dict.get('pzt_1'), 'color': self.settings['colors']['indianred']},
                      {'name': 'PZT Layer 2', 'idx': idx_dict.get('pzt_2'), 'color': self.settings['colors']['indianred']},
                      {'name': 'Steel Blocks', 'idx': idx_dict.get('steel_block'), 'color': self.settings['colors']['lightsteelblue']},
                ]
            else:    
                layers = [
                    {'name': 'PZT Layer 1', 'idx': idx_dict.get('pzt_1'), 'color': self.settings['colors']['indianred']},
                    {'name': 'PZT Layer 2', 'idx': idx_dict.get('pzt_2'), 'color': self.settings['colors']['indianred']},
                    {'name': 'Steel Blocks', 'idx': np.concatenate([idx_dict.get(key) for key in ['side_block_1', 'central_block', 'side_block_2'] if idx_dict.get(key) is not None]), 'color': self.settings['colors']['lightsteelblue']},
                    {'name': 'Grooves', 'idx': np.concatenate([idx_dict.get(key) for key in ['groove_sb1', 'groove_cb1', 'groove_cb2', 'groove_sb2'] if idx_dict.get(key) is not None]), 'color': self.settings['colors']['lightgrey']},
                    {'name': 'Gouge Layer 1', 'idx': idx_dict.get('gouge_1'), 'color': self.settings['colors']['sandybrown']},
                    {'name': 'Gouge Layer 2', 'idx': idx_dict.get('gouge_2'), 'color': self.settings['colors']['sandybrown']}
                ]

            for layer in layers:
                if layer['idx'] is not None and len(layer['idx']) > 0:
                    ax.axvspan(x[layer['idx'][0]], x[layer['idx'][-1]], color=layer['color'], alpha=0.3, label=layer['name'])

            # Plot transmitter and receiver positions
            x_tr = x[idx_dict['pzt_1'][-1]]
            y_tr = 0
            pzt_width = x[idx_dict['pzt_1'][-1]] - x[idx_dict['pzt_1'][0]]
            pzt_height = 4 * pzt_width
            ax.add_patch(Rectangle((x_tr - pzt_width, y_tr - pzt_height / 2), pzt_width, pzt_height, color=self.settings['colors']['teal']))
            ax.text(x_tr - pzt_width / 2, y_tr - pzt_height, 'Transmitter', ha='center', fontsize=self.settings['fontsize_labels'], color=self.settings['colors']['darkslategray'])

            x_rc = x[idx_dict['pzt_2'][0]]
            y_rc = 0
            ax.add_patch(Rectangle((x_rc, y_rc - pzt_height / 2), pzt_width, pzt_height, color=self.settings['colors']['teal']))
            ax.text(x_rc + pzt_width / 2, y_rc - pzt_height, 'Receiver', ha='center', fontsize=self.settings['fontsize_labels'], color=self.settings['colors']['darkslategray'])

            # Configure ax2 for the recorded signal
            ax2.set_ylim([t[0], t[-1]])
            ax2.set_xlim([-1, 1])  # Set x-limits to small range around zero
            ax2.set_ylabel("Recorded Signal", fontsize=self.settings['fontsize_labels'], color=self.settings['colors']['darkslategray'])
            ax2.axis('off')
            ax2.invert_yaxis()

            fig.tight_layout()

            # Initialize lines for animation
            line_wavefield, = ax.plot([], [], color=self.settings['colors']['darkslategray'], lw=self.settings['line_width'])
            line_recorded, = ax2.plot([], [], color=self.settings['colors']['darkslategray'], lw=self.settings['line_width'])

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

    def plot_velocity_model(self,
                            x: np.ndarray,
                            c: np.ndarray,
                            layer_starts: np.ndarray,
                            pzt_layer_width: float,
                            pla_layer_width: float,
                            outfile_path: Optional[str] = None) -> None:
        """
        Plot the velocity model with layers and smoothing.

        Args:
            x (np.ndarray): Spatial axis.
            c (np.ndarray): Velocity model array.
            layer_starts (np.ndarray): Cumulative positions along the sample.
            pzt_layer_width (float): Width of the PZT layer.
            pla_layer_width (float): Width of the pla layer.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If input arrays have incorrect dimensions or lengths.
        """
        if x.ndim != 1 or c.ndim != 1:
            raise ValueError("x and c must be 1D numpy arrays.")
        if len(x) != len(c):
            raise ValueError("x and c must have the same length.")
        if layer_starts.ndim != 1:
            raise ValueError("layer_starts must be a 1D numpy array.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        ax.plot(x, c, label='Velocity Model', color='black', linewidth=self.settings['line_width'])

        try: 
            layers = [
                {'name': 'pla Layer 1', 'start': layer_starts[0], 'end': layer_starts[1], 'color': self.settings['colors']['platinum']},
                {'name': 'PZT Layer 1', 'start': layer_starts[1], 'end': layer_starts[2], 'color': self.settings['colors']['indianred']},
                {'name': 'Side Block 1', 'start': layer_starts[2], 'end': layer_starts[3], 'color': self.settings['colors']['lightsteelblue']},
                {'name': 'Groove SB1', 'start': layer_starts[3], 'end': layer_starts[4], 'color': self.settings['colors']['lightgrey']},
                {'name': 'Gouge Layer 1', 'start': layer_starts[4], 'end': layer_starts[5], 'color': self.settings['colors']['sandybrown']},
                {'name': 'Groove CB1', 'start': layer_starts[5], 'end': layer_starts[6], 'color': self.settings['colors']['lightgrey']},
                {'name': 'Central Block', 'start': layer_starts[6], 'end': layer_starts[7], 'color': self.settings['colors']['lightsteelblue']},
                {'name': 'Groove CB2', 'start': layer_starts[7], 'end': layer_starts[8], 'color': self.settings['colors']['lightgrey']},
                {'name': 'Gouge Layer 2', 'start': layer_starts[8], 'end': layer_starts[9], 'color': self.settings['colors']['sandybrown']},
                {'name': 'Groove SB2', 'start': layer_starts[9], 'end': layer_starts[10], 'color': self.settings['colors']['lightgrey']},
                {'name': 'Side Block 2', 'start': layer_starts[10], 'end': layer_starts[11], 'color': self.settings['colors']['lightsteelblue']},
                {'name': 'PZT Layer 2', 'start': layer_starts[11], 'end': layer_starts[12], 'color': self.settings['colors']['indianred']},
                {'name': 'pla Layer 2', 'start': layer_starts[12], 'end': layer_starts[13], 'color': self.settings['colors']['platinum']},
            ]

        except IndexError:
            layers = [
                {'name': 'pla Layer 1', 'start': layer_starts[0], 'end': layer_starts[1], 'color': self.settings['colors']['platinum']},
                {'name': 'PZT Layer 1', 'start': layer_starts[1], 'end': layer_starts[2], 'color': self.settings['colors']['indianred']},
                {'name': 'Steel Block', 'start': layer_starts[2], 'end': layer_starts[3], 'color': self.settings['colors']['lightsteelblue']},
                {'name': 'PZT Layer 2', 'start': layer_starts[3], 'end': layer_starts[4], 'color': self.settings['colors']['indianred']},
                {'name': 'pla Layer 2', 'start': layer_starts[4], 'end': layer_starts[5], 'color': self.settings['colors']['platinum']},
            ]
        labels_used = set()

        for layer in layers:
            label = layer['name'] if layer['name'] not in labels_used else None
            labels_used.add(layer['name'])
            ax.axvspan(layer['start'], layer['end'], color=layer['color'], alpha=0.3, label=label)

        # Plot transmitter and receiver positions
        transmitter_pos = pzt_layer_width + pla_layer_width
        receiver_pos = x[-1] - pzt_layer_width - pla_layer_width
        ax.axvline(transmitter_pos, color="red", linestyle='-', label='Transmitter')
        ax.axvline(receiver_pos, color="green", linestyle='-', label='Receiver')

        ax.set_title("Velocity Model", fontsize=self.settings['fontsize_title'])
        ax.set_xlabel("Position (cm)", fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel("Velocity (cm/$\\mu$s)", fontsize=self.settings['fontsize_labels'])
        ax.grid(True)
        ax.legend(loc='upper center', fontsize=self.settings['fontsize_ticks'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_synthetic_spatial_function(self,
                                        x: np.ndarray,
                                        spatial_function: np.ndarray,
                                        outfile_path: Optional[str] = None) -> None:
        """
        Plot the synthetic spatial function.

        Args:
            x (np.ndarray): Spatial axis.
            spatial_function (np.ndarray): The synthetic spatial function values.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If x and spatial_function are not 1D arrays of the same length.
        """
        if x.ndim != 1 or spatial_function.ndim != 1:
            raise ValueError("x and spatial_function must be 1D numpy arrays.")
        if len(x) != len(spatial_function):
            raise ValueError("x and spatial_function must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        ax.plot(x, spatial_function, linewidth=self.settings['line_width'])
        ax.set_title("Synthetic Spatial Function", fontsize=self.settings['fontsize_title'])
        ax.set_xlabel("Position (cm)", fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel("Amplitude", fontsize=self.settings['fontsize_labels'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        ax.grid(alpha=0.3)

        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_velocity_and_stresses(self,
                                   x_values: np.ndarray,
                                   velocities: np.ndarray,
                                   normal_stress: np.ndarray,
                                   shear_stress: np.ndarray,
                                   x_label: str,
                                   velocity_label: str,
                                   stress_labels: Tuple[str, str],
                                   title: str,
                                   outfile_path: Optional[str] = None) -> None:
        """
        Plot estimated velocities and two stresses vs a common x-axis variable.

        Args:
            x_values (np.ndarray): The x-axis values (e.g., displacement or time).
            velocities (np.ndarray): Estimated velocities.
            normal_stress (np.ndarray): Normal stress values.
            shear_stress (np.ndarray): Shear stress values.
            x_label (str): Label for the x-axis.
            velocity_label (str): Label for the velocity y-axis.
            stress_labels (Tuple[str, str]): Labels for the stress y-axes.
            title (str): Title of the plot.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If input arrays are not 1D or lengths do not match.
        """
        # if not all(arr.ndim == 1 for arr in [x_values, velocities, normal_stress, shear_stress]):
        #     raise ValueError("All input arrays must be 1D numpy arrays.")
        # if not (len(x_values) == len(velocities) == len(normal_stress) == len(shear_stress)):
        #     raise ValueError("All input arrays must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot velocities on the left y-axis
        color1 = 'tab:blue'
        ax.set_xlabel(x_label, fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel(velocity_label, color=color1, fontsize=self.settings['fontsize_labels'])
        ax.plot(x_values, velocities, color=color1, label=velocity_label, linewidth=self.settings['line_width'])
        ax.tick_params(axis='y', labelcolor=color1, labelsize=self.settings['fontsize_ticks'])
        ax.tick_params(axis='x', labelsize=self.settings['fontsize_ticks'])

        # Create a second y-axis for normal stress
        ax2 = ax.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(stress_labels[0], color=color2, fontsize=self.settings['fontsize_labels'])
        ax2.plot(x_values, normal_stress, color=color2, linestyle='--', label=stress_labels[0], linewidth=self.settings['line_width'])
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=self.settings['fontsize_ticks'])

        # Adjust the position of ax2 to make room for a third y-axis
        ax2.spines['right'].set_position(('axes', 1.0))

        # Create a third y-axis for shear stress
        ax3 = ax.twinx()
        color3 = 'tab:green'
        ax3.set_ylabel(stress_labels[1], color=color3, fontsize=self.settings['fontsize_labels'])
        ax3.plot(x_values, shear_stress, color=color3, linestyle='-', label=stress_labels[1], linewidth=self.settings['line_width'])
        ax3.tick_params(axis='y', labelcolor=color3, labelsize=self.settings['fontsize_ticks'])

        # Offset the third y-axis
        ax3.spines['right'].set_position(('axes', 1.15))

        # Add grid, legend, and title
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left', fontsize=self.settings['fontsize_ticks'])

        ax.set_title(title, fontsize=self.settings['fontsize_title'])
        ax.grid(alpha=0.3)

        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_l2_norm_vs_velocity(self,
                                 velocity: np.ndarray,
                                 L2norm: np.ndarray,
                                 acquisition_time: int,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the L2 norm vs  velocity.

        Args:
            _velocity (np.ndarray): Array of  velocity values.
            L2norm (np.ndarray): Array of L2 norm values corresponding to the velocity.
            acquisition_time (int): acquisition time of the waveform for plot title.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If _velocity and L2norm are not 1D arrays of the same length.
        """
        if velocity.ndim != 1 or L2norm.ndim != 1:
            raise ValueError("_velocity and L2norm must be 1D numpy arrays.")
        if len(velocity) != len(L2norm):
            raise ValueError("_velocity and L2norm must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        
        # Plotting the L2 norm vs  velocity
        ax.plot(velocity, L2norm, linewidth=self.settings['line_width'])
        ax.set_xlabel('Velocity (cm/$\\mu$s)', fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel('L2 Norm of Residuals', fontsize=self.settings['fontsize_labels'])
        ax.set_title(f'L2 Norm vs  Velocity for Waveform at {acquisition_time} s', fontsize=self.settings['fontsize_title'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        ax.grid(alpha=0.3)
        
        self.output_path_choice(fig=fig, outfile_path=outfile_path)
       
    def plot_direct_and_reflections(
        self,
        direct_wave_time: np.ndarray,
        direct_wave_data: np.ndarray,
        reflection_info_list: list[dict],
        outfile_path: Optional[str] = None
    ) -> None:
        """
        Plots the direct wave snippet and each reflection wave snippet,
        showing correlation in the title.

        Parameters
        ----------
        direct_wave_time : np.ndarray
            Time axis for the direct wave snippet (1D).
        direct_wave_data : np.ndarray
            Waveform snippet for the direct wave (1D).
        reflection_info_list : list of dict
            Output from 'compute_reflections_correlation'. Each dict has:
            {
                'arrival_time': float,
                'reflection_time': np.ndarray,
                'reflection_data': np.ndarray,
                'corr_coeff': float,
            }
        outfile_path : str, optional
            If given, the figure is saved to this path; else displayed.

        Notes
        -----
        - The direct wave is plotted first, with time shifted to zero.
        - Each reflection is overlaid (also zero-based in time) and amplitude-scaled
        to match the direct wave’s maximum for easier visual comparison.
        - The correlation coefficient is displayed in the title for each subplot.
        """
        # Number of reflections
        n_reflections = len(reflection_info_list)

        # Create subplots: one for direct wave, plus one per reflection
        fig, axs = plt.subplots(
            nrows=n_reflections,
            figsize=self.settings['figure_size']
        )
        # If there's only 1 reflection, axs might not be a list
        if n_reflections  == 1:
            axs = [axs]

        # For each reflection
        for i, info in enumerate(reflection_info_list):
            reflection_time = info['reflection_time']
            reflection_data = info['reflection_data']
            arr_time        = info['arrival_time']
            corr_coeff      = info['corr_coeff']

            # Shift reflection time to start at 0
            overlay_time = reflection_time - reflection_time[0]

            # Amplitude scale reflection to match direct wave peak
            max_ref = np.max(reflection_data) if reflection_data.size else 1.0
            scale_factor = (np.max(direct_wave_data) / max_ref) if max_ref != 0 else 1.0

            axs[i].plot(
                overlay_time,
                direct_wave_data,
                label="Direct",
                linewidth=self.settings['line_width']
            )
            axs[i].plot(
                overlay_time,
                scale_factor * reflection_data,
                label="Reflection scaled",
                alpha=0.7,
                linewidth=self.settings['line_width']
            )

            title_str = f"Direct vs Reflection at {arr_time:.2f} μs, corr={corr_coeff:.3f}"
            axs[i].set_title(title_str,
                            fontsize=self.settings['fontsize_subplot_title'],
                            fontname=self.FONT_TYPE)
            axs[i].grid(True)
            axs[i].tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        fig.tight_layout()

        # Use the Plotter's output_path_choice method to save or show
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_reflection_windows(
        self,
        observed_time: np.ndarray,
        waveform: np.ndarray,
        t_start_direct: float,
        t_end_direct: float,
        reflection_info_list: list[dict],
        idx_Dstart: int,
        idx_Dend: int,
        title: str = "Reflections highlighted",
        x_label: str = "Time (μs)",
        y_label: str = "Amplitude",
        outfile_path: Optional[str] = None
    ) -> None:
        """
        Plots the main waveform and highlights the direct wave arrival window,
        plus each reflection window.

        Parameters
        ----------
        observed_time : np.ndarray
            The full time axis of the waveform.
        waveform : np.ndarray
            1D array of the full waveform.
        t_start_direct : float
            The picked start time (in the same units as observed_time) for the direct arrival.
        t_end_direct : float
            The picked end time for the direct arrival.
        reflection_info_list : list of dict
            Output from 'compute_reflections_correlation'. Each dict has:
            {
                'arrival_time': float,
                'reflection_time': np.ndarray,
                'reflection_data': np.ndarray,
                'corr_coeff': float,
            }
        idx_Dstart : int
            Index in observed_time corresponding to t_start_direct.
        idx_Dend : int
            Index in observed_time corresponding to t_end_direct.
        title : str, optional
            Plot title. Default "Reflections highlighted".
        x_label : str, optional
            X-axis label. Default "Time (μs)".
        y_label : str, optional
            Y-axis label. Default "Amplitude".
        outfile_path : str, optional
            If given, the figure is saved at this path; else displayed interactively.

        Returns
        -------
        None
            The function produces a plot, either saving or displaying it.
        """

        # 1) Create figure/axis using class settings
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # 2) Plot the main waveform
        ax.plot(observed_time, waveform, label="Waveform", linewidth=self.settings['line_width'])

        # 3) Highlight the direct arrival region
        ax.axvspan(t_start_direct, t_end_direct, facecolor='r', alpha=0.2, label="Direct Arrival")

        # 4) Basic labeling & grid
        ax.set_xlabel(x_label, fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel(y_label, fontsize=self.settings['fontsize_labels'])
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE
        )
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        # 5) For reflection i, snippet is from arr_time to arr_time + direct_arrival_span
        direct_arrival_span = idx_Dend - idx_Dstart
        for i, reflection_info in enumerate(reflection_info_list, start=1):
            arr_time = reflection_info['arrival_time']
            ref_start_idx = np.searchsorted(observed_time, arr_time)
            ref_end_idx   = ref_start_idx + direct_arrival_span
            if ref_end_idx > len(observed_time):
                break

            ax.axvspan(
                observed_time[ref_start_idx],
                observed_time[ref_end_idx - 1],
                facecolor='g',
                alpha=0.2,
                label="Reflection windows" if i == 1 else None
            )

        ax.legend(fontsize=self.settings['fontsize_ticks'])

        # 6) Adjust layout and let the class method handle saving/showing
        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_original_vs_updated_stf(self,
                                 t: np.ndarray,
                                 stf_updated: np.ndarray,
                                 stf_original: np.ndarray,
                                 min_time: float,
                                 max_time : float,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the simulated waveform against the recorded waveform.

        Args:
            t (np.ndarray): Time array.
            stf_updated (np.ndarray): Updated Source Time Function.
            stf_original (np.ndarray): Original STF.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If input arrays are not 1D or lengths do not match.
        """
        if not all(arr.ndim == 1 for arr in [t, stf_updated, stf_original]):
            raise ValueError("t, stf_updated, and stf_original must be 1D numpy arrays.")
        if not (len(t) == len(stf_updated) == len(stf_original)):
            raise ValueError("t, stf_updated, and stf_original must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        ax.plot(t, stf_original, label="Original STF", color=self.settings['colors']['platinum'], linewidth=2*self.settings['line_width'])
        ax.plot(t, stf_updated, label="Updated STF", color=self.settings['colors']['indianred'], linewidth=2*self.settings['line_width'],alpha=0.25)

        ax.set_title("Updating Source Time Function with FWI", fontsize=self.settings['fontsize_title'])
        ax.set_xlabel("Time [$\\mu s$]", fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel("Amplitude [a.u.]", fontsize=self.settings['fontsize_labels'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        ax.legend(fontsize=self.settings['fontsize_ticks'])
        ax.grid(alpha=0.3)
        ax.set_xlim(left=min_time,right=max_time)
        fig.tight_layout()

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_boxplot_parameters(
        self,
        param_matrix: np.ndarray,
        param_labels: List[str],
        title: str,
        ylabel: str,
        outfile_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Creates a box plot for the given parameter matrix.

        Args:
            param_matrix (np.ndarray): Shape (n_runs, n_parameters).
            param_labels (List[str]): Labels for each parameter (x-axis).
            title (str): Plot title.
            ylabel (str): Label for the y-axis.
            outfile_path (Union[str, Path], optional): If provided, saves plot to file. Otherwise shows it.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        box = ax.boxplot(
            param_matrix,
            patch_artist=True,
            showmeans=True
            # You can pass any other boxplot kwargs here
        )
        # Customize the boxplot
        for patch in box['boxes']:
            patch.set(facecolor="lightblue", alpha=0.5)
        for median in box['medians']:
            median.set(color="red", linewidth=2)
        for mean_line in box['means']:
            mean_line.set(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=5)

        # Set tick labels
        ax.set_xticks(np.arange(1, len(param_labels) + 1))
        ax.set_xticklabels(param_labels, rotation=0, fontsize=self.settings['fontsize_ticks'])

        ax.set_title(title, fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax.set_ylabel(ylabel, fontsize=self.settings['fontsize_labels'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_histogram_l2_distribution(
        self,
        data: np.ndarray,
        bins: int,
        title: str,
        xlabel: str,
        ylabel: str,
        outfile_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Creates a histogram for the given data.

        Args:
            data (np.ndarray): 1D array of data values.
            bins (int): Number of histogram bins.
            title (str): Plot title.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            outfile_path (Union[str, Path], optional): If provided, saves plot to file. Otherwise shows it.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        ax.hist(data, bins=bins, color="lightgreen", edgecolor="k", alpha=0.7)

        ax.set_title(title, fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        ax.set_xlabel(xlabel, fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel(ylabel, fontsize=self.settings['fontsize_labels'])
        ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
        ax.grid(alpha=0.1)

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_scatter_l2_vs_parameters(
        self,
        param_list: List[Tuple[str,np.ndarray]],
        l2_values: np.ndarray,
        title: str,
        best_index: int = 0,
        outfile_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Creates scatter plots of L2 misfit vs. each parameter in subplots.

        Args:
            param_list (List[np.ndarray]): List of arrays, each array is a parameter across runs.
            l2_values (np.ndarray): L2 misfit array across runs.
            param_labels (List[str]): Parameter names for each array in param_list.
            title (str): Plot title.
            best_index (int, optional): Index that indicates the best (lowest L2). Defaults to 0.
            outfile_path (Union[str, Path], optional): If provided, saves plot to file. Otherwise shows it.
        """
        n_params = len(param_list)
        # Example layout: 2 rows, 4 columns for up to 8 parameters
        n_cols = 4
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=self.settings['figure_size'], tight_layout=True)
        axs = axs.flatten()  # so we can index them linearly

        for i, (label, param_data) in enumerate(param_list):
            ax = axs[i]
            ax.scatter(param_data, l2_values, s=30, c="blue", alpha=0.7, edgecolors="k")
            ax.set_xlabel(label, fontsize=self.settings['fontsize_labels'])
            ax.set_ylabel("L2 Misfit", fontsize=self.settings['fontsize_labels'])

            # highlight best param with a red star
            best_val = param_data[best_index]
            best_l2 = l2_values[best_index]
            ax.scatter(best_val, best_l2, s=100, c="red", marker="*", zorder=5)

            ax.tick_params(axis='both', which='major', labelsize=self.settings['fontsize_ticks'])
            ax.grid(alpha=0.1)

            ax.set_ylim([0.9*np.amin(l2_values),3*np.amin(l2_values)])

        # Hide any leftover subplots if n_params < n_rows * n_cols
        for j in range(n_params, n_rows * n_cols):
            axs[j].axis('off')

        fig.suptitle(title, fontsize=self.settings['fontsize_title'], fontname=self.FONT_TYPE)
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

class InteractivePlotter(Plotter):
    """
    A specialized Plotter class that provides interactive methods for human-needed operations.
    Inherits from Plotter so it has self.settings, etc.
    """
    def manual_pick_arrival_times(
        self,
        observed_time: np.ndarray,
        observed_waveform: np.ndarray,
        start_time: Optional[float]= 0,
        outfile_path: Optional[Path] = None
    ) -> List[float]:
        """
        Manually pick arrival times from waveform data to estimate initial velocities.
        Uses the same self.settings as other plotting methods in this class.
        """
        # Validate inputs
        if observed_time.ndim != 1 or observed_waveform.ndim != 1:
            raise ValueError("observed_time and observed_waveform must be 1D numpy arrays.")
        if len(observed_time) != len(observed_waveform):
            raise ValueError("observed_time and observed_waveform must have the same length.")

        picked_times: List[float] = []
        picking_mode = [False]  # store in mutable for closure

        def onclick(event):
            """
            Only pick if in picking mode. Otherwise, let zoom/pan do its job.
            """

            if picking_mode[0] and event.button == 1 and event.inaxes:
                picking_mode[0] = not picking_mode[0]
                print(f"Picking mode = {picking_mode[0]}")
                picked_time = event.xdata
                picked_times.append(picked_time)
                print(f"Picked time: {picked_time:.6f}")
                event.inaxes.axvline(x=picked_time, color='r', linestyle='--')
                plt.draw()

        def start_picking_callback(event):
            """
            Button callback to enable picking mode.
            """
            picking_mode[0] = True
            print("Picking mode enabled. Left-click to pick arrival times.")

        # 1. Create figure/axes
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        plt.subplots_adjust(bottom=0.2)  # leave room for button

        # 2. Plot the waveform
        ax.plot(observed_time, observed_waveform, label='Waveform')
        ax.set_xlabel('Time [$\\mu s$]', fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel('Amplitude', fontsize=self.settings['fontsize_labels'])
        if outfile_path is not None:
            title_str = f"File: {outfile_path.name}"
        else:
            title_str = "Pick Arrival Times"
        ax.set_title(title_str, fontsize=self.settings['fontsize_title'])
        ax.set_xlim([observed_time[0], observed_time[-1]])
        ax.set_ylim([np.amin(observed_waveform), np.amax(observed_waveform)])
        ax.legend()
        ax.grid(True)

        # Visual reference up to start_time
        ax.axvspan(0, start_time, facecolor='0.2', alpha=0.3)
        ax.vlines(x=start_time, ymin=np.amin(observed_waveform), ymax=np.amax(observed_waveform), colors="k")

        # 3. Create a "Start Picking" button
        ax_button = plt.axes([0.7, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
        pick_button = Button(ax_button, "Click to Allow Picking")

        # 4. Connect callbacks
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        pick_button.on_clicked(start_picking_callback)

        # 5. Show the plot and wait for user interaction
        plt.show()

        # After the figure is closed, disable picking
        fig.canvas.mpl_disconnect(cid)

        # 6. Save the picked times if desired
        if outfile_path:
            outfile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(outfile_path, "wb") as f:
                pickle.dump(picked_times, f)
            print(f"Picked times saved to {outfile_path}")

        return picked_times
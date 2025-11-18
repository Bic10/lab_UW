# lab_uw/plotting.py

from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # or 'pdf', 'svg', anything non-interactive
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib import animation

import cycler

import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from tkinter import Button
import pickle

class Plotter:
    """
    A class containing methods for plotting various aspects of waveform data and simulations.
    """

    # Define default settings for plots
    FONT_TYPE = "Dejavu Sans"
    FONT_SIZE = 30
    FIGURE_SIZE = (16, 8)
    FORMAT = ".png"

    PALETTE = {
        # BLU shades
        'blu1': '#203a7a',
        'blu2': '#2d4e9e',
        'blu3': '#617fbe',
        'blu4': '#89a5d6',
        'blu5': '#b3c4e5',
        'blu6': '#cad4ec',

        # GRIGIO shades
        'gray1': '#58595b',
        'gray2': '#76777a',
        'gray3': '#9ea0a2',
        'gray4': '#b5b6b8',
        'gray5': '#d5d6d7',
        'gray6': '#ebebec',

        # VERDE shades
        'green1': '#134239',
        'green2': '#479482',
        'green3': '#6fa599',

        # GIALLO shades
        'yellow1': '#f2cb59',
        'yellow2': '#f5d781',
        'yellow3': '#f8e2a7',

        # ROSSO shades
        'red1':    '#d9795c',
        'red2':    '#d04837',
        'red3':    '#d9795c',

        # MARRONE shades
        'brown1':  '#bf9d8f',
        'brown2':  '#d5dbd2',
        'brown3':  '#f2cf9f',
    }

    SEMANTIC = {
        'primary':      'gray1',     # <— new!
        'background':    'gray6',     # very light grey
        'axes_face':     'gray6',

        'observed':      'blu2',      # darkest blue for data
        'synthetic':     'red2',      # bright red
        'highlight':     'yellow1',   # for emphasis

        'misfit_light':  'blu1',      # pale blue span
        'misfit_dark':   'blu6',   # contrasting span

        'accent':        'green2',    # for PZT blocks, etc.
        'shadow':        'gray3',     # grid & minor elements
    }

    DEFAULT_SETTINGS = {
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

        cycle = [
            self.get_color('observed'),
            self.get_color('synthetic'),
            self.get_color('highlight'),
            self.get_color('accent'),
        ]
        plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', cycle)

        # 2) set background & grid colors
        plt.rcParams['figure.facecolor'] = self.get_color('background')
        plt.rcParams['axes.facecolor']   = self.get_color('axes_face')
        plt.rcParams['grid.color']       = self.get_color('shadow')

    def get_color(self, role: str) -> str:
        """
        Return the hex code for a semantic color role.
        """
        key = self.SEMANTIC.get(role)
        if key is None:
            raise KeyError(f"Unknown semantic role '{role}'")
        return self.PALETTE[key]
    
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
            fig.clf()
            plt.close()
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

        time_ticks_waveforms = np.arange(time_ax_waveform[0],
                                         time_ax_waveform[-1],
                                         ticks_steps_waveforms)
        data_to_plot = data[::step_wf_to_plot]
        ymax = 1.3 * np.amax(data_to_plot)
        ymin = 1.3 * np.amin(data_to_plot)

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot all waveforms in the 'observed' role
        ax.plot(
            time_ax_waveform,
            data_to_plot.T,
            color=self.get_color('observed'),
            linewidth=0.8,
            alpha=0.5
        )
        # Highlight the selected subset in the 'highlight' role
        ax.plot(
            time_ax_waveform,
            data[highlight_start:highlight_end].T,
            color=self.get_color('highlight'),
            linewidth=self.settings['line_width']
        )

        ax.set_xlabel('Time [$\\mu s$]', fontsize=self.settings['fontsize_labels'])
        ax.set_ylabel('Amplitude [a.u.]', fontsize=self.settings['fontsize_labels'])
        ax.set_xticks(time_ticks_waveforms)
        ax.set_ylim([ymin, ymax])
        ax.set_xlim(time_ax_waveform[0], xlim_plot)
        ax.grid(alpha=0.1, color=self.get_color('shadow'))

        ax.set_title("Stacked Waveforms",
                     fontsize=self.settings['fontsize_title'],
                     fontname=self.FONT_TYPE)
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=self.settings['fontsize_ticks'])

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

        time_ax_waveform   = metadata['time_ax_waveform']
        first_sample_time  = time_ax_waveform[0]
        last_sample_time   = time_ax_waveform[-1]
        time_ax_acquisition = metadata['time_ax_acquisition']
        first_waveform_time = time_ax_acquisition[0]
        last_waveform_time  = time_ax_acquisition[-1]

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # allow overriding the heatmap colormap via settings, default to 'seismic'
        cmap_name = self.settings.get('cmap', 'seismic')
        cmap = plt.get_cmap(cmap_name)

        extent = [
            first_waveform_time, last_waveform_time,
            first_sample_time,    last_sample_time
        ]
        amp_scale_limit = max(-np.amin(data), np.amax(data))

        im = ax.imshow(
            data.T,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap=cmap,
            vmin=-amp_scale_limit,
            vmax=amp_scale_limit,
            extent=extent
        )

        # colorbar styling
        cbar = fig.colorbar(im, pad=0.04)
        cbar.set_label(
            "Relative Amplitude",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        cbar.ax.yaxis.set_tick_params(
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        plt.setp(cbar.ax.get_yticklabels(), color=self.get_color('shadow'))

        # axis titles & labels in primary color
        ax.set_title(
            "Amplitude Map",
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            'Experiment Time [s]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            'Waveform Travel Time [$\\mu s$]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        # tick styling
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )

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

        # Only plot half of the spectrum if it's symmetrical
        spectrum_length = wave_len // 2
        freqs = signal_freqs[:spectrum_length]
        amps  = amp_spectrum[:, :spectrum_length]

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # allow colormap override via settings
        cmap_name = self.settings.get('cmap_spectrum', 'plasma')
        cmap = plt.get_cmap(cmap_name)

        pcm = ax.pcolormesh(
            time_ax_acquisition,
            freqs,
            amps.T,
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=1e-3, vmax=amps.max())
        )
        ax.set_ylim([0, freq_cut])

        # colorbar
        cbar = fig.colorbar(pcm, pad=0.04)
        cbar.set_label(
            "Spectral Amplitude",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        cbar.ax.yaxis.set_tick_params(
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        plt.setp(cbar.ax.get_yticklabels(), color=self.get_color('shadow'))

        # titles and labels
        ax.set_title(
            'Amplitude Spectrum Map',
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Time [s]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Frequency [MHz]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        # ticks and grid
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.1, color=self.get_color('shadow'))

        fig.tight_layout()
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
        """
        if 'time_ax_acquisition' not in metadata:
            raise KeyError("metadata must contain 'time_ax_acquisition'.")

        # Check dimensions
        if amp_spectrum.ndim != 2:
            raise ValueError("amp_spectrum must be a 2D numpy array.")
        wave_num, wave_len = amp_spectrum.shape

        # Only use half the spectrum if it's symmetrical
        spectrum_length = wave_len // 2
        freqs = signal_freqs[:spectrum_length]
        amps  = amp_spectrum[:, :spectrum_length]

        time_ax_acquisition = metadata['time_ax_acquisition']
        if len(time_ax_acquisition) != wave_num:
            raise ValueError("time_ax_acquisition length does not match amp_spectrum waveforms.")

        # Mask to freq_cut
        mask = freqs <= freq_cut
        freqs_cut   = freqs[mask]
        amps_cut    = amps[:, mask]

        # Prepare colormap
        cmap_name = self.settings.get('cmap_distribution', 'plasma')
        cmap = plt.get_cmap(cmap_name)

        # Create subplots
        fig, axes = plt.subplots(
            nrows=3,
            figsize=self.settings['figure_size'],
            constrained_layout=True
        )

        # 1) Line plot with alpha
        axes[0].plot(
            freqs_cut,
            amps_cut.T,
            color=self.get_color('observed'),
            alpha=0.05,
            linewidth=self.settings['line_width']
        )
        axes[0].set_title(
            "Line Plot with Alpha",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        axes[0].set_xlabel(
            "Frequency [MHz]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        axes[0].set_ylabel(
            "Amplitude [a.u.]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        axes[0].set_xlim([0, freq_cut])
        axes[0].tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        axes[0].grid(alpha=0.1, color=self.get_color('shadow'))

        # Prepare histogram bins
        freq_flat = np.broadcast_to(freqs_cut, amps_cut.shape).ravel()
        amp_flat  = amps_cut.ravel()
        pos_mask  = amp_flat > 0
        freq_pos  = freq_flat[pos_mask]
        amp_pos   = amp_flat[pos_mask]

        num_freq_bins = 200
        num_amp_bins  = 200
        freq_bins = np.linspace(0, freq_cut, num_freq_bins)
        amp_min, amp_max = amp_pos.min(), amp_pos.max()
        amp_bins = np.logspace(np.log10(amp_min), np.log10(amp_max), num_amp_bins)

        h, xedges, yedges = np.histogram2d(
            freq_pos, amp_pos,
            bins=[freq_bins, amp_bins]
        )

        # 2) 2D histogram (log colorscale)
        pcm1 = axes[1].pcolormesh(
            xedges, yedges, h.T,
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=1, vmax=h.max()),
            rasterized=True
        )
        axes[1].set_title(
            "2D Histogram (Log Colorscale)",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        axes[1].set_xlabel(
            "Frequency [MHz]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        axes[1].set_ylabel(
            "Amplitude [a.u.]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        axes[1].set_xlim([0, freq_cut])
        axes[1].set_ylim([amp_min, amp_max])
        axes[1].tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        axes[1].grid(alpha=0.1, color=self.get_color('shadow'))
        cbar1 = fig.colorbar(pcm1, ax=axes[1], pad=0.01)
        cbar1.set_label(
            "# Points (log scale)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        cbar1.ax.yaxis.set_tick_params(
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        plt.setp(cbar1.ax.get_yticklabels(), color=self.get_color('shadow'))

        # 3) 2D histogram (linear colorscale)
        pcm2 = axes[2].pcolormesh(
            xedges, yedges, h.T,
            cmap=cmap,
            vmax=h.max(),
            rasterized=True
        )
        axes[2].set_title(
            "2D Histogram (Linear Colorscale)",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        axes[2].set_xlabel(
            "Frequency [MHz]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        axes[2].set_ylabel(
            "Amplitude [a.u.]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        axes[2].set_xlim([0, freq_cut])
        axes[2].set_ylim([amp_min, amp_max])
        axes[2].tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        axes[2].grid(alpha=0.1, color=self.get_color('shadow'))
        cbar2 = fig.colorbar(pcm2, ax=axes[2], pad=0.01)
        cbar2.set_label(
            "# Points (linear scale)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        cbar2.ax.yaxis.set_tick_params(
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        plt.setp(cbar2.ax.get_yticklabels(), color=self.get_color('shadow'))

        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def filtered_amp_and_phase_spectrum_plot(self,
                                             signal_freqs: np.ndarray,
                                             amp_spectrum: np.ndarray,
                                             phase_spectrum: np.ndarray = None,
                                             filtered_amp_spectrum: np.ndarray = None,
                                             lowpass_filter: np.ndarray = None,
                                             freq_cut: float = None,
                                             outfile_path: Optional[str] = None) -> None:
        """
        Plots the filtered amplitude and phase spectrum.
        """
        if signal_freqs.ndim != 1 or amp_spectrum.ndim != 1:
            raise ValueError("signal_freqs and amp_spectrum must be 1D arrays.")
        if len(signal_freqs) != len(amp_spectrum):
            raise ValueError("signal_freqs and amp_spectrum must have the same length.")

        max_idx = np.argmax(amp_spectrum)
        max_freq = signal_freqs[max_idx]

        fig, axes = plt.subplots(2, 1, figsize=self.settings['figure_size'])

        # --- Amplitude panel ---
        ax_amp = axes[0]
        ax_amp.semilogy(
            signal_freqs,
            amp_spectrum,
            label="Original Spectrum",
            color=self.get_color('observed'),
            linewidth=self.settings['line_width']
        )
        ax_amp.vlines(
            max_freq,
            ymin=np.amin(amp_spectrum),
            ymax=np.amax(amp_spectrum),
            colors=self.get_color('highlight'),
            linestyles='--',
            label=f"Peak @ {max_freq:.2f} MHz"
        )

        if lowpass_filter is not None and filtered_amp_spectrum is not None:
            ax_amp.semilogy(
                signal_freqs,
                lowpass_filter * np.amax(amp_spectrum),
                label="Filter Shape",
                color=self.get_color('accent'),
                linewidth=self.settings['line_width']
            )
            ax_amp.semilogy(
                signal_freqs,
                filtered_amp_spectrum,
                label="Filtered Spectrum",
                color=self.get_color('synthetic'),
                linewidth=self.settings['line_width']
            )
            if freq_cut is not None:
                ax_amp.vlines(
                    freq_cut,
                    ymin=np.amin(filtered_amp_spectrum),
                    ymax=np.amax(filtered_amp_spectrum),
                    colors=self.get_color('highlight'),
                    linestyles='-',
                    label=f"Cut-off = {freq_cut:.2f} MHz"
                )

        ax_amp.set_title(
            "Amplitude Spectrum",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax_amp.set_xlabel(
            "Frequency [MHz]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax_amp.set_ylabel(
            "Amplitude [a.u.]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax_amp.set_xlim([0, np.max(signal_freqs)])
        ax_amp.set_ylim([np.min(amp_spectrum), np.max(amp_spectrum)])
        ax_amp.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax_amp.grid(alpha=0.2, color=self.get_color('shadow'))
        ax_amp.legend(fontsize=self.settings['fontsize_ticks'], facecolor=self.get_color('background'))

        # --- Phase panel ---
        ax_ph = axes[1]
        if phase_spectrum is not None:
            ax_ph.plot(
                signal_freqs,
                phase_spectrum,
                color=self.get_color('accent'),
                linewidth=self.settings['line_width']
            )
            ax_ph.set_title(
                "Phase Spectrum",
                fontsize=self.settings['fontsize_title'],
                color=self.get_color('primary')
            )
            ax_ph.set_xlabel(
                "Frequency [MHz]",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )
            ax_ph.set_ylabel(
                "Phase [rad]",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )
            ax_ph.set_xlim([0, np.max(signal_freqs)])
            ax_ph.set_yticks(np.linspace(-np.pi, np.pi, 5))
            ax_ph.set_yticklabels(
                [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'],
                color=self.get_color('shadow')
            )
            ax_ph.tick_params(
                axis='both',
                which='major',
                labelsize=self.settings['fontsize_ticks'],
                colors=self.get_color('shadow')
            )
            ax_ph.grid(alpha=0.2, color=self.get_color('shadow'))

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
        """
        if 'time_ax_waveform' not in metadata:
            raise KeyError("metadata must contain 'time_ax_waveform'.")
        if single_waveform.ndim != 1 or single_waveform_filtered.ndim != 1:
            raise ValueError("single_waveform and single_waveform_filtered must be 1D numpy arrays.")
        if len(single_waveform) != len(single_waveform_filtered):
            raise ValueError("single_waveform and single_waveform_filtered must have the same length.")

        time_ax = metadata['time_ax_waveform']
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # original vs filtered using semantic colors
        ax.plot(
            time_ax,
            single_waveform,
            color=self.get_color('observed'),
            label="Original Waveform",
            linewidth=self.settings['line_width']
        )
        ax.plot(
            time_ax,
            single_waveform_filtered,
            color=self.get_color('synthetic'),
            label="Filtered Waveform",
            linewidth=self.settings['line_width']
        )

        ax.set_title(
            f"Effect of Lowpass Filtering at {freq_cut:.2f} MHz",
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            'Time [$\\mu s$]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            'Amplitude [a.u.]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        ax.legend(
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

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
        """
        if not all(arr.ndim == 1 for arr in [time, waveform, ratio]):
            raise ValueError("time, waveform, and ratio must be 1D numpy arrays.")
        if not (len(time) == len(waveform) == len(ratio)):
            raise ValueError("time, waveform, and ratio must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Recorded waveform
        ax.plot(
            time,
            waveform,
            label='Recorded Waveform',
            color=self.get_color('observed'),
            alpha=0.5,
            linewidth=self.settings['line_width']
        )

        # STA/LTA ratio scaled to waveform amplitude
        norm = np.amax(waveform) / np.amax(ratio)
        ax.plot(
            time,
            norm * ratio,
            label='STA/LTA Ratio',
            color=self.get_color('accent'),
            linewidth=self.settings['line_width']
        )

        # Mark maxima and minima
        ax.plot(
            time[index_max_list],
            norm * ratio[index_max_list],
            marker='o',
            linestyle='',
            label='Maxima',
            color=self.get_color('synthetic')
        )
        ax.plot(
            time[index_min_before_list],
            norm * ratio[index_min_before_list],
            marker='o',
            linestyle='',
            label='Minima Before',
            color=self.get_color('highlight')
        )
        ax.plot(
            time[index_min_after_list],
            norm * ratio[index_min_after_list],
            marker='o',
            linestyle='',
            label='Minima After',
            color=self.get_color('shadow')
        )

        ax.set_title(
            "Wavelet Selection Using STA/LTA",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            'Time [$\\mu s$]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            'Amplitude [a.u.]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

        ax.legend(
            loc="lower left",
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_simulation_waveform(self,
                                 t: np.ndarray,
                                 sp_simulated: np.ndarray,
                                 sp_recorded: np.ndarray,
                                 misfit_interval: np.ndarray,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the simulated waveform against the recorded waveform,
        with alternating misfit shading and simulation drawn last.
        """
        if not all(arr.ndim == 1 for arr in [t, sp_simulated, sp_recorded]):
            raise ValueError("t, sp_simulated, and sp_recorded must be 1D numpy arrays.")
        if not (len(t) == len(sp_simulated) == len(sp_recorded)):
            raise ValueError("t, sp_simulated, and sp_recorded must have the same length.")

        # bump up title/fonts for emphasis
        title_fs = int(1.2 * self.settings['fontsize_title'])
        label_fs = int(1.1 * self.settings['fontsize_labels'])

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # colors for shading
        light = self.get_color('misfit_light')
        dark  = self.get_color('misfit_dark')

        # --- alternating misfit shading ---
        if misfit_interval.size:
            idx     = np.sort(np.unique(misfit_interval))
            gaps    = np.where(np.diff(idx) > 1)[0]
            blocks  = np.split(idx, gaps + 1)

            # region before first block
            ax.axvspan(t[0], t[blocks[0][0]], color=light, alpha=0.3, zorder=1)

            # shade each misfit block (dark) and gap (light)
            for i, blk in enumerate(blocks):
                ax.axvspan(
                    t[blk[0]],
                    t[blk[-1]],
                    color=dark,
                    alpha=0.5,
                    zorder=2
                )
                if i + 1 < len(blocks):
                    nxt = blocks[i+1]
                    ax.axvspan(
                        t[blk[-1]],
                        t[nxt[0]],
                        color=light,
                        alpha=0.3,
                        zorder=1
                    )

            # region after last block
            ax.axvspan(
                t[blocks[-1][-1]],
                t[-1],
                color=light,
                alpha=0.3,
                zorder=1
            )

            # label the first misfit block
            y0 = np.min(sp_recorded)
            ax.text(
                t[blocks[0][0]],
                y0,
                'Misfit Evaluation Interval',
                ha='left',
                fontsize=label_fs,
                color=self.get_color('shadow'),
                zorder=3
            )
        else:
            print("Misfit interval is empty; cannot shade region.")

        # --- plot the data, simulation on top ---
        ax.plot(
            t,
            sp_recorded,
            label="Recorded Waveform",
            color=self.get_color('observed'),
            linewidth=self.settings['line_width'],
            zorder=2
        )
        ax.plot(
            t,
            sp_simulated,
            label="Simulated Waveform",
            color=self.get_color('synthetic'),
            linewidth=2 * self.settings['line_width'],
            alpha=0.8,
            zorder=4
        )

        # --- formatting ---
        ax.set_title(
            "Ultrasonic Wave Simulation",
            fontsize=title_fs,
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Time [$\\mu s$]",
            fontsize=label_fs,
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Amplitude [a.u.]",
            fontsize=label_fs,
            color=self.get_color('primary')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))
        
        ax.set_xlim(left=t[100],right=t[-1])

        ax.legend(
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow'),
            framealpha=0.8
        )

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
        """
        # Input validation unchanged...

        movie_sampling = 10  # Downsampling of the snapshot to speed up movie

        fig, (ax, ax2) = plt.subplots(
            1, 2,
            figsize=self.settings['figure_size'],
            gridspec_kw={'width_ratios': [10, 1]}
        )
        ylim = 1.3 * np.amax(np.abs(sp_field))

        # Axis limits and labels
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([-ylim, ylim])
        ax.set_title(
            "Ultrasonic Wavefield in DDS Experiment",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Sample Length [cm]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Relative Shear Wave Amplitude",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))

        # Shading layers
        is_simple = len(sample_dimensions) == 1
        layers = []
        if is_simple:
            layers = [
                ('PZT Layer 1', 'pzt_1', 'accent'),
                ('PZT Layer 2', 'pzt_2', 'accent'),
                ('Steel Blocks', 'steel_block', 'misfit_light'),
            ]
        else:
            layers = [
                ('PZT Layer 1', 'pzt_1', 'accent'),
                ('PZT Layer 2', 'pzt_2', 'accent'),
                ('Steel Blocks', ['side_block_1','central_block','side_block_2'], 'misfit_light'),
                ('Grooves',       ['groove_sb1','groove_cb1','groove_cb2','groove_sb2'], 'shadow'),
                ('Gouge Layer 1', 'gouge_1', 'highlight'),
                ('Gouge Layer 2', 'gouge_2', 'highlight'),
            ]

        for name, key, role in layers:
            idxs = idx_dict.get(key) if isinstance(key, str) else np.concatenate([idx_dict[k] for k in key if idx_dict.get(k) is not None])
            if idxs is not None and len(idxs):
                ax.axvspan(
                    x[idxs[0]],
                    x[idxs[-1]],
                    color=self.get_color(role),
                    alpha=0.3,
                    label=name
                )

        # Transmitter & Receiver patches
        p1, p2 = idx_dict['pzt_1'], idx_dict['pzt_2']
        pzt_w = x[p1[-1]] - x[p1[0]]
        pzt_h = 4 * pzt_w

        # transmitter
        x_tr = x[p1[-1]]
        ax.add_patch(Rectangle(
            (x_tr - pzt_w, -pzt_h/2),
            pzt_w, pzt_h,
            color=self.get_color('accent')
        ))
        ax.text(
            x_tr - pzt_w/2, -pzt_h,
            'Transmitter',
            ha='center',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('shadow')
        )

        # receiver
        x_rc = x[p2[0]]
        ax.add_patch(Rectangle(
            (x_rc, -pzt_h/2),
            pzt_w, pzt_h,
            color=self.get_color('accent')
        ))
        ax.text(
            x_rc + pzt_w/2, -pzt_h,
            'Receiver',
            ha='center',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('shadow')
        )

        # side panel for recorded signal
        ax2.set_ylim([t[0], t[-1]])
        ax2.set_xlim([-1, 1])
        ax2.set_ylabel(
            "Recorded Signal",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax2.axis('off')
        ax2.invert_yaxis()

        fig.tight_layout()

        # animation setup
        line_wf, = ax.plot([], [], color=self.get_color('shadow'), lw=self.settings['line_width'])
        line_rc, = ax2.plot([], [], color=self.get_color('shadow'), lw=self.settings['line_width'])

        sp_movie = sp_field[::movie_sampling]
        sp_rec   = sp_recorded[::movie_sampling] / np.amax(np.abs(sp_recorded))
        t_rec    = t[::movie_sampling]
        n_frames = len(sp_movie)

        def update(frame):
            line_wf.set_data(x, sp_movie[frame])
            line_rc.set_data(sp_rec[:frame], t_rec[:frame])
            return line_wf, line_rc

        ani = animation.FuncAnimation(
            fig, update, frames=n_frames,
            blit=True, interval=20
        )
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
        """
        if x.ndim != 1 or c.ndim != 1:
            raise ValueError("x and c must be 1D numpy arrays.")
        if len(x) != len(c):
            raise ValueError("x and c must have the same length.")
        if layer_starts.ndim != 1:
            raise ValueError("layer_starts must be a 1D numpy array.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot the velocity curve
        ax.plot(
            x,
            c,
            label='Velocity Model',
            color=self.get_color('synthetic'),
            linewidth= 2*self.settings['line_width']
        )

        # Define layers with semantic roles
        try:
            layers = [
                ('pla Layer 1', layer_starts[0], layer_starts[1], 'shadow'),
                ('PZT Layer 1', layer_starts[1], layer_starts[2], 'synthetic'),
                ('Side Block 1', layer_starts[2], layer_starts[3], 'misfit_light'),
                ('Groove SB1', layer_starts[3], layer_starts[4], 'shadow'),
                ('Gouge Layer 1', layer_starts[4], layer_starts[5], 'highlight'),
                ('Groove CB1', layer_starts[5], layer_starts[6], 'shadow'),
                ('Central Block', layer_starts[6], layer_starts[7], 'misfit_light'),
                ('Groove CB2', layer_starts[7], layer_starts[8], 'shadow'),
                ('Gouge Layer 2', layer_starts[8], layer_starts[9], 'highlight'),
                ('Groove SB2', layer_starts[9], layer_starts[10], 'shadow'),
                ('Side Block 2', layer_starts[10], layer_starts[11], 'misfit_light'),
                ('PZT Layer 2', layer_starts[11], layer_starts[12], 'synthetic'),
                ('pla Layer 2', layer_starts[12], layer_starts[13], 'shadow'),
            ]
        except IndexError:
            layers = [
                ('pla Layer 1', layer_starts[0], layer_starts[1], 'shadow'),
                ('PZT Layer 1', layer_starts[1], layer_starts[2], 'synthetic'),
                ('Steel Block', layer_starts[2], layer_starts[3], 'misfit_light'),
                ('PZT Layer 2', layer_starts[3], layer_starts[4], 'synthetic'),
                ('pla Layer 2', layer_starts[4], layer_starts[5], 'shadow'),
            ]

        used = set()
        for name, start, end, role in layers:
            label = name if name not in used else None
            used.add(name)
            ax.axvspan(
                start, end,
                color=self.get_color(role),
                alpha=0.3,
                label=label
            )

        # Transmitter and receiver lines
        transmitter_pos = pzt_layer_width + pla_layer_width
        receiver_pos = x[-1] - pzt_layer_width - pla_layer_width
        ax.axvline(
            transmitter_pos,
            color=self.get_color('highlight'),
            linestyle='-',
            label='Transmitter'
        )
        ax.axvline(
            receiver_pos,
            color=self.get_color('accent'),
            linestyle='-',
            label='Receiver'
        )

        # Labels, title, grid, ticks
        ax.set_title(
            "Velocity Model",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Position (cm)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Velocity (cm/$\\mu$s)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))
        ax.legend(
            loc='upper center',
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )

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

        # Plot using the 'synthetic' role color
        ax.plot(
            x,
            spatial_function,
            color=self.get_color('synthetic'),
            linewidth=self.settings['line_width']
        )

        ax.set_title(
            "Synthetic Spatial Function",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Position (cm)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Amplitude",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

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
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # -- Velocities on left y-axis --
        ax.set_xlabel(
            x_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            velocity_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('observed')
        )
        ax.plot(
            x_values,
            velocities,
            color=self.get_color('observed'),
            label=velocity_label,
            linewidth=self.settings['line_width']
        )
        ax.tick_params(
            axis='x',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.tick_params(
            axis='y',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('observed')
        )

        # -- Normal stress on second y-axis --
        ax2 = ax.twinx()
        ax2.set_ylabel(
            stress_labels[0],
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('accent')
        )
        ax2.plot(
            x_values,
            normal_stress,
            color=self.get_color('accent'),
            linestyle='--',
            label=stress_labels[0],
            linewidth=self.settings['line_width']
        )
        ax2.tick_params(
            axis='y',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('accent')
        )

        # -- Shear stress on third y-axis --
        ax2.spines['right'].set_position(('axes', 1.0))
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.15))
        ax3.set_ylabel(
            stress_labels[1],
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('synthetic')
        )
        ax3.plot(
            x_values,
            shear_stress,
            color=self.get_color('synthetic'),
            label=stress_labels[1],
            linewidth=self.settings['line_width']
        )
        ax3.tick_params(
            axis='y',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('synthetic')
        )

        # -- Legend combining all --
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(
            lines + lines2 + lines3,
            labels + labels2 + labels3,
            loc='upper left',
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

        # -- Title, grid, and layout --
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_l2_norm_vs_velocity(self,
                                 velocity: np.ndarray,
                                 L2norm: np.ndarray,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the L2 norm vs velocity.
        """
        if velocity.ndim != 1 or L2norm.ndim != 1:
            raise ValueError("velocity and L2norm must be 1D numpy arrays.")
        if len(velocity) != len(L2norm):
            raise ValueError("velocity and L2norm must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot using semantic colors
        ax.plot(
            velocity,
            L2norm,
            color=self.get_color('synthetic'),
            linewidth=self.settings['line_width']
        )

        ax.set_xlabel(
            'Velocity (cm/$\\mu$s)',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            'L2 Norm of Residuals',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_title(
            'L2 Norm vs Velocity',
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

        fig.tight_layout()
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
        """
        n_reflections = len(reflection_info_list)
        fig, axs = plt.subplots(nrows=n_reflections,
                                figsize=self.settings['figure_size'])
        if n_reflections == 1:
            axs = [axs]

        for ax, info in zip(axs, reflection_info_list):
            reflection_time = info['reflection_time']
            reflection_data = info['reflection_data']
            arr_time        = info['arrival_time']
            corr_coeff      = info['corr_coeff']

            overlay_time = reflection_time - reflection_time[0]
            max_ref = np.max(reflection_data) if reflection_data.size else 1.0
            scale = np.max(direct_wave_data) / max_ref if max_ref != 0 else 1.0

            # Direct waveform
            ax.plot(
                overlay_time,
                direct_wave_data,
                label="Direct",
                color=self.get_color('observed'),
                linewidth=self.settings['line_width']
            )
            # Scaled reflection
            ax.plot(
                overlay_time,
                scale * reflection_data,
                label="Reflection (scaled)",
                color=self.get_color('synthetic'),
                alpha=0.7,
                linewidth=self.settings['line_width']
            )

            ax.set_title(
                f"Direct vs Reflection at {arr_time:.2f} μs, corr={corr_coeff:.3f}",
                fontsize=self.settings['fontsize_subplot_title'],
                fontname=self.FONT_TYPE,
                color=self.get_color('primary')
            )
            ax.set_xlabel(
                "Time [$\\mu s$]",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )
            ax.set_ylabel(
                "Amplitude [a.u.]",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )

            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self.settings['fontsize_ticks'],
                colors=self.get_color('shadow')
            )
            ax.grid(alpha=0.3, color=self.get_color('shadow'))
            ax.legend(
                fontsize=self.settings['fontsize_ticks'],
                facecolor=self.get_color('background'),
                edgecolor=self.get_color('shadow')
            )

        fig.tight_layout()
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
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # -- plot main waveform --
        ax.plot(
            observed_time,
            waveform,
            label="Waveform",
            color=self.get_color('observed'),
            linewidth=self.settings['line_width']
        )

        # -- highlight direct arrival --
        ax.axvspan(
            t_start_direct,
            t_end_direct,
            color=self.get_color('highlight'),
            alpha=0.2,
            label="Direct Arrival"
        )

        # -- labels, title, grid, ticks --
        ax.set_xlabel(
            x_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            y_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.grid(color=self.get_color('shadow'), alpha=0.2)
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )

        # -- highlight each reflection window --
        direct_span = idx_Dend - idx_Dstart
        for i, info in enumerate(reflection_info_list, start=1):
            arr = info['arrival_time']
            start_idx = np.searchsorted(observed_time, arr)
            end_idx   = start_idx + direct_span
            if end_idx > len(observed_time):
                break
            ax.axvspan(
                observed_time[start_idx],
                observed_time[end_idx - 1],
                color=self.get_color('misfit_light'),
                alpha=0.2,
                label="Reflection windows" if i == 1 else None
            )

        # -- legend and finish --
        ax.legend(
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )
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
        """
        if not all(arr.ndim == 1 for arr in [t, stf_updated, stf_original]):
            raise ValueError("t, stf_updated, and stf_original must be 1D numpy arrays.")
        if not (len(t) == len(stf_updated) == len(stf_original)):
            raise ValueError("t, stf_updated, and stf_original must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot original and updated STF using semantic colors
        ax.plot(
            t,
            stf_original,
            label="Original STF",
            color=self.get_color('observed'),
            linewidth=self.settings['line_width'],
            zorder=2
        )
        ax.plot(
            t,
            stf_updated,
            label="Updated STF",
            color=self.get_color('synthetic'),
            linewidth=2 * self.settings['line_width'],
            alpha=1.0,
            zorder=4
        )

        # Title and labels
        ax.set_title(
            "Updating Source Time Function with FWI",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Time [$\\mu s$]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Amplitude [a.u.]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        # Axis limits, grid, ticks, legend
        ax.set_xlim(left=min_time, right=max_time)
        ax.grid(alpha=0.3, color=self.get_color('shadow'))
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.legend(
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

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
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        box = ax.boxplot(
            param_matrix,
            patch_artist=True,
            showmeans=True
        )

        # Customize box colors
        for patch in box['boxes']:
            patch.set(facecolor=self.get_color('highlight'), alpha=0.5)
        for median in box['medians']:
            median.set(color=self.get_color('accent'), linewidth=2)
        for mean_line in box['means']:
            mean_line.set(
                marker='o',
                markerfacecolor=self.get_color('synthetic'),
                markeredgecolor=self.get_color('synthetic'),
                markersize=6
            )

        # X‑axis labels
        ax.set_xticks(np.arange(1, len(param_labels) + 1))
        ax.set_xticklabels(
            param_labels,
            rotation=0,
            fontsize=self.settings['fontsize_ticks'],
            color=self.get_color('shadow')
        )

        # Title and y‑label
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            ylabel,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        ax.tick_params(
            axis='y',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.1, color=self.get_color('shadow'))

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
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Histogram bars with semantic colors
        ax.hist(
            data,
            bins=bins,
            color=self.get_color('highlight'),
            edgecolor=self.get_color('shadow'),
            alpha=0.7
        )

        # Title and axis labels
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            xlabel,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            ylabel,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        # Tick styling and grid
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.1, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_scatter_l2_vs_parameters(
        self,
        param_list: List[Tuple[str, np.ndarray]],
        l2_values: np.ndarray,
        title: str,
        best_index: int = 0,
        outfile_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Creates scatter plots of L2 misfit vs. each parameter in subplots.
        """
        n_params = len(param_list)
        n_cols = 4
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=self.settings['figure_size'],
            tight_layout=True
        )
        axs = axs.flatten()

        for i, (label, param_data) in enumerate(param_list):
            ax = axs[i]
            # scatter all points
            ax.scatter(
                param_data,
                l2_values,
                s=30,
                c=self.get_color('observed'),
                alpha=0.7,
                edgecolors=self.get_color('shadow')
            )
            ax.set_xlabel(
                label,
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )
            ax.set_ylabel(
                "L2 Misfit",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )

            # highlight best point
            best_val = param_data[best_index]
            best_l2 = l2_values[best_index]
            ax.scatter(
                best_val,
                best_l2,
                s=100,
                c=self.get_color('highlight'),
                marker="*",
                zorder=5
            )

            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self.settings['fontsize_ticks'],
                colors=self.get_color('shadow')
            )
            ax.grid(alpha=0.1, color=self.get_color('shadow'))

            min_l2 = np.nanmin(l2_values)
            max_l2 = np.nanmax(l2_values)
            max_ylim = max_l2 if max_l2 < 3*min_l2 else 3*min_l2
            ax.set_ylim([min_l2, max_ylim])

        # hide unused subplots
        for j in range(n_params, n_rows * n_cols):
            axs[j].axis('off')

        fig.suptitle(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def misfit_map(self,
                   misfit_grid: np.ndarray,
                   unique_damps: np.ndarray,
                   unique_vels: np.ndarray,
                   outfile_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plots a 2D misfit surface given a misfit grid.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Display misfit surface
        im = ax.imshow(
            misfit_grid,
            origin='lower',
            extent=[
                min(unique_damps), max(unique_damps),
                min(unique_vels),  max(unique_vels)
            ],
            aspect='auto',
            cmap=self.settings.get('cmap_misfit', 'viridis')
        )

        # Colorbar with semantic styling
        cbar = fig.colorbar(im, ax=ax, pad=0.04)
        cbar.set_label(
            "Misfit (L2 norm)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        cbar.ax.yaxis.set_tick_params(
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        plt.setp(cbar.ax.get_yticklabels(), color=self.get_color('shadow'))

        # Axes labels and title
        ax.set_xlabel(
            "Damping",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Velocity",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_title(
            "2D Misfit Surface",
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )

        # Tick and grid styling
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))

        fig.tight_layout()
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
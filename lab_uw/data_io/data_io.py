# lab_uw/data_io/data_io.py
import numpy as np
import json
import pandas as pd
from scipy.signal import find_peaks

import sys
import logging
from typing import Tuple, Dict, Optional, List, TextIO, Any
import re
from pathlib import Path

from lab_uw.data_io.schema import UWFrame
from lab_uw.data_io.readers.tsv_reader import load_tsv
from lab_uw.data_io.readers.tiepie_reader import load_tiepie_h5

from lab_uw.directory_manager import DirectoryManager
from lab_uw.plotting import Plotter

logger = logging.getLogger(__name__)

###############################################################################
# CLASS: UltrasonicDataHandler
###############################################################################

class UltrasonicDataHandler:
    def __init__(self, waveform_data: np.ndarray = None, metadata: Dict = None):
        if waveform_data is None: waveform_data = np.array([])
        if metadata is None: metadata = {}
        self.frame = UWFrame(waveform_data, metadata).ensure_2d()

    # --- explicit, symmetrical loaders ---
    @classmethod
    def from_tsv(cls, path: Path) -> "UltrasonicDataHandler":
        f = load_tsv(path)
        return cls(f.waveform_data, f.metadata)

    @classmethod
    def from_hdf5(cls, path: Path, **kwargs) -> "UltrasonicDataHandler":
        f = load_tiepie_h5(path, **kwargs)
        return cls(f.waveform_data, f.metadata)

    # Optional: unified chooser (still symmetrical—just dispatches on suffix)
    @classmethod
    def load(cls, path: Path, **h5_options) -> "UltrasonicDataHandler":
        ext = path.suffix.lower()
        if ext in {".tsv", ".txt"}:
            return cls.from_tsv(path)
        if ext in {".h5", ".hdf5"}:
            return cls.from_hdf5(path, **h5_options)
        raise ValueError(f"Unsupported file type: {path}")

    # --- properties matching your old access style ---
    @property
    def waveform_data(self): return self.frame.waveform_data
    @waveform_data.setter
    def waveform_data(self, v): self.frame.waveform_data = v

    @property
    def metadata(self): return self.frame.metadata
    @metadata.setter
    def metadata(self, v): self.frame.metadata = v

    # --- small processors (µs everywhere) ---
    def remove_mean(self) -> "UltrasonicDataHandler":
        self.frame.waveform_data = self.frame.waveform_data - np.mean(self.frame.waveform_data)
        return self

    def downsample_waveforms(self, number_of_waveforms2process: Optional[int]) -> "UltrasonicDataHandler":
        if not number_of_waveforms2process: return self
        n = self.metadata.get("number_of_waveforms", self.waveform_data.shape[0])
        step = max(1, round(n / number_of_waveforms2process))
        self.frame.waveform_data = self.frame.waveform_data[::step, :]
        if "time_ax_acquisition" in self.metadata:
            self.metadata["time_ax_acquisition"] = self.metadata["time_ax_acquisition"][::step]
        self.metadata["number_of_waveforms"] = int(self.frame.waveform_data.shape[0])
        return self

    def truncate_time(self, maxtime_us: float) -> "UltrasonicDataHandler":
        if maxtime_us and maxtime_us > 0:
            t = self.metadata["time_ax_waveform"]
            idx = int(np.searchsorted(t, maxtime_us))
            self.frame.waveform_data = self.frame.waveform_data[:, :idx]
            self.metadata["time_ax_waveform"] = t[:idx]
            self.metadata["number_of_samples"] = int(idx)
        return self

    def zero_before(self, t_us: float) -> "UltrasonicDataHandler":
        if t_us and t_us > 0:
            t = self.metadata["time_ax_waveform"]
            idx = int(np.searchsorted(t, t_us))
            self.frame.waveform_data[:, :idx] = 0.0
        return self

    def lowpass(self, cutoff_mhz: Optional[float]) -> "UltrasonicDataHandler":
        if cutoff_mhz:
            from scipy.signal import butter, lfilter
            def butter_bandpass(lowcut, highcut, fs, order=5):
                from scipy.signal import butter
                return butter(order, [lowcut, highcut], fs=fs, btype='band')
            def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                return lfilter(b, a, data)

            dt_us = self.metadata["sampling_rate"]
            fs_mhz = 1.0 / dt_us
            self.frame.waveform_data = butter_bandpass_filter(
                self.frame.waveform_data, 0.25, cutoff_mhz, fs=fs_mhz
            )
        return self

    @classmethod
    def load_stf(
        cls,
        dir_manager: DirectoryManager,
        machine_name_stf: str,
        experiment_name_stf: str,
        data_type_stf: str,
        stf_chosen: str,
        frequency_cutoff: float = None
    ) -> "UltrasonicDataHandler":
        """
        Creates an UltrasonicDataHandler instance by locating, loading, and processing
        an STF file (JSON or similar), returning a 1D array in waveform_data.
        """
        infile_path_stf_list = dir_manager.make_infile_path_list(
            machine_name=machine_name_stf,
            experiment_name=experiment_name_stf,
            data_type=data_type_stf
        )
        
        chosen_stf_path = None
        for infile_stf in infile_path_stf_list:
            if infile_stf.stem == stf_chosen:
                chosen_stf_path = infile_stf
                break
        if chosen_stf_path is None:
            raise FileNotFoundError(
                f"No stf file named '{stf_chosen}' found in {data_type_stf} "
                f"for experiment '{experiment_name_stf}'."
            )

        # 1) Instantiate the class
        stf_handler = cls()

        stf_handler.infile = chosen_stf_path

        stf_waveform_raw, stf_metadata = stf_handler.load_waveform_json(chosen_stf_path)
        stf_metadata["time_ax_waveform"] = (
            np.array(stf_metadata["time_ax_waveform"]) 
            - np.array(stf_metadata["time_ax_waveform"])[0]
        )

        if frequency_cutoff:
            stf_waveform_filt = butter_bandpass_filter(stf_waveform_raw, 0.25, frequency_cutoff, 1/stf_metadata["sampling_rate"])
            stf_waveform = stf_waveform_filt - stf_waveform_filt[0]

        stf_handler.waveform_data = stf_waveform_raw
        stf_handler.metadata = stf_metadata
        return stf_handler

    def load_waveform_json(self, infile_path: Path) -> Tuple[np.ndarray, Dict]:
        with open(infile_path, "r") as json_file:
            data_dict = json.load(json_file)
        data = np.array(data_dict["data"])
        metadata = data_dict["metadata"]
        return data, metadata

    def save_waveform_json(self, data: np.ndarray, metadata: dict, outfile_path: Path) -> None:
        serialized_metadata = {key: self.serialize_value(value) for key, value in metadata.items()}
        data_dict = {"metadata": serialized_metadata, "data": data.tolist()}
        with open(outfile_path, "w") as output_json:
            json.dump(data_dict, output_json)

    @staticmethod
    def serialize_value(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.str_):
            return str(value)
        return value

    def compute_amplitude_phase_spectrum(self) -> tuple:
            """
            Computes the amplitude and phase spectrum (via FFT) for each waveform
            stored in `self.waveform_data`.

            Returns
            -------
            freq : np.ndarray
                1D array of frequency bins corresponding to the FFT. The units depend on
                the units of 'sampling_rate' in metadata. For example, if 'sampling_rate'
                is in microseconds, `freq` will be in MHz.
            amplitude_spectrum : np.ndarray
                2D array of the amplitude spectrum of shape [n_waveforms, n_samples].
            phase_spectrum : np.ndarray
                2D array of the phase spectrum in radians of shape [n_waveforms, n_samples].
            """
            # Handle the case of empty data
            if self.waveform_data.size == 0:
                raise ValueError("No waveform data is present to compute spectra.")

            # If the data is 1D, reshape to 2D for uniform processing
            data_2d = self.waveform_data

            if self.waveform_data.ndim == 1:
                n_samples = len(data_2d)  # shape (1, n_samples)
            else:
                n_waveforms, n_samples = data_2d.shape  # shape (n_waveforms, n_samples)

            # Retrieve the sampling interval from metadata
            # e.g. if sampling_rate is in microseconds, freq will be in MHz
            if "sampling_rate" not in self.metadata:
                raise KeyError("metadata does not contain 'sampling_rate' key.")

            dt = self.metadata["sampling_rate"]

            # Construct the frequency axis (fftshift not used here; if you prefer a
            # shifted axis, you can use np.fft.fftshift and np.fft.fftfreq accordingly)
            self.frequencies = np.fft.rfftfreq(n_samples, d=dt)

            # Compute the FFT along the sample axis
            try:
                fft_data = np.fft.rfft(data_2d, axis=1)
            except:
                fft_data = np.fft.rfft(data_2d)

            # Compute amplitude and phase
            self.amplitude_spectrum = np.abs(fft_data)
            self.phase_spectrum = np.angle(fft_data)

            return self.frequencies, self.amplitude_spectrum, self.phase_spectrum

    def plot_amplitude_and_phase_spectrum(self):

        plotter = Plotter()
        plotter.filtered_amp_and_phase_spectrum_plot(
                                             signal_freqs = self.frequencies,
                                             amp_spectrum = self.amplitude_spectrum,
                                             phase_spectrum = self.phase_spectrum,
        )
###############################################################################
# CLASS: MechanicalDataHandler
###############################################################################
class MechanicalDataHandler:
    """
    Handles mechanical data operations including preprocessing and synchronization signal extraction.
    """
    def __init__(self, mech_data: pd.DataFrame, metadata: dict):
        self.mech_data = mech_data
        self.metadata = metadata

    @classmethod
    def load_mechanical_data(cls, infile_path: Path) -> "MechanicalDataHandler":
        try:
            mech_data = pd.read_csv(infile_path, engine="python", sep=None, skiprows=[1])
            metadata = {"file_path": infile_path}
            logger.info(f"Mechanical data loaded successfully from {infile_path}.")
            return cls(mech_data=mech_data, metadata=metadata)
        except Exception as e:
            logger.error(f"Error loading mechanical data from {infile_path}: {e}")
            raise

    @classmethod
    def locate_and_load_data(
        cls,
        dir_manager: DirectoryManager,
        machine_name: str,
        experiment_name: str,
        data_type_mech: str,
        mech_file_name: str
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Class-level approach: find the mechanical CSV via dir_manager, load it, then find sync.
        Returns (mech_data, sync_data, sync_peaks).
        """
        infile_path_list_mech = dir_manager.make_infile_path_list(
            machine_name, experiment_name, data_type=data_type_mech
        )
        mech_data_path = None
        for infile_path in infile_path_list_mech:
            if infile_path.name == mech_file_name:
                mech_data_path = infile_path
                break
        if mech_data_path is None:
            raise FileNotFoundError(f"{mech_file_name} not found in mechanical data.")

        # Instantiate
        handler = cls.load_mechanical_data(mech_data_path)
        mech_data = handler.mech_data
        sync_data, sync_peaks = handler.find_sync_values()
        return mech_data, sync_data, sync_peaks

    def find_sync_values(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            sync_data = self.mech_data['sync'].values
            sync_peaks, _ = find_peaks(sync_data, prominence=4.2, height=4)
            logger.info(f"Found {len(sync_peaks)} synchronization peaks.")
            return sync_data, sync_peaks
        except KeyError:
            logger.error("Synchronization column 'sync' is missing in the mechanical data.")
            return None, None

class BlockMetadataHandler:
    """
    Loads geometry and velocity metadata for blocks (side blocks, central block, etc.)
    from a JSON file (e.g., blocks_metadata.json).
    """

    def __init__(self, metadata_dict: dict):
        """
        Initialize the BlockMetadataHandler with a dictionary of block metadata.

        Parameters
        ----------
        metadata_dict : dict
            Dictionary loaded from JSON, containing multiple block entries
            (e.g., 'central_block1', 'mauro_side1', etc.).
        """
        self._metadata_dict = metadata_dict

    @classmethod
    def from_json(cls, config_path: Path) -> "BlockMetadataHandler":
        """
        Create a BlockMetadataHandler instance by loading from a JSON file.

        Parameters
        ----------
        config_path : Path
            Path to the blocks_metadata.json file.

        Returns
        -------
        BlockMetadataHandler
            A handler instance containing all blocks' metadata.
        """
        try:
            with config_path.open("r") as f:
                raw_data = json.load(f)
            return cls(raw_data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load block metadata from {config_path}: {e}")
            raise

    def get_block_params(self, block_key: str) -> dict:
        """
        Retrieve parameters for a specific block (e.g., 'mauro_side1').

        Parameters
        ----------
        block_key : str
            Key identifying the block in the loaded dictionary (e.g., 'pignalberi_side1').

        Returns
        -------
        dict
            Dictionary of parameters for that block.

        Raises
        ------
        KeyError
            If the specified block_key is not found.
        """
        if block_key not in self._metadata_dict:
            raise KeyError(f"Block '{block_key}' not found in metadata.")
        return self._metadata_dict[block_key]

    def rename_block_key(self, old_key: str, new_key: str) -> None:
        """
        Rename a block key in the internal metadata dictionary.
        
        Parameters
        ----------
        old_key : str
            Existing block key to rename.
        new_key : str
            New block key name to use.

        Raises
        ------
        KeyError
            If old_key is not found.
        """
        if old_key not in self._metadata_dict:
            raise KeyError(f"Block '{old_key}' not found in metadata.")
        # Move the data under 'old_key' to 'new_key' and remove 'old_key'
        self._metadata_dict[new_key] = self._metadata_dict.pop(old_key)

    def update_block_params(self, block_key: str, updates: dict) -> None:
        """
        Update (or add) parameters for a specific block in the dictionary.
        
        Parameters
        ----------
        block_key : str
            The block to be updated (e.g., 'mauro_side1').
        updates : dict
            A dictionary of key-value pairs to be merged into that block's metadata.
            If a key doesn't exist, it will be created; if it does exist, it will be overwritten.
        """
        if block_key not in self._metadata_dict:
            # Optionally raise an error instead of creating a new block:
            # raise KeyError(f"Block '{block_key}' not found, cannot update.")
            logger.info(f"Block '{block_key}' not found; creating a new block entry.")
            self._metadata_dict[block_key] = {}

        for k, v in updates.items():
            self._metadata_dict[block_key][k] = v

    def save_blocks_metadata(self, config_path: Path) -> None:
        """
        Save the internal metadata dictionary to a JSON file.

        Parameters
        ----------
        config_path : Path
            File path at which to save the JSON data.
        """
        try:
            with config_path.open("w") as f:
                json.dump(self._metadata_dict, f, indent=2)
            logger.info(f"Block metadata successfully saved to {config_path}")
        except OSError as e:
            logger.error(f"Failed to write block metadata to {config_path}: {e}")
            raise

    @classmethod
    def load_blocks_metadata(
        cls,
        dir_manager: Any,
        blocks_metadata_name: str,
        block_keys: Tuple[str, ...]
    ) -> Tuple[dict, ...]:
        """
        Class method that:
          1) Builds the path from a DirectoryManager + blocks_metadata_name
          2) Loads the JSON into a BlockMetadataHandler
          3) Retrieves a tuple of block dictionaries for the given 'block_keys'

        Parameters
        ----------
        dir_manager : DirectoryManager
            Directory manager for building paths.
        blocks_metadata_name : str
            Name of the blocks metadata JSON file (e.g. "blocks_metadata.json").
        block_keys : Tuple[str, ...]
            Keys in the JSON for the blocks (e.g. ("mauro_side1", "central_block1"))

        Returns
        -------
        Tuple[dict, ...]
            A tuple of dictionaries for each block key.
        """
        blocks_metadata_path = dir_manager.base_dir / "metadata" / blocks_metadata_name
        block_handler = cls.from_json(blocks_metadata_path)

        results = []
        for bk in block_keys:
            results.append(block_handler.get_block_params(bk))

        return tuple(results)
    
from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
# lab_uw/data_io.py

import numpy as np
import json
import sys
import pandas as pd
from scipy.signal import find_peaks
import logging
from typing import Tuple, Dict, Optional, List, TextIO, Any
import re
from pathlib import Path

from lab_uw.directory_manager import DirectoryManager
from lab_uw.signal_processing import SignalProcessor
from lab_uw.plotting import Plotter

logger = logging.getLogger(__name__)

###############################################################################
# CLASS: UltrasonicDataHandler
###############################################################################
class UltrasonicDataHandler:
    """
    Handles ultrasonic data operations such as reading, processing, and saving waveform data.
    """
    def __init__(self, waveform_data: np.ndarray = None, metadata: Dict = None):
        if waveform_data is None:
            waveform_data = np.array([])
        if metadata is None:
            metadata = {}
        self.waveform_data = waveform_data
        self.metadata = metadata

    @classmethod
    def load_multi_channel_UW_data(cls, infile_path: Path) -> Dict[str, "UltrasonicDataHandler"]:
        """
        Reads a TSV file that may contain multiple channel blocks,
        returning a dictionary of channel_name -> UltrasonicDataHandler.
        """
        channel_dict = {}

        with open(infile_path, "r", encoding='iso8859') as infile:
            while True:
                # 1) Read next channel's metadata
                header_info = cls.extract_next_channel_metadata(infile)
                if not header_info:
                    # No more channels found
                    break
                (
                    channel_name,
                    amplitude_info,
                    time_info,
                    axis_info
                ) = header_info

                offset = float(time_info[0])
                scale = float(time_info[1])
                n_samples = int(time_info[2])
                sampling_rate = float(time_info[3])  # microseconds or whatever

                time_ax_waveform = np.arange(offset, offset + scale*n_samples, sampling_rate)

                # 3) Read waveforms belonging to the current channel
                waveform_2d = cls.read_waveforms_until_next_channel(infile)
                data_array = np.array(waveform_2d, dtype=float)

                # 4) Build metadata dict
                metadata = {
                    "channel_name": channel_name,
                    "number_of_samples": n_samples,
                    "sampling_rate": sampling_rate,
                    "time_ax_waveform": time_ax_waveform,
                    # You can add amplitude_info, axis_info, etc.:
                    "amplitude_info": amplitude_info,
                    "axis_info": axis_info,
                }

                # 5) Create an UltrasonicDataHandler for this channel
                handler = cls(waveform_data=data_array, metadata=metadata)

                # 6) Store in dictionary
                channel_dict[channel_name] = handler

        return channel_dict

    @staticmethod
    def extract_next_channel_metadata(infile: TextIO) -> Optional[Tuple[str, List[float], List[float], List[float]]]:
        """
        Reads the next 4 lines of channel metadata from the file.
        Returns (channel_name, amplitude_info, time_info, axis_info)
        or None if no further channel found.
        """
        # Attempt to read the next four lines
        lines = []
        for _ in range(4):
            pos = infile.tell()
            line = infile.readline()
            if not line:
                # EOF or missing lines => no more channels
                return None
            # If this is not truly a header line, we might guess we reached next channel
            # or there's a data line in between. Usually the format is strict, so let's
            # just gather them for now.
            lines.append(line.strip())

        # Check that the first line has "channel:"
        if "channel:" not in lines[0]:
            # This means those lines are not a valid channel header
            return None

        # The lines structure presumably:
        # 0: "Bscan Image from channel:s1p1 ..."
        # 1: "[Amplitude Scale] ... min:-100%=-2048 max:..."
        # 2: "[Time Scale] ... offset=0 scale=250 sample=6250 sampling=..."
        # 3: "[Axis Scale] start=0 end=1 step=0.1"

        general_line = lines[0]
        amplitude_line = lines[1]
        time_line = lines[2]
        axis_line = lines[3]

        # Extract the channel_name
        # e.g. "Bscan Image from channel:s1p1 ..." => split at "channel:"
        channel_part = general_line.split("channel:")[1].strip()
        channel_name = channel_part.split()[0].rstrip(",")  # i.e: 's1p1'

        amplitude_nums = re.findall(r"[-]?\d+\.*\d*", amplitude_line)  # get numeric portions
        time_nums = re.findall(r"[-]?\d+\.*\d*", time_line)
        axis_nums = re.findall(r"[-]?\d+\.*\d*", axis_line)

        # Convert each to float
        amplitude_info = [float(x) for x in amplitude_nums]
        time_info = [float(x) for x in time_nums]
        axis_info = [float(x) for x in axis_nums]

        return (channel_name, amplitude_info, time_info, axis_info)

    @staticmethod
    def read_waveforms_until_next_channel(infile: TextIO) -> List[List[float]]:
        """
        Reads waveform lines until we hit another 'Bscan Image from channel:'
        or EOF. Returns a list of waveforms (each waveform is a list of floats).
        """
        waveforms = []
        while True:
            pos = infile.tell()
            line = infile.readline()
            if not line:
                # EOF
                break

            # Check if line looks like the next channel header
            if "Bscan Image from channel:" in line:
                # We have gone one line too far. Rewind and stop reading waveforms
                infile.seek(pos)
                break

            # Otherwise parse the line as numeric values, if possible:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                row_values = [float(val) for val in line.split()]
                waveforms.append(row_values)
            except ValueError:
                # Possibly a weird line or partial text.  You can decide to break or skip.
                logger.debug(f"Skipping non-numeric line: {line}")
                continue

        return waveforms
    
    @classmethod
    def load_UW_data(cls, infile_path: Path) -> "UltrasonicDataHandler":
        """
        Creates an UltrasonicDataHandler instance by loading data from a TSV file.
        (Raw ultrasonic waveforms + metadata)
        """
        with open(infile_path, "r", encoding='iso8859') as infile:
            acquisition_info, time_info, channel_name = cls.extract_metadata_from_tsv(infile)
            number_of_samples = int(time_info[2])
            sampling_rate = time_info[3]  # microseconds
            time_ax_waveform = np.arange(time_info[0], time_info[1], sampling_rate)
            acquisition_frequency = acquisition_info[2]
            time_ax_acquisition = np.arange(acquisition_info[0], acquisition_info[1], acquisition_frequency)

            waveform_list = cls.read_waveforms(infile)
            data = np.array(waveform_list).astype(float)
            corrected_number_of_waveforms = data.shape[0]
            time_ax_acquisition = time_ax_acquisition[:corrected_number_of_waveforms]

            metadata = {
                "number_of_samples": number_of_samples,
                "sampling_rate": sampling_rate,
                "time_ax_waveform": time_ax_waveform,
                "acquisition_frequency": acquisition_frequency,
                "number_of_waveforms": data.shape[0],
                "time_ax_acquisition": time_ax_acquisition,
                "channel_name": channel_name
            }

        return cls(waveform_data=data, metadata=metadata)

    @classmethod
    def load_and_process_uw(
        cls,
        infile_path: Path,
        remove_mean: bool = True,
        number_of_waveforms2process: int = None,
        maxtime2simulate: float = 0,
        zero_out_time: float = 0,
        frequency_cutoff: float = None,
        time_ax_acquisition_start: float = None
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """
        High-level method to:
        1) load the .tsv ultrasonic data
        2) remove mean,
        3) optionally truncate at maxtime2simulate,
        4) Downsampling the number of waveforms to analyze
        5) optionally zero out data up to 'zero_out_time'
        6) lowpass filter

        Returns
        -------
        observed_waveform_data : np.ndarray
            The 2D array of shape [n_waveforms, n_samples], after processing.
        metadata : Dict[str, Any]
            The original metadata from the .tsv file (with minor changes if truncated).
        """
        # Load raw data from .tsv
        handler = cls.load_UW_data(infile_path)
        observed_waveform_data = handler.waveform_data
        metadata = handler.metadata

        if remove_mean:
            observed_waveform_data = observed_waveform_data - np.mean(observed_waveform_data)

        # Downsampling the number of waveforms to analyze and edit the metadata accordingly
        if isinstance(number_of_waveforms2process,int) and number_of_waveforms2process>0:
            downsampling = max(1, round(metadata["number_of_waveforms"] / number_of_waveforms2process)) 
            print(f"Number of waveforms: {metadata['number_of_waveforms']}, wanting {number_of_waveforms2process}, downsampling factor: {downsampling}")

            observed_waveform_data = observed_waveform_data[::downsampling,:]
            metadata["time_ax_acquisition"] = metadata["time_ax_acquisition"][::downsampling]
            metadata["number_of_waveforms"] = len(observed_waveform_data)

        # Possibly reduce the number of samples (time-limiting the simulation of a waveform)
        if maxtime2simulate > 0:
            idx_maxtime = np.searchsorted(metadata["time_ax_waveform"], maxtime2simulate)
            observed_waveform_data = observed_waveform_data[:, :idx_maxtime]
            metadata["time_ax_waveform"] = metadata["time_ax_waveform"][:idx_maxtime]
            metadata["number_of_samples"] = len(metadata["time_ax_waveform"])

        # Zero out data up to zero_out_time
        if zero_out_time > 0:
            idx_zero_out = np.searchsorted(metadata["time_ax_waveform"], zero_out_time)
            observed_waveform_data[:, :idx_zero_out] = 0.0

        # Lowpass filtering
        if frequency_cutoff:
            observed_waveform_data = butter_bandpass_filter(observed_waveform_data, 0.25, frequency_cutoff, 1/metadata["sampling_rate"])

        if time_ax_acquisition_start:
            metadata["time_ax_acquisition"] = metadata["time_ax_acquisition"] + time_ax_acquisition_start

        handler.waveform_data = observed_waveform_data
        handler.metadata      = metadata

        return handler

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


    @staticmethod
    def extract_metadata_from_tsv(infile: TextIO) -> Tuple[List[float], List[float]]:
        infile.seek(0)
        general = infile.readline().strip()
        amplitude_scale = infile.readline().strip()
        time_scale = infile.readline().strip()
        acquisition_scale = infile.readline().strip()

        channel_name = general.split(":")[1]  # Euroscan allowes for saving multiple channels in the same tsv file
        acquisition_info = [float(entry) for entry in re.findall(r"\d+\.*\d*", acquisition_scale)]
        time_info = [float(entry) for entry in re.findall(r"\d+\.*\d*", time_scale)]
        return acquisition_info, time_info, channel_name

    @staticmethod
    def read_waveforms(infile: TextIO) -> List[List[float]]:
        waveform_list = []
        for line in infile:
            line = line.strip()
            if line:
                try:
                    waveform_list.append([float(value) for value in line.split()])
                except ValueError:                        
                    break
        return waveform_list

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
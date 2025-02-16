import numpy as np
import json
import pandas as pd
from scipy.signal import find_peaks
import logging
from typing import Tuple, Dict, Optional, List, TextIO, Any
import re
from pathlib import Path

from lab_uw.directory_manager import DirectoryManager
from lab_uw.signal_processing import SignalProcessor

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
    def load_UW_data(cls, infile_path: Path) -> "UltrasonicDataHandler":
        """
        Creates an UltrasonicDataHandler instance by loading data from a TSV file.
        (Raw ultrasonic waveforms + metadata)
        """
        with open(infile_path, "r", encoding='iso8859') as infile:
            acquisition_info, time_info = cls.extract_metadata_from_tsv(infile)
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
                "time_ax_acquisition": time_ax_acquisition
            }

        return cls(waveform_data=data, metadata=metadata)

    @classmethod
    def load_and_process_uw(
        cls,
        infile_path: Path,
        zero_out_time: float = 0,
        frequency_cutoff_MHz: float = None,
        maxtime2simulate: float = 0,
        number_of_waveforms_to_process: int = None
    ) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
        """
        High-level method to:
        1) load the .tsv ultrasonic data
        2) remove mean,
        3) optionally truncate at maxtime2simulate,
        4) compute downsampling factor
        5) optionally zero out data up to 'zero_out_time'
        6) lowpass filter

        Returns
        -------
        observed_waveform_data : np.ndarray
            The 2D array of shape [n_waveforms, n_samples], after processing.
        observed_time : np.ndarray
            The truncated time axis.
        downsampling : int
            The computed downsampling factor.
        metadata : Dict[str, Any]
            The original metadata from the .tsv file (with minor changes if truncated).
        """
        # 1) Load raw data from .tsv
        handler = cls.load_UW_data(infile_path)
        observed_waveform_data = handler.waveform_data
        metadata = handler.metadata
        observed_time = metadata["time_ax_waveform"]

        # 2) Remove mean
        observed_waveform_data = observed_waveform_data - np.mean(observed_waveform_data)

        # 4) Downsampling factor
        if number_of_waveforms_to_process > 0:
            total_waveforms = metadata["number_of_waveforms"]
            downsampling = max(1, round(total_waveforms / number_of_waveforms_to_process)) 

        # 3) Possibly reduce the number of samples (time-limiting)
        if maxtime2simulate > 0:
            idx_maxtime = np.searchsorted(observed_time, maxtime2simulate)
            observed_waveform_data = observed_waveform_data[:, :idx_maxtime]
            observed_time = observed_time[:idx_maxtime]
            # (Optionally update metadata if you need the truncated shape/time.)
            metadata["time_ax_waveform"] = observed_time
            metadata["number_of_samples"] = len(observed_time)

        # 5) Zero out data up to zero_out_time
        if zero_out_time > 0:
            idx_zero_out = np.searchsorted(observed_time, zero_out_time)
            observed_waveform_data[:, :idx_zero_out] = 0.0

        # 6) Lowpass filtering
        if frequency_cutoff_MHz:
            signal_processor = SignalProcessor()
            observed_waveform_data, _ = signal_processor.signal2noise_separation_lowpass(
                waveform_data=observed_waveform_data,
                metadata=metadata,
                freq_cut=frequency_cutoff_MHz
            )

        return observed_waveform_data, observed_time, downsampling, metadata

    @classmethod
    def load_stf(
        cls,
        dir_manager: DirectoryManager,
        machine_name_stf: str,
        experiment_name_stf: str,
        data_type_stf: str,
        stf_chosen: str,
        frequency_cutoff_MHz: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        High-level method to find, load, filter, and zero an STF from a JSON (or TSV).
        Returns (stf_waveform, stf_time, stf_duration).
        """
        # 1) Locate the stf file
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

        # 2) Load stf data (JSON or TSV) using an empty handler instance
        temp_handler = cls()
        stf_waveform_raw, stf_metadata = temp_handler.load_waveform_json(chosen_stf_path)

        stf_time = np.array(stf_metadata["time_ax_waveform"])
        signal_processor = SignalProcessor()

        # 3) Lowpass filter
        stf_waveform_filt, _ = signal_processor.signal2noise_separation_lowpass(
            waveform_data=stf_waveform_raw,
            metadata=stf_metadata,
            freq_cut=frequency_cutoff_MHz
        )

        # 4) Zero start
        stf_waveform = stf_waveform_filt - stf_waveform_filt[0]
        stf_duration = stf_time[-1] - stf_time[0]

        return stf_waveform, stf_time, stf_duration

    @staticmethod
    def extract_metadata_from_tsv(infile: TextIO) -> Tuple[List[float], List[float]]:
        infile.seek(0)
        general = infile.readline().strip()
        amplitude_scale = infile.readline().strip()
        time_scale = infile.readline().strip()
        acquisition_scale = infile.readline().strip()

        acquisition_info = [float(entry) for entry in re.findall(r"\d+\.*\d*", acquisition_scale)]
        time_info = [float(entry) for entry in re.findall(r"\d+\.*\d*", time_scale)]
        return acquisition_info, time_info

    @staticmethod
    def read_waveforms(infile: TextIO) -> List[List[float]]:
        waveform_list = []
        for line in infile:
            line = line.strip()
            if line:
                try:
                    waveform_list.append([float(value) for value in line.split()])
                except ValueError:
                    pass
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
            mech_data = pd.read_csv(infile_path, sep=',', skiprows=[1])
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
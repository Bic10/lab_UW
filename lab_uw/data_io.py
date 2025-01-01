# lab_uw/data_io.py

import numpy as np
import json
import pandas as pd
from scipy.signal import find_peaks
import logging
from typing import Tuple, Dict, Optional, List, TextIO
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Class 1: Ultrasonic Data Handler

class UltrasonicDataHandler:
    """
    Handles ultrasonic data operations such as reading, processing, and saving waveform data.
    """
    def __init__(self, waveform_data: np.ndarray= np.array([]), metadata: dict={}):
        """
        Initializes the UltrasonicDataHandler.

        Parameters
        ----------
        waveform_data : np.ndarray
            Numpy array containing waveform data.
        metadata : dict
            Metadata about the ultrasonic data.
        """
        self.waveform_data = waveform_data
        self.metadata = metadata

    @classmethod
    def make_UW_data(cls, infile_path: Path) -> "UltrasonicDataHandler":
        """
        Creates an UltrasonicDataHandler instance by loading data from a TSV file.

        Parameters
        ----------
        infile_path : Path
            Path to the input TSV file containing ultrasonic data.

        Returns
        -------
        UltrasonicDataHandler
            An instance of the UltrasonicDataHandler class initialized with loaded data.
        """
        with open(infile_path, "r", encoding='iso8859') as infile:
            acquisition_info, time_info = cls.extract_metadata_from_tsv(infile)
            number_of_samples = int(time_info[2])
            sampling_rate = time_info[3]  # [microseconds]
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
                'time_ax_acquisition': time_ax_acquisition
            }

        return cls(waveform_data=data, metadata=metadata)

    @staticmethod
    def extract_metadata_from_tsv(infile: TextIO) -> Tuple[List[float], List[float]]:
        """
        Extracts metadata from the header of a TSV file.
        """
        infile.seek(0)
        general = infile.readline().strip()
        amplitude_scale = infile.readline().strip()
        time_scale = infile.readline().strip()
        acquisition_scale = infile.readline().strip()

        acquisition_info = [float(entry) for entry in re.findall(r"\d+\.*\d*", acquisition_scale)]
        time_info = [float(entry) for entry in re.findall(r"\d+\.*\d*", time_scale)]
        return acquisition_info, time_info

    @staticmethod
    def read_waveforms(infile: TextIO) -> List[np.ndarray]: 
        """
        Reads waveform data from the body of a TSV file.
        """
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
        """
        Loads waveform data and metadata from a JSON file.

        Parameters
        ----------
        infile_path : Path
            Path to the JSON file containing waveform data.

        Returns
        -------
        Tuple[np.ndarray, Dict]
            A tuple containing:
            - numpy array of waveform data
            - metadata dictionary
        """
        with open(infile_path, "r") as json_file:
            data_dict = json.load(json_file)
        data = np.array(data_dict["data"])
        metadata = data_dict["metadata"]
        return data, metadata

    def save_waveform_json(self, data: np.ndarray, metadata: dict, outfile_path: Path) -> None:
        """
        Saves waveform data and metadata to a JSON file.

        Parameters
        ----------
        data : np.ndarray
            Waveform data to save.
        metadata : dict
            Dictionary of metadata to save.
        outfile_path : Path
            Path to the output JSON file.
        """
        serialized_metadata = {key: self.serialize_value(value) for key, value in metadata.items()}
        data_dict = {"metadata": serialized_metadata, "data": data.tolist()}
        with open(outfile_path, "w") as output_json:
            json.dump(data_dict, output_json)

    @staticmethod
    def serialize_value(value):
        """
        Serializes a value to make it compatible with JSON.

        Parameters
        ----------
        value : any
            Value to serialize (e.g., numpy types).

        Returns
        -------
        Any
            Serialized value compatible with JSON.
        """
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.str_):
            return str(value)
        return value



# Class 2: Mechanical Data Handler
class MechanicalDataHandler:
    """
    Handles mechanical data operations including preprocessing and synchronization signal extraction.
    """
    def __init__(self, mech_data: pd.DataFrame, metadata: dict):
        """
        Initializes the MechanicalDataHandler.

        Parameters
        ----------
        mech_data : pd.DataFrame
            Pandas DataFrame containing mechanical data, including a 'sync' column for synchronization signals.
        metadata : dict
            Metadata about the mechanical data.
        """
        self.mech_data = mech_data
        self.metadata = metadata

    @classmethod
    def make_mechanical_data(cls, infile_path: Path) -> "MechanicalDataHandler":
        """
        Creates a MechanicalDataHandler instance by loading mechanical data from a CSV file.

        Parameters
        ----------
        infile_path : Path
            Path to the input mechanical data CSV file.

        Returns
        -------
        MechanicalDataHandler
            An instance of the MechanicalDataHandler class initialized with loaded data.
        """
        try:
            mech_data = pd.read_csv(infile_path, sep=',', skiprows=[1])
            metadata = {"file_path": infile_path}
            logger.info(f"Mechanical data loaded successfully from {infile_path}.")
            return cls(mech_data=mech_data, metadata=metadata)
        except Exception as e:
            logger.error(f"Error loading mechanical data from {infile_path}: {e}")
            raise

    def find_sync_values(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Finds synchronization peaks in the 'sync' column of the mechanical data.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            A tuple containing:
            - Array of synchronization data
            - Indices of synchronization peaks

        Raises
        ------
        KeyError
            If the 'sync' column is missing from the DataFrame.
        """
        try:
            sync_data = self.mech_data['sync'].values  # Assuming 'sync' is a column in the DataFrame
            sync_peaks, _ = find_peaks(sync_data, prominence=4.2, height=4)
            logger.info(f"Found {len(sync_peaks)} synchronization peaks.")
            return sync_data, sync_peaks
        except KeyError:
            logger.error("Synchronization column 'sync' is missing in the mechanical data.")
            return None, None

# Class 3: Data Coupling Handler
class DataCouplingHandler:
    """
    Handles the integration of ultrasonic and mechanical data using synchronization peaks.
    """
    def align_data(
        self,
        ultrasonic_data: np.ndarray,
        ultrasonic_time: np.ndarray,
        mechanical_data: pd.DataFrame,
        mechanical_time: np.ndarray,
        sync_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Aligns ultrasonic and mechanical data using synchronization peaks.

        Parameters
        ----------
        ultrasonic_data : np.ndarray
            Ultrasonic waveform data.
        ultrasonic_time : np.ndarray
            Time axis for ultrasonic data.
        mechanical_data : pd.DataFrame
            Mechanical data as a DataFrame.
        mechanical_time : np.ndarray
            Time axis for mechanical data.
        sync_indices : np.ndarray
            Synchronization peak indices.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]
            Aligned ultrasonic data, time axis, and mechanical data.

        Raises
        ------
        ValueError
            If no synchronization indices are provided.
        """
        if not sync_indices.size:
            raise ValueError("No synchronization indices found for alignment.")

        start_sync_index = sync_indices[0]
        mech_start_time = mechanical_time[start_sync_index]

        ultrasonic_time_aligned = ultrasonic_time + mech_start_time

        logger.info("Data successfully aligned using synchronization peaks.")
        return ultrasonic_data, ultrasonic_time_aligned, mechanical_data

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

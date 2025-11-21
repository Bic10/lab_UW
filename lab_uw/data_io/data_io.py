# lab_uw/data_io/data_io.py

import numpy as np
import json
import pandas as pd
from scipy.signal import find_peaks

import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

from lab_uw.data_io.schema import UWFrame
from lab_uw.data_io.readers.tsv_reader import read_tsv
from lab_uw.data_io.readers.hdf5_reader import read_hdf5, read_stf_hdf5
from lab_uw.data_io.writers.hdf5_writer import write_hdf5, write_stf_hdf5

logger = logging.getLogger(__name__)

class UltrasonicDataHandler:
    """
    Thin façade around UWFrame with:
      - symmetric readers: read_tsv(), read_hdf5()
      - open() dispatcher by file extension
      - STF helpers: read_stf_hdf5(), write_stf_hdf5()
      - small, chainable processors (µs everywhere)
    """
    def __init__(self, waveform_data: np.ndarray = None, metadata: Dict = None):
        if waveform_data is None: waveform_data = np.array([])
        if metadata is None: metadata = {}
        self.frame = UWFrame(waveform_data, metadata).ensure_2d()

    # --------- symmetric readers ----------
    @classmethod
    def read_tsv(cls, path: Path) -> "UltrasonicDataHandler":
        f = read_tsv(path)
        return cls(f.waveform_data, f.metadata)

    @classmethod
    def read_hdf5(cls, path: Path, **kwargs) -> "UltrasonicDataHandler":
        f = read_hdf5(path, **kwargs)
        return cls(f.waveform_data, f.metadata)

    # unified dispatcher (by suffix)
    @classmethod
    def open(cls, path: Path, **h5_options) -> "UltrasonicDataHandler":
        ext = path.suffix.lower()
        if ext in {".tsv", ".txt"}:
            return cls.read_tsv(path)
        if ext in {".h5", ".hdf5"}:
            return cls.read_hdf5(path, **h5_options)
        raise ValueError(f"Unsupported file type: {path}")

    # --------- STF helpers (HDF5) ----------
    @classmethod
    def read_stf_h5(cls, h5_path: Path, *, name: str, cycle_index: int = 0) -> "UltrasonicDataHandler":
        """Read /stf/<name>/cycle_xxxxxx/source_waveform -> handler with (1, L)."""
        f = read_stf_hdf5(h5_path, name=name, cycle_index=cycle_index)
        return cls(f.waveform_data, f.metadata)

    # --------- (optional) acquisition writer convenience ----------
    def write_hdf5(
        self,
        out_path: Path,
        run_name: str,
        *,
        channel_count: int = 1,
        dataset: str = "active",
        add_timestamps: bool = False,
    ) -> None:
        """Thin wrapper around writers.write_hdf5 for acquisitions."""
        write_hdf5(
            frame=self.frame,
            out_path=out_path,
            run_name=run_name,
            channel_count=channel_count,
            dataset=dataset,
            add_timestamps=add_timestamps,
        )

    def write_stf_hdf5(
        self,
        out_path: Path,
        stf_name: str,
        *,
        cycle_index: int = 0,
        include_active_stub: bool = False,
        include_passive_stub: bool = False,
        t0_unix_s: Optional[float] = None,
    ) -> None:
        """Persist row 0 as /stf/<stf_name>/cycle_xxxxxx/source_waveform."""
        y = self.frame.waveform_data
        if y.ndim == 2:
            if y.shape[0] == 0:
                raise ValueError("No STF data to write.")
            if y.shape[0] > 1:
                logger.warning("STF writer: multiple rows detected; writing only row 0.")
            y = y[0]
        stf_frame = UWFrame(y[None, :], self.frame.metadata)
        write_stf_hdf5(
            stf=stf_frame,
            out_path=out_path,
            stf_name=stf_name,
            cycle_index=cycle_index,
            include_active_stub=include_active_stub,
            include_passive_stub=include_passive_stub,
            t0_unix_s=t0_unix_s,
        )

    # --------- properties ----------
    @property
    def waveform_data(self): return self.frame.waveform_data
    @waveform_data.setter
    def waveform_data(self, v): self.frame.waveform_data = v

    @property
    def metadata(self): return self.frame.metadata
    @metadata.setter
    def metadata(self, v): self.frame.metadata = v

    # --------- small processors (µs units) ----------
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
            def _butter_band(lowcut, highcut, fs, order=5):
                from scipy.signal import butter
                return butter(order, [lowcut, highcut], fs=fs, btype='band')
            def _filt(data, lowcut, highcut, fs, order=5):
                b, a = _butter_band(lowcut, highcut, fs, order=order)
                return lfilter(b, a, data)
            dt_us = float(self.metadata["sampling_rate"])
            fs_mhz = 1.0 / dt_us
            self.frame.waveform_data = _filt(self.frame.waveform_data, 0.25, cutoff_mhz, fs=fs_mhz)
        return self

    # --------- spectra ----------
    def compute_amplitude_phase_spectrum(self) -> tuple:
        if self.waveform_data.size == 0:
            raise ValueError("No waveform data is present to compute spectra.")
        data_2d = self.waveform_data
        n_samples = data_2d.shape[-1]
        if "sampling_rate" not in self.metadata:
            raise KeyError("metadata does not contain 'sampling_rate' key.")
        dt = float(self.metadata["sampling_rate"])        # µs/sample
        self.frequencies = np.fft.rfftfreq(n_samples, d=dt)
        try:
            fft_data = np.fft.rfft(data_2d, axis=1)
        except Exception:
            fft_data = np.fft.rfft(data_2d)
        self.amplitude_spectrum = np.abs(fft_data)
        self.phase_spectrum = np.angle(fft_data)
        return self.frequencies, self.amplitude_spectrum, self.phase_spectrum

        
#########s######################################################################
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
            metadata = {"file_path": Path(infile_path)}
            logger.info(f"Mechanical data loaded successfully from {infile_path}.")
            return cls(mech_data=mech_data, metadata=metadata)
        except Exception as e:
            logger.error(f"Error loading mechanical data from {infile_path}: {e}")
            raise

    # NEW: simple resolver by folder + file name (no DirectoryManager)
    @classmethod
    def find_and_load(
        cls,
        folder: Path,
        filename: str,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load mechanical CSV from an explicit folder/filename and return
        (mech_data, sync_data, sync_peaks).
        """
        path = Path(folder) / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        handler = cls.load_mechanical_data(path)
        sync_data, sync_peaks = handler.find_sync_values()
        return handler.mech_data, sync_data, sync_peaks

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
        config_path: Path,
        block_keys: Tuple[str, ...]
    ) -> Tuple[dict, ...]:
        """
        Path-first loader: pass the full JSON path (e.g., base/metadata/blocks_metadata.json)
        and the block keys to extract.
        """
        handler = cls.from_json(Path(config_path))
        return tuple(handler.get_block_params(bk) for bk in block_keys) 
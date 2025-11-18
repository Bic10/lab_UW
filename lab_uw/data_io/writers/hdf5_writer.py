# lab_uw/data_io/writers/hdf5_writer.py
import h5py, numpy as np
from pathlib import Path
from lab_uw.data_io.schema import UWFrame

def write_hdf5(
    frame: UWFrame,
    out_path: Path,
    run_name: str,
    channel_count: int = 1,
    dataset: str = "active",
):
    """
    Writes a UWFrame into canonical HDF5.
    We map each waveform -> one cycle_xxx, placing it under `run_name`.
    CH layout: we store receiver as CH1; if you need CH2 later, extend here.
    """
    fs_hz = 1e6 / float(frame.metadata["sampling_rate"])
    n_wave, n_samp = frame.waveform_data.shape

    with h5py.File(out_path, "a") as h5:
        run = h5.require_group(run_name)
        run.attrs["source"] = frame.metadata.get("channel_name", "unknown")
        run.attrs["sampling_rate_Hz"] = fs_hz

        for i in range(n_wave):
            cyc = run.create_group(f"cycle_{i:06d}")
            data = np.zeros((channel_count, n_samp), dtype=np.float32)
            data[0, :] = frame.waveform_data[i].astype(np.float32)
            dset = cyc.create_dataset(dataset, data=data, compression="gzip")
            dset.attrs["fs"] = fs_hz
            cyc.create_dataset("passive", data=np.zeros((channel_count, 1), np.float32))
            cyc.create_dataset("source_waveform", data=np.zeros((1,), np.float32))
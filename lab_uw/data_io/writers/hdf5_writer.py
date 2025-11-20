# -*- coding: utf-8 -*-
from __future__ import annotations

import h5py
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lab_uw.data_io.schema import UWFrame


def write_hdf5(
    frame: UWFrame,
    out_path: Path,
    run_name: str,
    channel_count: int = 1,
    dataset: str = "active",
    *,
    add_timestamps: bool = False,
) -> None:
    """
    Write a UWFrame into the canonical TiePie-like HDF5 layout.

    Layout:
      /<run_name>/cycle_000000/
          attrs:
              [optional] t0_unix_s, iso_time
              sample_rate_Hz
          datasets:
              <dataset>        float32 [C, N]   (CH1=receiver by convention)
              passive          float32 [C, 1]
              source_waveform  float32 [0]      (empty placeholder)

    Notes
    -----
    - Units: frame.metadata["sampling_rate"] is in microseconds.
      We convert to Hz for on-disk attributes.
    - We keep a placeholder "source_waveform" to stay compatible with downstream
      tools that expect the node to exist under cycles.
    """
    fs_hz = 1e6 / float(frame.metadata["sampling_rate"])  # µs -> Hz
    data = frame.waveform_data
    if data.ndim != 2:
        raise ValueError("UWFrame.waveform_data must be 2D [n_waveforms, n_samples].")

    n_wave, n_samp = data.shape
    now = datetime.now(tz=timezone.utc)

    with h5py.File(out_path, "a") as h5:
        run = h5.require_group(run_name)
        run.attrs["source"] = str(frame.metadata.get("channel_name", "unknown"))

        for i in range(n_wave):
            cyc = run.create_group(f"cycle_{i:06d}")
            if add_timestamps:
                cyc.attrs["t0_unix_s"] = float(now.timestamp())
                cyc.attrs["iso_time"] = now.isoformat().replace("+00:00", "Z")
            cyc.attrs["sample_rate_Hz"] = fs_hz

            # active/passive datasets
            active = np.zeros((channel_count, n_samp), dtype=np.float32)
            active[0, :] = data[i].astype(np.float32)
            ds_act = cyc.create_dataset(dataset, data=active, compression="gzip")
            ds_act.attrs["fs"] = fs_hz

            ds_pas = cyc.create_dataset("passive", data=np.zeros((channel_count, 1), np.float32), compression="gzip")
            ds_pas.attrs["fs"] = fs_hz

            # optional placeholder for compatibility
            cyc.create_dataset("source_waveform", data=np.zeros((0,), np.float32), compression="gzip")


def write_stf_hdf5(
    stf: UWFrame,
    out_path: Path,
    stf_name: str,
    *,
    cycle_index: int = 0,
    include_active_stub: bool = False,
    include_passive_stub: bool = False,
    t0_unix_s: Optional[float] = None,
) -> None:
    """
    Save a Source Time Function using the same cycle-based layout used by acquisitions:

      /stf/<stf_name>/cycle_000000/
          attrs:
              t0_unix_s
              iso_time
              sample_rate_Hz   (from STF sampling_rate in µs)
          datasets:
              source_waveform  float32 [L]       <-- REQUIRED
              [optional] active float32 [1, L]   (stub for symmetry)
              [optional] passive float32 [1, 1]  (stub)

    Notes
    -----
    - STF must be 1D (or 2D with shape (1, L)); we store it as 1D.
    - We also set ds_src.attrs["fs_src_Hz"] to make the STF’s own rate explicit.
    """
    dt_us = float(stf.metadata["sampling_rate"])  # µs
    fs_hz = 1e6 / dt_us

    y = stf.waveform_data
    if y.ndim == 2 and y.shape[0] == 1:
        y = y[0]
    if y.ndim != 1:
        raise ValueError("STF waveform must be 1D or (1, L).")

    L = int(y.size)
    now = datetime.fromtimestamp(t0_unix_s, tz=timezone.utc) if t0_unix_s is not None else datetime.now(tz=timezone.utc)

    with h5py.File(out_path, "a") as h5:
        grp = h5.require_group(f"/stf/{stf_name}")
        cyc = grp.create_group(f"cycle_{cycle_index:06d}")

        # attrs
        cyc.attrs["t0_unix_s"] = float(now.timestamp())
        cyc.attrs["iso_time"] = now.isoformat().replace("+00:00", "Z")
        cyc.attrs["sample_rate_Hz"] = fs_hz

        # source waveform
        ds_src = cyc.create_dataset("source_waveform", data=y.astype(np.float32), compression="gzip")
        ds_src.attrs["fs_src_Hz"] = fs_hz  # explicit STF sampling rate

        # optional stubs for symmetry with acquisitions
        if include_active_stub:
            ds_act = cyc.create_dataset("active", data=np.zeros((1, L), dtype=np.float32), compression="gzip")
            ds_act.attrs["fs"] = fs_hz
        if include_passive_stub:
            ds_pas = cyc.create_dataset("passive", data=np.zeros((1, 1), dtype=np.float32), compression="gzip")
            ds_pas.attrs["fs"] = fs_hz

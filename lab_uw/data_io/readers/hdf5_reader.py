# lab_uw/data_io/readers/hdf5_reader.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional, Sequence, List

import h5py
import numpy as np
from lab_uw.data_io.schema import UWFrame

CycleSel = Literal["all", "first", "last"]
    
def _pick_cycles(all_cycles: Sequence[str], cycles: CycleSel, count: int) -> list[str]:
    if not all_cycles:
        return []
    if cycles == "all":
        return list(all_cycles)
    if cycles == "first":
        return list(all_cycles[:max(1, int(count))])
    if cycles == "last":
        return list(all_cycles[-max(1, int(count)):])
    # fallback: first only
    return [all_cycles[0]]

def read_hdf5(
    h5_path: Path,
    *,
    group_regex: str = ".*",
    dataset: str = "active",
    channel: int = 1,                   # CH1 by convention in writer
    cycles: CycleSel = "all",
    count: int = 1,
    average: bool = True,
    group_mode: Literal["first","concat"] = "first",
) -> UWFrame:
    """
    Read acquisitions saved in the cycle-based layout:

      /<run>/cycle_000000/{active, passive, source_waveform}

    Returns a UWFrame with µs-based time axis in metadata.

    Parameters
    ----------
    group_regex : str
        Regex to select run groups (e.g. r"^run2$" or r"fc\\d+_bw.*").
    dataset : {"active","passive"}
        Which dataset to read from each cycle.
    channel : int
        1-based channel index to extract from (C, N) recording.
    cycles : {"all","first","last"}
        Which cycles to include.
    count : int
        How many cycles when cycles is "first"/"last".
    average : bool
        If True with multiple cycles, average to a single waveform.
    group_mode : {"first","concat"}
        If multiple run groups match:
          - "first": read only the first match
          - "concat": read cycles from all matches (in lexicographic group order)
    """
    ch = int(channel) - 1
    r = re.compile(group_regex)

    with h5py.File(h5_path, "r") as h5:
        run_groups = [k for k in h5.keys() if isinstance(h5[k], h5py.Group) and r.match(k)]
        if not run_groups:
            raise RuntimeError(f"No groups match '{group_regex}' in {h5_path}")

        groups_to_read = [run_groups[0]] if group_mode == "first" else sorted(run_groups)

        traces: list[np.ndarray] = []
        dt_us: Optional[float] = None
        n_min: Optional[int] = None

        for gname in groups_to_read:
            g = h5[gname]
            cycle_names = sorted([k for k in g.keys() if k.startswith("cycle_")])
            use_cycles = _pick_cycles(cycle_names, cycles, count)
            if not use_cycles:
                continue

            # discover sampling (prefer dataset attr; fallback to cycle attr; last-resort 5e6 Hz)
            # read first cycle/dataset to infer lengths
            ds0 = g[f"{use_cycles[0]}/{dataset}"]
            arr0 = ds0[()]  # (C, N)
            if arr0.ndim != 2 or arr0.shape[0] <= ch:
                raise RuntimeError(f"Bad dataset shape in /{gname}/{use_cycles[0]}/{dataset}")
            fs_hz = float(ds0.attrs.get("fs",
                       g[use_cycles[0]].attrs.get("sample_rate_Hz", 5e6)))
            dt_us_local = 1e6 / fs_hz
            if dt_us is None:
                dt_us = dt_us_local
            else:
                # light consistency check across groups
                if abs(dt_us - dt_us_local) > 1e-9:
                    raise RuntimeError(f"Sampling mismatch across groups: {dt_us} vs {dt_us_local} µs")

            # collect traces
            for c in use_cycles:
                y = g[f"{c}/{dataset}"][()][ch].astype(float)  # (N,)
                n_min = len(y) if n_min is None else min(n_min, len(y))
                traces.append(y)

        if not traces:
            raise RuntimeError("No valid cycles found after filtering.")

        # equalize length
        N = int(n_min)
        traces = [t[:N] for t in traces]
        data = np.stack(traces, axis=0)  # (n_waveforms, N)

        # average if requested
        if average and data.shape[0] > 1:
            data = data.mean(axis=0, keepdims=True)

        # build time axis (µs)
        t_us = np.arange(N, dtype=float) * float(dt_us)

        meta = {
            "channel_name": f"CH{channel}",
            "number_of_samples": int(N),
            "sampling_rate": float(dt_us),     # µs per sample (consistent with the rest of your code)
            "time_ax_waveform": t_us,          # µs
            "number_of_waveforms": int(data.shape[0]),
            "h5_path": str(h5_path),
            "groups_read": groups_to_read,
            "dataset": dataset,
        }
        return UWFrame(data, meta).ensure_2d()

def read_stf_hdf5(
    h5_path: Path,
    *,
    name: str,
    cycle_index: int = 0,
) -> UWFrame:
    """
    Read a Source Time Function written by write_stf_hdf5:

      /stf/<stf_name>/cycle_xxxxxx/source_waveform

    Returns a single-waveform UWFrame with µs time axis.
    """
    with h5py.File(h5_path, "r") as h5:
        grp = h5[f"/stf/{name}"]
        cyc = grp[f"cycle_{int(cycle_index):06d}"]

        y = np.asarray(cyc["source_waveform"][()], dtype=float)
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        if y.ndim != 1:
            raise RuntimeError("STF dataset must be 1D or (1, L).")

        fs_hz = float(cyc.attrs["sample_rate_Hz"])
        dt_us = 1e6 / fs_hz
        t_us  = np.arange(y.size, dtype=float) * dt_us

        meta = {
            "channel_name": f"STF:{name}",
            "number_of_samples": int(y.size),
            "sampling_rate": float(dt_us),     # µs / sample
            "time_ax_waveform": t_us,          # µs
            "number_of_waveforms": 1,
            "h5_path": str(h5_path),
            "stf_name": name,
            "cycle_index": int(cycle_index),
        }
        return UWFrame(y[None, :], meta).ensure_2d()


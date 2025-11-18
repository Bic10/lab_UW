# lab_uw/data_io/readers/tiepie_reader.py
import h5py, re
import numpy as np
from pathlib import Path
from lab_uw.data_io.schema import UWFrame

def load_tiepie_h5(
    h5_path: Path,
    group_regex: str = ".*",
    dataset: str = "active",
    channel: int = 2,
    cycles: str = "all",
    count: int = 1,
    average: bool = True,
) -> UWFrame:
    ch = channel - 1
    with h5py.File(h5_path, "r") as h5:
        groups = [k for k in h5.keys() if isinstance(h5[k], h5py.Group)]
        gnames = [g for g in groups if re.compile(group_regex).match(g)]
        if not gnames:
            raise RuntimeError(f"No groups match '{group_regex}' in {h5_path}")
        g = h5[gnames[0]]
        cycles_all = sorted([k for k in g.keys() if k.startswith("cycle_")])
        if not cycles_all:
            raise RuntimeError(f"No cycles in /{gnames[0]}")

        if cycles == "all":
            sel = cycles_all
        elif cycles == "first":
            sel = cycles_all[:max(1, int(count))]
        elif cycles == "last":
            sel = cycles_all[-max(1, int(count)):]
        else:
            sel = [cycles_all[0]]

        ds0 = g[f"{sel[0]}/{dataset}"][:]            # (C, N)
        if ds0.ndim != 2 or ds0.shape[0] <= ch:
            raise RuntimeError("Bad dataset shape")
        fs_hz = float(g[f"{sel[0]}/{dataset}"].attrs.get("fs", 5e6))
        dt_us = 1e6 / fs_hz

        traces = []
        min_len = ds0[ch].size
        for c in sel:
            y = g[f"{c}/{dataset}"][:][ch].astype(float)
            min_len = min(min_len, y.size)
            traces.append(y)
        traces = [t[:min_len] for t in traces]
        data = np.stack(traces, axis=0)  # (n_sel, N)

        if average and data.shape[0] > 1:
            data = data.mean(axis=0, keepdims=True)

        n_wave, n_samp = data.shape
        time_ax = np.arange(n_samp, dtype=float) * dt_us

        meta = {
            "channel_name": f"CH{channel}",
            "h5_group": gnames[0],
            "number_of_samples": int(n_samp),
            "sampling_rate": dt_us,            # µs/sample
            "time_ax_waveform": time_ax,       # µs
            "number_of_waveforms": int(n_wave),
        }
        return UWFrame(data, meta).ensure_2d()

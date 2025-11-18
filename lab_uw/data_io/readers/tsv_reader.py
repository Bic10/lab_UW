# lab_uw/data_io/readers/tsv_reader.py
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List
from lab_uw.data_io.schema import UWFrame

def _extract_metadata_from_tsv_header(lines: List[str]) -> Tuple[list, list, str]:
    general, amplitude_scale, time_scale, acquisition_scale = lines[:4]
    channel_name = general.split(":")[1]
    acquisition_info = [float(e) for e in re.findall(r"\d+\.*\d*", acquisition_scale)]
    time_info       = [float(e) for e in re.findall(r"\d+\.*\d*", time_scale)]
    return acquisition_info, time_info, channel_name

def load_tsv(infile_path: Path) -> UWFrame:
    with open(infile_path, "r", encoding="iso8859") as f:
        header = [f.readline().strip() for _ in range(4)]
        acquisition_info, time_info, channel_name = _extract_metadata_from_tsv_header(header)

        n_samples       = int(time_info[2])
        dt_us           = float(time_info[3])                     # µs/sample
        t0, t1          = float(time_info[0]), float(time_info[1])
        time_ax_wave    = np.arange(t0, t1, dt_us)

        acq_period_us   = float(acquisition_info[2])              # µs between B-scans
        acq_t0, acq_t1  = float(acquisition_info[0]), float(acquisition_info[1])
        time_ax_acq     = np.arange(acq_t0, acq_t1, acq_period_us)

        # data
        rows = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append([float(v) for v in s.split()])
            except ValueError:
                break
        data = np.array(rows, dtype=float)
        n_wave = data.shape[0]
        time_ax_acq = time_ax_acq[:n_wave]

    meta = {
        "number_of_samples": n_samples,
        "sampling_rate": dt_us,
        "time_ax_waveform": time_ax_wave,
        "acquisition_frequency": acq_period_us,
        "number_of_waveforms": n_wave,
        "time_ax_acquisition": time_ax_acq,
        "channel_name": channel_name,
    }
    return UWFrame(data, meta).ensure_2d()
# lab_uw/data_io/schema.py
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any

@dataclass
class UWFrame:
    """
    Normalized ultrasonic frame used across the codebase.

    waveform_data: (n_waveforms, n_samples) float64
    metadata:
      number_of_samples        : int
      number_of_waveforms      : int
      sampling_rate            : float   # µs per sample
      time_ax_waveform         : np.ndarray[float]  # µs
      # Optional (present for TSV):
      acquisition_frequency    : float   # µs between waveforms
      time_ax_acquisition      : np.ndarray[float]  # µs
      channel_name             : str
      # Optional (present for HDF5):
      h5_group                 : str
    """
    waveform_data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_2d(self) -> "UWFrame":
        x = self.waveform_data
        if x.ndim == 1:
            self.waveform_data = x[None, :]
            self.metadata["number_of_waveforms"] = 1
        return self
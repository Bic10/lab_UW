import numpy as np
import pytest
from lab_uw.signal_processing import SignalProcessor

def test_remove_starting_noise():
    data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
    metadata = {"number_of_samples": 5, "time_ax_waveform": np.array([0, 1, 2, 3, 4])}

    new_data, new_metadata = SignalProcessor.remove_starting_noise(data, metadata, remove_initial_samples=2)

    assert new_data.shape == (2, 3)
    assert new_metadata["number_of_samples"] == 3
    assert np.array_equal(new_metadata["time_ax_waveform"], [2, 3, 4])

def test_lowpass_mask():
    mask = SignalProcessor.lowpass_mask(100, 10)
    assert mask.shape == (100,)
    
    # Ensure that the Hanning window is correctly applied
    # Check that the first and last elements are zero as expected
    assert mask[0] == 0
    assert mask[-1] == 0
    
    # Check that the middle part of the mask has non-zero values
    assert np.all(mask[1:5] != 0)
    assert np.all(mask[-5:-1] != 0)
    assert np.all(mask[10:-10] == 0)

def test_signal2noise_separation_lowpass():
    waveform = np.random.randn(5, 100)  # Generate random waveform data
    metadata = {
        "sampling_rate": 1000,
        "number_of_samples": 100,
        "time_ax_waveform": np.linspace(0, 0.1, 100)
    }
    
    signal, noise = SignalProcessor.signal2noise_separation_lowpass(waveform, metadata, freq_cut=50)
    
    assert signal.shape == waveform.shape
    assert noise.shape == waveform.shape
    assert not np.allclose(signal, waveform)  # Ensure signal is not the same as input
    assert np.allclose(signal + noise, waveform)  # Ensure signal + noise = original waveform

def test_sta_lta():
    waveform = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    sta_lta_ratio = SignalProcessor.sta_lta(waveform, sta_window=2, lta_window=5)
    
    assert sta_lta_ratio.shape == waveform.shape
    assert np.max(sta_lta_ratio) > 1  # Ensure some peaks are detected

def test_select_wavelets_given_known_numbers_of_them():
    waveform = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0])
    index_max_list, index_min_before_list, index_min_after_list = SignalProcessor.select_wavelets_given_known_numbers_of_them(
        waveform, chunk_n=3, offset=0, tolerance=0.1)
    
    assert len(index_max_list) == 3
    assert len(index_min_before_list) == 3
    assert len(index_min_after_list) == 3
    assert index_max_list == [1, 5, 9]


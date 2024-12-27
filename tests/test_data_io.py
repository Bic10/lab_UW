import numpy as np
import pytest
from lab_uw.data_io import DataHandler

def test_load_waveform_json(tmp_path):
    data_handler = DataHandler()

    # Sample data to write to JSON
    sample_data = np.array([1, 2, 3, 4])
    sample_metadata = {"sampling_rate": 0.04, "number_of_samples": 4}

    # Save to JSON file
    json_file = tmp_path / "test_waveform.json"
    data_handler.save_waveform_json(sample_data, sample_metadata, str(json_file))

    # Load the saved JSON file
    loaded_data, loaded_metadata = data_handler.load_waveform_json(str(json_file))

    assert np.array_equal(loaded_data, sample_data)
    assert loaded_metadata == sample_metadata

def test_serialize_value():
    data_handler = DataHandler()
    assert data_handler.serialize_value(np.array([1, 2, 3])) == [1, 2, 3]
    assert data_handler.serialize_value(np.int32(10)) == 10
    assert data_handler.serialize_value(np.float64(3.14)) == 3.14

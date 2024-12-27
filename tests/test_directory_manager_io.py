import pytest
from lab_uw.directory_manager import DirectoryManager
from pathlib import Path

def test_make_infile_path_list(tmp_path):
    dir_manager = DirectoryManager(base_dir=tmp_path)
    (tmp_path / "experiments_test/machine_1/raw_data").mkdir(parents=True)
    sample_file = tmp_path / "experiments_test/machine_1/raw_data/sample.txt"
    sample_file.touch()

    file_list = dir_manager.make_infile_path_list("test", "machine_1", "raw_data")
    assert len(file_list) == 1
    assert file_list[0] == sample_file

def test_make_data_analysis_folders(tmp_path):
    dir_manager = DirectoryManager(base_dir=tmp_path)
    data_folders = dir_manager.make_data_analysis_folders("test_machine", "experiment_1", ["velocity", "stress"])
    assert len(data_folders) == 2
    assert all(folder.exists() for folder in data_folders)

def test_make_images_folders(tmp_path):
    dir_manager = DirectoryManager(base_dir=tmp_path)
    image_folders = dir_manager.make_images_folders("test_machine", "experiment_1", ["plots", "figures"])
    assert len(image_folders) == 2
    assert all(folder.exists() for folder in image_folders)

from pathlib import Path
from typing import Union

BASE_DIR = Path.cwd().parent  # Default to parent of current working directory

class DirectoryManager:
    """
    Class to manage directory structures for experiments, data analysis, and images.
    """
    def __init__(self, base_dir: Union[str, Path] = None):
        """
        Initialize the DirectoryManager with a base directory.

        Args:
            base_dir (Union[str, Path], optional): The base directory path.
                If None, defaults to BASE_DIR.
        """
        # Default to BASE_DIR if base_dir is None
        if base_dir is None:
            self.base_dir = BASE_DIR
        else:
            self.base_dir = Path(base_dir)

    def make_infile_path_list(self, machine_name: str, experiment_name: str, data_type: str) -> list[Path]:
        """
        Generates a list of input file paths based on the machine name, experiment name, and data type.
        """
        data_type = Path(data_type)
        
        indir_path = self.base_dir / f"experiments_{machine_name}" / experiment_name / data_type

        if not indir_path.exists():
            raise FileNotFoundError(f"Directory not found: {indir_path}")

        # Return list of all files in the directory
        return [file_path for file_path in indir_path.iterdir() if file_path.is_file()]

    def make_data_analysis_folders(self, machine_name: str, experiment_name: str, data_types: list[str]) -> list[Path]:
        """
        Creates folders for storing elaborated data based on machine name, experiment name, and data types.
        """
        folder_path = self.base_dir / f"experiments_{machine_name}" / experiment_name / "data_analysis"

        outdir_paths = []
        for data_type in data_types:
            outdir_path = folder_path / data_type
            outdir_paths.append(outdir_path)
            outdir_path.mkdir(parents=True, exist_ok=True)
        
        return outdir_paths

    def make_images_folders(self, machine_name: str, experiment_name: str, image_types: list[str]) -> list[Path]:
        """
        Creates folders for storing images based on machine name, experiment name, and image types.
        """
        folder_path = self.base_dir / f"experiments_{machine_name}" / experiment_name / "standard_images"

        outdir_paths = []
        for image_type in image_types:
            outdir_path = folder_path / image_type
            outdir_paths.append(outdir_path)
            outdir_path.mkdir(parents=True, exist_ok=True)
        
        return outdir_paths

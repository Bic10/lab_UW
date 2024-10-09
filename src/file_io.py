# src/file_io.py
import numpy as np
import os
import json
import re
# HANDLE TSV FILES FROM EUROSCAN
def make_UW_data(infile_path: str) -> tuple[np.ndarray, dict]:
    '''
    Reads data from a binary file and extracts data and metadata.

    Args:
    - infile_path: Path to the input binary file.

    Returns:
    - Tuple containing data array and metadata dictionary.
    '''
    with open(infile_path, "rb") as infile:
        acquisition_info, time_info = extract_metadata_from_tsv(infile, encoding='iso8859')

        number_of_samples = time_info[2]
        sampling_rate = time_info[3]
        time_ax_waveform = np.arange(time_info[0], time_info[1], sampling_rate)

        acquisition_frequency = acquisition_info[2]
        time_ax_acquisition = np.arange(acquisition_info[0], acquisition_info[1], acquisition_frequency)

        waveform_list = remove_empty_lines(infile, number_of_samples, encoding='iso8859')
        corrected_number_of_waveforms = len(waveform_list)
        time_ax_acquisition = time_ax_acquisition[:corrected_number_of_waveforms]

        metadata = {"number_of_samples": number_of_samples,
                    'sampling_rate': sampling_rate,
                    "time_ax_waveform": time_ax_waveform,
                    'number_of_waveforms': corrected_number_of_waveforms,
                    "acquisition_frequency": acquisition_frequency,
                    'time_ax_acquisition': time_ax_acquisition}

        data = np.array(waveform_list).astype(float)

    return data, metadata


def remove_empty_lines(infile: iter, number_of_samples: int, encoding: str = 'iso8859') -> list[list[str]]:
    '''
    Removes empty lines from input file.

    Args:
    - infile: Iterable containing lines from the input file.
    - number_of_samples: Number of samples per line.
    - encoding: Encoding type of the input file.

    Returns:
    - List of lines with sufficient number of samples.
    '''
    waveform_list = []             
    for line in infile:
        line = line.decode(encoding)
        line = line.split("\t")
        if len(line) < number_of_samples:
            continue
        waveform_list.append(line)
    return waveform_list


def extract_metadata_from_tsv(infile: iter, encoding: str = 'iso8859') -> tuple[list[float], list[float]]:
    '''
    Extracts metadata from a TSV file.    AT THE MOMENT WE ONLY MANAGE TO USE THIRD AND FORTH LINES.
    IMPLEMENT LATER THE REST

    Args:
    - infile: Iterable containing lines from the input file.
    - encoding: Encoding type of the input file.

    Returns:
    - Tuple containing acquisition information and time information extracted from the TSV file.
    '''
    # encoding = 'iso8859' #just to read greek letters: there is a "mu" in the euroscan file  

    general = next(infile).decode(encoding)
    amplitude_scale = next(infile).decode(encoding)
    time_scale = next(infile).decode(encoding)
    acquisition_scale = next(infile).decode(encoding)
    acquisition_info = re.findall(r"\d+\.*\d*", acquisition_scale)
    acquisition_info = [float(entry) for entry in acquisition_info]
    time_info = re.findall(r"\d+\.*\d*",time_scale)   # find all float in the string "time_scale" using regular expression:
                                            # the first argument of findall must be a "regular expression". It start with an r so
                                            # is an r-string, that means "\" can be read as a normal character
                                            # The regular expression "\d+\.*\d*" means "match all the digits that are adiacent as
                                            # a single entry and, if they are followed by a dot and some other digits, they are still
                                            # the same numbers". So it gets as a single number both 1, 10, 1.4, 10.42 etc.
    time_info = [float(entry) for entry in time_info]           # just convert the needed info in float number

    return acquisition_info, time_info

# DIRECTORY STRUCTURE
def make_infile_path_list(machine_name: str, experiment_name: str, data_type:str) -> list[str]:
    '''
    Generates a list of input file paths based on the machine name and experiment name.

    Args:
    - machine_name: Name of the machine.
    - experiment_name: Name of the experiment.

    Returns:
    - List of input file paths.
    '''
    code_path = os.getcwd()
    parent_folder = os.path.abspath(os.path.join(code_path, os.pardir))
    # obrobriosa soluzione tampone per "pareggiare" la profondita del path: questa davvero non posso lasciarla cosi
    while os.path.basename(parent_folder) != "active_source_implementation":
        parent_folder = os.path.abspath(os.path.join(parent_folder, os.pardir))

    indir_path = os.path.join(parent_folder,"experiments_"+ machine_name, experiment_name, data_type)

    infile_path_list = []   
    for infile_name in os.listdir(indir_path):
        file_path = os.path.join(indir_path, infile_name)
        infile_path_list.append(file_path)

    return infile_path_list

def make_data_analysis_folders(machine_name: str, experiment_name: str, data_types: list[str]) -> list[str]:
    '''
    Creates folders for storing elaborated data on the based on the machine name, experiment name, and data types.

    Args:
    - machine_name: Name of the machine.
    - experiment_name: Name of the experiment.
    - data_types: List of data types.

    Returns:
    - List of folder paths for each data type.
    '''
    folder_name = "data_analysis"

    code_path = os.getcwd()

    parent_folder = os.path.abspath(os.path.join(code_path, os.pardir))
    # obrobriosa soluzione tampone per "pareggiare" la profondita del path: questa davvero non posso lasciarla cosi
    while os.path.basename(parent_folder) != "active_source_implementation":
        parent_folder = os.path.abspath(os.path.join(parent_folder, os.pardir))

    folder_path = os.path.join(parent_folder,"experiments_"+ machine_name, experiment_name, folder_name)

    outdir_path_datas = []
    for im in data_types:
        outdir_path_data = os.path.join(folder_path, im)
        outdir_path_datas.append(outdir_path_data)

        if not os.path.exists(outdir_path_data):
            os.makedirs(outdir_path_data)  
    
    return outdir_path_datas    

def make_images_folders(machine_name: str, experiment_name: str, image_types: list[str]) -> list[str]:
    '''
    Creates folders for storing images based on the machine name, experiment name, and image types.

    Args:
    - machine_name: Name of the machine.
    - experiment_name: Name of the experiment.
    - image_types: List of image types.

    Returns:
    - List of folder paths for each image type.
    '''
    folder_name = "images"

    code_path = os.getcwd()
    parent_folder = os.path.abspath(os.path.join(code_path, os.pardir))
    parent_folder = os.path.abspath(os.path.join(parent_folder, os.pardir))

    folder_path = os.path.join(parent_folder,"experiments_"+ machine_name, experiment_name, folder_name)

    outdir_path_images = []
    for im in image_types:
        outdir_path_image = os.path.join(folder_path, im)
        outdir_path_images.append(outdir_path_image)

        if not os.path.exists(outdir_path_image):
            os.makedirs(outdir_path_image)  
    
    return outdir_path_images

# HANDLE JSON FILES 
def load_waveform_json(infile_path: str) -> tuple[np.ndarray, dict]:
    """
    Load waveform data from a JSON file.

    Args:
        infile_path (str): Path to the input JSON file.

    Returns:
        tuple[np.ndarray, dict]: A tuple containing the waveform data and its metadata.
    """
    with open(infile_path, "r") as json_file:
        data_dict = json.load(json_file)

    # Retrieve metadata and data from the loaded dictionary
    data = np.array(data_dict["data"])
    metadata = data_dict["metadata"]

    return data, metadata

def save_waveform_json(data: np.ndarray, metadata: dict, outfile_path: str) -> None:
    """
    Save waveform data and its metadata to a JSON file.

    Args:
        data (np.ndarray): The waveform data.
        metadata (dict): Metadata associated with the waveform data.
        outfile_path (str): Path to the output JSON file.
    """
    # Serialize wavelet_metadata
    serialized_metadata = {key: serialize_value(value) for key, value in metadata.items()}

    # SAVE THE wavelet
    with open(outfile_path, "w") as output_json:
        # Create a dictionary to hold both metadata and data
        data_dict = {"metadata": serialized_metadata, "data": data.tolist()}

        # Save the dictionary to a JSON file
        json.dump(data_dict, output_json)


def serialize_value(value):
    '''
    Helper function to convert non-serializable values to serializable ones
    Needed to save numpy arrays in json file
    '''
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(value)
    elif isinstance(value, (np.float16, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.complex64, np.complex128)):
        return {"real": value.real, "imag": value.imag}
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (np.str_, np.bytes_)):
        return str(value)
    elif isinstance(value, (np.void)):
        return None
    else:
        return value


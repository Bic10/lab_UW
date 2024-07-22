from src.file_io import make_infile_path_list, make_UW_data
from src.signal_processing import remove_starting_noise, signal2noise_separation_lowpass
from src.synthetic_data import synthetic_wavelets_in_noise
from src.plotting import uw_all_plot, amplitude_map, amplitude_spectrum_map
from src.LAB_UW_forward_modeling import DDS_UW_simulation
from global_optimization_velocity import find_mechanical_data, find_sync_values, process_waveform

def main():
    print("Project setup complete. Ready to implement functionality.")

if __name__ == "__main__":
    main()

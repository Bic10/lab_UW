import glob
import time as tm
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

from file_io import *
from signal_processing import *
from synthetic_data import *
from LAB_UW_forward_modeling import *
################################################################################################################################
from scipy.signal import correlate, correlation_lags

def cross_correlation(signal1, signal2):
    # Normalize signals
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    # Compute cross-correlation
    cross_corr_result = correlate(signal1, signal2, mode='full')
    return cross_corr_result

def find_time_delay(signal1, signal2, dt):
    cross_corr_result = cross_correlation(signal1, signal2)
    lags = correlation_lags(len(signal1), len(signal2), mode='full')
    max_index = np.argmax(cross_corr_result)
    lag = lags[max_index]
    time_delay = lag * dt
    return time_delay


def find_first_percentage_of_max(arr, percentage):
    # Find the maximum value in the array
    max_value = np.max(arr)
    
    # Calculate the target value, which is the given percentage of the max
    target_value = max_value * (percentage / 100.0)
    
    # Find the index of the first element that is greater than or equal to the target value
    index = np.where(arr >= target_value)[0]
    
    # Return the index of the first occurrence, or None if no such value is found
    if len(index) > 0:
        return index[0], arr[index[0]]  # Returning the index and the value itself
    else:
        print("NOTHING")
        return None  # No value meets the condition
    
def find_mechanical_data(file_path_list, pattern):
    """
    Trova un file specifico all'interno di una lista di percorsi dei file utilizzando un pattern.
    
    Args:
        file_path_list (list): Lista di percorsi dei file in cui cercare il file.
        pattern (str): Pattern per il nome del file da cercare.
    
    Returns:
        str: Percorso completo del file trovato, o None se non viene trovato nessun file corrispondente.
    """
    for file_path in file_path_list:
        if glob.fnmatch.fnmatch(file_path, pattern):
            print("MECHANICAL DATA CHOOSEN:", file_path)
            return file_path
    return None  # Nessun file trovato nella lista
    

def find_sync_values(mech_data_path):
    """
    Trova i valori di picco sincronizzazione all'interno di un file di dati meccanici.

    Questa funzione legge un file CSV contenente dati meccanici, estrae la colonna
    relativa alla sincronizzazione e individua i picchi di sincronizzazione in base
    ai parametri specificati.

    Args:
        mech_data_path (str): Percorso del file CSV contenente i dati meccanici.

    Returns:
        numpy.ndarray: Un array NumPy contenente gli indici dei picchi di sincronizzazione
                       trovati nei dati meccanici.
    """
    mech_data = pd.read_csv(mech_data_path, sep=',', skiprows=[1])
    sync_data = mech_data.sync
    
    # Trova i picchi di sincronizzazione nei dati sincronizzazione
    sync_peaks, _ = find_peaks(sync_data, prominence=4.2, height=4)
    return mech_data, sync_data, sync_peaks

def plot_sync_peaks(sync_data, sync_peaks, experiment_name):
    """
    Visualizza i picchi di sincronizzazione su un grafico dei dati di sincronizzazione.

    Questa funzione prende i dati di sincronizzazione e gli indici dei picchi di sincronizzazione,
    quindi crea un grafico per visualizzare sia i dati di sincronizzazione che i picchi di sincronizzazione
    evidenziati in rosso.

    Args:
        sync_data (numpy.ndarray): Array NumPy contenente i dati di sincronizzazione.
        sync_peaks (numpy.ndarray): Array NumPy contenente gli indici dei picchi di sincronizzazione.
        experiment_name (str): Nome dell'esperimento o della prova da visualizzare nel titolo del grafico.
    """
    
    plt.figure(figsize=(10, 4))
    plt.title(f'Sync Peaks {experiment_name}')
    

    # Plot dei picchi di sincronizzazione evidenziati in rosso
    plt.scatter(np.arange(0, len(sync_data))[sync_peaks], sync_data[sync_peaks], c='r', s = 10, zorder = 2, alpha = 0.8)
    
    # Plot dei dati di sincronizzazione
    plt.plot(sync_data, zorder = 1, c = 'k', linewidth = 0.8)
    
    plt.ylabel('Arduino voltage [V]', fontsize=12)
    plt.xlabel('records #', fontsize=12)
    
    plt.show()
    
##########################################################################################################################################
# GENERAL PARAMETERS
freq_cut = 2                  # [MHz]   maximum frequency of the data we want to reproduce  

## GET OBSERVED DATA

# Load and process the pulse waveform: it is going to be our time source function
machine_name = "on_bench"
experiment_name = "glued_pzt"
data_type = "data_analysis/wavelets_from_glued_pzt"
infile_path_list_pulse  = make_infile_path_list(machine_name=machine_name,experiment_name=experiment_name,data_type=data_type)

pulse_list = []
pulse_metadata_list = []
t_pulse_list = []
for pulse_path in sorted(infile_path_list_pulse): 
    pulse, pulse_metadata = load_waveform_json(pulse_path)
    t_pulse = pulse_metadata['time_ax_waveform']
    pulse, _  = signal2noise_separation_lowpass(pulse,
                                                pulse_metadata,
                                                freq_cut=freq_cut)
    pulse = pulse - pulse[0]    # make the pulse start from zero, as it should, physically

    pulse_list.append(pulse)
    pulse_metadata_list.append(pulse_metadata)
    t_pulse_list.append(t_pulse)

# JUST USE ONE OF THE PULSE AS A SOURCE TIME FUNCTION
choosen_pulse = 0
pulse = pulse_list[choosen_pulse]
t_pulse = t_pulse_list[choosen_pulse]
dt_pulse = t_pulse[1]-t_pulse[0]
pulse_duration = t_pulse[-1]-t_pulse[0]

#########################################################################################
# DATA FOLDERS
machine_name = "Brava_2"
experiment_name = "s0108"
data_type_uw = 'data_tsv_files'
data_type_mech = 'mechanical_data'
sync_file_pattern = '*s*_data_rp' #pattern to find specific experiment in mechanical data

# LOAD MECHANICAL DATA
infile_path_list_mech = make_infile_path_list(machine_name, experiment_name, data_type=data_type_mech)
#LOAD MECHANICAL DATA
mech_data_path= find_mechanical_data(infile_path_list_mech, sync_file_pattern)
    
mech_data, sync_data, sync_peaks = find_sync_values(mech_data_path)
# plot_sync_peaks(sync_data, sync_peaks, experiment_name)

########################################################################################################
### PICKED MANUALLY FOR PLOTTING POURPOSE: THEY ARE CATTED PRECISELY AROUND THE MECHANICAL DATA OF THE STEP
if experiment_name == "s0108":
    steps_carrara = [5582,8698,15050,17990,22000,23180,36229,39391,87940,89744,126306,128395,134000,135574,169100,172600,220980,223000,259432, 261425,266429,268647,279733,282787,331437,333778,369610,374824]
    sync_peaks = steps_carrara 

if experiment_name == "s0103":
    steps_mont = [4833,8929,15166,18100,22188,23495,36297,39000,87352,89959,154601,156625,162000,165000,168705,170490,182000,184900,233364,235558,411811,462252]
    sync_peaks = steps_mont
##############################################################################################################

#MAKE UW PATH LIST
infile_path_list_uw = sorted(make_infile_path_list(machine_name,experiment_name, data_type=data_type_uw))
outdir_path_l2norm= make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name,data_types=["global_optimization_velocity"])
print(f"The misfits calculated will be saved at path:\n\t {outdir_path_l2norm}")
outdir_path_images = make_images_folders(machine_name, experiment_name, "global_optimization_velocity_plots_testing")       

for choosen_uw_file, infile_path in enumerate(infile_path_list_uw):

    # CHOOSE OUTFILE_PATH
    outfile_name = os.path.basename(infile_path).split('.')[0] 
    outfile_path = os.path.join(outdir_path_l2norm[0], outfile_name)

    # choosen_uw_file = 0                  # which of the sorted UW files analized. It must correspond to the part of mechanical data extracted!
    # infile_path = infile_path_list_uw[choosen_uw_file]     
    
    print('PROCESSING UW DATA IN %s: ' %infile_path)
    start_time = tm.time()

    # LOAD UW DATA
    data_OBS,metadata = make_UW_data(infile_path)
    t_OBS = metadata['time_ax_waveform']

    # REMOVEVE EVERYTHING BEFORE initial_time_removed: given the velocity in plays, there can be only noise there.
    initial_time_removed = 0         # [mus]
    t_OBS,data_OBS = t_OBS[t_OBS>initial_time_removed], data_OBS[:,t_OBS>initial_time_removed]

    make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name,data_types=["global_optimization_velocity"])


    # FREQUENCY LOW PASS:
    # reduce computation time 
    # # assumtion: there is no point to simulate ABOVE the spectrum of data_OBS (visual ispection)
    data_OBS, _  = signal2noise_separation_lowpass(data_OBS,metadata,freq_cut=freq_cut)


    # DOWNSAMPLING THE WAVEFORMS: FOR PLOTING PURPOSE, WE DO NOT NEED TO PROCESS ALL THE WAVEFORMS
    number_of_waveforms_wanted = 20
    data_OBS = data_OBS[:sync_peaks[2*choosen_uw_file+1]-sync_peaks[2*choosen_uw_file]] # subsempling on around the step
    metadata['number_of_waveforms'] = len(data_OBS)
    downsampling = round(metadata['number_of_waveforms']/number_of_waveforms_wanted)
    print(f"number of waveforms in the selected subset: {metadata['number_of_waveforms']}\nNumber of waveforms wanted: {number_of_waveforms_wanted}\nDownsampling waveforms by a factor: {downsampling}")

    ### INPUT DATA ###
    # These are constants through the entire experiment.
    side_block_1 = 2               # [cm] width of first side block
    side_block_2 =2               # [cm] width of first gouge layer
    central_block = 4.8
    pzt_width = 0.1                                 # [cm] its important!
    pla_width = 0.1                                 # [cm] plate supporting the pzt
    csteel = 3374 * (1e2/1e6)       # [cm/mus]   steel s-velocity
    cpzt = 2000* (1e2/1e6)         # [cm/mus] s-velocity in piezo ceramic, beetween 1600 (bad coupling) and 2500 (good one). It matters!!!
                                    # according to https://www.intechopen.com/chapters/40134   
    cpla =  0.4*0.1392              # [cm/mus]   plate supporting the pzt


    # EXTRACT LAYER THICKNESS FROM MECHANICA DATA
    #Choose layer thickness from mechanical data using sync peaks indexes
    thickness_gouge_1_list = mech_data.rgt_lt_mm[sync_peaks[2*choosen_uw_file]: sync_peaks[2*choosen_uw_file+1]].values/10  #the velocities are in cm/mus, layer thickness in mm. Divided by ten!!!
    thickness_gouge_2_list = thickness_gouge_1_list

    # TRASMISSION AT 0 ANGLE: ONE RECEIVER, ONE TRASMITTER. 
    # Fix the zero of the ax at the beginning of side_block_1. 
    # The trasmitter is solidal with the side_block_1, so its coordinates are constant
    # The receiver moves with the side_block_1, so its coordinates must be corrected for the layer thickness. It is computed in the for loop
    # Must be made more efficient, bu at the mooment is at list clear
    x_trasmitter = 1                              # [cm] position of the trasmitter from the beginning of the sample: the distance from the beginning of the block is fixed.

    # GUESSED VELOCITY MODEL OF THE SAMPLE
    # S- velocity of gouge to probe. Extract from the literature!
    cmin = 600 * (1e2/1e6)        
    cmax = 1000 * (1e2/1e6) 
    c_step = 20*1e2/1e6
    c_gouge_list = np.arange(cmin, cmax,c_step) # choose of velocity in a reasonable range: from pressure-v in air to s-steel velocity

    # DEFINE EVALUATION INTERVAL FOR L2 NORM OF THE RESIDUALS
    # at the moment the gouge thickness is the same for both the layers, but still the implementation below allowed for differences.
    max_travel_time_list = 2*(side_block_1-x_trasmitter)/csteel + thickness_gouge_1_list/cmin+ central_block/csteel + thickness_gouge_1_list/cmin
    min_travel_time_list = 2*(side_block_1-x_trasmitter)/csteel + thickness_gouge_1_list/cmax+ central_block/csteel + thickness_gouge_1_list/cmax
    idx_travel_time_list = []
    for min_travel_time,max_travel_time in zip(min_travel_time_list,max_travel_time_list):
        idx_travel_time = np.where((t_OBS >min_travel_time) & (t_OBS < max_travel_time+pulse_duration))
        idx_travel_time_list.append(idx_travel_time)


    # for waveform in data_OBS[::downsampling]:
    #     time_delay = find_time_delay(waveform,pulse,sampling_rate=t_OBS[1]-t_OBS[0]) 
    #     print(f"Time delay: {time_delay}")
    #     plt.plot(t_OBS, waveform)
    #     pulse = pulse_list[0]
    #     t_pulse = np.arange(len(pulse))*(dt_pulse) + time_delay + initial_time_removed 
    #     plt.plot(t_pulse, pulse)
    #     plt.vlines(time_delay, min(waveform), max(waveform))
    #     plt.show()

    
    # CPU PARALLELIZED GLOBAL OPTIMIZATION ALGORITHM: 
    L2norm = np.zeros((len(data_OBS[::downsampling]), len(c_gouge_list)))
    # Define your parallel function

    # Prepare arguments for multiprocessing
    args_list = []
    for idx_waveform in range(len(data_OBS[::downsampling])):
        idx = idx_waveform * downsampling
        waveform_OBS = data_OBS[idx] - np.mean(data_OBS[idx])
        thickness_gouge_1 = thickness_gouge_1_list[idx]
        thickness_gouge_2 = thickness_gouge_2_list[idx]
        idx_travel_time = idx_travel_time_list[idx]
        
        # Package all necessary arguments into a tuple
        args = (
            idx_waveform,
            waveform_OBS,
            thickness_gouge_1,
            thickness_gouge_2,
            idx_travel_time,
            # Include any other variables needed
            t_OBS,
            t_pulse,
            pulse,
            c_gouge_list,
            # Constants can be included or passed as global if truly constant
        )
        args_list.append(args)
    
    # Define your multiprocessing function without global variables
    def process_waveform(args):
        (idx_waveform,
         waveform_OBS,
         thickness_gouge_1,
         thickness_gouge_2,
         idx_travel_time,
         t_OBS,
         t_pulse,
         pulse,
         c_gouge_list) = args
         
        sample_dimensions = [side_block_1, thickness_gouge_1, central_block, thickness_gouge_2, side_block_2]
        x_receiver = sum(sample_dimensions) - 1  # Adjust based on your setup
        
        result = np.zeros(len(c_gouge_list))
        for idx_gouge, c_gouge in enumerate(c_gouge_list):
            waveform_SYNT = DDS_UW_simulation(
                t_OBS, waveform_OBS, t_pulse, pulse,
                idx_travel_time, sample_dimensions, freq_cut,
                x_trasmitter, x_receiver, pzt_width, pla_width,
                csteel, c_gouge, cpzt, cpla,
                normalize=True, plotting=False)
            L2norm_new = compute_misfit(
                waveform_OBS=waveform_OBS,
                waveform_SYNT=waveform_SYNT,
                interval=idx_travel_time)
            result[idx_gouge] = L2norm_new

            print(f"Waveform: {idx_waveform}. Shear velocity: {c_gouge} => Misfit: {L2norm_new}")

            # Optional: print progress occasionally
        return idx_waveform, result
    
    # Set up multiprocessing inside the loop
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_waveform, args_list)
    
    # Collect and sort results
    results.sort(key=lambda x: x[0])
    L2norm = np.array([result[1] for result in results])
    
    # Save or process L2norm as needed
    np.save(outfile_path, L2norm, allow_pickle=True)

    print("--- %s seconds for processing %s---" % (tm.time() - start_time, outfile_name))


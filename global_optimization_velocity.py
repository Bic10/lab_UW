import glob
import time as tm
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

from src.file_io import *
from src.signal_processing import *
from src.LAB_UW_forward_modeling import *
################################################################################################################################
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
# GET OBSERVED DATA
# Load the pulse waveform: it is going to be our time source function
machine_name = "on_bench"
experiment_name = "glued_pzt"
data_type = "data_analysis/wavelets_from_glued_pzt"
infile_path_list_pulse  = make_infile_path_list(machine_name=machine_name,experiment_name=experiment_name,data_type=data_type)

pulse_list = []
pulse_metadata_list = []
t_pulse_list = []
time_span = []
for pulse_path in sorted(infile_path_list_pulse): 
    pulse, pulse_metadata = load_waveform_json(pulse_path)
    t_pulse = pulse_metadata['time_ax_waveform']
    time_span.append(np.ptp(t_pulse))
    pulse_list.append(pulse)
    pulse_metadata_list.append(pulse_metadata)
    t_pulse_list.append(t_pulse)
    # plt.plot(t_pulse, pulse)

time_span = min(time_span)

# Load Waveform data
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
##### PICKED MANUALLY FOR PLOTTING POURPOSE: THEY ARE CATTED PRECISELY AROUND THE MECHANICAL DATA OF THE STEP
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
    initial_time_removed = 10         # [mus]
    t_OBS,data_OBS = t_OBS[t_OBS>initial_time_removed], data_OBS[:,t_OBS>initial_time_removed]


    make_data_analysis_folders(machine_name=machine_name, experiment_name=experiment_name,data_types=["global_optimization_velocity"])

    # FREQUENCY LOW PASS:
    # reduce computation time 
    # # assumtion: there is no point to simulate anything that do not show up in the data_OBS
    freq_cut = 2                  # [MHz]   maximum frequency of the data we want to reproduce  
    # data_OBS_filtered, _  = signal2noise_separation_lowpass(data_OBS,metadata,freq_cut=freq_cut)

    for i in range(len(pulse_list)):
        pulse_list[i], _  = signal2noise_separation_lowpass(pulse_list[i],pulse_metadata_list[i],freq_cut=freq_cut)
        pulse_list[i] = pulse_list[i] - pulse_list[i][0] 

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
    # x_receiver_list = sum(saple_dimensions) - 0.95


    # GUESSED VELOCITY MODEL OF THE SAMPLE
    # S- velocity of gouge to probe. Extract from the literature!
    cmin = 600 * (1e2/1e6)        
    cmax = 2000 * (1e2/1e6) 
    c_step = 50*1e2/1e6
    c_gouge_list = np.arange(cmin, cmax,c_step) # choose of velocity in a reasonable range: from pressure-v in air to s-steel velocity
    # c_gouge_list = [cmax]


    # DEFINE EVALUATION INTERVAL FOR L2 NORM OF THE RESIDUALS
    # at the moment the gouge thickness is the same for both the layers, but still the implementation below allowed for differences.
    max_travel_time_list = 2*(side_block_1-x_trasmitter)/csteel + thickness_gouge_1_list/cmin+ central_block/csteel + thickness_gouge_1_list/cmin
    min_travel_time_list = 2*(side_block_1-x_trasmitter)/csteel + thickness_gouge_1_list/cmax+ central_block/csteel + thickness_gouge_1_list/cmax
    idx_travel_time_list = []
    for min_travel_time,max_travel_time in zip(min_travel_time_list,max_travel_time_list):
        idx_travel_time = np.where((t_OBS >min_travel_time) & (t_OBS < max_travel_time+time_span))
        idx_travel_time_list.append(idx_travel_time)

    # DOWNSAMPLING THE WAVEFORMS: FOR PLOTING PURPOSE, WE DO NOT NEED TO PROCESS ALL THE WAVEFORMS
    number_of_waveforms_wanted = 20
    data_OBS = data_OBS[:sync_peaks[2*choosen_uw_file+1]-sync_peaks[2*choosen_uw_file]] # subsempling on around the step
    metadata['number_of_waveforms'] = len(data_OBS)
    downsampling = round(metadata['number_of_waveforms']/number_of_waveforms_wanted)
    print(f"number of waveforms in the selected subset: {metadata['number_of_waveforms']}\nNumber of waveforms wanted: {number_of_waveforms_wanted}\nDownsampling waveforms by a factor: {downsampling}")

    # JUST USE ONE OF THE PULSE AS A SOURCE TIME FUNCTION
    choosen_pulse = 0
    pulse = pulse_list[choosen_pulse]
    t_pulse = t_pulse_list[choosen_pulse]

    # CPU PARALLELIZED GLOBAL OPTIMIZATION ALGORITHM: 
    L2norm = np.zeros((len(data_OBS[::downsampling]), len(c_gouge_list)))
    # Define your parallel function
    def process_waveform(idx_waveform):
        idx = idx_waveform * downsampling
        waveform_OBS = data_OBS[idx] - np.mean(data_OBS[idx])

        # there is a problem with synchronization: the number of waves are slightly different from the rec numbers
        # thy handling is a dute-tape, must be checked out the problem!
        sample_dimensions = [side_block_1, thickness_gouge_1_list[idx], central_block, thickness_gouge_2_list[idx], side_block_2]
        x_receiver = sum(sample_dimensions) - 1      # TO BE MODIFIED THE -1 IS THERE BECAUSE THE RECEVIER IS ONE CENTIMETER

        result = np.zeros(len(c_gouge_list))
        for idx_gouge, c_gouge in enumerate(c_gouge_list):
            L2norm_new = DDS_UW_simulation(t_OBS, waveform_OBS, t_pulse_list[choosen_pulse], pulse_list[choosen_pulse], idx_travel_time_list[idx], 
                            sample_dimensions, freq_cut, 
                            x_trasmitter, x_receiver, pzt_width, pla_width, 
                            csteel, c_gouge, cpzt, cpla, normalize=True, plotting=False)
            result[idx_gouge] = L2norm_new


        return idx_waveform, result



    # Set up multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes=num_processes)

    # Map the function to the data
    results = pool.map(process_waveform, range(len(data_OBS[::downsampling])))

    # Close the pool
    pool.close()
    pool.join()

    # Sort the results
    results.sort(key=lambda x: x[0])

    # Extract L2norm values
    L2norm = np.array([result[1] for result in results])

    np.save(outfile_path, L2norm, allow_pickle=True)    

    # plt.figure() 
    # right_v = c_gouge_list[np.argmin(L2norm, axis=1)]
    # plt.plot(right_v)
    # plt.savefig(outfile_path + "plot_velocity.jpg")

    print("--- %s seconds for processing %s---" % (tm.time() - start_time, outfile_name))
   

# # Capture all local variables: just to ave track of al the various parameters used in the simulation
# variables = {name: str(value) for name, value in locals().items()}
# outfile_variables_path = outfile_path.replace(outfile_name,"global_optimization_variables.json")

# # Remove system variables and other undesired variables if needed
# del variables['__name__']  # Example: Remove the '__name__' variable

# # Save variables to a file
# with open(outfile_variables_path, 'w') as f:
#     json.dump(variables, f)



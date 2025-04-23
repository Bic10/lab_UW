import pickle
from pathlib import Path

##########################################################
# analyze result from source receive simulation parameters
##########################################################
# infile_path = Path("/home/michele/Desktop/Dottorato/active_source_implementation/experiments_on_bench/STF_ss10_05/data_analysis/simulation_parameters_s_2025-04-10_only_STF_60s_stf_bandpass_from_local/width250_volt70_8")
# results_pkl = infile_path.with_suffix(".pkl")  # Or the specific path to the pickle file
# with open(results_pkl, "rb") as f:
#     loaded_data = pickle.load(f)

# # loaded_data is a dict with the keys you saved, e.g.:
# # {
# #     "global_search_space_result": result,
# #     "params": params
# # }

# global_search_space_result = loaded_data["global_search_space_result"]
# params = loaded_data["params"]

# # Now you can extract individual items:
# velocity_list_waveform = global_search_space_result["velocity_list_waveform"]
# L2norm_waveform = global_search_space_result["L2norm_waveform"]
# best_steel_velocity = global_search_space_result["best_steel_velocity"]
# best_pzt_velocity = global_search_space_result["best_pzt_velocity"]
# best_L2_misfit = global_search_space_result["best_L2_misfit"]
# best_spread_tx = global_search_space_result["best_spread_tx"]
# best_spread_rx = global_search_space_result["best_spread_rx"]
# best_position2edge_tx = global_search_space_result["best_position2edge_tx"]
# best_position2edge_rx = global_search_space_result["best_position2edge_rx"]
# best_radius_factor_tx = global_search_space_result["best_radius_factor_tx"]
# best_radius_factor_rx = global_search_space_result["best_radius_factor_rx"]

# # Use this values to update blocks_metadata.json file
# print(best_steel_velocity)
# print(best_pzt_velocity)
# # 'params' is whatever extra parameters you saved
# print("Params:", params)


##########################################################
# analyze result from local inverison:
##########################################################
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

infile_path = Path("/home/michele/Desktop/Dottorato/active_source_implementation/experiments_Brava_2/s0244suwanh3_30/data_analysis/2025_04_23_stf_local_inversion_freesurface_velgouge_damgouge/007_hold1000sec.pkl")
results_pkl = infile_path.with_suffix(".pkl") 
with open(results_pkl, "rb") as f:
    loaded_data = pickle.load(f)

# Loaded data e' un dizionario con due chiavi: una rec_n_list, l'altra best_velocity_list. 
# Direi che non e' difficile capire cosa sono i valori delle chiavi, giovanotto
# plt.plot(loaded_data["rec_n_list"], loaded_data["best_velocity_list"])
# plt.show()

for key in loaded_data:
    print(key)
    plt.plot(loaded_data[key]["gouge_velocity_model"])
plt.show()
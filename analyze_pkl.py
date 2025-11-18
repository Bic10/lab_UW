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

infile_path = Path("/home/michele/Desktop/Dottorato/active_source_implementation/experiments_Brava_2/s0244suwanh3_30/data_analysis/2025_04_25_EGU2025_shorter_duration/007_hold1000sec.pkl")
results_pkl = infile_path.with_suffix(".pkl") 
with open(results_pkl, "rb") as f:
    loaded_data = pickle.load(f)



for rec_n in loaded_data:
    # print(rec_n)
    plt.plot(loaded_data[rec_n]["gouge_velocity_model"][loaded_data[rec_n]["gouge_velocity_model"]<0.3])
    plt.xlabel("Points in the layers")
    plt.ylabel("Velocity [cm/mus]")
plt.show()

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Load the data
# infile_path = Path("/home/michele/Desktop/Dottorato/active_source_implementation/experiments_Brava_2/s0244suwanh3_30/data_analysis/2025_04_23_EGU2025/007_hold1000sec.pkl")
# results_pkl = infile_path.with_suffix(".pkl") 
# with open(results_pkl, "rb") as f:
#     loaded_data = pickle.load(f)

# # Create the data for the heatmap
# gouge_values = []  # To store gouge_velocity_model values
# rec_n_list = []    # To store corresponding rec_n values

# # Collect gouge_velocity_model values for each rec_n
# for rec_n in loaded_data:
#     gouge_velocity_model = loaded_data[rec_n]["gouge_velocity_model"]
    
#     # Filter out values above 0.3, as per your previous request
#     filtered_gouge_velocity_model = gouge_velocity_model[gouge_velocity_model < 0.3]
    
#     # Store the filtered gouge values and corresponding rec_n
#     gouge_values.append(filtered_gouge_velocity_model)
#     rec_n_list.append([rec_n] * len(filtered_gouge_velocity_model))  # Repeated rec_n for each data point

# # Convert gouge_values to a 2D numpy array
# gouge_values = [np.array(values) for values in gouge_values]

# # Now we will align the gouge velocity models for each rec_n into a grid
# # First, determine the maximum number of values for any rec_n (for consistent grid shape)
# max_length = max(len(gv) for gv in gouge_values)

# # Create an empty grid with max_length rows and enough columns for each rec_n
# data_grid = np.full((max_length, len(gouge_values)), np.nan)

# # Fill the grid with the gouge velocity values for each rec_n
# for col_idx, gouge_velocity in enumerate(gouge_values):
#     data_grid[:len(gouge_velocity), col_idx] = gouge_velocity

# # Create a plot for the heatmap
# fig, ax = plt.subplots(figsize=(10, 8))

# cmap = plt.get_cmap('YlOrRd')  # Use the 'YlOrRd' colormap
# cbar_label = "Gouge Velocity"

# # Plot the heatmap
# cax = ax.imshow(data_grid, aspect='auto', origin='lower', cmap=cmap, interpolation='none', extent=[0, len(loaded_data), 0, max_length])

# # Add a colorbar
# cbar = fig.colorbar(cax, ax=ax)
# cbar.set_label(cbar_label)

# # Set axis labels and title
# ax.set_xlabel('Rec N')
# ax.set_ylabel('Filtered Gouge Velocity Model')
# ax.set_title('Gouge Velocity Evolution')

# # Show the plot
# plt.tight_layout()
# plt.show()

#!/usr/bin/env python
"""
Create a movie of the gouge-velocity profiles (one frame per rec_n).

Output: gouge_velocity_evolution.mp4      (≈ 3 fps, H.264)
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ----------------------------------------------------------------------
# 1.  Load the data -----------------------------------------------------
# ----------------------------------------------------------------------
PKL_PATH = Path(
    "/home/michele/Desktop/Dottorato/active_source_implementation/"
    "experiments_Brava_2/s0244suwanh3_30/data_analysis/"
    "2025_04_23_EGU2025/007_hold1000sec.pkl"
)

with open(PKL_PATH, "rb") as f:
    loaded_data = pickle.load(f)

rec_keys = sorted(loaded_data)                # deterministic order

# ----------------------------------------------------------------------
# 2.  Pre-compute profiles + global axis limits ------------------------
# ----------------------------------------------------------------------
vel_by_rec = []
y_min, y_max = np.inf, -np.inf
x_max = 0

for k in rec_keys:
    v = loaded_data[k]["gouge_velocity_model"]
    v = v[v < 0.3]                            # keep < 0.3 only
    vel_by_rec.append(v)

    if v.size:
        y_min = min(y_min, v.min())
        y_max = max(y_max, v.max())
        x_max = max(x_max, v.size - 1)

if not vel_by_rec or not np.isfinite([y_min, y_max]).all():
    raise RuntimeError("No velocity points survived the <0.3 filter.")

# ----------------------------------------------------------------------
# 3.  Build the animation ----------------------------------------------
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], "-o")

ax.set_xlabel("Points in the layers")
ax.set_ylabel("Velocity [cm/μs]")
ax.set_xlim(0, x_max)
ax.set_ylim(y_min * 0.95, y_max * 1.05)

def init():
    line.set_data([], [])
    return line,

def update(i):
    y = vel_by_rec[i]
    x = np.arange(y.size)
    line.set_data(x, y)
    ax.set_title(f"Gouge velocity evolution – rec_n = {rec_keys[i]}")
    return line,

ani = FuncAnimation(
    fig, update,
    frames=len(rec_keys),
    init_func=init,
    blit=True,
    repeat=False
)

# ----------------------------------------------------------------------
# 4.  Save to MP4 -------------------------------------------------------
# ----------------------------------------------------------------------
OUT = "gouge_velocity_evolution.mp4"
writer = FFMpegWriter(fps=3, codec="libx264", bitrate=1800,
                      metadata={"title": "Gouge velocity evolution",
                                "artist": "Your Name"})
ani.save(OUT, writer=writer, dpi=200)
print(f"✓ Movie written to {OUT}")








# lab_uw/utils.py

import numpy as np
import pickle
from pathlib import Path
from lab_uw.data_io import UltrasonicDataHandler
from lab_uw.plotting import InteractivePlotter

def pick_arrival_times(
    dir_manager, machine_name, experiment_name, infile_path_list_uw, start_time=0
):
    """
    Prepare manual pick arrival times by processing UW files.

    Parameters:
        dir_manager (DirectoryManager): Manages directory paths.
        machine_name (str): Name of the machine used for the experiment.
        experiment_name (str): Name of the experiment.
        infile_path_list_uw (list[Path]): List of paths to ultrasonic waveform files.
        start_time (float, optional): Starting time for processing. Defaults to 0s.

    Returns:
        list: A list of manual pick arrival time intervals.
    """
    # Prepare output directory

    experiment_path = dir_manager.base_dir / f"experiments_{machine_name}" / experiment_name
    picked_travel_times_dir = experiment_path / 'data_analysis' / 'picked_travel_times'
    picked_travel_times_dir.mkdir(parents=True, exist_ok=True)

    arrival_times_list = []
    for infile_path_uw in infile_path_list_uw:
        stem = Path(infile_path_uw.stem).stem  # Get file stem
        new_file_name = f"{stem}.pkl"
        infile_path_travel_times = picked_travel_times_dir / new_file_name

        try:
            with open(infile_path_travel_times, 'rb') as f:
                arrival_times_list.append(pickle.load(f))
        except FileNotFoundError:
            waveform_choosed = 0
            
            # Instantiate UltrasonicDataHandler using the Path object
            ultrasonic_handler = UltrasonicDataHandler.load_UW_data(infile_path_uw)
            observed_waveform_data, metadata = ultrasonic_handler.waveform_data, ultrasonic_handler.metadata
            observed_waveform = observed_waveform_data[waveform_choosed]
            observed_time = metadata['time_ax_waveform']

            picked_times = InteractivePlotter().manual_pick_arrival_times(
                observed_time=observed_time,
                observed_waveform=observed_waveform,
                start_time=start_time,
                outfile_path=infile_path_travel_times
            )
            arrival_times_list.append(picked_times)

    return arrival_times_list

def solve_quadratic_equation(A, B, C, real_only=True, positive_only=False):
    """
    Solve A*x^2 + B*x + C = 0 for x, returning up to two solutions.

    Parameters
    ----------
    A : float
        Quadratic coefficient.
    B : float
        Linear coefficient.
    C : float
        Constant term.
    real_only : bool, optional
        If True, discard complex solutions. Default is True.
    positive_only : bool, optional
        If True, only return positive real solutions. Default is False.

    Returns
    -------
    solutions : list of float
        List of up to two valid solutions (depending on filters).
        If no valid solutions, returns an empty list.

    Notes
    -----
    - If A == 0, the equation is not quadratic. You might want to handle
      that as a special case or return an empty list.
    - If real_only is False, complex solutions (if any) are returned
      exactly as Python complex numbers.
    """
    solutions = []

    # Handle degenerate case (A=0 => linear or no solution)
    if abs(A) < 1e-14:
        # It's effectively B*x + C = 0 if B != 0
        if abs(B) > 1e-14:
            x = -C / B
            # Filter if needed
            if (not real_only) or (isinstance(x, float)):  # It's real
                if (not positive_only) or (x > 0):
                    solutions.append(x)
        return solutions

    # Compute discriminant
    discriminant = B**2 - 4*A*C

    if not real_only:
        # Return complex solutions as well
        sqrt_disc = np.sqrt(discriminant + 0j)  # Force complex
        x1 = (-B + sqrt_disc) / (2*A)
        x2 = (-B - sqrt_disc) / (2*A)
        # Filter for positive
        if positive_only:
            if (x1.real > 0) and (abs(x1.imag) < 1e-14):
                solutions.append(x1.real)
            if (x2.real > 0) and (abs(x2.imag) < 1e-14):
                solutions.append(x2.real)
        else:
            solutions.extend([x1, x2])
    else:
        # real_only=True
        if discriminant < 0:
            return solutions  # No real solutions
        sqrt_disc = np.sqrt(discriminant)
        x1 = (-B + sqrt_disc) / (2*A)
        x2 = (-B - sqrt_disc) / (2*A)

        # Keep real solutions
        for x in (x1, x2):
            if not positive_only or x > 0:
                solutions.append(x)

    return solutions

def cross_correlate_wavelet_signal(wavelet, signal, dt):
    """
    Cross-correlate a wavelet with a recorded signal.
    
    Parameters
    ----------
    wavelet : ndarray
        1D array containing the wavelet.
    signal : ndarray
        1D array containing the recorded signal.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    time_shift : float
        Estimated time shift (in seconds) of the wavelet within the signal.
    corr : ndarray
        Full cross-correlation array.
    lags : ndarray
        Array of sample lags corresponding to 'corr'.
    """
    from scipy.signal import correlate
    # 'full' mode returns cross-correlation at all possible lags
    corr = correlate(signal, wavelet, mode='full')
    # Lags: from -(len(wavelet)-1) to (len(signal)-1)
    lags = np.arange(-len(wavelet) + 1, len(signal))
    
    # Index of the maximum correlation
    imax = np.argmax(corr)
    # Convert sample lag to time shift
    best_lag = lags[imax]
    time_shift = best_lag * dt
    
    return time_shift, corr, lags

def plot_wavelet_over_signal(wavelet, signal, dt, time_shift):
    """
    Plot the recorded signal and overlay the wavelet shifted to the 
    location of maximum correlation.
    
    Parameters
    ----------
    wavelet : ndarray
        1D array containing the wavelet.
    signal : ndarray
        1D array containing the recorded signal.
    dt : float
        Sampling interval in seconds.
    time_shift : float
        Time shift (in seconds) at which to overlay the wavelet.
    """
    import matplotlib.pyplot as plt

    t_signal = np.arange(len(signal)) * dt
    t_wavelet = np.arange(len(wavelet)) * dt
    
    # You might want to align the *center* of the wavelet to the best match:
    # For instance, shift by half the wavelet length to place its center
    # at the correlation maximum:
    # shift_correction = (len(wavelet)//2) * dt
    # t_wavelet_shifted = t_wavelet + time_shift - shift_correction
    
    # If you prefer to overlay the wavelet's start at the best match, do:
    t_wavelet_shifted = t_wavelet + time_shift
    
    plt.figure(figsize=(8,4))
    plt.plot(t_signal, signal, label='Recorded Signal')
    plt.plot(t_wavelet_shifted, wavelet, label='Wavelet (Aligned)', color='red')
    plt.title("Wavelet Overlaid on Recorded Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

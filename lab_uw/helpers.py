# src/helpers.py
import numpy as np

def fourier_derivative_2nd(f: np.ndarray, dx: float) -> np.ndarray:
    '''
    Compute second spatial derivatives using Fourier transform.

    Args:
        f (np.ndarray): Input function.
        dx (float): Grid spacing.

    Returns:
        np.ndarray: Second spatial derivative.
    '''
    # Length of vector f
    nx = np.size(f)
    # Initialize k vector up to Nyquist wavenumber 
    kmax = np.pi / dx
    dk = kmax / (nx / 2)
    k = np.arange(float(nx))
    k[: int(nx/2)] = k[: int(nx/2)] * dk
     
    k[int(nx/2) :] = k[: int(nx/2)] - kmax
    
    # Fourier derivative
    ff = np.fft.fft(f)
    ff = (1j*k)**2 * ff
    df_num = np.real(np.fft.ifft(ff))
    return df_num

def calculate_scaling_factor(distance_from_source: float) -> float:
    scaling_factor = 1 / (4 * np.pi * (distance_from_source +1))
    scaling_factor = scaling_factor/np.amax(scaling_factor)
    return scaling_factor
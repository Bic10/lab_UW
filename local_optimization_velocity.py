import numpy as np
from LAB_UW_forward_modeling import DDS_UW_simulation, pseudospectral_1D_forward, compute_misfit
from synthetic_data import make_grid_1D, build_velocity_model, synthetic_source_spatial_function

def forward_modeling(velocity_model, source, t_OBS, dt, dx, nt, nx, isx, irx, sample_dimensions, freq_cut, x_trasmitter, x_receiver, pzt_width, pmma_width, csteel, cpzt, cpmma):
    """
    Solve the forward wave equation to compute synthetic seismograms.
    
    Args:
        velocity_model (np.ndarray): Current 1D velocity model.
        source (np.ndarray): Source time function.
        t_OBS (np.ndarray): Time axis of observed waveform.
        dt (float): Time step.
        dx (float): Spatial step.
        nt (int): Number of time steps.
        nx (int): Number of spatial steps.
        isx (int): Source index.
        irx (int): Receiver index.
        sample_dimensions (tuple): Dimensions of the sample.
        freq_cut (float): Frequency cutoff for filtering.
        x_trasmitter (float): Transmitter location.
        x_receiver (float): Receiver location.
        pzt_width (float): Width of the PZT.
        pmma_width (float): Width of the PMMA.
        csteel (float): Velocity in steel.
        cpzt (float): Velocity in the PZT.
        cpmma (float): Velocity in the PMMA.

    Returns:
        np.ndarray: The simulated wavefield.
    """
    # Prepare the synthetic wavefield using the DDS_UW_simulation function from LAB_UW_forward_modeling
    L2norm = DDS_UW_simulation(
        t_OBS=t_OBS, 
        waveform_OBS=np.zeros(len(t_OBS)),  # Placeholder since we only need the synthetic output
        t_pulse=source, 
        pulse=source, 
        interval=slice(None),  # Use full interval for forward modeling
        sample_dimensions=sample_dimensions,
        freq_cut=freq_cut, 
        x_trasmitter=x_trasmitter, 
        x_receiver=x_receiver, 
        pzt_width=pzt_width, 
        pmma_width=pmma_width, 
        c_max=np.max(velocity_model), 
        c_gouge=velocity_model, 
        c_pzt=cpzt, 
        c_pmma=cpmma, 
        normalize=False, 
        plotting=False
    )

    return L2norm

def compute_misfit(observed_data, synthetic_data):
    """
    Compute the misfit between observed and synthetic seismograms.
    
    Args:
        observed_data (np.ndarray): Observed seismograms.
        synthetic_data (np.ndarray): Synthetic seismograms.
    
    Returns:
        float: The computed misfit (e.g., L2 norm).
    """
    return np.linalg.norm(observed_data - synthetic_data)

def adjoint_state(observed_data, synthetic_data):
    """
    Compute the adjoint source based on the data misfit.
    
    Args:
        observed_data (np.ndarray): Observed seismograms.
        synthetic_data (np.ndarray): Synthetic seismograms.
    
    Returns:
        np.ndarray: The adjoint source (reversed time residuals).
    """
    residual = observed_data - synthetic_data
    adjoint_source = residual[::-1]  # Reverse time residual
    return adjoint_source

def compute_gradient(velocity_model, forward_wavefield, adjoint_wavefield, dt, dx):
    """
    Compute the gradient of the misfit function with respect to the velocity model.
    
    Args:
        velocity_model (np.ndarray): Current 1D velocity model.
        forward_wavefield (np.ndarray): Wavefield from the forward problem.
        adjoint_wavefield (np.ndarray): Wavefield from the adjoint problem.
        dt (float): Time step.
        dx (float): Spatial step.
    
    Returns:
        np.ndarray: The gradient of the misfit function.
    """
    # Compute the gradient using the formula derived from adjoint state method
    gradient = np.zeros_like(velocity_model)
    # Assuming the gradient is computed using some relationship between forward and adjoint fields
    # You might need to implement or derive this from your formulation
    # ...
    return gradient

def update_velocity_model(velocity_model, gradient, step_length):
    """
    Update the velocity model using the gradient descent method.
    
    Args:
        velocity_model (np.ndarray): Current 1D velocity model.
        gradient (np.ndarray): Gradient of the cost function.
        step_length (float): Step length for the gradient update.
    
    Returns:
        np.ndarray: Updated 1D velocity model.
    """
    updated_velocity = velocity_model - step_length * gradient
    return updated_velocity

def local_inversion(observed_data, initial_velocity_model, source, dt, dx, nt, nx, max_iterations, tolerance, sample_dimensions, freq_cut, x_trasmitter, x_receiver, pzt_width, pmma_width, csteel, cpzt, cpmma):
    """
    Perform local inversion to find the best 1D velocity model.
    
    Args:
        observed_data (np.ndarray): Observed seismograms.
        initial_velocity_model (np.ndarray): Initial 1D velocity model.
        source (np.ndarray): Source time function.
        dt (float): Time step.
        dx (float): Spatial step.
        nt (int): Number of time steps.
        nx (int): Number of spatial steps.
        max_iterations (int): Maximum number of iterations for the inversion.
        tolerance (float): Tolerance for convergence.
        sample_dimensions (tuple): Sample dimensions for the experiment.
        freq_cut (float): Frequency cutoff.
        x_trasmitter (float): Transmitter location.
        x_receiver (float): Receiver location.
        pzt_width (float): Width of PZT.
        pmma_width (float): Width of PMMA.
        csteel (float): Steel velocity.
        cpzt (float): PZT velocity.
        cpmma (float): PMMA velocity.
    
    Returns:
        np.ndarray: The optimized 1D velocity model.
    """
    velocity_model = initial_velocity_model.copy()
    for iteration in range(max_iterations):
        # Forward modeling to compute synthetic data
        synthetic_data = forward_modeling(
            velocity_model, source, observed_data, dt, dx, nt, nx, np.argmin(np.abs(x - x_trasmitter)), np.argmin(np.abs(x - x_receiver)),
            sample_dimensions, freq_cut, x_trasmitter, x_receiver, pzt_width, pmma_width, csteel, cpzt, cpmma
        )
        
        # Compute misfit
        misfit = compute_misfit(observed_data, synthetic_data)
        print(f"Iteration {iteration}: Misfit = {misfit}")

        # Check for convergence
        if misfit < tolerance:
            print("Convergence achieved.")
            break
        
        # Compute adjoint source and solve adjoint state equation
        adjoint_source = adjoint_state(observed_data, synthetic_data)
        adjoint_wavefield = forward_modeling(
            velocity_model, adjoint_source, observed_data, dt, dx, nt, nx, np.argmin(np.abs(x - x_trasmitter)), np.argmin(np.abs(x - x_receiver)),
            sample_dimensions, freq_cut, x_trasmitter, x_receiver, pzt_width, pmma_width, csteel, cpzt, cpmma
        )
        
        # Compute gradient
        gradient = compute_gradient(velocity_model, synthetic_data, adjoint_wavefield, dt, dx)
        
        # Update velocity model
        step_length = 0.1  # Set appropriate step length; can use line search for optimization
        velocity_model = update_velocity_model(velocity_model, gradient, step_length)
    
    return velocity_model

# Example usage
observed_data = np.load("observed_data.npy")  # Load observed data
initial_velocity_model = np.ones(100) * 1500  # Initial homogeneous velocity model
source = np.load("source.npy")  # Load source time function

dt = 0.001  # Time step
dx = 10     # Spatial step
nt = 1000   # Number of time steps
nx = 100    # Number of spatial steps
max_iterations = 50
tolerance = 1e-5
freq_cut = 2.0  # Frequency cutoff
x_trasmitter = 1.0  # Transmitter location
x_receiver = 5.0  # Receiver location
pzt_width = 0.1  # PZT width
pmma_width = 0.1  # PMMA width
csteel = 3374 * (1e2 / 1e6)  # Steel velocity in cm/mus
cpzt = 2000 * (1e2 / 1e6)  # PZT velocity in cm/mus
cpmma = 0.4 * 0.1392  # PMMA velocity in cm/mus
sample_dimensions = (2, 2, 4.8, 2, 2)  # Example sample dimensions

optimized_velocity_model = local_inversion(observed_data, initial_velocity_model, source, dt, dx, nt, nx, max_iterations, tolerance, sample_dimensions, freq_cut, x_trasmitter, x_receiver, pzt_width, pmma_width, csteel, cpzt, cpmma)

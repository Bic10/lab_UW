# lab_uw/simulation_setup.py
import numpy as np
from typing import Union, Tuple, Optional
from lab_uw.plotting import Plotter
from pathlib import Path
from scipy.signal.windows import kaiser
from numpy import convolve

class Grid1D:
    def __init__(self, cmin: float, fmax: float, grid_len: float, ppt: int):
        '''
        Initialize a 1D grid.

        Args:
            cmin (float): Minimum velocity.
            fmax (float): Maximum frequency.
            grid_len (float): Length of the grid.
            ppt (int): Points per the shortest wavelength.
        '''
        self.cmin = cmin
        self.fmax = fmax
        self.grid_len = grid_len
        self.ppt = ppt
        self.spatial_axis = self.make_grid()
        self.dx = self.spatial_axis[1] - self.spatial_axis[0]
        self.total_grid_points = len(self.spatial_axis)

    def make_grid(self) -> np.ndarray:
        '''
        Create a 1D grid.

        Returns:
            np.ndarray: 1D grid.
        '''
        lambda_min = self.cmin / self.fmax       # [cm] minimum wavelength of the simulation
        dx = lambda_min / self.ppt               # dx spacing x-axis
        x = np.arange(0, self.grid_len + dx, dx)      # [cm] space coordinates
        # if len(x) % 2:
        #     x = np.append(x, self.grid_len + dx)
        return x

    def plot(self):
        '''
        Plot the grid (if necessary).
        '''
        # Implement plotting if needed
        pass

class SimulationTime:
    def __init__(self, observed_time: np.ndarray, dx: float, max_velocity: float, cfl_factor: float = 0.49):
        '''
        Initialize simulation time variables.

        Args:
            observed_time (np.ndarray): Time array from the observed data.
            dx (float): Spatial step size.
            max_velocity (float): Maximum velocity in the medium.
            cfl_factor (float): CFL condition factor.
        '''
        self.observed_time = observed_time
        self.dx = dx
        self.max_velocity = max_velocity
        self.cfl_factor = cfl_factor
        self.simulation_time = None
        # self.dt = None
        # self.num_t = None
        self.prepare_time_variables()

    def prepare_time_variables(self):
        '''
        Prepare the time variables for the simulation based on the spatial grid and maximum velocity.
        '''
        # Calculate the raw time step based on CFL condition
        dt_raw = (self.cfl_factor * self.dx) / self.max_velocity
        # Extract the data sampling rate from observed_time
        dt_obs = self.observed_time[1] - self.observed_time[0]

        # Find the largest submultiple of dt_obs smaller than dt_raw
        self.dt = dt_obs / np.ceil(dt_obs / dt_raw)

        # Create the time axis for the simulation
        self.simulation_time = np.arange(0, self.observed_time[-1], self.dt)

        self.num_t = len(self.simulation_time)

class VelocityModel1D:
    def __init__(
        self,
        x: np.ndarray,
        sample_dimensions: Tuple[float, float, float, float, float],
        x_transmitter: float,
        x_receiver: float,
        pzt_layer_width: float,
        pla_layer_width: float,
        h_groove_side: float,
        h_groove_central: float,
        steel_velocity: float,
        gouge_velocity: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
        pzt_velocity: float,
        pla_velocity: float,
        outfile_path: Path = False
    ):
        '''
        Initialize the 1D velocity model.
        '''
        self.x = x
        self.sample_dimensions = sample_dimensions
        self.x_transmitter = x_transmitter
        self.x_receiver = x_receiver
        self.pzt_layer_width = pzt_layer_width
        self.pla_layer_width = pla_layer_width
        self.h_groove_side = h_groove_side
        self.h_groove_central = h_groove_central
        self.steel_velocity = steel_velocity
        self.gouge_velocity = gouge_velocity
        self.pzt_velocity = pzt_velocity
        self.pla_velocity = pla_velocity
        self.outfile_path = outfile_path
        self.layer_starts = None
        self.idx_dict = {}
        self.values = None  # Velocity model array

        self.build_velocity_model()

    def build_velocity_model(self):
        '''
        Build the 1D velocity model.
        '''
        self.compute_layer_positions()
        self.define_region_indices()
        self.initialize_velocity_model()
        self.assign_velocities()
        # self.apply_smoothing()

    def compute_layer_positions(self):
        '''
        Compute layer thicknesses and cumulative starts.
        '''
        # Unpack sample dimensions
        (side_block_1_total, gouge_1_length,
         central_block_total, gouge_2_length, side_block_2_total) = self.sample_dimensions

        # Adjust block lengths to exclude groove heights
        side_block_1 = side_block_1_total - self.h_groove_side
        side_block_2 = side_block_2_total - self.h_groove_side
        central_block = central_block_total - 2 * self.h_groove_central  # Subtract grooves on both sides

        # Compute cumulative positions along the sample
        self.layer_thicknesses = [
            self.pla_layer_width,
            self.pzt_layer_width,
            side_block_1,
            self.h_groove_side,
            gouge_1_length,
            self.h_groove_central,
            central_block,
            self.h_groove_central,
            gouge_2_length,
            self.h_groove_side,
            side_block_2,
            self.pzt_layer_width,
            self.pla_layer_width
        ]
        self.layer_starts = np.concatenate(([0.0], np.cumsum(self.layer_thicknesses)))

    def define_region_indices(self):
        '''
        Define indices for each region.
        '''
        x = self.x  # For brevity

        regions = [
            'pla_1',
            'pzt_1',
            'side_block_1',
            'groove_sb1',
            'gouge_1',
            'groove_cb1',
            'central_block',
            'groove_cb2',
            'gouge_2',
            'groove_sb2',
            'side_block_2',
            'pzt_2',
            'pla_2'
        ]

        for i, region in enumerate(regions):
            start = self.layer_starts[i]
            end = self.layer_starts[i + 1]
            indices = np.where((x >= start) & (x < end))[0]
            self.idx_dict[region] = indices

    def initialize_velocity_model(self):
        '''
        Initialize the velocity model with default velocities.
        '''
        self.values = self.steel_velocity * np.ones_like(self.x)

    def assign_velocities(self):
        '''
        Assign velocities to each region.
        '''
        self.assign_constant_velocity('pla_1', self.pla_velocity)
        self.assign_constant_velocity('pzt_1', self.pzt_velocity)
        # Side Block 1 remains steel_velocity

        self.assign_groove_velocity('groove_sb1', self.gouge_velocity[0], is_start=True)
        self.assign_gouge_velocity('gouge_1', self.gouge_velocity[0])
        self.assign_groove_velocity('groove_cb1', self.gouge_velocity[0], is_start=False)

        self.assign_constant_velocity('central_block', self.steel_velocity)

        self.assign_groove_velocity('groove_cb2', self.gouge_velocity[1], is_start=True)
        self.assign_gouge_velocity('gouge_2', self.gouge_velocity[1])
        self.assign_groove_velocity('groove_sb2', self.gouge_velocity[1], is_start=False)

        # Side Block 2 remains steel_velocity
        self.assign_constant_velocity('pzt_2', self.pzt_velocity)
        self.assign_constant_velocity('pla_2', self.pla_velocity)

    def assign_constant_velocity(self, region_name: str, velocity: float):
        '''
        Assign a constant velocity to a region.
        '''
        indices = self.idx_dict.get(region_name, [])
        self.values[indices] = velocity

    def assign_gouge_velocity(self, region_name: str, gouge_velocity: Union[float, np.ndarray]):
        '''
        Assign velocity to gouge regions.
        '''
        indices = self.idx_dict.get(region_name, [])
        if not indices.size:
            return
        if isinstance(gouge_velocity, np.ndarray):
            if len(gouge_velocity) != len(indices):
                raise ValueError(f"Length of gouge_velocity does not match the size of {region_name} region.")
            self.values[indices] = gouge_velocity
        else:
            self.values[indices] = gouge_velocity

    def assign_groove_velocity(self, region_name: str, adjacent_velocity: Union[float, np.ndarray], is_start: bool):
        '''
        Assign velocities in groove regions with linear gradients.
        '''
        indices = self.idx_dict.get(region_name, [])
        if not indices.size:
            return
        groove_length = len(indices)
        if groove_length == 1:
            self.values[indices] = adjacent_velocity[0] if isinstance(adjacent_velocity, np.ndarray) else adjacent_velocity
            return

        start_vel = self.steel_velocity if is_start else (adjacent_velocity[-1] if isinstance(adjacent_velocity, np.ndarray) else adjacent_velocity)
        end_vel = (adjacent_velocity[0] if isinstance(adjacent_velocity, np.ndarray) else adjacent_velocity) if is_start else self.steel_velocity

        self.values[indices] = np.linspace(start_vel, end_vel, groove_length)

    def apply_smoothing(self):
        '''
        Apply smoothing between pzt_velocity and steel_velocity around transmitter and receiver.
        '''
        x = self.x

        # Transmitter smoothing
        transmitter_smoothing_start = self.layer_starts[1]
        transmitter_smoothing_end = self.layer_starts[2] + self.pzt_layer_width
        transmitter_indices = np.where((x >= transmitter_smoothing_start) & (x < transmitter_smoothing_end))[0]
        if transmitter_indices.size > 0:
            self.values[transmitter_indices] = np.linspace(
                self.pzt_velocity, self.steel_velocity, len(transmitter_indices))

        # Receiver smoothing
        receiver_smoothing_start = self.layer_starts[11]  # Start of PZT layer
        receiver_smoothing_end = self.layer_starts[12] + self.pzt_layer_width  # End of smoothing region
        receiver_indices = np.where((x >= receiver_smoothing_start) & (x < receiver_smoothing_end))[0]
        if receiver_indices.size > 0:
            self.values[receiver_indices] = np.linspace(
                self.steel_velocity, self.pzt_velocity, len(receiver_indices))

    def plot(self, outfile_path: Optional[str] = None):
        '''
        Plot the velocity model.
        '''
        Plotter().plot_velocity_model(
            x=self.x,
            c=self.values,
            layer_starts=self.layer_starts,
            pzt_layer_width=self.pzt_layer_width,
            pla_layer_width=self.pla_layer_width,
            outfile_path=outfile_path
        )

class VelocityModel1D_SingleBlock:
    """
    A simple 1D velocity model for a single block with transmitter on one side
    and receiver on the other. The block is sandwiched by PLA and PZT layers.
    
    Layer layout (from left to right):
        PLA -> PZT -> STEEL_BLOCK -> PZT -> PLA
    
    The STEEL_BLOCK region is adjusted to account for x_transmitter and x_receiver offsets.
    
    Attributes
    ----------
    x : np.ndarray
        Spatial grid (e.g., in cm or mm).
    sample_dimensions : Tuple[float]
        Expected to have exactly one element: (block_total_length,).
    x_transmitter : float
        Offset subtracted from the left boundary of the block for the transmitter.
    x_receiver : float
        Offset subtracted from the right boundary of the block for the receiver.
    pzt_layer_width : float
        Thickness of PZT layer at each side (transmitter and receiver).
    pla_layer_width : float
        Thickness of PLA layer at each side (transmitter and receiver).
    steel_velocity : float
        Constant velocity for steel.
    pzt_velocity : float
        Constant velocity for PZT.
    pla_velocity : float
        Constant velocity for PLA.
    plotting : bool
        Whether to produce a plot when the model is built.
    outfile_path : Optional[Path]
        If provided and plotting=True, the final velocity model plot is saved here.
    
    """


    def __init__(
        self,
        x: np.ndarray,
        sample_dimensions: Tuple[float,],
        x_transmitter: float,
        x_receiver: float,
        pzt_layer_width: float,
        pla_layer_width: float,
        steel_velocity: float,
        pzt_velocity: float,
        pla_velocity: float,
        plotting: bool = False,
        outfile_path: Optional[Path] = None
    ):
        self.x = x
        self.sample_dimensions = sample_dimensions  # (block_total_length,)
        self.x_transmitter = x_transmitter
        self.x_receiver = x_receiver
        self.pzt_layer_width = pzt_layer_width
        self.pla_layer_width = pla_layer_width
        self.steel_velocity = steel_velocity
        self.pzt_velocity = pzt_velocity
        self.pla_velocity = pla_velocity
        self.plotting = plotting
        self.outfile_path = outfile_path

        # Internal data containers
        self.layer_thicknesses = []
        self.layer_starts = None
        self.idx_dict = {}
        self.values = None  # Will hold the final velocity model

        # Build the model
        self.build_velocity_model()

    def build_velocity_model(self):
        self.compute_layer_positions()
        self.define_region_indices()
        self.initialize_velocity_model()
        self.assign_velocities()

        # # Now apply smoothing for each boundary you care about:
        # self.apply_smoothing("pzt_1", "steel_block", 10)
        # self.apply_smoothing("steel_block", "pzt_2", 10)

    def compute_layer_positions(self):
        """
        Compute the thicknesses of each layer:
          [PLA, PZT, STEEL_BLOCK, PZT, PLA]

        The steel block thickness is adjusted by x_transmitter and x_receiver offsets.
        """
        if len(self.sample_dimensions) != 1:
            raise ValueError("sample_dimensions must have exactly one element (the total block length) for a single-block model.")

        block_total_length = self.sample_dimensions[0]
        steel_length = block_total_length
        
        # Build the layer thickness list
        self.layer_thicknesses = [
            self.pla_layer_width,
            self.pzt_layer_width,
            steel_length,
            self.pzt_layer_width,
            self.pla_layer_width
        ]
        # Compute the cumulative starts: [0, layer1, layer1+layer2, ...]
        self.layer_starts = np.concatenate(([0.0], np.cumsum(self.layer_thicknesses)))

    def define_region_indices(self):
        """
        Create a dictionary that maps layer names to the x-grid indices that lie within each layer's start/end.
        """
        x = self.x
        regions = ["pla_1", "pzt_1", "steel_block", "pzt_2", "pla_2"]
        self.idx_dict.clear()

        for i, region in enumerate(regions):
            start = self.layer_starts[i]
            end   = self.layer_starts[i+1]

            indices = np.where((x >= start) & (x <= end))[0]

            self.idx_dict[region] = indices

    def initialize_velocity_model(self):
        """ Fill the entire velocity array with steel_velocity by default. """
        self.values = np.ones_like(self.x)

    def assign_velocities(self):
        """
        Assign velocities for each of the five regions:
            pla_1 -> pla_velocity
            pzt_1 -> pzt_velocity
            steel_block -> steel_velocity (already assigned by default, but let's be explicit)
            pzt_2 -> pzt_velocity
            pla_2 -> pla_velocity
        """
        self.assign_constant_velocity("pla_1", self.pla_velocity)
        self.assign_constant_velocity("pzt_1", self.pzt_velocity)
        self.assign_constant_velocity("steel_block", self.steel_velocity)
        self.assign_constant_velocity("pzt_2", self.pzt_velocity)
        self.assign_constant_velocity("pla_2", self.pla_velocity)
        #  duct-tape solution for last index non-assignement
        self.values[-1] = self.values[-2]

    def assign_constant_velocity(self, region_name: str, velocity: float):
        """ Helper to set a uniform velocity in a specified region. """
        indices = self.idx_dict.get(region_name, [])
        if indices.size:
            self.values[indices] = velocity

    def apply_smoothing(
        self,
        region_from: str,
        region_to: str,
        n_smooth: int
    ):
        """
        Smooths the boundary transition from 'region_from' to 'region_to' 
        using up to 'n_smooth' points at the end of region_from and 'n_smooth'
        points at the start of region_to.
        
        1. Retrieves each region’s velocity (pzt_velocity, steel_velocity, etc.) 
        from a local dictionary that maps region_name -> velocity.
        2. Finds the last 'n_smooth' indices in region_from and the first 
        'n_smooth' indices in region_to.
        3. Builds a linear ramp from region_from's velocity to region_to's velocity
        across those boundary indices.
        4. Writes the ramp into self.values for that boundary zone.
        """

        # A small lookup so we don’t need separate velocity arguments
        region_velocity_map = {
            "pla_1":       self.pla_velocity,
            "pzt_1":       self.pzt_velocity,
            "steel_block": self.steel_velocity,
            "pzt_2":       self.pzt_velocity,
            "pla_2":       self.pla_velocity
        }

        if region_from not in self.idx_dict or region_to not in self.idx_dict:
            return  # one or both regions don't exist in the idx_dict

        idx_from = self.idx_dict[region_from]
        idx_to   = self.idx_dict[region_to]
        if idx_from.size == 0 or idx_to.size == 0:
            return  # region(s) are empty => no smoothing

        vel_from = region_velocity_map.get(region_from, None)
        vel_to   = region_velocity_map.get(region_to, None)
        if vel_from is None or vel_to is None:
            return  # unknown region => skip

        # Sort the indices so they're in ascending spatial order
        idx_from_sorted = np.sort(idx_from)
        idx_to_sorted   = np.sort(idx_to)

        # Up to n_smooth points at the end of region_from
        n_tail = min(n_smooth, idx_from_sorted.size)
        tail   = idx_from_sorted[-n_tail:]  # last portion of region_from

        # Up to n_smooth points at the start of region_to
        n_head = min(n_smooth, idx_to_sorted.size)
        head   = idx_to_sorted[:n_head]     # first portion of region_to

        boundary_indices = np.concatenate([tail, head])
        if boundary_indices.size < 2:
            return  # nothing to ramp

        # Build a linear ramp from vel_from to vel_to
        ramp = np.linspace(vel_from, vel_to, boundary_indices.size)

        self.values[boundary_indices] = ramp

    def plot(self, outfile_path: Optional[str] = None):
        '''
        Plot the velocity model.
        '''
        Plotter().plot_velocity_model(
            x=self.x,
            c=self.values,
            layer_starts=self.layer_starts,
            pzt_layer_width=self.pzt_layer_width,
            pla_layer_width=self.pla_layer_width,
            outfile_path=outfile_path
        )

class Source1D:
    def __init__(self, stf_time: np.ndarray, stf_waveform: np.ndarray, position: float, radius: int, 
                    extension: float,
                    pzt_layer_width: float,
):
        '''
        Initialize the 1D source.

        Args:
            stf_time (np.ndarray): Time array of the source time function.
            stf_waveform (np.ndarray): Source time function waveform.
            position (float): Position of the source on the spatial grid.
            radius (int): Radius for the spatial function (number of grid points).
        '''
        self.stf_time = stf_time
        self.stf_waveform = stf_waveform
        self.position = position
        self.radius = radius
        self.extension = extension
        self.pzt_layer_width = pzt_layer_width
        self.time_function = None  # Will be set after interpolation
        self.spatial_function = None  # Will be set after being created on the grid

    def interpolate_time_function(self, dt: float, simulation_time: np.ndarray):
        '''
        Interpolate the source time function to match the simulation time discretization.

        Args:
            dt (float): Time step size of the simulation.
            simulation_time (np.ndarray): Simulation time array.
        '''
        interpolated_stf_time = np.arange(self.stf_time[0], self.stf_time[-1], dt)
        interpolated_stf = np.interp(interpolated_stf_time, self.stf_time, self.stf_waveform)
        self.time_function = np.zeros(len(simulation_time))
        self.time_function[:len(interpolated_stf)] = interpolated_stf

    def create_spatial_function(self, spatial_axis: np.ndarray, dx: float):
        # self.spatial_function = convolved_sinc_gaussian_filter(
        #     spatial_axis=spatial_axis,
        #     dx=dx,
        #     position=self.position,
        #     pzt_layer_width=self.pzt_layer_width,
        #     spreading_factor=self.spreading_factor,
        #     radius=self.radius,
        #     flip_side=flip_side
        # )
        self.spatial_function = arbitrary_source_and_receiver_positioning(
            spatial_axis=spatial_axis,
            dx=dx,
            position=self.position,
            pzt_layer_width=self.pzt_layer_width,
            extension=self.extension,
            radius=self.radius,
        )

    def plot_spatial_function(self, x: np.ndarray, spatial_function: np.ndarray, outfile_path: Optional[str] = None):
        '''
        Plot the synthetic spatial function.

        Args:
            x (np.ndarray): Spatial axis.
            spatial_function (np.ndarray): Spatial function values.
            outfile_path (str, optional): Path to save the plot.
        '''
        Plotter().plot_synthetic_spatial_function(
            x=x,
            spatial_function=spatial_function,
            outfile_path=outfile_path
        )

class Receiver1D:
    def __init__(self, position: float, radius: int, 
                    extension: float,
                    pzt_layer_width: float,
                ):        
        '''
        Initialize the 1D receiver.

        Args:
            position (float): Position of the receiver on the spatial grid.
            radius (int): Radius for the spatial function (number of grid points).
        '''
        self.position = position
        self.radius = radius
        self.extension = extension
        self.pzt_layer_width = pzt_layer_width
        self.spatial_function = None  # Will be set after being created on the grid

    def create_spatial_function(self, spatial_axis: np.ndarray, dx: float):
        # self.spatial_function = convolved_sinc_gaussian_filter(
        #     spatial_axis=spatial_axis,
        #     dx=dx,
        #     position=self.position,
        #     pzt_layer_width=self.pzt_layer_width,
        #     spreading_factor=self.spreading_factor,
        #     radius=self.radius,
        #     flip_side=flip_side
        # )

        self.spatial_function = arbitrary_source_and_receiver_positioning(
            spatial_axis=spatial_axis,
            dx=dx,
            position=self.position,
            pzt_layer_width=self.pzt_layer_width,
            extension=self.extension,
            radius=self.radius,
        )

def convolved_sinc_gaussian_filter(
    spatial_axis: np.ndarray,
    dx: float,
    position: float,
    pzt_layer_width: float,
    spreading_factor: float,
    radius: int,
    flip_side: str = None,
) -> np.ndarray:
    """
    Create a spatial distribution that is the convolution of a sub-grid Sinc
    and a Gaussian of std = spreading_factor*(pzt_layer_width/dx). We place this kernel
    into the global domain array. If 'flip_side' is 'left' or 'right', we reflect
    (fold) any amplitude that lies beyond the domain boundary (index < 0 or >= N)
    back inside by mirroring around the boundary index (0 or N-1).

    Parameters
    ----------
    spatial_axis : np.ndarray
        Global x-coordinates, e.g. np.arange(0, L, dx) of length N.
    dx : float
        Spatial step size.
    position : float
        Sub-grid source location in the same units as spatial_axis.
    pzt_layer_width : float
        Physical width of the PZT layer (in the same units). Used to set Gaussian sigma.
    spreading_factor : float
        Multiplier for sigma = spreading_factor * (pzt_layer_width / dx).
    radius : int
        Kernel half-width in grid points before/after the center node.
    flip_side : str, optional
        "left" or "right" => reflect out-of-bound amplitudes about index 0 or N-1.
        If None, no boundary folding is done.

    Returns
    -------
    filter_array : np.ndarray
        Length N array with the final distribution.
    """
    N = len(spatial_axis)
    grid_indices = spatial_axis / dx
    position_index = position / dx

    # Nearest integer node
    closest_node_index = int(round(position_index))
    frac_offset = position_index - closest_node_index

    # Build local kernel [-radius ... +radius], length = 2*radius+1
    local_indices = np.arange(2 * radius + 1)
    x_i = (local_indices - radius) + frac_offset  # sub-grid offset

    # Sinc, plus Gaussian
    sinc_array = np.sinc(x_i)
    sigma = spreading_factor * (pzt_layer_width / dx)
    gaussian_array = np.exp(-0.5 * (x_i / sigma) ** 2)

    # Convolve them (discrete)
    convolved_local = convolve(sinc_array, gaussian_array, mode='same')  # still length 2*radius+1

    # Place into a global array ignoring boundary for now
    filter_array = np.zeros(N, dtype=float)

    start_idx = closest_node_index - radius
    end_idx   = closest_node_index + radius + 1  # slice end is exclusive
    # local array covers convolved_local[0 : 2*radius+1]

    # Figure out overlap with the domain [0, N)
    global_start = max(start_idx, 0)
    global_end   = min(end_idx, N)
    if global_end > global_start:
        # The portion inside the domain:
        local_start = global_start - start_idx
        local_end   = local_start + (global_end - global_start)
        filter_array[global_start:global_end] = convolved_local[local_start:local_end]

    # ------------------------------------------------
    # BOUNDARY FOLDING: reflect out-of-bound indices
    # ------------------------------------------------
    if flip_side == "left":

        # We'll do a second pass for the negative portion:
        # negative region = [start_idx, 0), in local coordinates that means local indices < local_start
        neg_end = min(start_idx + 2*radius + 1, 0)  # where local kernel ends
        if neg_end > start_idx:
            # There's a portion that tries to go below 0
            num_neg = neg_end - start_idx  # how many negative indices
            # local portion that is negative is convolved_local[0 : num_neg]
            # we reflect it about i=0 => new indices = +1, +2, ...
            # so the global target is [ - (i), ... ] => i' = -(i+1) or i' = -i?
            # Typically, reflection about 0 means index -1 => +0, -2 => +1, etc.
            # Let's define i < 0 => i' = -(i+1), for example, so -1 => 0, -2 => 1, ...
            # We'll do it carefully below:

            # We'll build an array of out-of-bound indices in the global domain:
            out_of_bounds_i = np.arange(start_idx, neg_end)  # negative region
            local_index_offset = 0

            # For each i in out_of_bounds_i, find i' in the domain by reflection
            # i' = -1-i (that is reflection about -0.5, we might want i' = -(i+1)
            # or reflect about i=0 => i' = -i
            # There's some nuance. Usually for "mirror boundary" you'd do i' = -1 - i or i' = -i - 2, etc.
            # Let's do i' = -(i+1) so that i=-1 => i'=0, i=-2 => i'=1, etc.
            for local_i in out_of_bounds_i:
                # local_i in [start_idx ... neg_end-1]
                # figure out local kernel index => local_i - start_idx
                k = local_i - start_idx
                amplitude = convolved_local[k]
                if amplitude == 0.0:
                    continue

                i_reflected = -(local_i + 1)
                # i_reflected must be in [0, N). If i_reflected >= N, it's also out-of-bounds
                if 0 <= i_reflected < N:
                    filter_array[i_reflected] += amplitude
                # else ignore or do another reflection if you want multiple folds
            # done

    elif flip_side == "right":
        # domain boundary = N-1
        # amplitude that tries to exceed N-1 gets reflected around i=N-1
        # i >= N => i' = 2*(N-1) - i, e.g. i=N => i'=N-2

        # Because we wrote [global_start : global_end], anything >= N wasn't written.
        # We'll do a second pass over the local kernel portion that extends beyond N-1.
        pos_start = max(end_idx, 0)
        pos_end = start_idx + (2*radius + 1)  # total local coverage

        if pos_end > N:
            # There's a portion that extends beyond the domain
            # We'll build the array of out-of-bound global indices: i in [N, pos_end)
            out_of_bounds_i = np.arange(N, pos_end)
            for local_i in out_of_bounds_i:
                k = local_i - start_idx
                if 0 <= k < len(convolved_local):
                    amplitude = convolved_local[k]
                    if amplitude == 0.0:
                        continue

                    # reflect about N-1 => i' = 2*(N-1) - i
                    i_reflected = 2*(N-1) - local_i
                    if 0 <= i_reflected < N:
                        filter_array[i_reflected] += amplitude
                    # else ignore or keep folding, etc.

    return filter_array


def arbitrary_source_and_receiver_positioning(
    spatial_axis        : np.ndarray,
    dx                  : float,
    position            : float,
    pzt_layer_width     : float,
    extension           : float = None,
    beta                : float = 6.31,
    radius              : int = None,
    free_surface_left   : float = None,
    free_surface_right  : float = None    
) -> np.ndarray:
    """
    Implementation in python of the method reported in Hicks 2002:
    Arbitrary source and receiver positioning in finite-difference
    schemes using Kaiser windowed sinc functions
    """
    from numpy import sinc, kaiser
    import matplotlib.pyplot as plt
    if not free_surface_left:
        free_surface_left  = spatial_axis[0]
    if not free_surface_right:
        free_surface_right = spatial_axis[-1]

    free_surface_left_idx  = np.searchsorted(spatial_axis, free_surface_left)
    free_surface_right_idx = np.searchsorted(spatial_axis, free_surface_right)

    if not radius:
        radius = round((pzt_layer_width/2) / dx)
    if not extension:
        extension = pzt_layer_width

    window_len = 2*radius+1
    filter_window = kaiser(window_len,beta)
    filter_window = filter_window - filter_window[0]
    filter = np.zeros(spatial_axis.shape)

    if position >= spatial_axis[0] and position <= spatial_axis[-1]:
        pzt_start = position - extension /2
        pzt_end = position + extension/2
    else:
        raise ValueError("Position ouside the spatial axis simulated")

    pzt_positions = np.arange(pzt_start, pzt_end, dx)   
    finite_spatial_function = np.zeros(spatial_axis.shape)
    for delta_position in pzt_positions:
        raw_delta_approx = sinc(spatial_axis-delta_position)
        delta_position_idx = np.searchsorted(spatial_axis, delta_position)
        start_idx = delta_position_idx - radius
        end_idx   = delta_position_idx - radius + window_len

        if start_idx >= free_surface_left_idx and end_idx <= free_surface_right_idx:
            filter[start_idx:end_idx] = filter_window

        elif start_idx < free_surface_left_idx:

            folding_len = free_surface_left_idx -start_idx
            folding_window = np.zeros(window_len-folding_len)
            folding_window[:folding_len+1] = (
                filter_window[folding_len:2*folding_len+1] 
                - np.flip(filter_window[:folding_len+1])
                )
            folding_window[folding_len:] = filter_window[2*folding_len:]
            filter[free_surface_left_idx:end_idx] = folding_window

            # plt.plot(spatial_axis,filter)
            # plt.show()
        elif end_idx > free_surface_right_idx:

            folding_len = end_idx - free_surface_right_idx
            folding_window = np.zeros(window_len-folding_len)

            try:
                folding_window[:folding_len+1] = (
                    filter_window[folding_len:2*folding_len+1] 
                    - np.flip(filter_window[:folding_len+1])
                    )
            except ValueError:
                continue
            folding_window[folding_len:] = filter_window[2*folding_len:]
            filter[start_idx:free_surface_right_idx] = np.flip(folding_window)
            # plt.plot(spatial_axis,filter)
            # plt.show()
        optimal_delta_approx = raw_delta_approx*filter  
        finite_spatial_function += optimal_delta_approx 

    return finite_spatial_function

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dx = 0.01
    assembly_start = -1
    assembly_end = 11 
    position = -0.98
    pzt_layer_width = 0.5
    spatial_axis = np.arange(assembly_start,assembly_end, dx) 
    if spatial_axis[-1] < assembly_end:
        spatial_axis = np.arange(assembly_start,assembly_end+dx, dx) 

    finite_spatial_function = arbitrary_source_and_receiver_positioning(spatial_axis=spatial_axis,
                                                                        dx=dx,
                                                                        position=position,
                                                                        pzt_layer_width=pzt_layer_width,
                                                                        )
    
    plt.plot(spatial_axis,finite_spatial_function)
    plt.show()

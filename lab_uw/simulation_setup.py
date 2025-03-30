# lab_uw/simulation_setup.py
import numpy as np
from numpy import convolve
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Dict

from lab_uw.plotting import Plotter

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

class VelocityModel1DBase(ABC):
    """
    A base class for building 1D velocity models. 
    Subclasses must implement the standard method sequence:
      - build_velocity_model()
         * compute_layer_positions()
         * define_region_indices()
         * initialize_velocity_model()
         * assign_velocities()
         * apply_smoothing()
      - plot()

    Shared attributes:
    ------------------
    x : np.ndarray
        The spatial grid (1D).
    sample_dimensions : Tuple[float, ...]
        Dimensional parameters controlling the geometry.
    pzt_layer_width : float
        Thickness of the PZT layer(s).
    pla_layer_width : float
        Thickness of the PLA layer(s).
    steel_velocity : float
        Velocity used for steel regions.
    pzt_velocity : float
        Velocity used for PZT regions.
    pla_velocity : float
        Velocity used for PLA regions.
    outfile_path : Optional[Path]
        If plotting is desired, an output file path can be used.

    Internal:
    ---------
    layer_starts : np.ndarray
        Cumulative boundaries for each layer/region.
    idx_dict : Dict[str, np.ndarray]
        Mapping from region name to the array of x-indices in that region.
    values : np.ndarray
        Final velocity array for the entire domain.
    """

    def __init__(
        self,
        x: np.ndarray,
        sample_dimensions: Tuple[float, ...],
        pzt_layer_width: float,
        pla_layer_width: float,
        steel_velocity: float,
        pzt_velocity: float,
        pla_velocity: float,
        outfile_path: Optional[Path] = None
    ):
        self.x = x
        self.sample_dimensions = sample_dimensions

        self.pzt_layer_width = pzt_layer_width
        self.pla_layer_width = pla_layer_width

        self.steel_velocity = steel_velocity
        self.pzt_velocity   = pzt_velocity
        self.pla_velocity   = pla_velocity

        self.outfile_path = outfile_path

        # Data structures populated by the builder methods:
        self.layer_starts: np.ndarray = None
        self.idx_dict: Dict[str, np.ndarray] = {}
        self.values: np.ndarray = None

    @abstractmethod
    def build_velocity_model(self):
        pass

    @abstractmethod
    def compute_layer_positions(self):
        pass

    @abstractmethod
    def define_region_indices(self):
        pass

    @abstractmethod
    def initialize_velocity_model(self):
        pass

    @abstractmethod
    def assign_velocities(self):
        pass

    @abstractmethod
    def apply_smoothing_between(self):
        pass

    def assign_constant_velocity(self, region_name: str, velocity: float):
        """
        Assign a uniform velocity to 'region_name' if it exists in self.idx_dict.
        """
        indices = self.idx_dict.get(region_name, [])
        if indices.size:
            self.values[indices] = velocity


    def plot(self, outfile_path: Optional[str] = None):
        if outfile_path is None:
            outfile_path = self.outfile_path

        Plotter().plot_velocity_model(
            x=self.x,
            c=self.values,
            layer_starts=self.layer_starts,
            pzt_layer_width=self.pzt_layer_width,
            pla_layer_width=self.pla_layer_width,
            outfile_path=outfile_path
        )

class VelocityModel1D_SingleBlock(VelocityModel1DBase):
    """
    Single-block 1D velocity model:
      PLA -> PZT -> STEEL_BLOCK -> PZT -> PLA
    with the 'apply_smoothing(region_from, region_to, n_smooth)' approach.
    """

    def __init__(
        self,
        x: np.ndarray,
        sample_dimensions: Tuple[float],
        x_transmitter: float,
        x_receiver: float,
        pzt_layer_width: float,
        pla_layer_width: float,
        steel_velocity: float,
        pzt_velocity: float,
        pla_velocity: float,
        plotting: bool = True,
        outfile_path: Optional[Path] = None
    ):
        super().__init__(
            x=x,
            sample_dimensions=sample_dimensions,
            pzt_layer_width=pzt_layer_width,
            pla_layer_width=pla_layer_width,
            steel_velocity=steel_velocity,
            pzt_velocity=pzt_velocity,
            pla_velocity=pla_velocity,
            outfile_path=outfile_path
        )
        self.x_transmitter = x_transmitter
        self.x_receiver    = x_receiver
        self.plotting      = plotting

        # Build the model
        self.build_velocity_model()

    def build_velocity_model(self):
        self.compute_layer_positions()
        self.define_region_indices()
        self.initialize_velocity_model()
        self.assign_velocities()

    def compute_layer_positions(self):
        """
        [PLA, PZT, STEEL_BLOCK, PZT, PLA].
        Possibly adjust steel length by x_transmitter/x_receiver if desired.
        """
        if len(self.sample_dimensions) != 1:
            raise ValueError("Expected exactly 1 item in sample_dimensions for single-block model.")

        block_total_length = self.sample_dimensions[0]
        # steel_length = block_total_length - (self.x_transmitter + self.x_receiver)
        steel_length = block_total_length

        self.layer_thicknesses = [
            self.pla_layer_width,
            self.pzt_layer_width,
            steel_length,
            self.pzt_layer_width,
            self.pla_layer_width
        ]
        self.layer_starts = np.concatenate(([0.0], np.cumsum(self.layer_thicknesses)))

    def define_region_indices(self):
        x = self.x
        regions = ["pla_1", "pzt_1", "steel_block", "pzt_2", "pla_2"]
        self.idx_dict.clear()
        for i, region in enumerate(regions):
            start = self.layer_starts[i]
            end   = self.layer_starts[i+1]
            idx = np.where((x >= start) & (x <= end))[0]
            self.idx_dict[region] = idx

    def initialize_velocity_model(self):
        self.values = self.steel_velocity * np.ones_like(self.x)

    def assign_velocities(self):
        self.assign_constant_velocity("pla_1",       self.pla_velocity)
        self.assign_constant_velocity("pzt_1",       self.pzt_velocity)
        self.assign_constant_velocity("steel_block", self.steel_velocity)
        self.assign_constant_velocity("pzt_2",       self.pzt_velocity)
        self.assign_constant_velocity("pla_2",       self.pla_velocity)

        # Patch final index if needed
        if len(self.x) > 1:
            self.values[-1] = self.values[-2]

    def apply_smoothing_between(self, region_from: str, region_to: str, n_smooth: int):
        """
        Direct copy of your original single-block 'apply_smoothing':
        """
        region_velocity_map = {
            "pla_1":       self.pla_velocity,
            "pzt_1":       self.pzt_velocity,
            "steel_block": self.steel_velocity,
            "pzt_2":       self.pzt_velocity,
            "pla_2":       self.pla_velocity
        }

        if region_from not in self.idx_dict or region_to not in self.idx_dict:
            return

        idx_from = np.sort(self.idx_dict[region_from])
        idx_to   = np.sort(self.idx_dict[region_to])
        if idx_from.size == 0 or idx_to.size == 0:
            return

        vel_from = region_velocity_map.get(region_from, None)
        vel_to   = region_velocity_map.get(region_to, None)
        if vel_from is None or vel_to is None:
            return

        # Tail = last n_smooth of region_from
        n_tail = min(n_smooth, idx_from.size)
        tail   = idx_from[-n_tail:]

        # Head = first n_smooth of region_to
        n_head = min(n_smooth, idx_to.size)
        head   = idx_to[:n_head]

        boundary_indices = np.concatenate([tail, head])
        if boundary_indices.size < 2:
            return

        # Linear ramp from vel_from to vel_to
        ramp = np.linspace(vel_from, vel_to, boundary_indices.size)
        self.values[boundary_indices] = ramp

class VelocityModel1D_DDS(VelocityModel1DBase):
    """
    A 1D velocity model for a double-direct-shear (DDS) test sample 
    that includes grooves, gouge layers, side blocks, central block, etc.

    Layer layout (from left to right):
        1.  pla_1
        2.  pzt_1
        3.  side_block_1
        4.  groove_sb1
        5.  gouge_1
        6.  groove_cb1
        7.  central_block
        8.  groove_cb2
        9.  gouge_2
        10. groove_sb2
        11. side_block_2
        12. pzt_2
        13. pla_2

    The lengths of each region come from `sample_dimensions` and the various
    groove/gouge offsets. At the end, we optionally apply smoothing between 
    two specific boundaries, in the same style as the single-block approach.
    """

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
        # Gouge velocity can be a tuple of floats or arrays: (gouge_1_vel, gouge_2_vel)
        gouge_velocity: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
        pzt_velocity: float,
        pla_velocity: float,
        outfile_path: Optional[Path] = None,
        plotting: bool = True
    ):
        # Store main attributes
        self.x = x
        self.sample_dimensions = sample_dimensions
        self.x_transmitter    = x_transmitter
        self.x_receiver       = x_receiver
        self.pzt_layer_width  = pzt_layer_width
        self.pla_layer_width  = pla_layer_width
        self.h_groove_side    = h_groove_side
        self.h_groove_central = h_groove_central

        self.steel_velocity = steel_velocity
        self.gouge_velocity = gouge_velocity  # (gouge_1, gouge_2)
        self.pzt_velocity   = pzt_velocity
        self.pla_velocity   = pla_velocity

        self.outfile_path = outfile_path
        self.plotting     = plotting

        # Internals
        self.layer_thicknesses = []
        self.layer_starts: np.ndarray = None
        self.idx_dict = {}
        self.values: np.ndarray = None

        self.build_velocity_model()

    def build_velocity_model(self):
        """
        Master method that calls all steps in order.
        """
        self.compute_layer_positions()
        self.define_region_indices()
        self.initialize_velocity_model()
        self.assign_velocities()

        # Patch final index if needed
        if len(self.x) > 1:
            self.values[-1] = self.values[-2]

        # Now apply smoothing at boundaries you care about:
        # For example, a boundary from "pzt_1" to "side_block_1"
        # and from "side_block_2" to "pzt_2":
        self.apply_smoothing_between("pzt_1", "side_block_1", 10)
        self.apply_smoothing_between("side_block_2", "pzt_2", 10)

    def compute_layer_positions(self):
        """
        Calculate thicknesses for the 13 sub-layers:
          [pla_1, pzt_1, side_block_1, groove_sb1, gouge_1, 
           groove_cb1, central_block, groove_cb2, gouge_2, 
           groove_sb2, side_block_2, pzt_2, pla_2]
        Then compute their cumulative starts in `self.layer_starts`.
        """
        (side_block_1_total, gouge_1_length,
         central_block_total, gouge_2_length, side_block_2_total) = self.sample_dimensions

        # Subtract the groove heights from each steel block portion
        side_block_1  = side_block_1_total - self.h_groove_side
        side_block_2  = side_block_2_total - self.h_groove_side
        central_block = central_block_total - 2.0 * self.h_groove_central

        # Build the list of layer thicknesses in order
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

        # Now get cumulative starts
        self.layer_starts = np.concatenate(([0.0], np.cumsum(self.layer_thicknesses)))

    def define_region_indices(self):
        """
        Fill self.idx_dict with each region's x-indices.
        The region names must match the order used in compute_layer_positions().
        """
        x = self.x
        region_names = [
            "pla_1", "pzt_1", "side_block_1", "groove_sb1", "gouge_1",
            "groove_cb1", "central_block", "groove_cb2", "gouge_2",
            "groove_sb2", "side_block_2", "pzt_2", "pla_2"
        ]
        self.idx_dict.clear()

        for i, region in enumerate(region_names):
            start = self.layer_starts[i]
            end   = self.layer_starts[i + 1]
            indices = np.where((x >= start) & (x <= end))[0]
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

    def apply_smoothing_between(self, region_from: str, region_to: str, n_smooth: int):
        """
        Smooth the boundary transition from `region_from` to `region_to`.
        This is exactly the single-block style:

         1) region_velocity_map: region -> float velocity
         2) tail indices from region_from, head indices from region_to
         3) linear ramp from velocity_from -> velocity_to
        """

        # 1) We define a map from each region to a single float velocity
        #    for boundary smoothing. If a region uses array velocities,
        #    we can put None or skip it. 
        region_velocity_map = {
            "pla_1":         self.pla_velocity,
            "pzt_1":         self.pzt_velocity,
            "side_block_1":  self.steel_velocity,
            "groove_sb1":    self.steel_velocity,
            "gouge_1":       None,   # if it's array-based, or pick a float if you want
            "groove_cb1":    self.steel_velocity,
            "central_block": self.steel_velocity,
            "groove_cb2":    self.steel_velocity,
            "gouge_2":       None,
            "groove_sb2":    self.steel_velocity,
            "side_block_2":  self.steel_velocity,
            "pzt_2":         self.pzt_velocity,
            "pla_2":         self.pla_velocity
        }

        # 2) Basic checks
        if region_from not in self.idx_dict or region_to not in self.idx_dict:
            return
        idx_from = self.idx_dict[region_from]
        idx_to   = self.idx_dict[region_to]
        if idx_from.size == 0 or idx_to.size == 0:
            return

        vel_from = region_velocity_map.get(region_from, None)
        vel_to   = region_velocity_map.get(region_to, None)
        # If either is None, skip
        if not isinstance(vel_from, (int, float)) or not isinstance(vel_to, (int, float)):
            return

        # 3) Sort the region indices in ascending x-position
        idx_from_sorted = np.sort(idx_from)
        idx_to_sorted   = np.sort(idx_to)

        # 4) Tail = last n_smooth of 'region_from'
        tail_size = min(n_smooth, idx_from_sorted.size)
        tail      = idx_from_sorted[-tail_size:]

        # 5) Head = first n_smooth of 'region_to'
        head_size = min(n_smooth, idx_to_sorted.size)
        head      = idx_to_sorted[:head_size]

        boundary_indices = np.concatenate([tail, head])
        if len(boundary_indices) < 2:
            return

        # 6) Build a linear ramp
        ramp = np.linspace(vel_from, vel_to, len(boundary_indices))

        # 7) Write the ramp into self.values
        self.values[boundary_indices] = ramp

class Source1D:
    def __init__(self, stf_time: np.ndarray, 
                 stf_waveform: np.ndarray,
                 position: float, 
                 pzt_layer_width: float,
                 radius: int, 
                 extension: float = None,

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
    def __init__(self, 
                 position: float, 
                 pzt_layer_width: float,
                 radius: int, 
                 extension: float = None,
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

        self.spatial_function = arbitrary_source_and_receiver_positioning(
            spatial_axis=spatial_axis,
            dx=dx,
            position=self.position,
            pzt_layer_width=self.pzt_layer_width,
            extension=self.extension,
            radius=self.radius,
        )

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

##################################
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

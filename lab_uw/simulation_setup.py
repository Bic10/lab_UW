# lab_uw/simulation_setup.py
import numpy as np
from typing import Union, Tuple, Optional
from lab_uw.plotting import Plotter
from pathlib import Path

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
        plotting: bool = False,
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
        self.apply_smoothing()

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
            side_block_1 - self.x_transmitter,
            self.h_groove_side,
            gouge_1_length,
            self.h_groove_central,
            central_block,
            self.h_groove_central,
            gouge_2_length,
            self.h_groove_side,
            side_block_2 - self.x_receiver,
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

import numpy as np
from scipy.signal.windows import kaiser

class Source1D:
    def __init__(self, stf_time: np.ndarray, stf_waveform: np.ndarray, position: float, radius: int):
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

    def create_spatial_function(self, spatial_axis: np.ndarray, dx: float, flip_side: str = None):
        '''
        Create the spatial function of the source on the grid.

        Args:
            spatial_axis (np.ndarray): The spatial axis of the grid.
            dx (float): Spatial step size.
            flip_side (str): 'left' or 'right' to indicate which side to flip and fold (if any).
        '''
        self.spatial_function = self._arbitrary_position_filter(
            spatial_axis=spatial_axis,
            dx=dx,
            position=self.position,
            radius=self.radius,
            flip_side=flip_side
        )

    @staticmethod
    def _arbitrary_position_filter(
        spatial_axis: np.ndarray,
        dx: float,
        position: float,
        radius: int,
        flip_side: str = None  # Either 'left', 'right', or None
    ) -> np.ndarray:
        """
        Create a Kaiser-windowed sinc filter for arbitrary source/receiver positioning on a 1D grid.
        If flip_side is specified ('left' or 'right'), the values of the windowed sinc function on that side
        of the closest grid node are flipped and added to the values on the opposite side.
        
        Parameters:
        - spatial_axis (np.ndarray): The spatial axis of the grid.
        - dx (float): Spatial step size.
        - position (float): Exact position of the source/receiver.
        - radius (int): Radius of the windowed sinc function (number of grid points).
        - flip_side (str): 'left' or 'right' to indicate which side to flip and fold.
        
        Returns:
        - windowed_sinc (np.ndarray): The windowed sinc filter adjusted for the free surface.
        """
        # Normalize positions to grid indices
        grid_indices = spatial_axis / dx
        num_points = len(grid_indices)
        position_index = position / dx

        # Create sinc function centered at the arbitrary position
        sinc_function = np.sinc(grid_indices - position_index)

        # Apply Kaiser window to the sinc function
        beta = 6.0  # Kaiser window parameter
        kaiser_window = kaiser(2 * radius + 1, beta)

        # Find the grid node closest to the desired position
        closest_node_index = np.argmin(np.abs(grid_indices - position_index))

        # Apply windowed sinc filter centered on the position_index
        start_idx = max(0, closest_node_index - radius)
        end_idx = min(num_points, closest_node_index + radius + 1)
        
        windowed_sinc = np.zeros_like(sinc_function)
        window_indices = np.arange(start_idx, end_idx)
        windowed_sinc[window_indices] = sinc_function[window_indices] * kaiser_window[:end_idx - start_idx]

        # Implement the flip and fold
        if flip_side == 'right':
            # Indices on the left side
            left_indices = np.arange(start_idx, closest_node_index)
            num_left = len(left_indices)
            # Indices on the right side
            right_indices = np.arange(closest_node_index, closest_node_index + num_left)
            # Adjust right_indices to not exceed end_idx
            right_indices = right_indices[right_indices < end_idx]

            # Flip the left values
            flipped_left_values = windowed_sinc[left_indices][::-1]
            flipped_left_values = flipped_left_values[:len(right_indices)]  # Adjust length

            # Add to the right side
            windowed_sinc[right_indices] += flipped_left_values

            # Zero out the left side
            windowed_sinc[left_indices] = 0.0

        elif flip_side == 'left':
            # Indices on the right side
            right_indices = np.arange(closest_node_index + 1, end_idx)
            num_right = len(right_indices)
            # Indices on the left side
            left_indices = np.arange(closest_node_index - num_right, closest_node_index)
            left_indices = left_indices[left_indices >= start_idx]  # Ensure within bounds

            # Flip the right values
            flipped_right_values = windowed_sinc[right_indices][::-1]
            flipped_right_values = flipped_right_values[:len(left_indices)]  # Adjust length

            # Add to the left side
            windowed_sinc[left_indices] += flipped_right_values

            # Zero out the right side
            windowed_sinc[right_indices] = 0.0

        return windowed_sinc

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
    def __init__(self, position: float, radius: int):
        '''
        Initialize the 1D receiver.

        Args:
            position (float): Position of the receiver on the spatial grid.
            radius (int): Radius for the spatial function (number of grid points).
        '''
        self.position = position
        self.radius = radius
        self.spatial_function = None  # Will be set after being created on the grid

    def create_spatial_function(self, spatial_axis: np.ndarray, dx: float, flip_side: str = None):
        '''
        Create the spatial function of the receiver on the grid.

        Args:
            spatial_axis (np.ndarray): The spatial axis of the grid.
            dx (float): Spatial step size.
            flip_side (str): 'left' or 'right' to indicate which side to flip and fold (if any).
        '''
        self.spatial_function = self._arbitrary_position_filter(
            spatial_axis=spatial_axis,
            dx=dx,
            position=self.position,
            radius=self.radius,
            flip_side=flip_side
        )

    @staticmethod
    def _arbitrary_position_filter(
        spatial_axis: np.ndarray,
        dx: float,
        position: float,
        radius: int,
        flip_side: str = None  # Either 'left', 'right', or None
    ) -> np.ndarray:
        """
        Create a Kaiser-windowed sinc filter for arbitrary source/receiver positioning on a 1D grid.
        If flip_side is specified ('left' or 'right'), the values of the windowed sinc function on that side
        of the closest grid node are flipped and added to the values on the opposite side.
        
        Parameters:
        - spatial_axis (np.ndarray): The spatial axis of the grid.
        - dx (float): Spatial step size.
        - position (float): Exact position of the source/receiver.
        - radius (int): Radius of the windowed sinc function (number of grid points).
        - flip_side (str): 'left' or 'right' to indicate which side to flip and fold.
        
        Returns:
        - windowed_sinc (np.ndarray): The windowed sinc filter adjusted for the free surface.
        """
        # Normalize positions to grid indices
        grid_indices = spatial_axis / dx
        num_points = len(grid_indices)
        position_index = position / dx

        # Create sinc function centered at the arbitrary position
        sinc_function = np.sinc(grid_indices - position_index)

        # Apply Kaiser window to the sinc function
        beta = 6.0  # Kaiser window parameter
        kaiser_window = kaiser(2 * radius + 1, beta)

        # Find the grid node closest to the desired position
        closest_node_index = np.argmin(np.abs(grid_indices - position_index))

        # Apply windowed sinc filter centered on the position_index
        start_idx = max(0, closest_node_index - radius)
        end_idx = min(num_points, closest_node_index + radius + 1)
        
        windowed_sinc = np.zeros_like(sinc_function)
        window_indices = np.arange(start_idx, end_idx)
        windowed_sinc[window_indices] = sinc_function[window_indices] * kaiser_window[:end_idx - start_idx]

        # Implement the flip and fold
        if flip_side == 'right':
            # Indices on the left side
            left_indices = np.arange(start_idx, closest_node_index)
            num_left = len(left_indices)
            # Indices on the right side
            right_indices = np.arange(closest_node_index, closest_node_index + num_left)
            # Adjust right_indices to not exceed end_idx
            right_indices = right_indices[right_indices < end_idx]

            # Flip the left values
            flipped_left_values = windowed_sinc[left_indices][::-1]
            flipped_left_values = flipped_left_values[:len(right_indices)]  # Adjust length

            # Add to the right side
            windowed_sinc[right_indices] += flipped_left_values

            # Zero out the left side
            windowed_sinc[left_indices] = 0.0

        elif flip_side == 'left':
            # Indices on the right side
            right_indices = np.arange(closest_node_index + 1, end_idx)
            num_right = len(right_indices)
            # Indices on the left side
            left_indices = np.arange(closest_node_index - num_right, closest_node_index)
            left_indices = left_indices[left_indices >= start_idx]  # Ensure within bounds

            # Flip the right values
            flipped_right_values = windowed_sinc[right_indices][::-1]
            flipped_right_values = flipped_right_values[:len(left_indices)]  # Adjust length

            # Add to the left side
            windowed_sinc[left_indices] += flipped_right_values

            # Zero out the right side
            windowed_sinc[right_indices] = 0.0

        return windowed_sinc





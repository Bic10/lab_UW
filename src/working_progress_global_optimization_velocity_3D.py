# global_optimization_velocity.py
from daos.sensor_dao import SensorDao
from daos.gouge_dao import GougeDao
from daos.block_dao import BlockDao
from daos.machine_dao import MachineDao
from daos.experiment_dao import ExperimentDao

# Initialize DAOs
sensor_dao = SensorDao()
gouge_dao = GougeDao()
block_dao = BlockDao()
machine_dao = MachineDao()
experiment_dao = ExperimentDao()

# Fetch experiment details
experiment_id = "your_experiment_id"  # Replace with the actual ID
experiment = experiment_dao.find_experiment_by_id(experiment_id)

# Fetch blocks and gouges
blocks = experiment_dao.find_blocks(experiment_id)
gouges = experiment.get('gouges', [])

# Fetch sensor data
sensor_data = []
for block in blocks:
    for sensor in block.get('sensors', []):
        sensor_detail = sensor_dao.find_sensor_by_id(sensor["_id"])
        sensor_data.append(sensor_detail)

# Fetch material properties for gouges
gouge_properties = []
for gouge in gouges:
    gouge_detail = gouge_dao.find_gouge_by_id(gouge["_id"])
    gouge_properties.append(gouge_detail)

# Use the retrieved data in the simulation
side_block_1 = blocks[0]['dimensions']['width']  # Example usage
side_block_2 = blocks[1]['dimensions']['width']
central_block = experiment.get('central_block', None)  # Example usage if it's directly stored

# Fetch pulse waveform and velocity models
pulse_id = "your_pulse_id"  # Replace with the actual pulse ID
time_axis, source_pulse = sensor_dao.find_sensor_by_id(pulse_id).get('waveform')

# Assuming velocity model is stored as part of the experiment or blocks
velocity_model_id = "your_velocity_model_id"  # Replace with the actual model ID
velocity_model = block_dao.find_block_by_id(velocity_model_id).get('velocities')

# Now run the simulation using these inputs
# (Rest of the simulation code using fetched data)

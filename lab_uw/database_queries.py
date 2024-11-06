# src/database_queries.py
from daos.experiment_dao import ExperimentDao
import matplotlib.pyplot as plt

class DatabaseQueries:
    def __init__(self):
        self.experiment_dao = ExperimentDao()

    def get_ultrasonic_waveforms(self, experiment_id, uw_sequence_name, start_uw, end_uw):
        uw_dict = self.experiment_dao.find_additional_measurements(
            experiment_id=experiment_id,
            measurement_type="ultrasonic_waveforms",
            measurement_sequence_id=uw_sequence_name,
            start_uw=start_uw,
            end_uw=end_uw
        )
        return uw_dict

    def get_centralized_measurements(self, experiment_id, group_name, channel_name):
        measurements_dict = self.experiment_dao.find_centralized_measurements(
            experiment_id=experiment_id,
            group_name=group_name,
            channel_name=channel_name
        )
        return measurements_dict

    def plot_ultrasonic_waveforms(self, uw_dict):
        data = uw_dict["data"]
        for waveform in data:
            plt.plot(waveform)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f'Ultrasonic Waveforms')
        plt.show()

    def plot_centralized_measurements(self, x_values, y_values, x_field, y_field):
        plt.plot(x_values, y_values)
        plt.xlabel(x_field)
        plt.ylabel(y_field)
        plt.title(f'Plot of {x_field} vs {y_field}')
        plt.show()

# Example usage
if __name__ == "__main__":
    db_queries = DatabaseQueries()
    
    # Example to get ultrasonic waveforms and plot
    experiment_id = "s0108sw06car102030"
    uw_sequence_name = "001_run_in_10MPa"
    uw_data = db_queries.get_ultrasonic_waveforms(experiment_id, uw_sequence_name, start_uw=0, end_uw=500)
    db_queries.plot_ultrasonic_waveforms(uw_data)
    
    # Example to get centralized measurements and plot
    x_field = "Time"
    y_field = "Vertical Load"
    x_data = db_queries.get_centralized_measurements(experiment_id, "ADC", x_field)
    y_data = db_queries.get_centralized_measurements(experiment_id, "ADC", y_field)
    db_queries.plot_centralized_measurements(x_data["data"], y_data["data"], x_field, y_field)

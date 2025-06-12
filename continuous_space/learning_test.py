import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

from learnable_movement_cell import LearnableMovementCell

def run_simulation(num_test, place_cell_ampa, instructive_signal_ampa, num_place_cell_connections, num_instructive_signal_connections, num_left, num_right, steps, dt, place_cell_spikes_over_time, instructive_signal_spikes_over_time):

    # === Run simulation ===
    cell = LearnableMovementCell(num_test=num_test, 
                                 place_cell_ampa=place_cell_ampa,
                                 instructive_signal_ampa=instructive_signal_ampa,
                                 place_cell_connections=num_place_cell_connections,
                                 total_num_place_cells=sum(num_place_cell_connections),
                                 instructive_signal_connections=num_instructive_signal_connections,
                                 num_left=num_left, 
                                 num_right=num_right, 
                                 dt=dt)
    cell.run_with_custom_spikes(place_cell_spikes_over_time, instructive_signal_spikes_over_time)

    # === Retrieve and plot voltage traces for all compartments ===
    V = cell.get_voltage_trace()  # shape: [time, total compartments]
    time_axis = np.arange(V.shape[0]) * float(dt/ms)

    # === Generate labels (no clamps anymore) ===
    labels = []
    total_N = V.shape[1]
    for i in range(total_N):
        if i < num_left:
            labels.append(f"Left Buffer {i}")
        elif i < num_left + num_test:
            labels.append(f"Test Compartment {i - num_left}")
        else:
            labels.append(f"Right Buffer {i - (num_left + num_test)}")

    # === Plotting ===
    plt.figure(figsize=(12, 7))
    for i in range(V.shape[1]):
        plt.plot(time_axis, V[:, i]/mV, label=labels[i])

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage Traces Across MovementCell with Buffers Only')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def two_compartments():
    # === Simulation parameters ===
    num_test = 2
    num_left = 3
    num_right = 3
    steps = 10000
    dt = 0.001 * ms

    # === Define connections ===
    place_cell_ampa = 2.0 * nsiemens
    instructive_signal_ampa = 8.0 * nsiemens

    num_place_cell_connections = [10, 20] # per compartment
    num_instructive_signal_connections = [5, 10] # per compartment

    place_total = sum(num_place_cell_connections)
    instructive_total = sum(num_instructive_signal_connections)

    # === Define spike pattern (only for test compartments) === 
    place_spike_dict = {
        5000: [(15,)],
        5140: [(2,)]
    }
    instructive_spike_dict = {
        5000: [(12,)], 
        5140: [(2,)]
    }

    place_cell_spikes_over_time = []
    instructive_signal_spikes_over_time = []
    for t in range(steps):
        place_spikes = np.zeros(place_total, dtype=int)
        instructive_spikes = np.zeros(instructive_total, dtype=int)

        if t in place_spike_dict:
            for i in place_spike_dict[t]:
                place_spikes[i] = 1
        if t in instructive_spike_dict:
            for i in instructive_spike_dict[t]:
                instructive_spikes[i] = 1

        place_cell_spikes_over_time.append(place_spikes)
        instructive_signal_spikes_over_time.append(instructive_spikes)

    # === Run the simulation and plot results ===
    run_simulation(num_test, place_cell_ampa, instructive_signal_ampa, num_place_cell_connections, num_instructive_signal_connections, num_left, num_right, steps, dt, place_cell_spikes_over_time, instructive_signal_spikes_over_time)

def main():
    two_compartments()

if __name__ == '__main__':
    main()
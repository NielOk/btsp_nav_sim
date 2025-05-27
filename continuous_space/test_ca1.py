import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

from ca_1 import Ca1

def base_simulation():
    # === Simulation parameters ===
    num_compartments = 5
    steps = 10000
    dt = 0.01 * ms

    # === Define custom spike pattern ===
    # spikes_over_time[timestep] = spike array for that step
    # Example: spike in compartment 0 at t=10, and in compartment 2 at t=40
    spikes_over_time = []
    for t in range(steps):
        spikes = np.zeros(num_compartments, dtype=int)
        if t == 1000:
            spikes[2] = 1
        if t == 2000:
            spikes[3] = 1
        if t == 3000:
            spikes[4] = 1
        spikes_over_time.append(spikes)

    # === Run simulation ===
    ca1 = Ca1(num_place_cells=num_compartments, dt=dt)
    ca1.run_with_custom_spikes(spikes_over_time)

    # === Plot results ===
    V = ca1.get_voltage_trace()  # shape [time, compartments]
    time_axis = np.arange(V.shape[0]) * float(dt/ms)

    plt.figure(figsize=(10, 6))
    for i in range(num_compartments):
        plt.plot(time_axis, V[:, i]/mV, label=f'Compartment {i}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage Traces for CA1 Dendritic Compartments')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    base_simulation()

if __name__ == '__main__':
    main()
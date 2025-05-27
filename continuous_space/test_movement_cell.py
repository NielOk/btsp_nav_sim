import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

from movement_cell import MovementCell

def base_simulation():
    # === Simulation parameters ===
    num_internal = 5  # internal compartments
    steps = 10000
    dt = 0.01 * ms

    # === Define spike pattern ===
    spikes_over_time = []
    for t in range(steps):
        spikes = np.zeros(num_internal, dtype=int)
        if t == 1000:
            spikes[0] = 1
        if t == 1500:
            spikes[1] = 1
        if t == 2000:
            spikes[2] = 1
        if t == 2500:
            spikes[3] = 1
        if t == 3000:
            spikes[4] = 1
        spikes_over_time.append(spikes)

    # === Run simulation ===
    cell = MovementCell(num_compartments=num_internal, dt=dt)
    cell.run_with_custom_spikes(spikes_over_time)

    # === Retrieve and plot voltage traces for all compartments ===
    V = cell.get_voltage_trace()  # shape: [time, total_compartments = internal + 2]
    time_axis = np.arange(V.shape[0]) * float(dt/ms)

    plt.figure(figsize=(10, 6))
    for i in range(V.shape[1]):  # all compartments, including clamped ones
        if i == 0:
            label = "Left Clamp (0 mV)"
        elif i == V.shape[1] - 1:
            label = "Right Clamp (-90 mV)"
        else:
            label = f"Compartment {i - 1}"
        plt.plot(time_axis, V[:, i]/mV, label=label)

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage Traces for All Compartments in MovementCell')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    base_simulation()

if __name__ == '__main__':
    main()
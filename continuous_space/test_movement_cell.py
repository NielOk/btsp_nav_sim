import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

from movement_cell import MovementCell

def base_simulation():
    # === Simulation parameters ===
    num_test = 5
    num_left = 3
    num_right = 3
    steps = 25000
    dt = 0.005 * ms

    # === Define spike pattern (only for test compartments) ===
    spikes_over_time = []
    for t in range(steps):
        spikes = np.zeros(num_test, dtype=int)
        if t == 3000:
            spikes[0] = 1
        if t == 6000:
            spikes[1] = 1
        if t == 9000:
            spikes[2] = 1
        if t == 12000:
            spikes[3] = 1
        if t == 15000:
            spikes[4] = 1
        spikes_over_time.append(spikes)

    # === Run simulation ===
    cell = MovementCell(num_test=num_test, num_left=num_left, num_right=num_right, dt=dt)
    cell.run_with_custom_spikes(spikes_over_time)

    # === Retrieve and plot voltage traces for all compartments ===
    V = cell.get_voltage_trace()  # shape: [time, total compartments]
    time_axis = np.arange(V.shape[0]) * float(dt/ms)

    # === Generate labels ===
    labels = []
    total_N = V.shape[1]
    for i in range(total_N):
        if i == 0:
            labels.append("Clamp Left (0 mV)")
        elif i == total_N - 1:
            labels.append("Clamp Right (-90 mV)")
        elif 1 <= i <= num_left:
            labels.append(f"Left Buffer {i}")
        elif num_left + 1 <= i <= num_left + num_test:
            labels.append(f"Test Compartment {i - (num_left + 1)}")
        else:
            labels.append(f"Right Buffer {i - (num_left + num_test + 1)}")

    # === Plotting ===
    plt.figure(figsize=(12, 7))
    for i in range(V.shape[1]):
        plt.plot(time_axis, V[:, i]/mV, label=labels[i])

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage Traces Across MovementCell with Buffers and Clamps')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    base_simulation()

if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt

from learnable_movement_cell import LearnableMovementCell

import numpy as np

def generate_mouse_path(num_steps_per_cycle, num_cycles):
    """
    Generate an infinity-sign (lemniscate) shaped path for the mouse to follow.
    """
    total_steps = num_steps_per_cycle * num_cycles
    ts = np.linspace(0, 2 * np.pi * num_cycles, total_steps)

    x_vals = np.cos(ts)
    y_vals = np.sin(ts) * np.cos(ts)

    return x_vals, y_vals

def main():
    num_steps_per_cycle = 100
    num_cycles = 1
    x_vals, y_vals = generate_mouse_path(num_steps_per_cycle, num_cycles)
    plt.plot(x_vals, y_vals, label='Mouse Path')
    plt.show()

if __name__ == '__main__':
    main()
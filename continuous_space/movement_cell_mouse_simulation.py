import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

from learnable_movement_cell import LearnableMovementCell
from place_cell import PlaceCell

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

def create_place_cells(num_place_cells, bounds=(-5, 5, -5, 5)):
    """
    Create a list of PlaceCell instances within the specified bounds.
    """
    place_cells = []
    for _ in range(num_place_cells):
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[2], bounds[3])
        std = np.random.uniform(0.3, 0.7)
        place_cells.append(PlaceCell(x, y, std))
    return place_cells

def create_movement_cells(num_movement_cells, 
                          place_cell_ampa, 
                          instructive_signal_ampa, 
                          dt,
                          num_place_cell_connections=[10, 15], 
                          num_instructive_signal_connections=[5, 10], 
                          num_test = 2, 
                          num_left = 3,
                          num_right = 3):
    """
    Make a list of learnable movement cell instances.
    """
    movement_cells = []
    for _ in range(num_movement_cells):
        cell = LearnableMovementCell(num_test=num_test,
                                     place_cell_ampa=place_cell_ampa,
                                     instructive_signal_ampa=instructive_signal_ampa,
                                     place_cell_connections=num_place_cell_connections,
                                     instructive_signal_connections=num_instructive_signal_connections,
                                     num_left=num_left, 
                                     num_right=num_right,
                                     dt=dt
        )
        movement_cells.append(cell)

    return movement_cells

def main():
    num_steps_per_cycle = 100
    num_cycles = 3
    num_place_cells = 50
    bounds = (-5, 5, -5, 5)

    x_vals, y_vals = generate_mouse_path(num_steps_per_cycle, num_cycles)
    place_cells = create_place_cells(num_place_cells, bounds)

if __name__ == '__main__':
    main()
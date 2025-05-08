import numpy as np

class Ca1():

    def __init__(self, num_place_cells, num_grid_cells):
        """
        num_place_cells: number of place cells
        num_grid_cells: number of grid cells
        """
        self.place_cell_weights = np.random.randn(num_place_cells)
        self.grid_cell_weights = np.random.randn(num_grid_cells)

    def compute_ca1_activity(self, place_cell_activations, grid_cell_activations):
        """
        Compute CA1 activity based on place cell and grid cell activations.
        """
        ca1_activity = np.dot(place_cell_activations, self.place_cell_weights)
        ca1_activity += np.dot(grid_cell_activations, self.grid_cell_weights)
        return ca1_activity

    def update_weights(self, place_cell_activations, grid_cell_activations):
        pass
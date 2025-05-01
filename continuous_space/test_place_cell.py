import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from place_cell import PlaceCell

def main():
    # Parameters
    num_cells = 5
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)
    std_bounds = (0.5, 2.0)
    grid_res = 100

    # Create some place cells with random centers and random std devs
    np.random.seed(42)
    place_cells = []
    for _ in range(num_cells):
        cx = np.random.uniform(*x_bounds)
        cy = np.random.uniform(*y_bounds)
        std = np.random.uniform(*std_bounds)
        place_cells.append(PlaceCell(cx, cy, std))

    # Create a grid of locations
    x_vals = np.linspace(*x_bounds, grid_res)
    y_vals = np.linspace(*y_bounds, grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Plot firing probabilities for each place cell
    fig, axs = plt.subplots(1, num_cells, figsize=(4 * num_cells, 4))
    if num_cells == 1:
        axs = [axs]

    for i, cell in enumerate(place_cells):
        Z = cell.gaussian_firing(X, Y)
        ax = axs[i]
        c = ax.contourf(X, Y, Z, levels=50, cmap=cm.viridis)
        ax.set_title(f'Cell {i+1}\nCenter=({cell.center_x:.2f},{cell.center_y:.2f})\nσ={cell.standard_deviation:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        fig.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
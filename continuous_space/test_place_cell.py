import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from place_cell import PlaceCell

def simulate_firing(place_cell, x_vals, y_vals, trials_per_point):
    """
    Simulates binary firing using fire_at at each (x, y) in the grid.
    Returns empirical firing probability heatmap.
    """
    grid_res = len(x_vals)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(grid_res):
        for j in range(grid_res):
            x = X[i, j]
            y = Y[i, j]
            fires = [place_cell.fire_at(x, y) for _ in range(trials_per_point)]
            Z[i, j] = np.mean(fires)

    return X, Y, Z

def plot_distributions(num_cells, x_bounds, y_bounds, std_bounds, grid_res, trials_per_point):

    # Create some place cells with random centers and std devs
    np.random.seed(42)
    place_cells = []
    for _ in range(num_cells):
        cx = np.random.uniform(*x_bounds)
        cy = np.random.uniform(*y_bounds)
        std = np.random.uniform(*std_bounds)
        place_cells.append(PlaceCell(cx, cy, std))

    # Create grid
    x_vals = np.linspace(*x_bounds, grid_res)
    y_vals = np.linspace(*y_bounds, grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Plot layout: 2 rows (theoretical + empirical), num_cells columns
    fig, axs = plt.subplots(2, num_cells, figsize=(4 * num_cells, 8))
    if num_cells == 1:
        axs = np.array([[axs[0]], [axs[1]]])  # force 2D structure

    for i, cell in enumerate(place_cells):
        # Theoretical distribution
        Z_theory = cell.gaussian_firing(X, Y)
        ax1 = axs[0, i]
        c1 = ax1.contourf(X, Y, Z_theory, levels=50, cmap=cm.viridis)
        ax1.set_title(f'Theoretical\nCenter=({cell.center_x:.2f},{cell.center_y:.2f})\nσ={cell.standard_deviation:.2f}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        fig.colorbar(c1, ax=ax1)

        # Empirical distribution from fire_at()
        _, _, Z_empirical = simulate_firing(cell, x_vals, y_vals, trials_per_point)
        ax2 = axs[1, i]
        c2 = ax2.contourf(X, Y, Z_empirical, levels=50, cmap=cm.inferno)
        ax2.set_title('Empirical (sampled)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        fig.colorbar(c2, ax=ax2)

    plt.tight_layout()
    plt.show()

    return place_cells

if __name__ == '__main__':
    # Parameters
    num_cells = 5
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)
    std_bounds = (0.5, 2.0)
    grid_res = 50  # adjust for speed vs resolution
    trials_per_point = 50  # adjust for accuracy vs speed

    place_cells = plot_distributions(num_cells, x_bounds, y_bounds, std_bounds, grid_res, trials_per_point)
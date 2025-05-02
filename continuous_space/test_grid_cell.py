import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from grid_cell import GridCell  # assumes you saved your class in grid_cell.py


def simulate_firing(grid_cell, x_vals, y_vals, trials_per_point):
    grid_res = len(x_vals)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(grid_res):
        for j in range(grid_res):
            x = X[i, j]
            y = Y[i, j]
            fires = [grid_cell.fire_at(x, y) for _ in range(trials_per_point)]
            Z[i, j] = np.mean(fires)

    return X, Y, Z


def evaluate_activation(grid_cell, x_vals, y_vals):
    grid_res = len(x_vals)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(grid_res):
        for j in range(grid_res):
            Z[i, j] = grid_cell.activation(X[i, j], Y[i, j])

    return X, Y, Z


def plot_distributions(num_cells, x_bounds, y_bounds, spacing_bounds, std_bounds, grid_res, trials_per_point):
    np.random.seed(42)
    grid_cells = []
    for _ in range(num_cells):
        spacing = np.random.uniform(*spacing_bounds)
        std = np.random.uniform(*std_bounds)
        phase_x = np.random.uniform(0, spacing)
        phase_y = np.random.uniform(0, spacing * np.sin(np.pi / 3))
        grid_cells.append(GridCell(spacing, std, phase_x, phase_y, bounds=(*x_bounds, *y_bounds)))

    x_vals = np.linspace(*x_bounds, grid_res)
    y_vals = np.linspace(*y_bounds, grid_res)

    fig, axs = plt.subplots(2, num_cells, figsize=(4 * num_cells, 8))
    if num_cells == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    for i, cell in enumerate(grid_cells):
        X, Y, Z_theory = evaluate_activation(cell, x_vals, y_vals)
        ax1 = axs[0, i]
        c1 = ax1.contourf(X, Y, Z_theory, levels=50, cmap=cm.viridis)
        ax1.set_title(f'Theoretical\nspacing={cell.spacing:.2f}, σ={cell.std:.2f}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        fig.colorbar(c1, ax=ax1)

        _, _, Z_empirical = simulate_firing(cell, x_vals, y_vals, trials_per_point)
        ax2 = axs[1, i]
        c2 = ax2.contourf(X, Y, Z_empirical, levels=50, cmap=cm.inferno)
        ax2.set_title('Empirical (sampled)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        fig.colorbar(c2, ax=ax2)

    plt.tight_layout()
    plt.show()
    return grid_cells


# Add fire_at() method directly to your GridCell if it's not already there:
def fire_at(self, x, y):
    prob = min(self.activation(x, y), 1.0)
    return np.random.rand() < prob

# Dynamically patch GridCell if needed
if not hasattr(GridCell, "fire_at"):
    GridCell.fire_at = fire_at


if __name__ == '__main__':
    num_cells = 5
    x_bounds = (-5, 5)
    y_bounds = (-5, 5)
    spacing_bounds = (1.5, 2.5)
    std_bounds = (0.2, 0.6)
    grid_res = 50
    trials_per_point = 50

    plot_distributions(num_cells, x_bounds, y_bounds, spacing_bounds, std_bounds, grid_res, trials_per_point)
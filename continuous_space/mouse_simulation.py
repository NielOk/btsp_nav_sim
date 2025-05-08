import numpy as np
import matplotlib.pyplot as plt

from grid_cell import GridCell
from place_cell import PlaceCell
from ca_1 import Ca1

def generate_circle_path(radius, num_steps_per_lap, laps=1):
    total_steps = num_steps_per_lap * laps
    thetas = np.linspace(0, 2 * np.pi * laps, total_steps)
    x_vals = radius * np.cos(thetas)
    y_vals = radius * np.sin(thetas)
    return x_vals, y_vals

def simulate_mouse_and_cells(num_place_cells=3, num_grid_cells=2, num_steps_per_lap=100, radius=3.0, laps=1):
    bounds = (-5, 5, -5, 5)
    total_steps = num_steps_per_lap * laps

    # Create place cells
    place_cells = []
    for _ in range(num_place_cells):
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[2], bounds[3])
        std = np.random.uniform(0.3, 0.7)
        place_cells.append(PlaceCell(x, y, std))

    # Create grid cells
    grid_cells = []
    for _ in range(num_grid_cells):
        spacing = np.random.uniform(1.5, 2.5)
        std = np.random.uniform(0.2, 0.6)
        phase_x = np.random.uniform(0, spacing)
        phase_y = np.random.uniform(0, spacing * np.sin(np.pi / 3))
        center = (phase_x, phase_y)
        grid_cells.append(GridCell(spacing, std, bounds=bounds, center=center))

    # Patch in fire_at if not present
    if not hasattr(GridCell, "fire_at"):
        def fire_at(self, x, y):
            prob = min(self.activation(x, y), 1.0)
            return np.random.rand() < prob
        GridCell.fire_at = fire_at

    # Generate mouse path
    x_path, y_path = generate_circle_path(radius, num_steps_per_lap, laps=laps)

    # Log firings
    log = []
    for t in range(total_steps):
        x = x_path[t]
        y = y_path[t]
        step_firings = {"t": t, "x": x, "y": y, "place": [], "grid": []}
        for i, pc in enumerate(place_cells):
            if pc.fire_at(x, y):
                step_firings["place"].append(1)
            else:
                step_firings["place"].append(0)
        for i, gc in enumerate(grid_cells):
            if gc.fire_at(x, y):
                step_firings["grid"].append(1)
            else:
                step_firings["grid"].append(0)
        log.append(step_firings)

    return log, place_cells, grid_cells, x_path, y_path, bounds

def plot_cells_and_path(place_cells, grid_cells, x_path, y_path, bounds, resolution=300):
    x_vals = np.linspace(bounds[0], bounds[1], resolution)
    y_vals = np.linspace(bounds[2], bounds[3], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Place cell activation map
    place_Z = np.zeros_like(X)
    for pc in place_cells:
        place_Z += np.exp(-((X - pc.center_x) ** 2 + (Y - pc.center_y) ** 2) / (2 * pc.standard_deviation ** 2))

    # Grid cell activation map
    grid_Z = np.zeros_like(X)
    for gc in grid_cells:
        for (cx, cy) in gc.centers:
            grid_Z += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * gc.std ** 2))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axs:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')

    axs[0].contourf(X, Y, place_Z, levels=50, cmap='Blues')
    axs[0].set_title("Place Cell Firing Probability")
    axs[0].plot(x_path, y_path, 'r--', label='Mouse Path')
    axs[0].legend()

    axs[1].contourf(X, Y, grid_Z, levels=50, cmap='Greens')
    axs[1].set_title("Grid Cell Firing Probability")
    axs[1].plot(x_path, y_path, 'r--', label='Mouse Path')
    axs[1].legend()

    combined_Z = place_Z + grid_Z
    axs[2].contourf(X, Y, combined_Z, levels=50, cmap='viridis')
    axs[2].set_title("Combined Firing Probability")
    axs[2].plot(x_path, y_path, 'r--', label='Mouse Path')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def plot_ca1_activity(ca1_activity, x_path, y_path, bounds):
    # --- Plot activity over space ---
    plt.figure(figsize=(6, 6))
    plt.scatter(x_path, y_path, c=ca1_activity, cmap='hot', s=20)
    plt.colorbar(label='CA1 firing rate')
    plt.title('CA1 neuron activity across space')
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.show()

def main():
    num_place_cells = 20
    num_grid_cells = 1
    num_steps_per_lap = 100
    laps = 5

    log, place_cells, grid_cells, x_path, y_path, bounds = simulate_mouse_and_cells(
        num_place_cells=num_place_cells,
        num_grid_cells=num_grid_cells,
        num_steps_per_lap=num_steps_per_lap,
        laps=laps
    )

    ca_1_cell = Ca1(num_place_cells, num_grid_cells)
    ca_1_activity_log = []

    for entry in log:
        
        t = entry['t']
        x = entry['x']
        y = entry['y']
        place_cells_fire_pattern = np.array(entry['place'])
        grid_cells_fire_pattern = np.array(entry['grid'])
        
        ca_1_activity = ca_1_cell.compute_ca1_activity(place_cells_fire_pattern, grid_cells_fire_pattern)

        ca_1_activity_log.append(ca_1_activity)

    plot_cells_and_path(place_cells, grid_cells, x_path, y_path, bounds)
    plot_ca1_activity(ca_1_activity_log, x_path, y_path, bounds)
    print(bounds)

# Run it all
if __name__ == "__main__":
    main()
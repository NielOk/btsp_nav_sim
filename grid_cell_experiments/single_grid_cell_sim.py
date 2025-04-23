import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Configurations
GRID_SIZE = 10
N_ROUNDS = 2
N_GRID_CELLS = 1
VISUALIZE = True

def setup_simulation(grid_size, n_grid_cells):
    # Generate grid cells with firing locations
    grid_cells = []
    for i in range(n_grid_cells):
        spacing = 2
        offset = [1, 1]
        locations = {(x, y)
                    for x in range(offset[0], grid_size, spacing)
                    for y in range(offset[1], grid_size, spacing)}
        grid_cells.append(locations)

    # Store history
    firings = []

    return grid_cells, firings

def generate_circular_trajectory(grid_size, n_rounds):
    """Generates a circular trajectory (square perimeter walk)."""
    trajectory = []

    total_steps = 0
    for i in range(n_rounds):
        for x in range(1, grid_size - 1):  # Right
            trajectory.append((x, 1))
            total_steps += 1
        for y in range(1, grid_size):  # Up
            trajectory.append((grid_size - 1, y))
            total_steps += 1
        for x in reversed(range(1, grid_size - 1)):  # Left
            trajectory.append((x, grid_size - 1))
            total_steps += 1
        for y in reversed(range(1, grid_size - 1)):  # Down
            trajectory.append((1, y))
            total_steps += 1

    return trajectory, total_steps

# Use the trajectory and grid cells to figure out which steps for which the grid cells fire. binary 1 for fire, 0 for no fire
def get_firing_steps(trajectory, grid_cells):

    firing_steps = []
    for x, y in trajectory:
        for i, locations in enumerate(grid_cells):
            if (x, y) in locations:
                firing_steps.append(1)
            else:
                firing_steps.append(0)

    return firing_steps

def animate_firing(trajectory, firing_steps):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks(np.arange(0, GRID_SIZE, 1))
        ax.set_yticks(np.arange(0, GRID_SIZE, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title(f"Step {frame + 1} | Firing: {firing_steps[frame]}")

        # Draw all grid cell firing fields (for visual context)
        for loc in grid_cells[0]:  # Only one grid cell in your setup
            ax.add_patch(plt.Circle(loc, 0.2, color='blue', alpha=0.2))

        # Draw mouse
        x, y = trajectory[frame]
        if firing_steps[frame]:
            ax.plot(x, y, 'ro', markersize=12, label="Mouse (Firing)")
        else:
            ax.plot(x, y, 'ko', markersize=10, label="Mouse")

        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=300, repeat=False)
    plt.show()

if __name__ == '__main__':
    grid_cells, firings = setup_simulation(GRID_SIZE, N_GRID_CELLS)
    trajectory, total_steps = generate_circular_trajectory(GRID_SIZE, N_ROUNDS)
    firing_steps = get_firing_steps(trajectory, grid_cells)

    firing_steps = get_firing_steps(trajectory, grid_cells)

    if VISUALIZE:
        animate_firing(trajectory, firing_steps)

    print(firing_steps)
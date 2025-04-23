import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Configurations
GRID_SIZE = 10
N_ROUNDS = 2
N_PLACE_CELLS = 20
VISUALIZE = True

def setup_simulation(grid_size, n_place_cells):
    """Generate N place cells, each with a unique firing location."""
    possible_locations = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    random.shuffle(possible_locations)
    selected_locations = possible_locations[:n_place_cells]
    place_cells = selected_locations  # Each location corresponds to one cell
    return place_cells

def generate_circular_trajectory(grid_size, n_rounds):
    """Generates a circular trajectory (square perimeter walk)."""
    trajectory = []
    for _ in range(n_rounds):
        for x in range(1, grid_size - 1):  # Right
            trajectory.append((x, 1))
        for y in range(1, grid_size):  # Up
            trajectory.append((grid_size - 1, y))
        for x in reversed(range(1, grid_size - 1)):  # Left
            trajectory.append((x, grid_size - 1))
        for y in reversed(range(1, grid_size - 1)):  # Down
            trajectory.append((1, y))
    return trajectory, len(trajectory)

def get_firing_steps(trajectory, place_cells):
    """Returns a list where each entry is the index of the firing place cell, or -1 if no firing."""
    firing_steps = []
    for x, y in trajectory:
        if (x, y) in place_cells:
            firing_steps.append(place_cells.index((x, y)))  # Which cell fired
        else:
            firing_steps.append(-1)  # No firing
    return firing_steps

def animate_firing(trajectory, firing_steps, place_cells):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks(np.arange(0, GRID_SIZE, 1))
        ax.set_yticks(np.arange(0, GRID_SIZE, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        firing = firing_steps[frame]
        title = f"Step {frame + 1} | Firing: {'None' if firing == -1 else f'Place Cell {firing}'}"
        ax.set_title(title)

        # Draw place cell firing locations
        for i, loc in enumerate(place_cells):
            ax.add_patch(plt.Circle(loc, 0.3, color='blue', alpha=0.2))
            ax.text(loc[0], loc[1], f'{i}', fontsize=6, ha='center', va='center')

        # Draw mouse
        x, y = trajectory[frame]
        if firing != -1:
            ax.plot(x, y, 'ro', markersize=12, label=f"Mouse (Firing Cell {firing})")
        else:
            ax.plot(x, y, 'ko', markersize=10, label="Mouse")

        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=300, repeat=False)
    plt.show()

if __name__ == '__main__':
    place_cells = setup_simulation(GRID_SIZE, N_PLACE_CELLS)
    trajectory, total_steps = generate_circular_trajectory(GRID_SIZE, N_ROUNDS)
    firing_steps = get_firing_steps(trajectory, place_cells)

    if VISUALIZE:
        animate_firing(trajectory, firing_steps, place_cells)

    print(firing_steps)
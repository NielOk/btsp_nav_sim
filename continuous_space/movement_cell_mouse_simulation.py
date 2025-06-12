import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from tqdm import tqdm
import pickle
import itertools

from learnable_movement_cell import LearnableMovementCell
from place_cell import PlaceCell

import numpy as np

def generate_mouse_path(num_steps_per_cycle, num_cycles, scale=4.0):
    """
    Generate an infinity-sign (lemniscate) shaped path for the mouse to follow.
    """
    total_steps = num_steps_per_cycle * num_cycles
    ts = np.linspace(0, 2 * np.pi * num_cycles, total_steps)

    x_vals = np.cos(ts)
    y_vals = np.sin(ts) * np.cos(ts)

    x_vals *= scale
    y_vals *= scale

    return x_vals, y_vals

def compute_average_speed(x_vals, y_vals, dt):
    """
    Compute the average speed of the mouse along the path.
    """
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    distances = np.sqrt(dx**2 + dy**2)
    total_distance = np.sum(distances)
    num_steps = len(x_vals)
    print(num_steps)
    average_speed = total_distance / (num_steps * dt)
    return average_speed

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
                          total_num_place_cells,
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
                                     total_num_place_cells=total_num_place_cells,
                                     instructive_signal_connections=num_instructive_signal_connections,
                                     num_left=num_left, 
                                     num_right=num_right,
                                     dt=dt
        )
        movement_cells.append(cell)

    return movement_cells

def get_place_spikes_over_time(place_cells, x_vals, y_vals, num_steps):
    place_spikes_over_time = []

    # Figure out place cell spikes
    for t in range(num_steps):
        x = x_vals[t]
        y = y_vals[t]

        place_spikes = np.zeros(len(place_cells), dtype=int)
        
        for i, pc in enumerate(place_cells):
            if pc.fire_at(x, y):
                place_spikes[i] = 1

        place_spikes_over_time.append(place_spikes)

    return place_spikes_over_time

def run_simulation(place_cells, movement_cells, x_vals, y_vals, instructive_signal_probability=0.005, buffer_steps=500):
    num_steps = len(x_vals)# buffer steps to allow for settling of learning at end
    
    place_spikes_over_time = get_place_spikes_over_time(place_cells, x_vals, y_vals, num_steps)

    # Send in place cell spikes and instructive signal spikes to movement cells
    for t in tqdm(range(num_steps)):
        place_spikes = place_spikes_over_time[t]

        for mc in movement_cells:
            mc_place_spikes = place_spikes[mc.place_cell_indices]
            
            # One spike per instructive connection, with independent probability
            mc_signal_spikes = (np.random.rand(len(mc.signal_map)) < instructive_signal_probability).astype(int)

            mc.receive_spikes(mc_place_spikes, mc_signal_spikes)
            mc.update()
            mc.apply_ampa_learning(learning_rate=7e-1, nmda_openness_threshold=0.7)

    # Additional buffer steps to allow for settling of learning
    for t in tqdm(range(buffer_steps)):
        for mc in movement_cells:
            mc.update()
            mc.apply_ampa_learning(learning_rate=7e-1, nmda_openness_threshold=0.7)

    return movement_cells

def plot_place_cells_and_path(place_cells, x_path, y_path, bounds=(-5, 5, -5, 5), resolution=300):
    """
    Plot place cell activation map and mouse path with one arrow per point indicating direction.
    """
    x_vals = np.linspace(bounds[0], bounds[1], resolution)
    y_vals = np.linspace(bounds[2], bounds[3], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Compute place cell activation map
    place_Z = np.zeros_like(X)
    for pc in place_cells:
        place_Z += np.exp(-((X - pc.center_x) ** 2 + (Y - pc.center_y) ** 2) / (2 * pc.standard_deviation ** 2))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')

    # Contour plot for place cell density
    ax.contourf(X, Y, place_Z, levels=50, cmap='Blues')

    # Convert path to arrays
    X_path = np.array(x_path)
    Y_path = np.array(y_path)

    # Compute deltas between each point and the next
    dx = np.diff(X_path)
    dy = np.diff(Y_path)

    # Remove last point to match length of dx/dy
    X_center = X_path[:-1]
    Y_center = Y_path[:-1]

    # Plot quiver arrows from each point to the next
    ax.quiver(X_center, Y_center, dx, dy, scale_units='xy', angles='xy', scale=1, color='red', width=0.003, alpha=0.8)

    ax.set_title("Place Cell Firing Map and Mouse Path Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def experiment(movement_cell_data_save_path, place_cells_data_save_path, mouse_path_save_path):
    # parameters for mouse path and place cells
    num_steps_per_cycle = 10000
    num_cycles = 1
    num_place_cells = 20
    bounds = (-5, 5, -5, 5)
    mouse_path_scale = 4.0
    dt = 0.001 * ms

    # Generate mouse path and place cells
    x_vals, y_vals = generate_mouse_path(num_steps_per_cycle, num_cycles, scale=mouse_path_scale)
    average_speed = compute_average_speed(x_vals, y_vals, dt)
    place_cells = create_place_cells(num_place_cells, bounds)

    num_movement_cells = 10
    num_test = 2
    num_left = 3
    num_right = 3
    place_cell_ampa = 2.0 * nsiemens
    instructive_signal_ampa = 8.0 * nsiemens
    num_place_cell_connections = [3, 1] # per compartment [10, 15]
    num_instructive_signal_connections = [5, 5] # per compartment

    per_cycle_time = num_steps_per_cycle * dt
    print(f"One cycle takes {per_cycle_time}.") 
    print(f"Average speed of mouse: {average_speed * second} units/ms")

    # Generate movement cells
    movement_cells = create_movement_cells(num_movement_cells, 
                                            place_cell_ampa, 
                                            len(place_cells),
                                            instructive_signal_ampa, 
                                            dt,
                                            num_place_cell_connections=num_place_cell_connections,
                                            num_instructive_signal_connections=num_instructive_signal_connections,
                                            num_test=num_test, 
                                            num_left=num_left, 
                                            num_right=num_right)
    
    # Plot place cells and mouse path
    plot_place_cells_and_path(place_cells, x_vals, y_vals, bounds)

    for mc in movement_cells:
        print(mc.place_cell_g_acts)
    
    instructive_signal_probability = 0.0005 # per step probability of firing
    buffer_steps = 500
    movement_cells = run_simulation(place_cells, movement_cells, x_vals, y_vals, instructive_signal_probability, buffer_steps)

    for mc in movement_cells:
        mc.reset_state() # Reset cell activity before saving
        print(mc.place_cell_g_acts)

    # Save the movement cells and place cells to files.
    with open(movement_cell_data_save_path, 'wb') as f:
        pickle.dump(movement_cells, f)
    with open(place_cells_data_save_path, 'wb') as f:
        pickle.dump(place_cells, f)

    # Save mouse path
    with open(mouse_path_save_path, 'wb') as f:
        pickle.dump((x_vals, y_vals), f)

def save_movement_cell_activity(movement_cell_data_save_path, place_cell_data_save_path, mouse_path_save_path, spiked_permutations_save_path):
    # Load movement cells and place cells from files
    with open(movement_cell_data_save_path, 'rb') as f:
        movement_cells = pickle.load(f)
    with open(place_cell_data_save_path, 'rb') as f:
        place_cells = pickle.load(f)
    with open(mouse_path_save_path, 'rb') as f:
        x_vals, y_vals = pickle.load(f)
    
    dt = 0.001 * ms
    average_speed = compute_average_speed(x_vals, y_vals, dt)
    print(f"Average speed of mouse: {average_speed * second} units/ms")

    print(len(movement_cells), "movement cells loaded.")
    print(len(place_cells), "place cells loaded.")

    # Get permutations of place cells of lenght 2
    index_list = list(range(len(place_cells)))
    place_cell_permutations = list(itertools.permutations(index_list, 2))
    print("Number of place cell permutations:", len(place_cell_permutations))
    
    # Plot activity of each movement cell by sending in signals separated by 1 ms for each place cell
    simulation_steps = 5000
    all_spiked_permutations = []
    for i in range(len(movement_cells)): # limit to first movement cell for testing
        mc = movement_cells[i]
        print(f"Recording activity for movement cell {i+1}/{len(movement_cells)}...")
        print(mc.place_cell_indices)

        # for each permutation of place cells, run sim. no learning and no instructive signals here
        for j in tqdm(range(len(place_cell_permutations))):
            place_cell_pair = place_cell_permutations[j]

            # Figure out distance between place cells and figure out, if mouse moved directly from one to another, how long it would take
            pc1 = place_cells[place_cell_pair[0]]
            pc2 = place_cells[place_cell_pair[1]]
            distance = np.sqrt((pc1.center_x - pc2.center_x) ** 2 + (pc1.center_y - pc2.center_y) ** 2)
            time_to_move = distance / average_speed

            if place_cell_pair[0] not in mc.place_cell_indices or place_cell_pair[1] not in mc.place_cell_indices:
                continue # Skip if place cell not connected to movement cell
            pc1_index = mc.place_cell_indices.index(place_cell_pair[0])
            pc2_index = mc.place_cell_indices.index(place_cell_pair[1])


            if pc1_index <= 2 and pc2_index > 2 and mc.place_cell_g_acts[pc1_index] >  8.0 * nS and mc.place_cell_g_acts[pc2_index] > 8.0 * nS:
                all_spiked_permutations.append([place_cell_pair, 0.0*mV])

            '''
            for t in range(simulation_steps):
                place_spikes = np.zeros(len(place_cells), dtype=int)
                if t == 1:
                    place_spikes[place_cell_pair[0]] = 1
                if t == 3000: #1 + int(time_to_move / dt):
                    place_spikes[place_cell_pair[1]] = 1
                mc_place_spikes = place_spikes[mc.place_cell_indices]
                mc.receive_spikes(mc_place_spikes, np.zeros(len(mc.signal_map), dtype=int))
                mc.update()
            if mc.V[4] > -30.0 * mV:
                all_spiked_permutations.append([place_cell_pair, mc.V[4]])
            '''
            
            # Reset state of movement cell
            mc.reset_state()

        print(f"Number of permutations after movement cell {i}: {len(all_spiked_permutations)}")
    
    print("Number of permutations that spiked:", len(all_spiked_permutations))

    # Save the activity data
    with open(spiked_permutations_save_path, 'wb') as f:
        pickle.dump(all_spiked_permutations, f)

def plot_spiked_permutations(spiked_permutations_save_path, place_cells_data_save_path, mouse_path_save_path):
    # Load the spiked permutations and place cells and mouse path from files
    with open(spiked_permutations_save_path, 'rb') as f:
        all_spiked_permutations = pickle.load(f)
    with open(place_cells_data_save_path, 'rb') as f:
        place_cells = pickle.load(f)
    with open(mouse_path_save_path, 'rb') as f:
        x_vals, y_vals = pickle.load(f)

    print("Number of spiked permutations:", len(all_spiked_permutations))

    # Plot the spiked permutations
    plt.figure(figsize=(12, 6))
    for i in tqdm(range(len(all_spiked_permutations))):
        pair, voltage = all_spiked_permutations[i]
        pc1 = place_cells[pair[0]]
        pc2 = place_cells[pair[1]]
        x1, y1 = pc1.center_x, pc1.center_y
        x2, y2 = pc2.center_x, pc2.center_y

        dx = x2 - x1
        dy = y2 - y1

        # Draw arrow from pc1 to pc2
        plt.arrow(x1, y1, dx, dy, head_width=0.1, length_includes_head=True, alpha=0.5)
    
    plt.plot(x_vals, y_vals, color='black', linewidth=0.5, label='Mouse Path')

    plt.title("Spiked Place Cell Permutations")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()
    plt.show()
            
def main():
    movement_cell_data_save_path = "movement_cells.pkl"
    place_cells_data_save_path = "place_cells.pkl"
    spiked_permutations_save_path = "all_spiked_permutations.pkl"
    mouse_path_save_path = "mouse_path.pkl"
    experiment(movement_cell_data_save_path, place_cells_data_save_path, mouse_path_save_path)
    save_movement_cell_activity(movement_cell_data_save_path, place_cells_data_save_path, mouse_path_save_path, spiked_permutations_save_path)
    plot_spiked_permutations(spiked_permutations_save_path, place_cells_data_save_path, mouse_path_save_path)

if __name__ == '__main__':
    main()
import numpy as np

class GridCell():
    def __init__(self, spacing, std, phase_offset_x=0.0, phase_offset_y=0.0, bounds=(-5, 5, -5, 5)):
        """
        spacing: distance between adjacent grid fields (center-to-center)
        std: standard deviation of each Gaussian bump
        phase_offset_x, phase_offset_y: shift the grid in x/y direction
        bounds: (x_min, x_max, y_min, y_max) to define grid extent
        """
        self.spacing = spacing
        self.std = std
        self.phase_offset_x = phase_offset_x
        self.phase_offset_y = phase_offset_y
        self.bounds = bounds
        self.centers = self._generate_centers()

    def _generate_centers(self):
        s = self.spacing
        h = s * np.sqrt(3) / 2  # vertical step size
        centers = []

        # Define triangle size based on vertical space
        x_min, x_max, y_min, y_max = self.bounds
        total_height = y_max - y_min
        num_rows = int(total_height / h)

        # Compute triangle in canonical upright position (centered at origin)
        triangle_points = []
        for row in range(num_rows):
            y = row * h
            num_cols = row + 1
            x_start = -0.5 * s * row
            for col in range(num_cols):
                x = x_start + col * s
                triangle_points.append(np.array([x, y]))

        triangle_points = np.array(triangle_points)

        # Random small rotation
        theta = np.random.uniform(-np.pi / 12, np.pi / 12)  # ±15°
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        triangle_points = triangle_points @ rotation_matrix.T

        # Center the triangle in the bounding box
        centroid = triangle_points.mean(axis=0)
        box_center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2
        ])
        shift = box_center - centroid
        triangle_points += shift

        # Clip points outside the box (optional)
        for cx, cy in triangle_points:
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                centers.append((cx, cy))

        return centers
            
    def single_gaussian_activation(self, cx, cy, x, y):
        return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * self.std**2))

    def activation(self, x, y):
        activation = 0.0
        for cx, cy in self.centers:
            activation += self.single_gaussian_activation(cx, cy, x, y)
        return activation
import numpy as np

class GridCell():
    def __init__(self, spacing, std, phase_offset_x=0.0, phase_offset_y=0.0,
                 bounds=(-5, 5, -5, 5), center=(0.0, 0.0)):
        """
        spacing: distance between adjacent grid fields (center-to-center)
        std: standard deviation of each Gaussian bump
        phase_offset_x, phase_offset_y: shift the grid in x/y direction
        bounds: (x_min, x_max, y_min, y_max) to define grid extent
        center: (x_center, y_center) to place the centroid of the grid
        """
        self.spacing = spacing
        self.std = std
        self.phase_offset_x = phase_offset_x
        self.phase_offset_y = phase_offset_y
        self.bounds = bounds
        self.center = np.array(center)
        self.centers = self._generate_centers()

    def _generate_centers(self):
        s = self.spacing
        h = s * np.sqrt(3) / 2  # vertical step size
        centers = []

        x_min, x_max, y_min, y_max = self.bounds
        total_height = y_max - y_min
        num_rows = int(total_height / h)

        triangle_points = []
        for row in range(num_rows):
            y = row * h
            num_cols = row + 1
            x_start = -0.5 * s * row
            for col in range(num_cols):
                x = x_start + col * s
                triangle_points.append(np.array([x, y]))

        triangle_points = np.array(triangle_points)

        # Center the triangle on the user-specified center
        centroid = triangle_points.mean(axis=0)
        shift = self.center - centroid
        triangle_points += shift

        # Clip to bounding box
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
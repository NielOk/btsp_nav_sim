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
        h = s * np.sqrt(3) / 2  # vertical distance between rows
        centers = []

        x_min, x_max, y_min, y_max = self.bounds
        margin = max(2 * self.std, s)  # safety buffer

        y = y_min - margin
        row_idx = 0
        while y <= y_max + margin:
            x_offset = 0 if row_idx % 2 == 0 else s / 2
            x = x_min - margin
            while x <= x_max + margin:
                cx = x + x_offset
                cy = y
                centers.append((cx, cy))
                x += s
            y += h
            row_idx += 1

        return centers

    def single_gaussian_activation(self, cx, cy, x, y):
        return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * self.std**2))

    def activation(self, x, y):
        activation = 0.0
        for cx, cy in self.centers:
            activation += self.single_gaussian_activation(cx, cy, x, y)
        return activation
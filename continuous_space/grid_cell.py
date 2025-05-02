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
        x_min, x_max, y_min, y_max = self.bounds

        dx = self.spacing
        dy = self.spacing * np.sin(np.pi / 3)  # sin(60°) ≈ 0.866

        centers = []
        y = y_min
        row = 0
        while y <= y_max:
            x_offset = (dx / 2) if (row % 2 == 1) else 0
            x = x_min
            while x <= x_max:
                cx = x + x_offset + self.phase_offset_x
                cy = y + self.phase_offset_y
                centers.append((cx, cy))
                x += dx
            y += dy
            row += 1
        return centers
        
    def single_gaussian_activation(self, cx, cy, x, y):
        return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * self.std**2))

    def activation(self, x, y):
        activation = 0.0
        for cx, cy in self.centers:
            activation += self.single_gaussian_activation(cx, cy, x, y)
        return activation
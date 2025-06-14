'''
Place Cell Class
'''

import numpy as np

class PlaceCell():

    def __init__(self, center_x, center_y, standard_deviation):
        """
        center_x, center_y: coordinates of the place cell's center
        standard_deviation: standard deviation of the Gaussian firing rate
        """
        self.center_x = center_x
        self.center_y = center_y
        self.standard_deviation = standard_deviation

    def gaussian_firing(self, x, y):
        """Returns the firing rate of the place cell when the agent is at (x, y)."""

        firing_rate = np.exp(-((x - self.center_x)**2 + (y - self.center_y)**2) / (2 * self.standard_deviation**2))
        return firing_rate
    
    def fire_at(self, x, y):
        """Returns True if the place cell fires at (x, y), otherwise False."""
        probability_fire = self.gaussian_firing(x, y)

        # Sample from a uniform distribution to determine if the cell fires
        if np.random.uniform(0, 1) < probability_fire / 100.0:
            return True
        else:
            return False
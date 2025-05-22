'''
multi-compartment dendrite model of CA1
'''

import numpy as np
from brian2 import *

global_RA = 400*ohm*cm
# Define section length and diameter with units
section_length = 50 * um
diameter = 1 * um

# Calculate cross-sectional area
area = pi * (diameter / 2) ** 2

# Calculate axial resistance
R_axial = (4 * global_RA * section_length) / (pi * diameter ** 2)

class Ca1():

    def __init__(self, num_place_cells, dt=0.1, C=1.0*uF/cm**2):
        """
        num_place_cells: number of place cells
        num_grid_cells: number of grid cells
        """
        self.num_place_cells = num_place_cells
        self.N = num_place_cells # number of compartments in dendrite
        self.dt = dt
        self.surface_area = pi * diameter * section_length
        self.C = C * self.surface_area # capacitance for each compartment
        self.Gs = (1 / R_axial) # axial conductance for each compartment
    
        self.V = np.ones(self.N) * -70.0 * mV

        # Channel parameters
        self.E_AMPA = 0.0*mvolt
        self.E_NMDA = 0.0*mvolt
        self.E_KIR = -90.0*mvolt

        # Channel state conductances
        self.g_AMPA = np.zeros(self.N) * nS
        self.g_NMDA = np.zeros(self.N) * nS
        self.g_KIR = np.ones(self.N)  * 6.0 * nS  # constant KIR

        # Active conductances
        self.g_AMPA_act = 8.0*nsiemens
        self.g_NMDA_act = 2.0*nsiemens

        # Synaptic decay time constants
        self.tau_AMPA = 5.0 * ms
        self.tau_NMDA = 100.0 * ms

        self.V_trace = []  # store voltage over time

    def receive_spikes(self, spikes):
        """
        we overwrite to g_act only if spike occurred.
        """
        assert len(spikes) == self.N, "Mismatch in number of compartments and spike input"
        self.g_AMPA[spikes == 1] = self.g_AMPA_act
        self.g_NMDA[spikes == 1] = self.g_NMDA_act

    def update(self):
        """
        Perform one Euler integration step:
        - Update voltage based on synaptic and axial currents
        - Apply exponential decay to AMPA and NMDA conductances
        """

        # === Ionic synaptic currents (per compartment) ===
        I_ion = (
            self.g_AMPA * (self.V - self.E_AMPA) +
            self.g_NMDA * (self.V - self.E_NMDA) +
            self.g_KIR  * (self.V - self.E_KIR)
        )

        # === Axial currents from adjacent compartments ===
        V_left = Quantity(np.concatenate((self.V[0:1], self.V)), dim = self.V.dim)
        V_right = Quantity(np.concatenate((self.V, self.V[-1:])), dim = self.V.dim)

        V_diff = V_left[:-1] - 2.0 * self.V + V_right[1:]
        I_axial = self.Gs * V_diff  # units: A

        # === Euler update of membrane potential ===
        dVdt = (-I_ion + I_axial) / self.C
        self.V += (self.dt * dVdt)

        # === Exponential decay of synaptic conductances ===
        self.g_AMPA -= self.dt * self.g_AMPA / self.tau_AMPA
        self.g_NMDA -= self.dt * self.g_NMDA / self.tau_NMDA

        # === Store current voltages (for later analysis) ===
        self.V_trace.append(self.V.copy())

    def run_with_custom_spikes(self, spikes_over_time):
        for spikes in spikes_over_time:
            self.receive_spikes(spikes)
            self.update()

    def get_voltage_trace(self):
        return np.array(self.V_trace)
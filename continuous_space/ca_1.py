'''
multi-compartment dendrite model of CA1
'''

import numpy as np
from brian2 import *

global_RA = 400*ohm*cm
section_length = 50*um
diameter = 1*um
L_cm = section_length / cm
D_cm = diameter / cm
R_axial = (4 * global_RA * L_cm) / (pi * D_cm**2)

class Ca1():

    def __init__(self, num_place_cells, dt=0.1, C=1.0*uF):
        """
        num_place_cells: number of place cells
        num_grid_cells: number of grid cells
        """
        self.num_place_cells = num_place_cells
        self.N = num_place_cells # number of compartments in dendrite
        self.dt = dt
        self.C = C # capacitance per cm^2
        self.Gs = 1 / R_axial # axial conductance per cm^2
    
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
        V_pad = np.pad(self.V, (1, 1), mode='edge')  # Neumann boundary: zero derivative
        V_diff = V_pad[:-2] - 2 * self.V + V_pad[2:]
        I_axial = self.Gs * V_diff

        # === Euler update of membrane potential ===
        dVdt = (-I_ion + I_axial) / self.C
        self.V += self.dt * dVdt

        # === Exponential decay of synaptic conductances ===
        self.g_AMPA -= self.dt * self.g_AMPA / self.tau_AMPA
        self.g_NMDA -= self.dt * self.g_NMDA / self.tau_NMDA

        # === Store current voltages (for later analysis) ===
        self.V_trace.append(self.V.copy())
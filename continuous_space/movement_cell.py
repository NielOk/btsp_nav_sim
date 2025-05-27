'''
movement cell dendrite model, designed like ca_1 
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

class MovementCell():

    def __init__(self, num_compartments, dt=0.1, C=1.0*uF/cm**2):
        self.num_compartments = num_compartments
        self.N = num_compartments + 2  # total compartments including boundaries
        self.dt = dt
        self.surface_area = pi * diameter * section_length
        self.C = C * self.surface_area
        self.Gs = (1 / R_axial)

        # Initialize voltages
        self.V = np.ones(self.N) * -70.0 * mV
        self.V[0] = 0.0 * mV     # left boundary
        self.V[-1] = -90.0 * mV  # right boundary

        # Channel parameters
        self.E_AMPA = 0.0*mvolt
        self.E_NMDA = 0.0*mvolt
        self.E_KIR = -90.0*mvolt

        # Conductances (internal only)
        self.g_AMPA = np.zeros(self.N) * nS
        self.g_NMDA = np.zeros(self.N) * nS
        self.g_KIR = np.ones(self.N) * 6.0 * nS

        self.g_AMPA_act = 8.0 * nsiemens
        self.g_NMDA_act = 2.0 * nsiemens

        # Figure out thevenin equivalent conductances for ends
        self.Gthevenin_right = self.calculate_Gthevenin(G_L=self.g_KIR[-1])
        self.right_clamp_length = self.calculate_adjusted_section_length(G_L=self.g_KIR[-1])
        self.Gthevenin_left = self.calculate_Gthevenin(G_L=self.g_NMDA_act)
        self.left_clamp_length = self.calculate_adjusted_section_length(G_L=self.g_NMDA_act)

        self.tau_AMPA = 5.0 * ms
        self.nmda_duration = 500.0 * ms
        self.nmda_timer = np.zeros(self.N) * ms

        # Openness constants
        self.nmda_vh = -23.7
        self.nmda_vs = 12.5
        self.kir_vh = -80.0
        self.kir_vs = -10.0

        self.V_trace = []

    def calculate_Gthevenin(self, G_L=0.1*nS):
        """
        Calculate the thevenin equivalent conductance for infinite dendrite stretch. 
        If closer to soma than sequence, then G_L is G_Kir. 
        If further, then G_L is G_NMDA. 
        """

        Gthevenin = (-G_L + (G_L**2 + 4*self.Gs*G_L)**0.5) / 2

        return Gthevenin
    
    def calculate_Gb(self, G_L=0.1*nS):
        """
        Calculate the thevenin equivalent conductance for infinite dendrite stretch.
        If closer to soma than sequence, then G_L is G_Kir.
        If further, then G_L is G_NMDA.
        """
        Gthevenin = self.calculate_Gthevenin(G_L)
        Gb = (Gthevenin * self.Gs) / (2*self.Gs - Gthevenin)

        return Gb
    
    def calculate_adjusted_section_length(self, G_L=0.1*nS):
        """
        Figure out the adjusted section length for the clamped compartment after the thevenin equivalent conductance.
        """
        Gb = self.calculate_Gb(G_L)
        adjusted_section_length = ((np.pi * (diameter ** 2)) / (4 * Gb * global_RA))

        return adjusted_section_length
    
    def receive_spikes(self, spikes):
        
        assert len(spikes) == self.N - 2, "Spikes should match internal compartments only"
        self.g_AMPA[1:-1][spikes == 1] = self.g_AMPA_act
        self.g_NMDA[1:-1][spikes == 1] = self.g_NMDA_act
        self.nmda_timer[1:-1][spikes == 1] = self.nmda_duration

    def update(self):
        V_mV = self.V / mV

        # Openness (internal compartments only)
        nmda_open = (1 + np.exp(-(self.E_NMDA / mV - self.nmda_vh) / self.nmda_vs)) / \
                    (1 + np.exp(-(V_mV - self.nmda_vh) / self.nmda_vs))
        kir_open = (1 + np.exp(-(self.E_KIR / mV - self.kir_vh) / self.kir_vs)) / \
                   (1 + np.exp(-(V_mV - self.kir_vh) / self.kir_vs))

        # Ionic current (internal compartments only)
        I_ion = (
            self.g_AMPA * (self.V - self.E_AMPA) +
            self.g_NMDA * nmda_open * (self.V - self.E_NMDA) +
            self.g_KIR  * kir_open  * (self.V - self.E_KIR)
        )

        # Axial current (using neighbors)
        V_left = self.V[:-2]
        V_mid  = self.V[1:-1]
        V_right = self.V[2:]

        V_diff = V_left - 2 * V_mid + V_right
        I_axial = self.Gs * V_diff

        # Euler update for internal compartments only
        dVdt = (-I_ion[1:-1] + I_axial) / self.C
        self.V[1:-1] += self.dt * dVdt

        # Exponential decay for AMPA
        self.g_AMPA[1:-1] -= self.dt * self.g_AMPA[1:-1] / self.tau_AMPA

        # Timer-based NMDA decay
        self.nmda_timer[1:-1] -= self.dt
        mask_expired = self.nmda_timer <= 0*ms
        self.g_NMDA[mask_expired] = 0.0 * nS

        # Clamp boundary voltages
        self.V[0] = 0.0 * mV
        self.V[-1] = -90.0 * mV

        self.V_trace.append(self.V.copy())

    def run_with_custom_spikes(self, spikes_over_time):
        for spikes in spikes_over_time:
            self.receive_spikes(spikes)
            self.update()

    def get_voltage_trace(self):
        return np.array(self.V_trace)
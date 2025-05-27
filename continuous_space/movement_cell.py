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

    def __init__(self, num_test, num_left=1, num_right=1, dt=0.1, C=1.0*uF/cm**2):
        self.num_test = num_test
        self.num_left = num_left
        self.num_right = num_right
        self.N = 2 + num_left + num_test + num_right  # clampL + buffers + test + clampR

        self.dt = dt
        self.surface_area = pi * diameter * section_length
        self.C = C * self.surface_area
        self.Gs_vector = np.ones(self.N - 1) * (1 / R_axial)

        # === Voltage Initialization ===
        self.V = np.ones(self.N) * -70.0 * mV
        self.V[0] = 0.0 * mV     # Clamp L
        self.V[-1] = -90.0 * mV  # Clamp R

        # === Channels ===
        self.E_AMPA = 0.0 * mV
        self.E_NMDA = 0.0 * mV
        self.E_KIR = -90.0 * mV

        self.g_AMPA = np.zeros(self.N) * nS
        self.g_NMDA = np.zeros(self.N) * nS
        self.g_KIR = np.ones(self.N) * 6.0 * nS

        self.g_AMPA_act = 8.0 * nsiemens
        self.g_NMDA_act = 2.0 * nsiemens

        self.tau_AMPA = 5.0 * ms
        self.nmda_duration = 500.0 * ms
        self.nmda_timer = np.zeros(self.N) * ms

        # === Gating Constants ===
        self.nmda_vh = -23.7
        self.nmda_vs = 12.5
        self.kir_vh = -80.0
        self.kir_vs = -10.0

        self.V_trace = []

    def receive_spikes(self, spikes):
        assert len(spikes) == self.num_test, "Spikes must match test compartments"
        # Left buffer indices: [1 : 1 + num_left] → receives spikes
        # Test indices: [1 + num_left : 1 + num_left + num_test]
        test_start = 1 + self.num_left
        test_end = test_start + self.num_test

        self.g_AMPA[1:self.num_left+1] = self.g_AMPA_act        # Left buffer spikes
        self.g_NMDA[1:self.num_left+1] = self.g_NMDA_act
        self.nmda_timer[1:self.num_left+1] = self.nmda_duration

        self.g_AMPA[test_start:test_end][spikes == 1] = self.g_AMPA_act
        self.g_NMDA[test_start:test_end][spikes == 1] = self.g_NMDA_act
        self.nmda_timer[test_start:test_end][spikes == 1] = self.nmda_duration

    def update(self):
        V_mV = self.V / mV

        nmda_open = (1 + np.exp(-(self.E_NMDA / mV - self.nmda_vh) / self.nmda_vs)) / \
                    (1 + np.exp(-(V_mV - self.nmda_vh) / self.nmda_vs))
        kir_open = (1 + np.exp(-(self.E_KIR / mV - self.kir_vh) / self.kir_vs)) / \
                   (1 + np.exp(-(V_mV - self.kir_vh) / self.kir_vs))

        I_ion = (
            self.g_AMPA * (self.V - self.E_AMPA) +
            self.g_NMDA * nmda_open * (self.V - self.E_NMDA) +
            self.g_KIR  * kir_open  * (self.V - self.E_KIR)
        )

        V_left  = self.V[0:-2]
        V_mid   = self.V[1:-1]
        V_right = self.V[2:]

        G_left  = self.Gs_vector[0:self.N - 2]
        G_right = self.Gs_vector[1:self.N - 1]

        I_axial = G_left * (V_left - V_mid) + G_right * (V_right - V_mid)

        dVdt = (-I_ion[1:-1] + I_axial) / self.C
        self.V[1:-1] += self.dt * dVdt

        self.g_AMPA[1:-1] -= self.dt * self.g_AMPA[1:-1] / self.tau_AMPA
        self.nmda_timer[1:-1] -= self.dt
        self.g_NMDA[self.nmda_timer <= 0 * ms] = 0.0 * nS

        self.V[0] = 0.0 * mV
        self.V[-1] = -90.0 * mV

        self.V_trace.append(self.V.copy())

    def run_with_custom_spikes(self, spikes_over_time):
        for spikes in spikes_over_time:
            self.receive_spikes(spikes)
            self.update()

    def get_voltage_trace(self):
        return np.array(self.V_trace)
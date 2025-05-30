import numpy as np
from brian2 import *

global_RA = 400 * ohm * cm
section_length = 53 * um
diameter = 1 * um

area = pi * (diameter / 2) ** 2
R_axial = (4 * global_RA * section_length) / (pi * diameter ** 2)

class LearnableMovementCell:

    def __init__(self, num_test, place_cell_ampa, instructive_signal_ampa, 
                 place_cell_connections, instructive_signal_connections,
                 num_left=1, num_right=1, dt=0.1, C=1.0*uF/cm**2):
        
        assert len(place_cell_connections) == num_test
        assert len(instructive_signal_connections) == num_test

        self.num_test = num_test
        self.num_left = num_left
        self.num_right = num_right
        self.N = num_left + num_test + num_right
        self.dt = dt
        self.surface_area = pi * diameter * section_length
        self.C = C * self.surface_area
        self.Gs_vector = np.ones(self.N - 1) * (1 / R_axial)

        self.place_cell_map = []
        for comp_idx, num_conns in enumerate(place_cell_connections):
            self.place_cell_map.extend([comp_idx + self.num_left] * num_conns)

        self.signal_map = []
        for comp_idx, num_conns in enumerate(instructive_signal_connections):
            self.signal_map.extend([comp_idx + self.num_left] * num_conns)

        self.V = np.ones(self.N) * -70.0 * mV

        self.E_AMPA = 0.0 * mV
        self.E_NMDA = 0.0 * mV
        self.E_KIR = -90.0 * mV

        self.g_KIR = np.ones(self.N) * 55.571 * nS

        self.g_AMPA_place = np.zeros(len(self.place_cell_map)) * nS
        self.g_NMDA_place = np.zeros(len(self.place_cell_map)) * nS

        self.place_cell_g_acts = np.full(len(self.place_cell_map), place_cell_ampa) * nS

        self.g_AMPA_act_signal = instructive_signal_ampa
        self.g_AMPA_act_max = 8.0 * nS
        self.g_NMDA_act_max = 18.523 * nS

        self.tau_AMPA = 5.0 * ms
        self.nmda_duration = 500.0 * ms
        self.nmda_timer_place = np.zeros(len(self.place_cell_map)) * ms

        self.nmda_vh = -23.7
        self.nmda_vs = 12.5
        self.kir_vh = -80.0
        self.kir_vs = -10.0

        self.V_trace = []

    def receive_spikes(self, place_cell_spikes, signal_spikes):
        assert len(place_cell_spikes) == len(self.place_cell_map)
        assert len(signal_spikes) == len(self.signal_map)

        # Place cells have both nmda and ampa
        for i, spike in enumerate(place_cell_spikes):
            if spike:
                self.g_AMPA_place[i] += self.place_cell_g_acts[i]
                self.g_NMDA_place[i] = self.g_NMDA_act_max
                self.nmda_timer_place[i] = self.nmda_duration
        
        # Instructive signal is pure AMPA
        for i, spike in enumerate(signal_spikes):
            if spike:
                self.g_AMPA_place[i] += self.g_AMPA_act_signal
        
    def update(self):
        V_mV = self.V / mV

        nmda_openness = (1 + np.exp(-(self.E_NMDA / mV - self.nmda_vh) / self.nmda_vs)) / \
                    (1 + np.exp(-(V_mV - self.nmda_vh) / self.nmda_vs)) # This captures openness based on voltage dependence. Still need to capture based on glutamate.
        self.nmda_openness = nmda_openness
        kir_openness = (1 + np.exp(-(self.E_KIR / mV - self.kir_vh) / self.kir_vs)) / \
                    (1 + np.exp(-(V_mV - self.kir_vh) / self.kir_vs))
    
        g_AMPA_comp = np.zeros(self.N) * nS
        g_NMDA_comp = np.zeros(self.N) * nS

        for i, comp in enumerate(self.place_cell_map):
            g_AMPA_comp[comp] += self.g_AMPA_place[i]
            g_NMDA_comp[comp] += self.g_NMDA_place[i] * nmda_openness[comp]

        # Keep left sides at plateau
        g_AMPA_comp[0:self.num_left] = self.g_AMPA_act_max
        g_NMDA_comp[0:self.num_left] = self.g_NMDA_act_max

        # Figure out currents for each compartment
        I_ion = (
            g_AMPA_comp * (self.V - self.E_AMPA) +
            g_NMDA_comp * (self.V - self.E_NMDA) +
            self.g_KIR * kir_openness * (self.V - self.E_KIR)
        )

        I_axial = np.zeros(self.N) * amp
        I_axial[0] = self.Gs_vector[0] * (self.V[1] - self.V[0])

        G_left = self.Gs_vector[0:self.N - 2]
        G_right = self.Gs_vector[1:self.N - 1]
        V_left = self.V[0:-2]
        V_mid = self.V[1:-1]
        V_right = self.V[2:]
        I_axial[1:-1] = G_left * (V_left - V_mid) + G_right * (V_right - V_mid)
        I_axial[-1] = self.Gs_vector[-1] * (self.V[-2] - self.V[-1])

        dVdt = (-I_ion + I_axial) / self.C
        self.V += self.dt * dVdt

        self.g_AMPA_place -= self.dt * self.g_AMPA_place / self.tau_AMPA
        self.nmda_timer_place -= self.dt
        self.g_NMDA_place[self.nmda_timer_place <= 0 * ms] = 0.0 * nS

        self.V_trace.append(self.V.copy())

        return nmda_openness

    def apply_ampa_learning(self, learning_rate, nmda_openness_threshold):
        for i in range(len(self.place_cell_map)):
            comp = self.place_cell_map[i]

            nmda_open = self.nmda_openness[comp]
                        
            # Captures both glutamate and voltage dependence
            if self.nmda_timer_place[i] > 0 * ms and nmda_open >= nmda_openness_threshold:
                self.place_cell_g_acts[i] += learning_rate * nS

    def run_with_custom_spikes(self, place_cell_spikes_over_time, signal_spikes_over_time):

        for t in range(len(place_cell_spikes_over_time)):
            self.receive_spikes(place_cell_spikes_over_time[t], signal_spikes_over_time[t])
            self.update()
            self.apply_ampa_learning(learning_rate=1e-12, nmda_openness_threshold=0.7)

            if t == 5000 or t == 10000:
                print(f"Time step {t}: V = {self.V}")
                print(f"Place cell weights: {self.place_cell_g_acts}")
                print(f"AMPA weights: {self.place_cell_g_acts}")
                print(f"NMDA conductances: {self.g_NMDA_place}")

    def get_voltage_trace(self):
        return np.array(self.V_trace)
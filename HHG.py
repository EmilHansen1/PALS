# %% Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import quad
from scipy.optimize import root
from matplotlib.colors import LogNorm

# %% HHG class

class HighHarmonicGeneration:
    def __init__(self, settings_dict):
        self.Ip = settings_dict['Ip']
        self.N_cycles = settings_dict['N_cycles']
        self.cep = settings_dict['cep']

        intensity = settings_dict['Intensity']
        lambd = settings_dict['Wavelength']
        E_max = np.sqrt(intensity / 3.50945e16)  # Max electric field amplitude (a.u.)
        self.omega = 2. * np.pi * 137.036 / (lambd * 1.e-9 / 5.29177e-11)

        self.Up = E_max ** 2 / (4 * self.omega ** 2)
        self.rtUp = np.sqrt(self.Up)
        self.period = 2 * np.pi * self.N_cycles / self.omega

        self.SPA_time_integral = settings_dict['SPA_time_integral']

    def A_field_sin2(self, t):
        """
        Sin^2 field. One possible choice of 'build in' pulse forms. Uses the field parameters of the class.
        :param t: Time (a.u.)
        """
        return 2 * self.rtUp * np.sin(self.omega * t / (2 * self.N_cycles)) ** 2 * np.cos(self.omega * t + self.cep)

    def E_field_sin2(self, t):
        """
        Electric field for the sin2 field.
        """
        term1 = 2*np.sqrt(self.Up) * np.sin(self.omega*t/(2*self.N_cycles))**2 * np.sin(self.omega*t + self.cep)
        term2 = -np.sqrt(self.Up)/self.N_cycles * np.sin(self.omega*t/self.N_cycles) * np.cos(self.omega*t + self.cep)
        return self.omega * (term1 + term2)

    def AI_sin2(s, t):
        """
        Integral of sin2 vector potential
        :param t: time
        """
        if s.N_cycles == 1:
            return -s.rtUp * (2*np.cos(s.cep) * t*s.omega + 3*np.sin(s.cep) - 4*np.sin(s.omega*t + s.cep) +
                        np.sin(2*s.omega*t + s.cep))/(4*s.omega)
        else:
            return -s.rtUp/(2*s.omega*(s.N_cycles**2-1)) * (s.N_cycles*(s.N_cycles+1)*np.sin((s.omega*t*(s.N_cycles-1) + s.cep*s.N_cycles)/s.N_cycles)
                                        + s.N_cycles*(s.N_cycles-1)*np.sin((s.omega*t*(s.N_cycles+1) + s.cep*s.N_cycles)/s.N_cycles)
                                        - 2*s.N_cycles**2*np.sin(s.omega*t+s.cep) - 2*np.sin(s.cep) + 2*np.sin(s.omega*t+s.cep))

    def AI2_sin2(s, t):
        """
        Integral of sin2 vector potential squared
        :param t: time
        """
        if s.N_cycles == 1:
            return -s.Up/(96*s.omega) * (-12*np.cos(2*s.cep)*t*s.omega - 72*s.omega*t + 96*np.sin(s.omega*t) - 12*np.sin(2*s.omega*t)
                        + 48*np.sin(s.omega*t+2*s.cep) + 16*np.sin(3*s.omega*t+2*s.cep) - 3*np.sin(4*s.omega*t+2*s.cep)
                        - 36*np.sin(2*s.omega*t+2*s.cep) - 25*np.sin(2*s.cep))
        else:
            fac = 24*s.Up / (32*s.N_cycles**4*s.omega - 40*s.N_cycles**2*s.omega + 8*s.omega)
            term1 = -1/6 * (s.N_cycles-0.5) * (-s.N_cycles**2 + np.cos(2*s.omega*t+2*s.cep) + 1) * (s.N_cycles+0.5) * s.N_cycles * np.sin(2*s.omega*t/s.N_cycles)
            term2 = (1/6*s.N_cycles**4 - 1/24*s.N_cycles**2) * np.sin(2*s.omega*t + 2*s.cep) * np.cos(2*s.omega*t/s.N_cycles)
            term3 = -2/3 * (s.N_cycles-1) * (s.N_cycles+1) * (s.N_cycles**2*np.cos(s.omega*t/s.N_cycles) - 3*s.N_cycles**2/4 + 3/16) * np.sin(2*s.omega*t+2*s.cep)
            term4 = (1/3*s.N_cycles**3 - 1/3*s.N_cycles) * np.sin(s.omega*t/s.N_cycles) * np.cos(2*s.omega*t+2*s.cep)
            term5 = (-4/3*s.N_cycles**5 + 5/3*s.N_cycles**3 - 1/3*s.N_cycles) * np.sin(s.omega*t/s.N_cycles)
            rest_terms =s.N_cycles**4*s.omega*t - 5*s.N_cycles**2*s.omega*t/4 + s.omega*t/4 - np.sin(s.cep)*np.cos(s.cep)/4
            return fac * (term1 + term2 + term3 + term4 + term5 + rest_terms)


    def A_bicircular(self, t):
        envelope = np.sin(self.omega*t/(2*self.N_cycles))**2
        prefactor = np.sqrt(2*self.Up)
        A1 = np.array([np.cos(self.omega*t + self.cep), np.sin(self.omega*t + self.cep), 0])
        A2 = np.array([np.cos(2*self.omega*t + self.cep), -np.sin(2*self.omega*t + self.cep), 0])
        return prefactor * (A1 + A2) * envelope

    def E_bicircular(self, t):
        cg = self.N_cycles
        cg1 = self.Up
        cg5 = self.omega
        cg7 = self.cep
        cg9 = t
        cg3 = np.array([-np.sqrt(2) * (np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) * np.cos(cg5 * cg9 + cg7) * cg5 / cg * np.cos(cg5 * cg9 / cg / 2) - np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) ** 2 * cg5 * np.sin(cg5 * cg9 + cg7) + np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) * np.cos(2 * cg5 * cg9 + cg7) * cg5 / cg * np.cos(cg5 * cg9 / cg / 2) - 2 * np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) ** 2 * cg5 * np.sin(2 * cg5 * cg9 + cg7)) / 2,-np.sqrt(2) * (np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) * np.sin(cg5 * cg9 + cg7) * cg5 / cg * np.cos(cg5 * cg9 / cg / 2) + np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) ** 2 * cg5 * np.cos(cg5 * cg9 + cg7) - np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) * np.sin(2 * cg5 * cg9 + cg7) * cg5 / cg * np.cos(cg5 * cg9 / cg / 2) - 2 * np.sqrt(2) * np.sqrt(cg1) * np.sin(cg5 * cg9 / cg / 2) ** 2 * cg5 * np.cos(2 * cg5 * cg9 + cg7)) / 2,0])
        return cg3

    def AI_bicircular(self, t):
        cg = self.N_cycles
        cg1 = self.Up
        cg5 = self.omega
        cg7 = self.phi
        cg9 = t
        cg3 = np.array([-np.sqrt(cg1) * (-np.sin(cg7) * np.cos(cg5 * cg9) ** 2 - np.sin(cg5 * cg9) * np.cos(cg7) - np.cos(cg5 * cg9) * np.sin(cg7) - cg * np.sin(cg5 * cg9) * np.sin(cg7) * np.sin(cg5 * cg9 / cg) + cg * np.cos(cg5 * cg9) * np.cos(cg7) * np.sin(cg5 * cg9 / cg) + 2 * cg * np.cos(cg5 * cg9) ** 2 * np.cos(cg7) * np.sin(cg5 * cg9 / cg) + 4 * cg ** 4 * np.sin(cg5 * cg9) * np.cos(cg7) * np.cos(cg5 * cg9 / cg) + 4 * cg ** 4 * np.cos(cg5 * cg9) * np.sin(cg7) * np.cos(cg5 * cg9 / cg) + 4 * cg ** 4 * np.cos(cg5 * cg9) ** 2 * np.sin(cg7) * np.cos(cg5 * cg9 / cg) - 4 * cg ** 4 * np.cos(cg7) * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) + 4 * cg ** 3 * np.sin(cg5 * cg9) * np.sin(cg7) * np.sin(cg5 * cg9 / cg) - 4 * cg ** 3 * np.cos(cg5 * cg9) * np.cos(cg7) * np.sin(cg5 * cg9 / cg) - 2 * cg ** 3 * np.cos(cg5 * cg9) ** 2 * np.cos(cg7) * np.sin(cg5 * cg9 / cg) - cg ** 2 * np.sin(cg5 * cg9) * np.cos(cg7) * np.cos(cg5 * cg9 / cg) - cg ** 2 * np.cos(cg5 * cg9) * np.sin(cg7) * np.cos(cg5 * cg9 / cg) - 4 * cg ** 2 * np.cos(cg5 * cg9) ** 2 * np.sin(cg7) * np.cos(cg5 * cg9 / cg) + 5 * cg ** 2 * np.cos(cg7) * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) + 5 * cg ** 2 * np.sin(cg5 * cg9) * np.cos(cg7) + 5 * cg ** 2 * np.cos(cg5 * cg9) * np.sin(cg7) + 5 * cg ** 2 * np.sin(cg7) * np.cos(cg5 * cg9) ** 2 - cg * np.cos(cg7) * np.sin(cg5 * cg9 / cg) - 2 * cg ** 4 * np.sin(cg7) * np.cos(cg5 * cg9 / cg) - np.cos(cg7) * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) - 4 * cg ** 4 * np.sin(cg5 * cg9) * np.cos(cg7) - 4 * cg ** 4 * np.cos(cg5 * cg9) * np.sin(cg7) - 4 * cg ** 4 * np.sin(cg7) * np.cos(cg5 * cg9) ** 2 + cg ** 3 * np.cos(cg7) * np.sin(cg5 * cg9 / cg) + 2 * cg ** 2 * np.sin(cg7) * np.cos(cg5 * cg9 / cg) + 4 * cg ** 4 * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.cos(cg7) * np.cos(cg5 * cg9 / cg) + 2 * cg ** 3 * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.sin(cg7) * np.sin(cg5 * cg9 / cg) - 4 * cg ** 2 * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.cos(cg7) * np.cos(cg5 * cg9 / cg) - 2 * cg * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.sin(cg7) * np.sin(cg5 * cg9 / cg) - 7 * cg ** 2 * np.sin(cg7) + 2 * cg ** 4 * np.sin(cg7) + 2 * np.sin(cg7)) / cg5 / (4 * cg ** 4 - 5 * cg ** 2 + 1) / 2,-np.sqrt(cg1) * (2 * cg ** 4 * np.cos(cg7) + cg ** 2 * np.cos(cg7) - 4 * cg ** 4 * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.sin(cg7) * np.cos(cg5 * cg9 / cg) + 2 * cg ** 3 * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.cos(cg7) * np.sin(cg5 * cg9 / cg) + 4 * cg ** 2 * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.sin(cg7) * np.cos(cg5 * cg9 / cg) - 2 * cg * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) * np.cos(cg7) * np.sin(cg5 * cg9 / cg) - 4 * cg ** 4 * np.cos(cg5 * cg9) * np.cos(cg7) * np.cos(cg5 * cg9 / cg) + 4 * cg ** 4 * np.sin(cg5 * cg9) * np.sin(cg7) * np.cos(cg5 * cg9 / cg) + 4 * cg ** 4 * np.cos(cg5 * cg9) ** 2 * np.cos(cg7) * np.cos(cg5 * cg9 / cg) + 4 * cg ** 4 * np.sin(cg7) * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) - 4 * cg ** 3 * np.cos(cg5 * cg9) * np.sin(cg7) * np.sin(cg5 * cg9 / cg) - 4 * cg ** 3 * np.sin(cg5 * cg9) * np.cos(cg7) * np.sin(cg5 * cg9 / cg) + 2 * cg ** 3 * np.cos(cg5 * cg9) ** 2 * np.sin(cg7) * np.sin(cg5 * cg9 / cg) + cg ** 2 * np.cos(cg5 * cg9) * np.cos(cg7) * np.cos(cg5 * cg9 / cg) - cg ** 2 * np.sin(cg5 * cg9) * np.sin(cg7) * np.cos(cg5 * cg9 / cg) - 4 * cg ** 2 * np.cos(cg5 * cg9) ** 2 * np.cos(cg7) * np.cos(cg5 * cg9 / cg) - 5 * cg ** 2 * np.sin(cg7) * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) + cg * np.cos(cg5 * cg9) * np.sin(cg7) * np.sin(cg5 * cg9 / cg) + cg * np.sin(cg5 * cg9) * np.cos(cg7) * np.sin(cg5 * cg9 / cg) - 2 * cg * np.cos(cg5 * cg9) ** 2 * np.sin(cg7) * np.sin(cg5 * cg9 / cg) - np.cos(cg7) * np.cos(cg5 * cg9) ** 2 + np.cos(cg5 * cg9) * np.cos(cg7) - np.sin(cg5 * cg9) * np.sin(cg7) + np.sin(cg7) * np.sin(cg5 * cg9) * np.cos(cg5 * cg9) - 2 * cg ** 4 * np.cos(cg7) * np.cos(cg5 * cg9 / cg) + 4 * cg ** 4 * np.cos(cg5 * cg9) * np.cos(cg7) - 4 * cg ** 4 * np.sin(cg5 * cg9) * np.sin(cg7) - 4 * cg ** 4 * np.cos(cg7) * np.cos(cg5 * cg9) ** 2 - cg ** 3 * np.sin(cg7) * np.sin(cg5 * cg9 / cg) + 2 * cg ** 2 * np.cos(cg7) * np.cos(cg5 * cg9 / cg) - 5 * cg ** 2 * np.cos(cg5 * cg9) * np.cos(cg7) + 5 * cg ** 2 * np.sin(cg5 * cg9) * np.sin(cg7) + 5 * cg ** 2 * np.cos(cg7) * np.cos(cg5 * cg9) ** 2 + cg * np.sin(cg7) * np.sin(cg5 * cg9 / cg)) / cg5 / (4 * cg ** 4 - 5 * cg ** 2 + 1) / 2,0])
        return cg3

    def AI2_bicircular(self, t):
        cg = self.N_cycles
        cg1 = self.Up
        cg3 = self.omega
        cg5 = self.cep
        cg7 = t
        cg2 = cg1 * (288 * cg ** 3 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.sin(cg5) * np.cos(cg5) * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) - 32 * cg * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.sin(cg5) * np.cos(cg5) * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 12 * cg3 * cg7 - 48 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg5) ** 2 * np.cos(cg3 * cg7 / cg) ** 2 - 48 * cg ** 2 * np.cos(cg3 * cg7) ** 3 * np.sin(cg5) * np.cos(cg5) * np.cos(cg3 * cg7 / cg) ** 2 + 36 * cg ** 2 * np.cos(cg3 * cg7) * np.sin(cg5) * np.cos(cg5) * np.cos(cg3 * cg7 / cg) ** 2 + 384 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg3 * cg7 / cg) * np.cos(cg5) ** 2 + 384 * cg ** 2 * np.cos(cg3 * cg7) ** 3 * np.cos(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) - 288 * cg ** 2 * np.cos(cg3 * cg7) * np.cos(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) + 32 * cg * np.cos(cg3 * cg7) ** 3 * np.cos(cg5) ** 2 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) - 24 * cg * np.cos(cg3 * cg7) * np.cos(cg5) ** 2 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) - 32 * cg * np.sin(cg3 * cg7) * np.sin(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) + 432 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg5) ** 2 * np.cos(cg3 * cg7 / cg) ** 2 + 432 * cg ** 4 * np.cos(cg3 * cg7) ** 3 * np.sin(cg5) * np.cos(cg5) * np.cos(cg3 * cg7 / cg) ** 2 - 324 * cg ** 4 * np.cos(cg3 * cg7) * np.sin(cg5) * np.cos(cg5) * np.cos(cg3 * cg7 / cg) ** 2 - 864 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg3 * cg7 / cg) * np.cos(cg5) ** 2 - 864 * cg ** 4 * np.cos(cg3 * cg7) ** 3 * np.cos(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) + 648 * cg ** 4 * np.cos(cg3 * cg7) * np.cos(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) - 288 * cg ** 3 * np.cos(cg3 * cg7) ** 3 * np.cos(cg5) ** 2 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 216 * cg ** 3 * np.cos(cg3 * cg7) * np.cos(cg5) ** 2 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 72 * cg ** 3 * np.sin(cg3 * cg7) * np.sin(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) - 8 * np.sin(cg5) * np.cos(cg5) - 72 * cg ** 3 * np.sin(cg3 * cg7) * np.sin(cg5) * np.cos(cg5) * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) - 288 * cg ** 3 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.sin(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) + 8 * cg * np.sin(cg3 * cg7) * np.sin(cg5) * np.cos(cg5) * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 128 * cg * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.sin(cg3 * cg7 / cg) * np.sin(cg5) * np.cos(cg5) + 180 * cg ** 3 * np.sin(cg3 * cg7 / cg) - 16 * cg * np.sin(cg3 * cg7 / cg) - 16 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 - 8 * np.sin(cg3 * cg7) * np.cos(cg5) ** 2 - 324 * cg ** 5 * np.sin(cg3 * cg7 / cg) + 54 * cg ** 4 * np.sin(cg3 * cg7) - 42 * cg ** 2 * np.sin(cg3 * cg7) + 64 * cg * np.cos(cg3 * cg7) ** 3 * np.sin(cg3 * cg7 / cg) - 48 * cg * np.cos(cg3 * cg7) * np.sin(cg3 * cg7 / cg) - 45 * cg ** 3 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 4 * cg * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 81 * cg ** 5 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) - 144 * cg ** 3 * np.cos(cg3 * cg7) ** 3 * np.sin(cg3 * cg7 / cg) + 108 * cg ** 3 * np.cos(cg3 * cg7) * np.sin(cg3 * cg7 / cg) - 6 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7 / cg) ** 2 + 168 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 + 84 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg5) ** 2 + 48 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7 / cg) + 54 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7 / cg) ** 2 - 216 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 - 108 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg5) ** 2 - 108 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7 / cg) + 32 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg5) ** 2 + 32 * np.cos(cg3 * cg7) ** 3 * np.sin(cg5) * np.cos(cg5) - 24 * np.cos(cg3 * cg7) * np.sin(cg5) * np.cos(cg5) + 243 * cg ** 4 * cg3 * cg7 - 135 * cg ** 2 * cg3 * cg7 + 4 * np.sin(cg3 * cg7) - 192 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg3 * cg7 / cg) - 96 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7 / cg) * np.cos(cg5) ** 2 - 16 * cg * np.cos(cg3 * cg7) ** 3 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 12 * cg * np.cos(cg3 * cg7) * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) - 128 * cg * np.cos(cg3 * cg7) ** 3 * np.sin(cg3 * cg7 / cg) * np.cos(cg5) ** 2 + 96 * cg * np.cos(cg3 * cg7) * np.sin(cg3 * cg7 / cg) * np.cos(cg5) ** 2 + 432 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg3 * cg7 / cg) + 216 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7 / cg) * np.cos(cg5) ** 2 + 144 * cg ** 3 * np.cos(cg3 * cg7) ** 3 * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) - 108 * cg ** 3 * np.cos(cg3 * cg7) * np.sin(cg3 * cg7 / cg) * np.cos(cg3 * cg7 / cg) + 288 * cg ** 3 * np.cos(cg3 * cg7) ** 3 * np.sin(cg3 * cg7 / cg) * np.cos(cg5) ** 2 - 216 * cg ** 3 * np.cos(cg3 * cg7) * np.sin(cg3 * cg7 / cg) * np.cos(cg5) ** 2 + 24 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg3 * cg7 / cg) ** 2 + 12 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg5) ** 2 * np.cos(cg3 * cg7 / cg) ** 2 - 336 * cg ** 2 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg5) ** 2 - 336 * cg ** 2 * np.cos(cg3 * cg7) ** 3 * np.sin(cg5) * np.cos(cg5) + 252 * cg ** 2 * np.cos(cg3 * cg7) * np.sin(cg5) * np.cos(cg5) - 216 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg3 * cg7 / cg) ** 2 - 108 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg5) ** 2 * np.cos(cg3 * cg7 / cg) ** 2 + 432 * cg ** 4 * np.sin(cg3 * cg7) * np.cos(cg3 * cg7) ** 2 * np.cos(cg5) ** 2 + 432 * cg ** 4 * np.cos(cg3 * cg7) ** 3 * np.sin(cg5) * np.cos(cg5) - 324 * cg ** 4 * np.cos(cg3 * cg7) * np.sin(cg5) * np.cos(cg5)) / cg3 / (81 * cg ** 4 - 45 * cg ** 2 + 4) / 4
        return cg2

    def action(self, k, t_re, t_ion):
        """
        Calculation of the SFA action
        :param t_re: Recombination time
        :param t_ion: Ionization time
        :param k: Momentum value along z
        """
        AI = self.AI_sin2(t_re) - self.AI_sin2(t_ion)
        AI2 = self.AI2_sin2(t_re) - self.AI2_sin2(t_ion)
        return (self.Ip + k**2/2) * (t_re - t_ion) + k*AI + AI2/2

    def dipole_matrix_element(self, k_saddle, t):
        k_tilde = k_saddle + self.A_field_sin2(t)
        return 1j * 2**(3/2) * 4/np.pi * k_tilde / (k_tilde**2 + 1)**3

    def dipole_matrix_element_SPA(self, k_saddle, t_ion):
        k_tilde = k_saddle + self.A_field_sin2(t_ion)
        action_double_derivative = - k_tilde * self.E_field_sin2(t_ion)
        return -4*(1+1j)/np.sqrt(2*np.pi) * action_double_derivative**(-3/2) * k_tilde

    def k_saddle(self, t_re, t_ion):
        """
        Value of the momentum vector in the saddle point.
        :param t_re: Recombination time
        :param t_ion: Ionization time
        """
        alpha_re = self.AI_sin2(t_re)
        alpha_ion = self.AI_sin2(t_ion)
        return -(alpha_re - alpha_ion) / (t_re - t_ion)  # Might need to make Taylor of this when dt gets small?

    def get_saddle_guess(self, tr_lims, ti_lims, Nr, Ni):
        """
        Obtains the saddle point times through user input. User clicks on plot to identify the position of the saddle
        points.
        :param tr_lims: List of lower and upper limit of real saddle time used in plot
        :param ti_lims: List of lower and upper limit of imaginary saddle time used in plot
        :param Nr: Nr. of data points on real axis of plot
        :param Ni: Nr. of data points on imag axis of plot
        :param p_vec: Momentum vector for which saddle points are calculated
        """
        tr_list = np.linspace(tr_lims[0], tr_lims[1], Nr)
        ti_list = np.linspace(ti_lims[0], ti_lims[1], Ni)
        res_grid = np.zeros((len(ti_list), len(tr_list)), dtype=complex)
        t_re = self.period * 1.4

        # Should try to do this with numpy array magic
        for i, ti in enumerate(ti_list):
            for j, tr in enumerate(tr_list):
                res_grid[i, j] = self.t_ion_saddle_eq(t_re, tr + 1j*ti)

        res_grid = np.log10(np.abs(res_grid) ** 2)

        # Stuff needed to make interactive plot work also on Jupyter notebook
        tr_guess = []
        ti_guess = []
        self.guess_saddle_points = []

        def on_click(event):  # Handles events for the saddle point plot
            tr_guess.append(event.xdata)
            ti_guess.append(event.ydata)
            self.guess_saddle_points.append(event.xdata + 1j * event.ydata)
            self.user_points_plot.set_data(tr_guess, ti_guess)
            self.user_points_plot.figure.canvas.draw()

        fig, ax = plt.subplots()
        ax.imshow(np.flip(res_grid, 0), cmap='twilight', interpolation='bicubic', aspect='auto',
                   extent=(tr_lims[0], tr_lims[1], ti_lims[0], ti_lims[1]))
        self.user_points_plot, = ax.plot([], [], 'ob')   # Have to be class memeber, else cannot interact with plot in Jupyter notebook
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

    def t_ion_saddle_eq(self, t_re, t_ion):
        return 0.5 * (self.A_field_sin2(t_ion) + self.k_saddle(t_re, t_ion))**2 + self.Ip

    def t_ion_saddle_eq_real_imag(self, ts_list, t_re):
        """
        Simple wrapper function used to make scipy's root finder happy.
        """
        val = self.t_ion_saddle_eq(t_re, ts_list[0] + 1j*ts_list[1])
        return [val.real, val.imag]

    def find_saddle_times(self, guess_times, t_re):
        """
        Function to determine the saddle point times. Uses scipy's root function to determine the roots of the
        derivative of the action.
        """
        root_list = []
        for guess_time in guess_times:
            guess_i = np.array([guess_time.real, guess_time.imag])
            sol = root(self.t_ion_saddle_eq_real_imag, guess_i, args=(t_re,))
            tr = sol.x[0]
            ti = sol.x[1]

            # Restrict real part within first period
            if tr > self.period:
                tr -= self.period

            root_list.append(tr + 1j*ti)
        return root_list

    def perform_FFT(self, t_list, data_list):
        """
        Performs the Fourier transformation using scipy's FFT algorithm. Here we employ the correct normalization to
        get the same result as the continues unitary Fourier transform.
        :param t_list: List of times for which the data is sampled
        :param data_list: List of data points of the function to be Fourier transformed
        """
        dt = t_list[1] - t_list[0]
        Nt = len(t_list)
        fft_omega_list = fftfreq(Nt, dt) * 2*np.pi  # To convert to angular frequency
        fft_res = fft(data_list) * dt / np.sqrt(2*np.pi)  # To convert to unitary Fourier transform
        return np.abs(fft_res), fft_omega_list

    def fourier_trapz(self, t_list, data_list):
        omega_list = np.linspace(0, 3 * 3.17 * self.Up + self.Ip, 1000)
        res_list = []
        for omega in omega_list:
            res_list.append(np.trapz(np.exp(-1j * omega * t_list) * data_list, t_list))
        return np.array(res_list), np.array(omega_list)

    def integrand_HHG(self, t_ion, t_re):
        # First get the momentum vector in the saddle point
        k_s = self.k_saddle(t_re, t_ion)
        # Then obtain the dipole matrix elements
        d_t_re = self.dipole_matrix_element(k_s, t_re)
        d_t_ion = np.conj(self.dipole_matrix_element_SPA(k_s, t_ion))
        # Get the action
        S = self.action(k_s, t_re, t_ion)
        # Stability factors
        k_stability_fac = (2*np.pi / (1j * (t_re - t_ion)))**3/2  # Epsilon or...?
        action_double_derivative = -(k_s + self.A_field_sin2(t_ion)) * self.E_field_sin2(t_ion)
        t_stability_fac = np.sqrt(2*np.pi * 1j / action_double_derivative)
        # Combine it all to get the integrand
        return k_stability_fac * t_stability_fac * d_t_re * d_t_ion * self.E_field_sin2(t_ion) * np.exp(-1j * S)

    def calculate_HHG_spectrum(self):
        # Outer operation: Perfrom FFT on recomb. time integral.
        # For each datapoint in FFT: Calcualte the ionization integral (trapz is perhaps good enough?)
        # For each datapoint in inner integral: Determine k saddle points + prefactors + action
        t_recomb_list = np.linspace(0, 1*self.period, 50000)
        print(t_recomb_list[1] - t_recomb_list[0])
        fft_list = []

        if not np.any(self.guess_saddle_points):
            print('Everybody panic!')

        guess_times = self.guess_saddle_points
        for t_re in t_recomb_list:
            t_ion_saddle = self.find_saddle_times(guess_times, t_re)
            guess_times = t_ion_saddle
            res = 0
            for t_ion in t_ion_saddle:
                res += self.integrand_HHG(t_ion, t_re)
            fft_list.append(res)

        fft_list = np.array(fft_list)
        fft_list = fft_list + np.conj(fft_list)

        print('Now calculating the spectrum!')
        # Perform the FFT
        #fft_res, fft_omega = self.perform_FFT(t_recomb_list, fft_list)
        #return fft_res, fft_omega

        np.save('fft_list', fft_list)
        # Test with trapz
        res_list, omega_list = self.fourier_trapz(t_recomb_list, fft_list)
        return res_list, omega_list

    def get_dipole(self):
        # Outer operation: Perfrom FFT on recomb. time integral.
        # For each datapoint in FFT: Calcualte the ionization integral (trapz is perhaps good enough?)
        # For each datapoint in inner integral: Determine k saddle points + prefactors + action
        t_recomb_list = np.linspace(0, 1 * self.period, 50000)
        print(t_recomb_list[1] - t_recomb_list[0])
        dipole_list = []

        if not np.any(self.guess_saddle_points):
            print('Everybody panic!')

        guess_times = self.guess_saddle_points
        for t_re in t_recomb_list:
            t_ion_saddle = self.find_saddle_times(guess_times, t_re)
            guess_times = t_ion_saddle
            res = 0
            for t_ion in t_ion_saddle:
                res += self.integrand_HHG(t_ion, t_re)
            dipole_list.append(res)

        dipole_list = np.array(dipole_list)
        dipole_list = dipole_list + np.conj(dipole_list)
        return t_recomb_list, dipole_list



    def get_spectrogram(self, width, N_every, max_harmonic_order, N_omega, times=None, dip=None):
        """
        Performs a time-frequency analysis
        :param width: the width of the Gaussian window function - typically in the range of a couple of pi
        :param N_every: takes N_every entry of the dipole and the recombination times, so the entire signal is not used. Speeds up calculations enourmously!
        :params max_harmonic_order: the maximum harmonic order to be seen in the spectrum; i.e. a cutoff in energy at omega_cutoff = max_harmonic_order * omega
        :params N_omega: the number of points in the energy grid between [0, max_harmonic_order * omega]. Computation time scales linearly with this
        :returns: the spectrogram as a matrix to be plottet by, e.g., plt.imshow or plt.meshgrid. Also returns the time and energy lists.
        """

        # Get dipole acceleration
        if times is None and dip is None:
            rec_times, dipole = self.get_dipole()
        else:
            rec_times = times
            dipole = dip
        dipole_acceleration = np.gradient(np.gradient(dipole, rec_times), rec_times) 

        # We employ a Gaussian window
        def window(t, tau):
            return np.exp(-(tau - t)**2 / 2 / width**2)

        # Define the Gabor transform  function
        def gabor(signal, t, tau, omega):
            return np.trapz(signal*window(t, tau)*np.cos(omega*t), t) \
                    - 1j * np.trapz(signal*window(t, tau)*np.sin(omega*t), t)
        
        # Prepare for integration by taking a smaller number of times to speed up computation.
        # It will converge at some number of times, so N_every should not be too small, else
        # things will start looking not all right.
        t_lst = rec_times[::N_every]
        d_acc = dipole_acceleration[::N_every]
        w_lst = np.linspace(0, max_harmonic_order*HHG.omega, N_omega)

        # Calculate the spectrogram as entries of a matrix
        G_spec = np.zeros((len(t_lst), len(w_lst)), dtype=complex)
        for i, t in enumerate(t_lst):
            for j, w in enumerate(w_lst):
                G_spec[i, j] = gabor(d_acc, t_lst, t, w)

        return G_spec, t_lst, w_lst

# %% Run some stuff

settings_dict = {
    'Ip': 0.5,              # Ionization potential (a.u.)
    'Wavelength': 800,      # (nm)
    'Intensity': 2e14,      # (W/cm^2)
    'cep': 0,         # Carrier envelope phase
    'N_cycles': 2,          # Nr of cycles
    'N_cores': 4,           # Nr. of cores to use in the multiprocessing calculations
    'SPA_time_integral': True
}

HHG = HighHarmonicGeneration(settings_dict)
#HHG.get_saddle_guess([0, HHG.period], [0, 80], 400, 400)
#np.save('test_guess', HHG.guess_saddle_points)
HHG.guess_saddle_points = np.load('test_guess.npy')

print('Cutoff: ', HHG.Ip + 3.17*HHG.Up)

fft_res, fft_omega = HHG.calculate_HHG_spectrum()
filter_list = fft_omega >= 0
fft_res = fft_res[filter_list]
fft_omega = fft_omega[filter_list]

plt.axvline((HHG.Ip + 3.17 * HHG.Up) / HHG.omega, ls='--', c='gray', alpha=0.3)
plt.plot(fft_omega / HHG.omega, fft_omega**2 * np.abs(fft_res)**2)
plt.xlabel(r'Harmonic order $\omega/\omega_L$')
plt.ylabel(r'Yield (arb. units)')
#plt.xlim(0,1.19)
plt.yscale('log')
plt.minorticks_on()
plt.show()

# %%

dip_times, dip = HHG.get_dipole()
plt.plot(dip_times, dip)
plt.show()

# %%
plt.plot(dip_times, np.real(dip))

d_acc = np.gradient(np.gradient(dip, dip_times), dip_times)
plt.plot(dip_times, d_acc)

plt.show()
# %%

G, ts, ws = HHG.get_spectrogram(2*np.pi, 50, 40, 100, times=dip_times, dip=dip)
# %%
M = np.abs(G)**2
M_max = np.max(M)
vmin, vmax = M_max*1e-6, M_max
M[M < vmin] = vmin

plt.imshow(np.flip(M.T, 0), extent=(ts[0], ts[-1], ws[0]/HHG.omega, ws[-1]/HHG.omega),
           aspect='auto', norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='bicubic',
           cmap='Spectral_r')
plt.colorbar()

plt.axhline((HHG.Ip + 3.17*HHG.Up)/HHG.omega, ls='--', color='white')
plt.xlabel('Time (a.u.)')
plt.ylabel('Harmonic order')
#plt.xlim(100, 300)

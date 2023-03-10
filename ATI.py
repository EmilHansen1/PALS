# %% LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker

# %% ATI CLASS

class AboveThresholdIonization:
    def __init__(self, omega0: float, intensity: float, Ip: float, vec_pot: callable) -> None:
        """
        Very descriptive description
        :param omega0: Laser carrier frequency
        :param intensity: Laser intensity in W/cm²
        :param Ip: The ionization potential in a.u.
        :param vec_pot: The laser vector potential in a.u.
        """
        self.omega0 = omega0
        self.intensity = intensity
        self.Ip = Ip
        self.vec_pot = vec_pot

    def get_saddle_times(self, n_cycles: int) -> list[complex]:
        times = []


        return [2 + 1j]




def vector_potential(t: float, omega: float, Up: float):
    return 2 * np.sqrt(Up) * np.cos(omega * t)


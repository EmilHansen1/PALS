import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import quad
from scipy.optimize import root

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

settings_dict = {
    'Ip': 0.5,              # Ionization potential (a.u.)
    'Wavelength': 800,      # (nm)
    'Intensity': 1e14,      # (W/cm^2)
    'cep': np.pi/2,         # Carrier envelope phase
    'N_cycles': 15,          # Nr of cycles
    'N_cores': 4,           # Nr. of cores to use in the multiprocessing calculations
    'SPA_time_integral': True
}

HHG = HighHarmonicGeneration(settings_dict)
HHG.get_saddle_guess([0, HHG.period], [0, 80], 400, 400)
np.save('test_guess', HHG.guess_saddle_points)
#HHG.guess_saddle_points = np.load('test_guess.npy')

print('Cutoff: ', HHG.Ip + 3.17*HHG.Up)

fft_res, fft_omega = HHG.calculate_HHG_spectrum()
filter_list = fft_omega >= 0
fft_res = fft_res[filter_list]
fft_omega = fft_omega[filter_list]

plt.axvline(HHG.Ip + 3.17 * HHG.Up, ls='--', c='gray', alpha=0.3)
plt.plot(fft_omega, fft_omega**2 * np.abs(fft_res)**2)
#plt.xlim(0,1.19)
plt.yscale('log')
plt.show()
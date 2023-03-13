# %% LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from scipy.optimize import root
from matplotlib.colors import LogNorm
from scipy.integrate import quad
from multiprocessing import Pool

# %% ATI CLASS

class AboveThresholdIonization:
    def __init__(self, settings_dict=None):
        """
        Very descriptive description
        :param Ip: The ionization potential in a.u.
        """
        if settings_dict is None:  # No settings dictionary - use the standard values
            self.Ip = 0.5
            self.set_field_params(lambd=800, intensity=1e14, cep=np.pi/2, N_cycles=2)
            self.set_fields(build_in_field='sin2')  # Set the standard field type
            self.set_momentum_bounds(px_start=-1.5, px_end=1.5, py_start=0., py_end=1.5, pz=0., Nx=200, Ny=100)
            self.N_cores = 4
        else:
            self.Ip = settings_dict['Ip']
            self.set_field_params(lambd=settings_dict['Wavelength'], intensity=settings_dict['Intensity'],
                                  cep=settings_dict['cep'], N_cycles=settings_dict['N_cycles'])
            self.set_fields(build_in_field='sin2') #TODO: use custom field if neccesary
            self.set_momentum_bounds(px_start=settings_dict['px_start'], px_end=settings_dict['px_end'],
                                     py_start=settings_dict['py_start'], py_end=settings_dict['py_end'],
                                     pz=settings_dict['pz'], Nx=settings_dict['Nx'], Ny=settings_dict['Ny'])

            self.N_cores = settings_dict['N_cores']

        self.guess_saddle_points = np.array([])  # List to be filled with initial values for saddle point times

    def set_field_params(self, lambd, intensity, cep, N_cycles):
        """
        Function to set field parameters used for the standard build-in fields. If own field is provided, this is not
        needed.
        :param lambd: Laser carrier wavelength in nm
        :param intensity: Laser intensity in W/cm²
        :param cep: Carrier envelope phase
        :param N_cycles: Number of cycles
        """
        self.cep = cep
        self.N_cycles = N_cycles

        # Calculate omega and Up. Do the conversion to a.u.
        E_max = np.sqrt(intensity / 3.50945e16)  # Max electric field amplitude (a.u.)
        self.omega = 2. * np.pi * 137.036 / (lambd * 1.e-9 / 5.29177e-11)
        self.Up = E_max**2 / (4 * self.omega**2)
        self.rtUp = np.sqrt(self.Up)
        self.period = 2*np.pi * N_cycles/self.omega

    def set_fields(self, build_in_field='', custom_A_field=None, custom_E_field=None):
        """
        Function to set the field type used in calculations. Allows for custom fields. Note, if custom fields is used
        then both E and A field must be given.
        :param build_in_field: String choosing the type of in-built field. Leave '' if custom field is used.
        :param custom_A_field: Custom A-field. Must have form A_field(t), with t being time in a.u.
        :param custom_E_field: Custom E-field. Must have form E_field(t), with t being time in a.u.
        """
        self.build_in = ''
        if build_in_field:
            self.build_in = build_in_field
            if build_in_field == 'sin2':
                self.A_field = self.A_field_sin2
                self.E_field = self.E_field_sin2

                # More build-in fields can be added here:
            else:
                raise Exception('The build-in field specified does not exist!')

        elif custom_A_field is not None:
            self.A_field = custom_A_field
            self.E_field = custom_E_field
        else:
            raise Exception("Attempt at setting field type without anything provided!")

    def set_momentum_bounds(self, px_start, px_end, py_start, py_end, pz, Nx, Ny):
        """
        Function to set the bounds of the momentum values for which the transition amplitude will be calculated.
        :param px_start: Initial momentum value in the x direction of the momentum grid.
        :param px_end: Final momentum value in the x direction of the momentum gird.
        :param Nx: Nr. of momentum values in the x direction of the momentum grid
        :param pz: The value of momentum along z the calculations are performed for
        """
        self.px_start = px_start
        self.px_end = px_end
        self.py_start = py_start
        self.py_end = py_end
        self.pz = pz
        self.Nx = Nx
        self.Ny = Ny

    def A_field_sin2(self, t):
        """
        Sin^2 field. One possible choice of 'build in' pulse forms. Uses the field parameters of the class.
        :param t: Time (a.u.)
        """
        return np.array([2 * self.rtUp * np.sin(self.omega * t / (2*self.N_cycles))**2 * np.cos(self.omega * t + self.cep), 0, 0])

    def E_field_sin2(self, t):
        """
        Electric field for the sin2 field.
        """
        term1 = 2*np.sqrt(self.Up) * np.sin(self.omega*t/(2*self.N_cycles))**2 * np.sin(self.omega*t + self.cep)
        term2 = -np.sqrt(self.Up)/self.N_cycles * np.sin(self.omega*t/self.N_cycles) * np.cos(self.omega*t + self.cep)
        return np.array([self.omega * (term1 + term2), 0, 0])

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

    def A_integrals(self, t, p_vec):
        """
        Calculation of the vector potential integrals needed in the action. Redirects for analytical calculation or
        performs the numerical integration directly.
        :param t: time
        :param p_vec: momentum vector
        """
        if self.build_in:
            return self.analytic_A_integrals(t, p_vec)

        # This is if no analytical expression for the vector field integrals are known - quite a bit slower!
        t_list = np.linspace(0, t, 100)
        return np.trapz([2*np.dot(p_vec, self.A_field(t)) + np.sum(self.A_field(t)**2) for t in t_list], t_list)

    def analytic_A_integrals(self, t, p_vec):
        """
        Wrapper for the different analytical integrals of the vector potentials for the different field types
        :param t: time
        :param p_vec: momentum vector
        """
        if self.build_in == 'sin2':
            return 2*p_vec[0]*self.AI_sin2(t) + self.AI2_sin2(t)
        # One can add additional field types here
        else:
            raise Exception("Analytical integrals for this field type does not exist!")

    def action(self, t, p_vec):
        """
        Calculation of the SFA action
        :param t: time
        :param p_vec: momentum vector
        """
        p2 = p_vec[0]**2 + p_vec[1]**2 + p_vec[2]**2
        return (self.Ip + p2) * t + self.A_integrals(t, p_vec)

    def action_derivative(self, p_vec, ts):
        """
        Calculates the derivative of the SFA action
        :param p_vec: momentum vector
        :param ts:
        :return:
        """
        val = p_vec + self.A_field(ts)
        return 0.5 * (val[0]**2 + val[1]**2 + val[2]**2) + self.Ip  #np.linalg.norm(p_vec + self.A_field(ts))**2 + 2 * self.Ip

    def action_derivative_real_imag(self, ts_list, p_vec):
        """
        Simple wrapper function used to make scipy's root finder happy.
        """
        val = self.action_derivative(p_vec, ts_list[0] + 1j*ts_list[1])
        return [val.real, val.imag]

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
        p_vec = np.array([self.px_start, self.py_start, self.pz])

        # Should try to do this with numpy array magic
        for i, ti in enumerate(ti_list):
            for j, tr in enumerate(tr_list):
                res_grid[i, j] = self.action_derivative(p_vec, tr + 1j*ti)

        res_grid = np.log10(np.abs(res_grid) ** 2)
        fig, ax = plt.subplots()
        ax.imshow(np.flip(res_grid, 0), cmap='twilight', interpolation='bicubic', aspect='auto',
                   extent=(tr_lims[0], tr_lims[1], ti_lims[0], ti_lims[1]))
        klicker = clicker(ax, ["Saddle Points"], markers=["o"])
        plt.show()

        guess_sp = klicker.get_positions()['Saddle Points']
        self.guess_saddle_points = guess_sp[:,0] + 1j * guess_sp[:,1]

    def find_saddle_times(self, guess_times, p_vec):
        """
        Function to determine the saddle point times. Uses scipy's root function to determine the roots of the
        derivative of the action.
        """
        root_list = []
        for guess_time in guess_times:
            guess_i = np.array([guess_time.real, guess_time.imag])
            sol = root(self.action_derivative_real_imag, guess_i, args=(p_vec,))
            tr = sol.x[0]
            ti = sol.x[1]

            # Restrict real part within first period
            if tr > self.period:
                tr -= self.period

            root_list.append(tr + 1j*ti)
        return root_list

    def calculate_transition_amplitude(self, p_vec, saddle_times=None):
        """
        Function to calculate the transition amplitude in a given momentum point, given the saddle point times.
        :param saddle_times: list of complex saddle point times
        :param p_vec: the final momentum 3-vector
        :return: the transition amplitude M(p) for a given momentum p
        """
        #TODO: prefactors? Hell nah.

        p_vec = np.array(p_vec)  # Just to be sure...
        amplitude = 0
        if saddle_times is not None:
            # Use the saddle-point approximation
            for ts in saddle_times:
                action_double_derivative = np.dot(-(p_vec + self.A_field(ts)), self.E_field(ts))
                amplitude += np.sqrt(2*np.pi*1j/action_double_derivative) * np.exp(1j * self.action(ts, p_vec))
            return amplitude
        else:
            # Numerical integration of the time integral
            t_end = self.N_cycles * 2*np.pi / self.omega
            t_list = np.linspace(0, t_end, int(1e4)*self.N_cycles)
            amplitude = quad(lambda t: np.exp(-1j*self.action(t_list, p_vec)), 0, t_end)  # np.trapz(np.exp(-1j*self.action(t_list, p_vec)), t_list)
            return amplitude

    def calculate_pmd_py_slice_SPA(self, saddle_times_start, py):
        """
        Calculates the pmd for all px values at a given py value using the saddle point approximation.
        Done in this way to propagate the saddle time solutions through the momentum grid properly.
        """
        p_vec = np.array([0., py, self.pz])
        pmd_slice = []
        guess_points = saddle_times_start

        for i, px in enumerate(self.px_list):
            p_vec[0] = px

            # Obtain the saddle times if not the first px value (here they are already found)
            if i != 0:
                saddle_points = self.find_saddle_times(guess_points, p_vec)
                guess_points = saddle_points
            else:
                saddle_points = guess_points

            # Calculate the transition amplitude for the momentum point
            trans_amp = self.calculate_transition_amplitude(p_vec, saddle_times=saddle_points)
            pmd_slice.append(trans_amp)

        return pmd_slice

    def calculate_pmd_SPA(self):
        """
        Function to calculate the photoelectron momentum distribution using the saddle point approximation (SPA)
        for the momentum grid over px, py.
        """

        self.px_list = np.linspace(self.px_start, self.px_end, self.Nx)
        py_list = np.linspace(self.py_start, self.py_end, self.Ny)

        # Check the initial guess have been found by user - else make them find them!
        if not np.any(self.guess_saddle_points):
            raise Exception("I don't know what you are doing, and at this point I am too afraid to ask.")

        # Get the saddle points along the left px edge
        guess_points = self.guess_saddle_points
        edge_list = []
        p_vec = np.array([self.px_start, 0., self.pz])
        for py_i in py_list:
            p_vec[1] = py_i
            exact_points = self.find_saddle_times(guess_points, p_vec)
            edge_list.append(exact_points[:])  # Make copy just to be sure?
            guess_points = exact_points

        # Now loop over all the py-'slices' and calculate transition amplitude for each, finding saddle times for each
        # Done using multiprocessing starmap to speed up calculations a bit
        iter_param_list = [(edge_times_i, py_i) for edge_times_i, py_i in zip(edge_list,py_list)]
        with Pool(processes=4) as pool:
            pmd = pool.starmap(self.calculate_pmd_py_slice_SPA, iter_param_list)
        return np.array(pmd)


settings_dict = {
    'Ip': 0.5,              # Ionization potential (a.u.)
    'Wavelength': 800,      # (nm)
    'Intensity': 1e14,      # (W/cm^2)
    'cep': np.pi/2,         # Carrier envelope phase
    'N_cycles': 2,          # Nr of cycles
    'build_in_field': 'sin2',   # Build in field type to use. If using other field methods leave as a empty string ''.
    'px_start': -1.5, 'px_end': 1.5,  # Momentum bounds in x direction (a.u.)
    'py_start': 0., 'py_end': 1.5,    # Momentum bounds in y direction (a.u.)
    'pz': 0.,               # Momentum in z direction (a.u.)
    'Nx': 200, 'Ny': 100,   # Grid resolution in the x and y direction
    'N_cores': 4,           # Nr. of cores to use in the multiprocessing calculations
}

if __name__ == "__main__":
    ATI = AboveThresholdIonization(settings_dict=settings_dict)

    ATI.get_saddle_guess([0, ATI.N_cycles * 2*np.pi/ATI.omega], [0, 80], 400, 400)
    PMD = ATI.calculate_pmd_SPA()
    M = np.abs(PMD)**2
    plt.imshow(np.flip(M,0), norm=LogNorm(vmax=np.max(M), vmin=np.max(M)*1e-8), aspect='equal', extent=(ATI.px_start, ATI.px_end, ATI.py_start, ATI.py_end), interpolation='bicubic')
    plt.colorbar()

    plt.show()


"""
# Test integrals of A
t_list = np.linspace(0, ATI.N_cycles * 2*np.pi/ATI.omega, 100)
A_num_list = []
for ti in t_list:
    trapz_list = np.linspace(0, ti, 100)
    A_num_list.append(np.trapz(ATI.A_field(trapz_list), trapz_list))

plt.plot(t_list, ATI.AI_sin2(t_list))
plt.plot(t_list, np.array(A_num_list))
plt.show()
"""
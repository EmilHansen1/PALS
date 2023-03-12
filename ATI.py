# %% LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from scipy.optimize import root

# %% ATI CLASS

class AboveThresholdIonization:
    def __init__(self, settings_dict=None) -> None:
        """
        Very descriptive description
        :param Ip: The ionization potential in a.u.
        """
        if settings_dict is None:  # No settings dictionary - use the standard values
            self.Ip = 0.5
            self.set_field_params(lambd=800, intensity=1e14, cep=0, N_cycles=2)
            self.set_fields(build_in_field='sin2')  # Set the standard field type
            self.set_momentum_bounds(px_start=-2., px_end=2., py_start=-2., py_end=2., pz=0., Nx=100, Ny=100)
        else:
            pass  # Same as above but with the settings_dict values!

        self.guess_saddle_points = []  # List to be filled with initial values for saddle point times

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

    def set_fields(self, build_in_field='', custom_A_field=None, custom_E_field=None):
        """
        Function to set the field type used in calculations. Allows for custom fields. Note, if custom fields is used
        then both E and A field must be given.
        :param build_in_field: String choosing the type of in-built field. Leave '' if custom field is used.
        :param custom_A_field: Custom A-field. Must have form A_field(t), with t being time in a.u.
        :param custom_E_field: Custom E-field. Must have form E_field(t), with t being time in a.u.
        """
        if build_in_field:
            if build_in_field == 'sin2':
                self.A_field = self.A_field_sin2
                self.E_field = self.E_field_sin2

                # More build-in fields can be added here:

        elif custom_A_field is not None:
            self.A_field = custom_A_field
            self.E_field = custom_E_field
        else:
            print("Attempt at setting field type without anything provided! Don't panic! Using standard sin2 field.")

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
        return np.array([0, 0, 2 * self.rtUp * np.sin(self.omega * t / (2*self.N_cycles))**2 * np.cos(self.omega * t + self.cep)])

    def E_field_sin2(self, t):
        pass

    def action_derivative(self, p_vec, ts):
        val = p_vec + self.A_field(ts)
        return (val[0]**2 + val[1]**2 + val[2]**2) + 2 * self.Ip  #np.linalg.norm(p_vec + self.A_field(ts))**2 + 2 * self.Ip

    def action_derivative_real_imag(self, ts_list, p_vec):
        val = self.action_derivative(p_vec, ts_list[0] + 1j*ts_list[1])
        return [val.real, val.imag]

    def get_saddle_guess(self, tr_lims, ti_lims, Nr, Ni, p_vec):
        """
        Obtains the saddle point times through user input. User clicks on plot to identify the position of the saddle
        points.
        :param tr_lims: List of over and upper limit of real saddle time used in plot
        :param ti_lims: List of over and upper limit of imaginary saddle time used in plot
        :param Nr: Nr. of data points on real axis of plot
        :param Ni: Nr. of data points on imag axis of plot
        :param p_vec: Momentum vector for which saddle points are calculated
        """
        tr_list = np.linspace(tr_lims[0], tr_lims[1], Nr)
        ti_list = np.linspace(ti_lims[0], ti_lims[1], Ni)
        res_grid = np.zeros((len(ti_list), len(tr_list)), dtype=complex)

        # Should try to do this with numpy array magic
        for i, ti in enumerate(ti_list):
            for j, tr in enumerate(tr_list):
                res_grid[i,j] = self.action_derivative(p_vec, tr + 1j*ti)

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
            sol = root(self.action_derivative_real_imag, guess_i, args=(p_vec))

            # TODO Restrict solution to lie within the pulse

            root_list.append(sol.x[0] + 1j*sol.x[1])
        return root_list

    def calculate_transition_amplitude(self):
        """
        Function to calculate the transition amplitude in a given momentum point, given the saddle point times.
        """
        # TODO implement this
        pass

    def calculate_pmd_py_slice(self, saddle_times_start, px_list, py):
        """
        Calculates the pmd for all px values at a given py value. This is done to propagate the saddle time solution
        through the momentum grid properly.
        """
        p_vec = np.array([0., py, self.pz])
        pmd_slice = []
        guess_points = saddle_times_start
        for i, px in enumerate(px_list):
            p_vec[0] = px

            # Obtain the saddle times if not the first px value (here they are already found)
            if i != 0:
                saddle_points = self.find_saddle_times(guess_points, p_vec)
                guess_points = saddle_points
            else:
                saddle_points = guess_points

            # Calculate the transition amplitude for the momentum point
            trans_amp = self.calculate_transition_amplitude()
            pmd_slice.append(trans_amp)

        return pmd_slice

    def calculate_pmd(self):
        """
        Function to calculate the photoelectron momentum distribution for the set momentum grid over px,py.
        """
        pmd = []
        px_list = np.linspace(self.px_start, self.px_end, self.Nx)
        py_list = np.linspace(self.py_start, self.py_end, self.Ny)

        # Check the initial guess have been found by user - else make them find them!
        if not self.guess_saddle_points:
            # TODO throw some error here
            print("I don't know what you are doing, and at this point I am too afraid to ask.")

        # Get the saddle points along the left px edge
        guess_points = self.guess_saddle_points
        edge_list = []
        p_vec = np.array([px_list[0], 0., self.pz])
        for py_i in py_list:
            p_vec[1] = py_i
            exact_points = self.find_saddle_times(guess_points, p_vec)
            edge_list.append(exact_points[:])  # Make copy just to be sure?
            guess_points = exact_points

        # Now loop over all the py-'slices' and calculate transition amplitude for each, finding saddle times for each
        # (This should be done in parallel! And is thus made into its own function for ease of parallelization)
        for (edge_times_i, py) in zip(edge_list, py_list):
            pmd.append(self.calculate_pmd_py_slice(edge_times_i, px_list, py))

        return np.array(pmd)

settings_dict = {}  # This could be a very nice feature
ATI = AboveThresholdIonization()
#SP_guess = ATI.get_saddle_guess((0., 2*np.pi * ATI.N_cycles/ATI.omega), (0., 80), 400, 400)
#print(SP_guess)
#root_list = ATI.find_saddle_times(SP_guess, np.array([0.5, 0., 0.5]))
#print(root_list)

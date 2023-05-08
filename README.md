# StrongFieldTools
This is a small toolbox containing code for performing simple above threshold ionization (ATI) and high harmonic generation (HHG) calculations, within the strong field approximation (SFA). The code is designed to primarily work with the saddle point approximation (SPA). 

The code is organized into two classes, one for performing ATI calculations, and one for performing HHG calculations. The classes come with some standard field types, but one should be able to add extra quite simply (at least for the ATI class).

The classes are configured for different settings using a dictionary, which should be provided as a parameter when creating an instance of the class. If no dictionary is provided the code will run default settings. The settings can also be changed after instantiation by calling the different class members and changing their values.

Two example Jupyter notebooks are provided, which contain simple examples of how to generate spectra and photoelectron momentum distributions using the code. They also provide an example of the settings dictionary used (which can also be found at the bottom of the class files)

A small list of the most important methods from the two classes is found below.

## A short list of some relevant members (both classes):

    - Ip (ionization potential)
    - Up (pondermotive potential)
    - omega (angular frequency of carrier field (a.u.))
    - period (length of the laser pulse in time (a.u.))
    - cep (carrier-envelope phase)
    - N_cycles (nr. of cycles within the envelope of the field)
    - guess_saddle_points (set manually or using get_saddle_guess)


## ATI class

### Important methods

    get_saddle_guess(tr_lims, ti_lims, Nr, Ni)

- tr_lims: Real time limits of the guess plot.
- ti_lims: Imaginary time limits of the guess plot.
- Nr: Number of data points along the real axis of the plot. 
- Ni: Number of data points along the imaginary axis of the plot.

Function used to obtain the initial guess for the saddle point times, provided by the user by clicking on a graph window. These guesses are then used to find the saddle times for the rest of the momentum grid.


    calculate_PMD()

Function to calculate the photoelectron momentum distribution using the saddle point approximation (SPA) for the momentum grid over px, py.

    ATI_angular_dist(N_phi, pz=0, energy_bounds_Up=(0,3), N_energy_trapz=100)

- N_phi: Number of points to sample the angular distribution over.
- pz: Value of z momentum to perform the calculation for (usually assumed 0).
- energy_bounds_Up: List of minimum and maximum energy to sample the energy integral over (in units of Up)
- N_energy_trapz: Nr. of points to sample the energy integral over

Calculates the angular distribution of the ATI signal using the saddle point approximation.

    ATI_spectrum(energy_bounds_Up=(0,3), N_points=100, N_phi_samples=100)

- energy_bounds_Up: List of minimum and maximum energy to calculate for (in units of Up)
- N_points: Number of points to sample the spectrum over
- N_phi_samples: Number of points to sample the angular integral over

Function for calculating the ATI spectrum using the saddle point approximation.

    A_field(t)
    E_field(t)
- t: Time in atomic units.

Function to access the vector potential / electric field used selected. 


## HHG class

### Important methods

    get_saddle_guess(tr_lims, ti_lims, Nr, Ni)

- tr_lims: Real time limits of the guess plot.
- ti_lims: Imaginary time limits of the guess plot.
- Nr: Number of data points along the real axis of the plot. 
- Ni: Number of data points along the imaginary axis of the plot.

Function used to obtain the initial guess for the saddle point times, provided by the user by clicking on a graph window. These guesses are then used to find the saddle times for the rest of the momentum grid.

    calculate_HHG_spectrum()

Generates an HHG spectrum. Run times might be around 5 min. for a 2 cycles pulse, and gets longer for longer pulses. (Also depends on your CPU of course).  

    get_dipole()

Returns the dipole as a function of time over the pulse.

    get_spectrogram(width, N_every, max_harmonic_order, N_omega, times=None, dip=None)

- Width: Width of Gaussian window function. 
- N_every: Perform calculation for every N_every value of the dipole matrix element.
- max_harmonic_order: Maximum harmonic order in the spectrum.
- N_omega: Number of points in the energy grid between [0, max_harmonic_order * omega]
- times: List of recombination times.
- dip: List of dipole matrix element values.

Performs a Gabor transformation on the dipole signal. Is essentially a Fourirer transformation with a sliding time window. More information can be found on e.g. wikipedia.

## Required packages
The required packages are the standard ones used in the Python scientific community:
 - Numpy
 - Matplotlib
 - Scipy
 - Jupyter Notebook (for viewing the examples)
 - PyQt5 (or similar)
 
 These are most easily installed using pip.


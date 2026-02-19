from POSEIDON.constants import R_Sun, R_J, R_E, M_E
from POSEIDON.core import create_star, create_planet, load_data, define_model, \
                          wl_grid_constant_R, set_priors, read_opacities, make_atmosphere, \
                          compute_spectrum
from POSEIDON.visuals import plot_data, plot_spectra_retrieved, plot_PT_retrieved, \
                             plot_chem_retrieved, plot_spectra
from POSEIDON.retrieval import run_retrieval
from POSEIDON.utility import read_retrieved_spectrum, read_retrieved_PT, \
                             read_retrieved_log_X, plot_collection, write_spectrum
from POSEIDON.corner import generate_cornerplot
from POSEIDON.instrument import generate_syn_data_from_file

import numpy as np
import os
from pathlib import Path

from scipy.constants import parsec as pc

RUN_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = RUN_DIR / 'data'
os.chdir(RUN_DIR)


one_planet = True

#***** Model wavelength grid *****#

wl_min = 0.58      # Minimum wavelength (um)           0.58
wl_max = 5.30      # Maximum wavelength (um)           5.30
R = 20000          # Spectral resolution of grid

# We need to provide a model wavelength grid to initialise instrument properties
wl = wl_grid_constant_R(wl_min, wl_max, R)

#***** Define stellar properties *****#

R_s = 0.38*R_Sun      # Stellar radius (m)
T_s = 3506.0          # Stellar effective temperature (K)
err_T_s = 70          # Value in ExoMast
Met_s = -0.20         # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.872       # Stellar log surface gravity (log10(cm/s^2) by convention)


# Create the stellar object
# star = create_star(R_s, T_s, log_g_s, Met_s)
star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error = err_T_s, wl = wl)


#***** Define planet properties *****#

if one_planet:
    planet_name = 'TOI-270e'  # Planet name used for plots, output files etc.
else:
    planet_names = ['TOI-270e','TOI-270f','TOI-270g','TOI-270h','TOI-270i','TOI-270j','TOI-270k','TOI-270l','TOI-270m','TOI-270n',]  # Planet names used for plots, output files etc.
    planets = []
    colour_list = ['green', 'red', 'black', 'darkgrey', 'navy', 'brown', 'goldenrod', 'magenta', 'cyan', 'orange']
    list_of_spectra = []

R_p = 2.133*R_E     # Planetary radius (m)
if one_planet:
    # M_p = 4.78*M_E      # Planet mass
    masses = np.logspace(0.3010299956639812,1.3010299956639813,10)
    M_p = masses*M_E
T_eq = 387.8       # Equilibrium temperature (K)
d = 22.453*pc       # Distance to system (m)

if one_planet:
    # Create the planet object
    # planet = create_planet(planet_name, R_p, mass = M_p, gravity = g_p, T_eq = T_eq)
    planet = create_planet(planet_name, R_p, mass = M_p, T_eq = T_eq, d = d)


    #***** Define model *****#

    model_name = 'H2_H2O_Forward_Model'  # Model name used for plots, output files etc.

    bulk_species = ['H2', 'He']      # H2 + He comprises the bulk atmosphere
    param_species = ['H2O']   # The trace gases are H2O and CH4

    # Create the model object
    model = define_model(model_name, bulk_species, param_species, 
                        PT_profile = 'isotherm', cloud_model = 'cloud-free',
                        radius_unit = 'R_E', surface = False,
                        stellar_contam = None)


                        # Specify the pressure grid of the atmosphere
    P_min = 1.0e-7    # 0.1 ubar
    P_max = 100       # 100 bar
    N_layers = 100    # 100 layers

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure and radius
    P_ref = 10.0   # Reference pressure (bar)
    R_p_ref = R_p  # Radius at reference pressure

    # Provide a specific set of model parameters for the atmosphere 
    PT_params = np.array([1000])              # T (K)
    log_X_params = np.array([-3.3])     # log(H2O)

    # Generate the atmosphere
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                PT_params, log_X_params)


    #***** Read opacity data *****#

    opacity_treatment = 'opacity_sampling'

    # First, specify limits of the fine temperature and pressure grids for the 
    # pre-interpolation of cross sections. These fine grids should cover a
    # wide range of possible temperatures and pressures for the model atmosphere.

    # Define fine temperature grid (K)
    T_fine_min = 100     # 400 K lower limit suffices for a typical hot Jupiter
    T_fine_max = 1000    # 2000 K upper limit suffices for a typical hot Jupiter
    T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
    log_P_fine_max = 2.0    # 100 bar is the highest pressure in the opacity database
    log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                        log_P_fine_step)

    # Now we can pre-interpolate the sampled opacities (may take up to a minute)
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)


    # Generate our first transmission spectrum
    spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                spectrum_type = 'transmission')

    # Write to file
    spectrum_file = write_spectrum(planet_name, model_name, spectrum, wl)


    # Add the spectrum we want to plot to an empty spectra plot collection
    spectra = plot_collection(spectrum, wl, collection = [])

    # Produce figure and save to file
    fig = plot_spectra(spectra, planet, R_to_bin = 100, plt_label = planet_name)

for i in range(0,len(M_p)):
    # print(i)
    # Create the planet object
    # planet = create_planet(planet_name, R_p, mass = M_p, gravity = g_p, T_eq = T_eq)
    planet = create_planet(planet_names[i], R_p, mass = M_p[i], T_eq = T_eq, d = d)
    planets.append(planet)

    #***** Define model *****#

    model_name = 'H2_H2O_Forward_Model_' + str(i+1)  # Model name used for plots, output files etc.

    bulk_species = ['H2', 'He']      # H2 + He comprises the bulk atmosphere
    param_species = ['H2O']   # The trace gases are H2O and CH4

    # Create the model object
    model = define_model(model_name, bulk_species, param_species, 
                        PT_profile = 'isotherm', cloud_model = 'cloud-free',
                        radius_unit = 'R_E', surface = False,
                        stellar_contam = None)


                        # Specify the pressure grid of the atmosphere
    P_min = 1.0e-7    # 0.1 ubar
    P_max = 100       # 100 bar
    N_layers = 100    # 100 layers

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure and radius
    P_ref = 10.0   # Reference pressure (bar)
    R_p_ref = R_p  # Radius at reference pressure

    # Provide a specific set of model parameters for the atmosphere 
    PT_params = np.array([1000])              # T (K)
    # log_X_params = np.array([-3.3])     # log(H2O)
    # log_X_params = np.array([-0.0004148557908770917, -0.0004148557908770917, -0.12483854208890861, -0.5075795689696349, -1.001908614024893,
    #                          -1.640359126573138, -2.464949672183401, -3.529949315140789, -4.905449247243431, -6.681975723746822])     # log(H2O)
    log_X_params = np.array([[-0.0004148557908770917], [-0.0004148557908770917], [-0.12483854208890861], [-0.5075795689696349], [-1.001908614024893],
                             [-1.640359126573138], [-2.464949672183401], [-3.529949315140789], [-4.905449247243431], [-6.681975723746822]])     # log(H2O)


    # Generate the atmosphere
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                PT_params, log_X_params[i])

    if i == 0:
        #***** Read opacity data *****#

        opacity_treatment = 'opacity_sampling'

        # First, specify limits of the fine temperature and pressure grids for the 
        # pre-interpolation of cross sections. These fine grids should cover a
        # wide range of possible temperatures and pressures for the model atmosphere.

        # Define fine temperature grid (K)
        T_fine_min = 100     # 400 K lower limit suffices for a typical hot Jupiter
        T_fine_max = 1000    # 2000 K upper limit suffices for a typical hot Jupiter
        T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

        T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

        # Define fine pressure grid (log10(P/bar))
        log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
        log_P_fine_max = 2.0    # 100 bar is the highest pressure in the opacity database
        log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

        log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                            log_P_fine_step)

        # Now we can pre-interpolate the sampled opacities (may take up to a minute)
        opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)


    # Generate our first transmission spectrum
    spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                spectrum_type = 'transmission')
    list_of_spectra.append(spectrum)

    # Write to file
    spectrum_file = write_spectrum(planet_names[i], model_name, spectrum, wl)


    # Add the spectrum we want to plot to an empty spectra plot collection
    if i == 0:
        spectra = plot_collection(spectrum, wl, collection = [])
    else:
        spectra = plot_collection(spectrum, wl, collection = spectra)

    # # Produce figure
    # fig_spec = plot_spectra(spectra, planet, R_to_bin = 100,
    #                         plot_full_res = False,
    #                         spectra_labels = ['First model', 'No CH$_4$'])
    # Produce figure and save to file
    fig = plot_spectra(spectra, planet, R_to_bin = 100, plt_label = planet_names[i])

# Produce figure
fig_spec = plot_spectra(spectra, planet, R_to_bin = 100,
                        plot_full_res = False, colour_list = colour_list,
                        spectra_labels = planet_names, plt_label = "Full Range of Masses")
# , y_min = 0.25e-2, y_max = 1.75e-2
print(spectra[i])

data_included = 'NIRISS_G395H_Tiberius'
planet_name = 'TOI-270n'  # Planet name used for plots, output files etc.
model_name = 'H2_H2O_Forward_Model'

data_dir = str(DATA_ROOT / planet_name)

datasets = []
instruments = []

if ('NIRISS' in data_included):

    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_ExoTEP.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_ExoTEP.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')

if ('G395H_Eureka-feature' in data_included):

    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Eureka.dat',)
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Eureka-CS2.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('G395H_Eureka' in data_included):

    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Eureka.dat',)
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Eureka.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('G395H_Tiberius-feature' in data_included):

    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Tiberius.dat',)
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Tiberius-CS2.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('G395H_Tiberius' in data_included):

    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Tiberius.dat',)
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Tiberius.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')

data = load_data(data_dir, datasets, instruments, wl)
print(data)

# Specify number of transits observed by each instrument
N_trans = [1, 1, 1, 1]

# Specify spectral resolution for binning each raw dataset
R_to_bin = [None, None, None, None]   # Let's bin down to R = 100 for clarity and retrieval speed

Gauss_scatter = True

for i in range(0,len(planets)):
    model_name = 'H2_H2O_Forward_Model_' + str(i+1)  # Model name used for plots, output files etc.

    # Generate simulated data for the model using errors from data file
    generate_syn_data_from_file(planets[i], wl, spectra[i], data_dir, data, R_to_bin = R_to_bin,
                                N_trans = N_trans, label = model_name, Gauss_scatter = Gauss_scatter)

    if Gauss_scatter == True:
        datasets_new = [planet_names[i] + '_SYNTHETIC_JWST_NIRISS_SOSS_Ord2_' + model_name + '_N_trans_' + str(N_trans[0]) + '.dat',
                        planet_names[i] + '_SYNTHETIC_JWST_NIRISS_SOSS_Ord1_' + model_name + '_N_trans_' + str(N_trans[1]) + '.dat',
                        planet_names[i] + '_SYNTHETIC_JWST_NIRSpec_G395H_NRS1_' + model_name + '_N_trans_' + str(N_trans[2]) + '.dat',
                        planet_names[i] + '_SYNTHETIC_JWST_NIRSpec_G395H_NRS2_' + model_name + '_N_trans_' + str(N_trans[3]) + '.dat']
    else:
        datasets_new = [planet_names[i] + '_SYNTHETIC_JWST_NIRISS_SOSS_Ord2_' + model_name + '_N_trans_' + str(N_trans[0]) + '_no_gauss.dat',
                    planet_names[i] + '_SYNTHETIC_JWST_NIRISS_SOSS_Ord1_' + model_name + '_N_trans_' + str(N_trans[1]) + '_no_gauss.dat',
                    planet_names[i] + '_SYNTHETIC_JWST_NIRSpec_G395H_NRS1_' + model_name + '_N_trans_' + str(N_trans[2]) + '_no_gauss.dat',
                    planet_names[i] + '_SYNTHETIC_JWST_NIRSpec_G395H_NRS2_' + model_name + '_N_trans_' + str(N_trans[3]) + '_no_gauss.dat']

    data_new = load_data(data_dir, datasets_new, instruments, wl)


    # Plot each one individually here
    # Create plot collection
    spectrum_collection = []   
    spectrum_collection = plot_collection(list_of_spectra[i], wl, collection = spectra)

    # Produce figure and save to file
    fig_spec = plot_spectra(spectrum_collection, planet, data_properties = data_new, R_to_bin = 100, 
                            plot_full_res = False, show_data = True,
                            spectra_labels = ['Model'], colour_list = ['navy'], 
                            data_labels = ['NIRISS SOSS Ord2', 'NIRISS SOSS Ord1',
                                        'NIRSpec G395H NRS1', 'NIRSpec G395H NRS2'],
                            data_colour_list = ['royalblue', 'forestgreen',
                                                'orange', 'crimson'],
                            y_min = 2.7e-3, y_max = 3.2e-3,
                            figure_shape = 'wide', wl_axis = 'linear',
                            show_data_cap=False,
                            plt_label = 'Simulated JWST Data vs. True Model')
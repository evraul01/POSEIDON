from POSEIDON.constants import R_Sun, R_J, R_E, M_E
from POSEIDON.core import create_star, create_planet, load_data, define_model, \
                          wl_grid_constant_R, set_priors, read_opacities
from POSEIDON.visuals import plot_data, plot_spectra_retrieved, plot_PT_retrieved, \
                             plot_chem_retrieved
from POSEIDON.retrieval import run_retrieval
from POSEIDON.utility import read_retrieved_spectrum, read_retrieved_PT, \
                             read_retrieved_log_X, plot_collection
from POSEIDON.corner import generate_cornerplot

import numpy as np

from scipy.constants import parsec as pc

do_retrieval = True

#data_included = 'NIRISS'
#data_included = 'G395H'
data_included = 'NIRISS_G395H'

#***** Model wavelength grid *****#

wl_min = 0.58      # Minimum wavelength (um)           2.8
wl_max = 5.30      # Maximum wavelength (um)           5.3
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
star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error = err_T_s, wl = wl)
#star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error = err_T_s, 
#                   stellar_grid = 'phoenix', interp_backend = 'pymsg', wl = wl)

#***** Define planet properties *****#

planet_name = 'TOI-270d'  # Planet name used for plots, output files etc.

R_p = 2.133*R_E     # Planetary radius (m)
M_p = 4.78*M_E      # Planet mass
T_eq = 387.8       # Equilibrium temperature (K)
d = 22.453*pc       # Distance to system (m)

# Create the planet object
planet = create_planet(planet_name, R_p, mass = M_p, T_eq = T_eq, d = d)

#***** Specify data location and instruments *****#

data_dir = './data/' + planet_name

datasets = []
instruments = []

if ('NIRISS' in data_included):

    datasets.append(planet_name + '_NIRISS_SOSS_Ord2.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')

if ('G395H' in data_included):

    datasets.append(planet_name + '_NIRSpec_G395H_NRS1.dat',)
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')

data = load_data(data_dir, datasets, instruments, wl)

#***** Define model *****#

model_name = 'HCN_H2S_SOSS_' + data_included

bulk_species = ['H2', 'He']     # H2 + He comprises the bulk atmosphere

param_species = ['H2O', 'CH4', 'CO', 'CO2', 'HCN', 'H2S']

# Create the model object
model = define_model(model_name, bulk_species, param_species, 
                     PT_profile = 'isotherm', cloud_model = 'MacMad17',
                     cloud_type = 'deck_haze', cloud_dim = 1,
                     radius_unit = 'R_E', surface = False,
                     stellar_contam = None, 
                     )
    
#***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types['R_p_ref'] = 'uniform'
prior_types['T'] = 'uniform'
prior_types['log_X'] = 'uniform'
prior_types['log_a'] = 'uniform'
prior_types['gamma'] = 'uniform'
prior_types['log_P_cloud'] = 'uniform'
prior_types['f_cloud'] = 'uniform'
prior_types['T_het'] = 'uniform'
prior_types['f_het'] = 'uniform'
prior_types['T_phot'] = 'gaussian'

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges['R_p_ref'] = [0.60*R_p, 1.15*R_p]
prior_ranges['T'] = [100, 1000]
prior_ranges['a1'] = [0.02, 2.0]
prior_ranges['a2'] = [0.02, 2.0]
prior_ranges['log_P1'] = [-7, 2]
prior_ranges['log_P2'] = [-7, 2]
prior_ranges['log_P3'] = [-2, 2]
prior_ranges['T_deep'] = [100, 1000]
prior_ranges['log_X'] = [-12, 0.0]
prior_ranges['log_a'] = [-4, 8]
prior_ranges['gamma'] = [-20, 2]
prior_ranges['log_P_cloud'] = [-7, 2]
prior_ranges['f_cloud'] = [0, 1]
prior_ranges['f_het'] = [0.0, 0.5]
prior_ranges['T_het'] = [2300, 1.2*T_s]
prior_ranges['T_phot'] = [T_s, err_T_s]

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

#***** Read opacity data *****#

opacity_treatment = 'opacity_sampling'

# Define fine temperature grid (K)
T_fine_min = 100     # Same as prior range for T
T_fine_max = 1000    # Same as prior range for T
T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2.0    # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                       log_P_fine_step)

#***** Specify fixed atmospheric settings for retrieval *****#

# Atmospheric pressure grid
P_min = 1.0e-7   # 10 nbar
P_max = 100       # 10 bar
N_layers = 100   # 100 layers

# Let's space the layers uniformly in log-pressure
P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure
P_ref = 10.0   # Retrieved R_p_ref parameter will be the radius at 10 bar

#***** Run atmospheric retrieval *****#

# Run atmospheric retrieval
if (do_retrieval == True):

    # Pre-interpolate the opacities
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

    run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R = R, 
                  spectrum_type = 'transmission', sampling_algorithm = 'MultiNest', 
                  N_live = 200, verbose = True, resume = False)

#***** Make corner plot *****#

fig_corner = generate_cornerplot(planet, model)

#***** Plot retrieved transmission spectrum *****#

# Read retrieved spectrum confidence regions
wl, spec_low2, spec_low1, spec_median, \
spec_high1, spec_high2 = read_retrieved_spectrum(planet_name, model_name)

# Create composite spectra objects for plotting
spectra_median = []
spectra_low2 = []
spectra_low1 = []
spectra_high1 = []
spectra_high2 = []

# Add retrieved spectra to composite objects
spectra_median = plot_collection(spec_median, wl, collection = spectra_median)
spectra_low1 = plot_collection(spec_low1, wl, collection = spectra_low1) 
spectra_low2 = plot_collection(spec_low2, wl, collection = spectra_low2) 
spectra_high1 = plot_collection(spec_high1, wl, collection = spectra_high1) 
spectra_high2 = plot_collection(spec_high2, wl, collection = spectra_high2)

# Plot retrieved spectra

if (data_included == 'NIRISS'):
    data_colour_list = ['navy', 'forestgreen']
    data_labels = ['NIRISS SOSS (Ord 2)', 'NIRISS SOSS (Ord 1)']
    data_marker_list = ['o', 'o']
    data_marker_size_list = [5, 5]
elif (data_included == 'G395H'):
    data_colour_list = ['crimson', 'orange']
    data_labels = ['NIRSPec G395H (NRS1)', 'NIRSPec G395H (NRS2)']
    data_marker_list = ['o', 'o']
    data_marker_size_list = [5, 5]
else:
    data_colour_list = ['navy', 'forestgreen', 
                        'crimson', 'orange']
    data_labels = ['NIRISS SOSS (Ord 2)', 'NIRISS SOSS (Ord 1)',
                   'NIRSPec G395H (NRS1)', 'NIRSPec G395H (NRS2)']
    data_marker_list = ['o', 'o', 'o', 'o']
    data_marker_size_list = [5, 5, 5, 5]

fig_spec = plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1, 
                                spectra_high1, spectra_high2, planet_name,
                                data, R_to_bin = 100, show_ymodel = False,
                                plt_label = model_name,
                                data_colour_list = data_colour_list,
                                data_labels = data_labels,
                                data_marker_list = data_marker_list,
                                data_marker_size_list = data_marker_size_list,
                                figure_shape = 'wide', wl_axis = 'linear',
                                wl_min = wl_min, wl_max = wl_max,
                                )

#***** Plot retrieved P-T profile *****#

# Read retrieved P-T profile confidence regions
P, T_low2, T_low1, \
T_median, T_high1, T_high2 = read_retrieved_PT(planet_name, model_name)

# Create composite P-T objects for plotting
PT_median = []
PT_low2 = []
PT_low1 = []
PT_high1 = []
PT_high2 = []

# Add retrieved spectra to composite objects
PT_median = plot_collection(T_median, P, collection = PT_median)
PT_low1 = plot_collection(T_low1, P, collection = PT_low1) 
PT_low2 = plot_collection(T_low2, P, collection = PT_low2) 
PT_high1 = plot_collection(T_high1, P, collection = PT_high1) 
PT_high2 = plot_collection(T_high2, P, collection = PT_high2)

# Produce figure
fig_PT = plot_PT_retrieved(planet_name, PT_median, PT_low2, PT_low1, PT_high1,
                           PT_high2, plt_label = model_name,
                           PT_labels = ['P-T profile'])

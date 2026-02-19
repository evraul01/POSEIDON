from POSEIDON.constants import R_Sun, R_J, R_E, M_E
from POSEIDON.core import create_star, create_planet, load_data, define_model, \
                          wl_grid_constant_R, read_opacities, compute_spectrum
from POSEIDON.utility import read_retrieved_spectrum, read_retrieved_PT, \
                             plot_collection, write_spectrum
from POSEIDON.corner import generate_cornerplot
from POSEIDON.visuals import plot_spectra_retrieved, plot_PT_retrieved, plot_histograms
from POSEIDON.retrieval import get_retrieved_atmosphere, Bayesian_model_comparison
from POSEIDON.stellar import stellar_contamination


import numpy as np
from spectres import spectres

from scipy.constants import parsec as pc

# data_included = 'NIRISS_Visit1_exoTEDRF_R100'
data_included = 'NIRSpec_Stacked_exoTEDRF_R100'

model_name = 'multigas_OS_2000_' + data_included

bulk_species = ['H2', 'He']     # H2 + He comprises the bulk atmosphere
param_species = ['H2O', 'CH4', 'CO', 'HCN', 'NH3', 'H2S', 'CO2', 'SO2']

stellar_contam = None
if '_OS_' in model_name:
    stellar_contam = 'one_spot'
elif '_OSfree_' in model_name:
    stellar_contam = 'one_spot_free_log_g'
if '_TS_' in model_name:
    stellar_contam = 'two_spots'
elif '_TSfree_' in model_name:
    stellar_contam = 'two_spots_free_log_g'

#***** Model wavelength grid *****#

if 'joint' in model_name:
    wl_min = 0.5      # Minimum wavelength (um)
    wl_max = 5.4      # Maximum wavelength (um)
elif 'NIRSpec' in data_included:
    wl_min = 2.7
    wl_max = 5.4
elif 'NIRISS' in data_included:
    wl_min = 0.5
    wl_max = 2.9
R = 20000          # Spectral resolution of grid

# We need to provide a model wavelength grid to initialise instrument properties
wl = wl_grid_constant_R(wl_min, wl_max, R)

#***** Define stellar properties *****#

R_s = 0.516*R_Sun      # Stellar radius (m)
T_s = 3556.0          # Stellar effective temperature (K)
err_T_s = 70          # Value in ExoMast
Met_s = -0.06         # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.727       # Stellar log surface gravity (log10(cm/s^2) by convention)
err_log_g_s = 0.029

print("Job started!")

# Create the stellar object
#star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error = err_T_s, 
                   #stellar_grid = 'phoenix', wl = wl)
star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error = err_T_s, 
                  stellar_grid = 'phoenix', interp_backend = 'pymsg', wl = wl)
#star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error = err_T_s, 
#                   log_g_error = err_log_g_s, # stellar_grid = 'cbk04')
#                   stellar_grid = 'phoenix', interp_backend = 'pymsg',
#                   wl = wl)

#***** Define planet properties *****#

planet_name = 'GJ-3090b'  # Planet name used for plots, output files etc.

R_p = 2.13*R_E     # Planetary radius (m)
M_p = 3.34*M_E      # Planet mass
T_eq = 693       # Equilibrium temperature (K)
d = 22.4751*pc       # Distance to system (m)

# Create the planet object
planet = create_planet(planet_name, R_p, mass = M_p, T_eq = T_eq, d = d)

#***** Specify data location and instruments *****#

data_dir = './data/' + planet_name
#data_dir = './data/' + planet_name + '/Old Data'

datasets = []
instruments = []

if ('NIRISS_Visit1_exoTEDRF_R50R25' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit1_exoTEDRF_R50R25.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit1_exoTEDRF_R50R25.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Visit1_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit1_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit1_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Visit2_exoTEDRF_R50R25' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit2_exoTEDRF_R50R25.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit2_exoTEDRF_R50R25.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Visit2_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit2_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit2_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Visit1_NAMELESS_R50' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit1_NAMELESS_R50.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit1_NAMELESS_R50.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Visit2_NAMELESS_R50' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit2_NAMELESS_R50.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit2_NAMELESS_R50.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Stacked_exoTEDRF_R50R25' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Stacked_exoTEDRF_R50R25.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Stacked_exoTEDRF_R50R25.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Stacked_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Stacked_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Stacked_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Combined_exoTEDRF_R50R25' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit1_exoTEDRF_R50R25.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit1_exoTEDRF_R50R25.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit2_exoTEDRF_R50R25.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit2_exoTEDRF_R50R25.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
elif ('NIRISS_Combined_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit1_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit1_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord2_Visit2_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRISS_SOSS_Ord1_Visit2_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')
    instruments.append('JWST_NIRISS_SOSS_Ord2')
    instruments.append('JWST_NIRISS_SOSS_Ord1')

if ('NIRSpec_Visit1_exoTEDRF_R250' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit1_exoTEDRF_R250.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit1_exoTEDRF_R250.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Visit1_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit1_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit1_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Visit2_exoTEDRF_R250' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit2_exoTEDRF_R250.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit2_exoTEDRF_R250.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Visit2_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit2_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit2_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Stacked_exoTEDRF_R250' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Stacked_exoTEDRF_R250.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Stacked_exoTEDRF_R250.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Stacked_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Stacked_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Stacked_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Stacked_Eureka_R250' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Stacked_Eureka_R250.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Stacked_Eureka_R250.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Stacked_Eureka_Fixed_R100' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Stacked_Eureka_Fixed_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Stacked_Eureka_Fixed_R100.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Stacked_Eureka_Free_R100' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Stacked_Eureka_Free_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Stacked_Eureka_Free_R100.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Combined_exoTEDRF_R250' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit1_exoTEDRF_R250.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit1_exoTEDRF_R250.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit2_exoTEDRF_R250.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit2_exoTEDRF_R250.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
elif ('NIRSpec_Combined_exoTEDRF_R100' in data_included):
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit1_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit1_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS1_Visit2_exoTEDRF_R100.dat')
    datasets.append(planet_name + '_NIRSpec_G395H_NRS2_Visit2_exoTEDRF_R100.dat')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')
    instruments.append('JWST_NIRSPec_G395H_NRS1')
    instruments.append('JWST_NIRSPec_G395H_NRS2')


data = load_data(data_dir, datasets, instruments, wl,
                #  offset_1_datasets = [datasets[2], datasets[3]],
                #  offset_2_datasets = [datasets[4], datasets[5]],
                #  offset_3_datasets = [datasets[6], datasets[7]],
                 )

# data = load_data(data_dir, datasets, instruments, wl,
#                  offset_1_datasets = [datasets[2], datasets[3]],
#                  )


#***** Define model *****#

cloud_model = 'MacMad17'
surface = False
disable_atmosphere = False
if 'flat_' in model_name:
    param_species = []
    surface = True
    disable_atmosphere = True

cloud_type = 'deck_haze'
if 'no-haze_' in model_name:
    cloud_type = 'deck'

if 'clear_' in model_name:
    cloud_model = 'cloud-free'

# Create the model object
model = define_model(model_name, bulk_species, param_species, 
                     PT_profile = 'isotherm', cloud_model = cloud_model,
                     cloud_type = cloud_type, cloud_dim = 1,
                     radius_unit = 'R_E', surface = surface,
                     disable_atmosphere = disable_atmosphere,
                     stellar_contam = stellar_contam, 
                     offsets_applied = None
                     )

# Create composite spectra objects for plotting
spectra_median = []
spectra_low2 = []
spectra_low1 = []
spectra_high1 = []
spectra_high2 = []

# Read retrieved spectrum confidence regions
wl, spec_low2, spec_low1, spec_median, \
spec_high1, spec_high2 = read_retrieved_spectrum(planet_name, model_name)

# Add retrieved spectra to composite objects
spectra_median = plot_collection(spec_median, wl, collection = spectra_median)
spectra_low1 = plot_collection(spec_low1, wl, collection = spectra_low1) 
spectra_low2 = plot_collection(spec_low2, wl, collection = spectra_low2) 
spectra_high1 = plot_collection(spec_high1, wl, collection = spectra_high1) 
spectra_high2 = plot_collection(spec_high2, wl, collection = spectra_high2)


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

# Pre-interpolate the opacities
opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)
print("Done Reading Opacities!")

#***** Make Best-fitting Model *****#

atmosphere_best = get_retrieved_atmosphere(planet, model, P, P_ref_set = P_ref)
print("Done Creating Atmosphere Dictionary!")

# REPLACE WITH YOUR BEST-FITTING VALUES
# f_het = 0.14
# T_het = 2773.4
# T_phot = 3553.7

# Create the best-fitting stellar object
# star_best = create_star(R_s, T_phot, log_g_s, Met_s, T_eff_error = err_T_s, 
#                         interp_backend = 'pymsg', stellar_grid = 'PHOENIX', stellar_contam = 'one_spot',
#                         f_het = f_het, T_het = T_het)
print("Done Creating Star!")

spectrum_best = compute_spectrum(planet, star, model, atmosphere_best, opac, wl,
                                 spectrum_type = 'transmission', save_spectrum = True)
# spectrum_best = compute_spectrum(planet, star_best, model, atmosphere_best, opac, wl,
#                                  spectrum_type = 'transmission', save_spectrum = True)
print("Done Computing Spectrum!")

# Compute wavelength-dependent stellar contamination factor
# epsilon_best = stellar_contamination(star_best, wl)
# print("Done Computing Wavelength-Dependent Stellar Contamination Factor!")
# print("eps_best:", epsilon_best)

# Produce atmosphere * TLS spectrum
spectrum_best = spectrum_best
# spectrum_best = spectrum_best * epsilon_best
print("Done Producing Best Spectrum!")

# Write to file
spectrum_file = write_spectrum(planet_name, model_name, spectrum_best, wl)
print("All Done!")
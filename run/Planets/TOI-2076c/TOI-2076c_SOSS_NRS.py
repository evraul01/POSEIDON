from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_E, M_E
from POSEIDON.core import load_data, wl_grid_constant_R
from POSEIDON.core import define_model
from POSEIDON.core import set_priors
from POSEIDON.core import read_opacities
from POSEIDON.retrieval import run_retrieval
from POSEIDON.utility import read_retrieved_spectrum
from POSEIDON.corner import generate_cornerplot
from POSEIDON.utility import bin_spectrum

import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

#***** Model wavelength grid *****#
wl_min = 0.8      # Minimum wavelength (um)
wl_max = 4.7      # Maximum wavelength (um)
R = 10000         # Spectral resolution of grid

# We need to provide a model wavelength grid to initialise instrument properties
wl = wl_grid_constant_R(wl_min, wl_max, R)


#***** Define stellar properties *****#
R_s = 0.7960*R_Sun     # Stellar radius (m)
T_s = 5192.0           # Stellar effective temperature (K)
err_T_s = 61           # Error in Stellar effective temperature (K)
Met_s = 0.02           # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.566        # Stellar log surface gravity (log10(cm/s^2) by convention)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error=err_T_s, interp_backend='pysynphot', 
                   stellar_grid='phoenix', wl=wl)

#***** Define planet properties *****#
planet_name = 'TOI-2076c'  
model_name = 'TOI-2076c_SOSS-NRS_LowRes'  

R_p = 3.54*R_E      # Planetary radius (m) -- Wang+ 26
m_p = 7.2*M_E        # Gravitational field of planet (m/s^2) -- Assuming Wang+ 26 mass
T_eq = 581          # Equilibrium temperature (K)

# Create the planet object
planet = create_planet(planet_name, R_p, mass=m_p, T_eq=T_eq)


#***** Specify data location and instruments  *****#
data_dir = 'data/TOI-2076c/'         
datasets = ['TOI-2076c_NIRISS_SOSS_R50.csv',
            'TOI-2076c_NIRSpec_G395H_NRS1_R100.csv',
            'TOI-2076c_NIRSpec_G395H_NRS2_R100.csv']  
instruments = ['NIRISS_SOSS', 'NIRSPEC_G395H_NRS1', 'NIRSPEC_G395H_NRS2']            

# Load dataset, pre-load instrument PSF and transmission function
data = load_data(data_dir, datasets, instruments, wl,
                 offset_1_datasets=['TOI-2076c_NIRSpec_G395H_NRS1_R100.csv'],
                 offset_2_datasets=['TOI-2076c_NIRSpec_G395H_NRS2_R100.csv'])

#***** Define model *****#
bulk_species = ['H2', 'He']    
param_species = ['C2H2', 'C2H4', 'CH4', 'CO', 'CO2', 'H2O', 'H2S', 'HCN', 'K', 'NH3', 'NO', 'NO2', 'Na', 'O2', 'O3', 'OCS', 'OH', 'PH3', 'SO2'] 

# Create the model object
model = define_model(model_name, bulk_species, param_species, PT_profile='isotherm', X_profile='chem_eq',
                     cloud_model='MacMad17', cloud_type='deck_haze', cloud_dim=1,
                     offsets_applied='two_datasets', disable_atmosphere=False, radius_unit='R_E', mass_unit='M_E',
                     stellar_contam='two_spots_free_log_g', mass_setting='free')

#***** Set priors for retrieval *****#
prior_types = {}
# Star
prior_types['T_phot'] = 'gaussian'
prior_types['log_g_phot'] = 'gaussian'
prior_types['f_spot'] = 'uniform'
prior_types['f_fac'] = 'uniform'
prior_types['T_spot'] = 'uniform'
prior_types['T_fac'] = 'uniform'
prior_types['log_g_fac'] = 'uniform'
prior_types['log_g_spot'] = 'uniform'
# Planet
prior_types['T'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['M_p'] = 'gaussian'
prior_types['log_Met'] = 'uniform'
prior_types['C_to_O'] = 'uniform'
prior_types['log_a'] = 'uniform'
prior_types['gamma'] = 'uniform'
prior_types['log_P_cloud'] = 'uniform'
prior_types['delta_rel_1'] = 'uniform'      
prior_types['delta_rel_2'] = 'uniform' 

prior_ranges = {}
# Star
prior_ranges['T_phot'] = [T_s, err_T_s]
prior_ranges['log_g_phot'] = [4.56, 0.02]
prior_ranges['f_spot'] = [0.0, 0.5]
prior_ranges['f_fac'] = [0.0, 0.5]
prior_ranges['log_g_fac'] = [3.5, 4.6]
prior_ranges['log_g_spot'] = [4.4, 5.0]
prior_ranges['T_spot'] = [2400, T_s+3*err_T_s]
prior_ranges['T_fac'] = [T_s-3*err_T_s, 1.2*T_s]
# Planet
prior_ranges['T'] = [400, 1600]
prior_ranges['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges['M_p'] = [7.2*M_E, 1.4*M_E]
prior_ranges['log_Met'] = [-1, 4]
prior_ranges['C_to_O'] = [0.2, 1.5]
prior_ranges['log_a'] = [-5, 10]
prior_ranges['gamma'] = [-20, 1]
prior_ranges['log_P_cloud'] = [-6, 6]
prior_ranges['delta_rel_1'] = [-5000, 5000]       
prior_ranges['delta_rel_2'] = [-5000, 5000]  

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

#***** Read opacity data *****#
opacity_treatment = 'opacity_sampling'

# Define fine temperature grid (K)
T_fine_min = 100     # Same as prior range for T
T_fine_max = 2000    # Same as prior range for T
T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM
T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2.0    # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM
log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step),
                       log_P_fine_step)

# Pre-interpolate the opacities
opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

#***** Specify fixed atmospheric settings for retrieval *****#
# Atmospheric pressure grid
P_min = 1.0e-7    # 0.1 ubar
P_max = 100       # 100 bar
N_layers = 100    # 100 layers

P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)
P_ref = 10.0   # Retrieved R_p_ref parameter will be the radius at 10 bar

#***** Run atmospheric retrieval *****#
run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R=R,
              spectrum_type='transmission', sampling_algorithm='MultiNest',
              N_live=500, verbose=True)

# Do summary plots
wl, spec_low2, spec_low1, spec_median, spec_high1, spec_high2 = read_retrieved_spectrum(planet_name, model_name)

plt.figure(figsize=(8, 4))
plt.errorbar(data['wl_data'], data['ydata']*1e6, yerr=data['err_data']*1e6, xerr=data['half_bin'],
             fmt='o', mfc='white', mec='black', ecolor='black')

out_med = bin_spectrum(wl, spec_median, 50)
plt.plot(out_med[0], out_med[1]*1e6, c='blue')
out_l1 = bin_spectrum(wl, spec_low2, 50)
out_u1 = bin_spectrum(wl, spec_high2, 50)
plt.fill_between(out_u1[0], out_l1[1]*1e6, out_u1[1]*1e6, alpha=0.25, color='blue')
vmax = np.max((data['ydata'] + data['err_data'])*1e6)
vmin = np.min((data['ydata'] - data['err_data'])*1e6)
plt.ylim(vmin*0.995, vmax*1.005)
plt.xlabel('Wavelength [µm]', fontsize=12)
plt.ylabel('Transit Depth [ppm]', fontsize=12)
plt.savefig('POSEIDON_output/TOI-2076c/retrievals/results/{}_spectrum.pdf'.format(model_name))

fig_corner = generate_cornerplot(planet, model)

# Save a copy of this file.
thispath = os.path.realpath(__file__)
thisfile = thispath.split('/')[-1]
shutil.copy(thisfile, 'POSEIDON_output/TOI-2076c/retrievals/results/' + thisfile)

print('Done')

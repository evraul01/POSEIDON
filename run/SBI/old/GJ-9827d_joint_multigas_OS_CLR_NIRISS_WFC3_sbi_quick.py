from POSEIDON.constants import R_Sun, R_E, M_E
from POSEIDON.core import create_star, create_planet, load_data, define_model, \
                          wl_grid_constant_R, set_priors, read_opacities
from POSEIDON.retrieval import run_retrieval, postprocess_sbi_raw

import numpy as np
import time
import os
from pathlib import Path
from scipy.constants import parsec as pc
from sbi.analysis import pairplot
import matplotlib.pyplot as plt

# Same setup as run/GJ-9827d_joint_multigas_OS_CLR_NIRISS_WFC3.py, but uses SBI.
# Supports either full SBI run or postprocessing-only from existing SBI_raw files.

do_retrieval = True
do_sbi_postprocess_only = False   # True: reuse existing SBI_raw and regenerate outputs only

data_included = 'NIRISS_WFC3'
model_name = 'joint_multigas_OS_CLR_' + data_included + '_sbi_quick'
stellar_contam = 'one_spot'

RUN_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = RUN_DIR / 'data'
OUTPUT_ROOT = RUN_DIR / 'POSEIDON_output'
os.chdir(RUN_DIR)

# Wavelength grid
wl_min = 0.58
wl_max = 2.90
R = 20000
wl = wl_grid_constant_R(wl_min, wl_max, R)

# Star
R_s = 0.58 * R_Sun
T_s = 4236.0
err_T_s = 12
Met_s = -0.29
log_g_s = 4.719
err_log_g_s = 0.02

star = create_star(R_s, T_s, log_g_s, Met_s, T_eff_error=err_T_s,
                   stellar_grid='phoenix', wl=wl)

# Planet
planet_name = 'GJ-9827d'
R_p = 1.89 * R_E
M_p = 3.02 * M_E
T_eq = 599.68
d = 29.6610 * pc
planet = create_planet(planet_name, R_p, mass=M_p, T_eq=T_eq, d=d)

# Data
data_dir = str(DATA_ROOT / planet_name)
datasets = [planet_name + '_NIRISS_SOSS_Ord2.dat',
            planet_name + '_NIRISS_SOSS_Ord1.dat',
            planet_name + '_WFC3_G141.dat']
instruments = ['JWST_NIRISS_SOSS_Ord2', 'JWST_NIRISS_SOSS_Ord1', 'WFC3_G141']

data = load_data(data_dir, datasets, instruments, wl,
                 offset_datasets=[planet_name + '_WFC3_G141.dat'])

# Model
bulk_species = ['H2', 'He']
param_species = ['N2', 'HCN', 'H2O', 'CO', 'CO2', 'CH4', 'NH3', 'H2S']
model = define_model(model_name, bulk_species, param_species,
                     radius_unit='R_E', surface=False,
                     PT_profile='isotherm', cloud_model='MacMad17',
                     cloud_type='deck_haze', stellar_contam=stellar_contam,
                     offsets_applied='single_dataset')

# Priors
prior_types = {
    'R_p_ref': 'uniform', 'T': 'uniform', 'log_X': 'CLR',
    'log_P_surf': 'uniform', 'log_a': 'uniform', 'gamma': 'uniform',
    'log_P_cloud': 'uniform', 'f_cloud': 'uniform', 'delta_rel': 'uniform',
    'T_het': 'uniform', 'f_het': 'uniform', 'T_phot': 'gaussian',
    'log_g_het': 'uniform', 'log_g_phot': 'gaussian',
    'f_spot': 'uniform', 'f_fac': 'uniform', 'T_spot': 'uniform',
    'T_fac': 'uniform', 'log_g_spot': 'uniform', 'log_g_fac': 'uniform',
}

prior_ranges = {
    'R_p_ref': [0.60 * R_p, 1.15 * R_p],
    'T': [100, 1000],
    'a1': [0.02, 2.0], 'a2': [0.02, 2.0],
    'log_P1': [-7, 2], 'log_P2': [-7, 2], 'log_P3': [-2, 2],
    'T_deep': [100, 1000],
    'log_X': [-12, 0.0],
    'log_a': [-4, 8], 'gamma': [-20, 2],
    'log_P_cloud': [-7, 2], 'f_cloud': [0, 1],
    'f_het': [0.0, 0.5], 'T_het': [2300, 1.2 * T_s],
    'T_phot': [T_s, err_T_s], 'log_g_het': [3.0, 5.4],
    'log_g_phot': [log_g_s, err_log_g_s],
    'delta_rel': [-500, +500],
    'f_spot': [0.0, 0.5], 'f_fac': [0.0, 0.5],
    'T_spot': [2300, T_s + 3 * err_T_s],
    'T_fac': [T_s - 3 * err_T_s, 1.2 * T_s],
    'log_g_spot': [3.0, 5.4], 'log_g_fac': [3.0, 5.4],
}

priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

# Opacities
opacity_treatment = 'opacity_sampling'
T_fine = np.arange(100, 1000 + 10, 10)
log_P_fine = np.arange(-6.0, 2.0 + 0.2, 0.2)
opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

# Atmosphere grid
P = np.logspace(np.log10(100), np.log10(1.0e-7), 100)
P_ref = 10.0

# Quick SBI benchmark knobs.
# For closer quality comparison with larger MultiNest runs, increase round sizes.
sbi_round_sizes = (2000, 1000, 1000)

if do_retrieval:
    print('Starting SBI quick benchmark with round sizes:', sbi_round_sizes)
    t0 = time.perf_counter()

    run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R=R,
                  spectrum_type='transmission', sampling_algorithm='sbi',
                  verbose=True, resume=False,
                  sbi_round_sizes=sbi_round_sizes,
                  sbi_training_batch_size=256,
                  sbi_posterior_samples=10000,
                  sbi_device='cpu',
                  sbi_density_estimator='nsf',
                  sbi_hidden_features=128,
                  sbi_num_transforms=5,
                  sbi_seed=0)

    t1 = time.perf_counter()
    print('Total wall-time (hours):', (t1 - t0) / 3600.0)

if do_sbi_postprocess_only:
    postprocess_sbi_raw(planet, star, model, opac, data, wl, P, priors=priors,
                        P_ref=P_ref, R=R, spectrum_type='transmission',
                        N_output_samples=1000, sbi_round_sizes=sbi_round_sizes)

# Pairplot from saved SBI samples (based on file availability).
sbi_samples_file = OUTPUT_ROOT / planet_name / 'retrievals' / 'samples' / (model_name + '_samples.txt')
if sbi_samples_file.exists():
    sbi_samples = np.loadtxt(str(sbi_samples_file), skiprows=1)
    sbi_samples = np.atleast_2d(sbi_samples)
    valid_rows = np.isfinite(sbi_samples).all(axis=1)
    sbi_samples = sbi_samples[valid_rows]

    n_params = min(sbi_samples.shape[1], len(model['param_names']))
    labels = list(model['param_names'])[:n_params]
    sbi_samples = sbi_samples[:, :n_params]

    if sbi_samples.shape[0] < 2:
        print('Skipping pairplot: need at least 2 valid posterior samples, found',
              sbi_samples.shape[0])
    elif sbi_samples.shape[1] < 2:
        print('Skipping pairplot: need at least 2 parameters to plot, found',
              sbi_samples.shape[1])
    else:
        fig_pair, _ = pairplot(
            sbi_samples,
            labels=labels,
            figsize=(11, 11),
        )
        pairplot_file = OUTPUT_ROOT / planet_name / 'plots' / (model_name + '_sbi_pairplot_quick.png')
        plt.savefig(str(pairplot_file), dpi=200)
        plt.close()
        print('Saved pairplot to ' + str(pairplot_file))
else:
    print('No samples file found; skipping pairplot:', str(sbi_samples_file))

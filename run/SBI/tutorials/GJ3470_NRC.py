import sys
import matplotlib.pyplot as plt
import numpy as np
from spectres import spectres
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.planet import Planet
from petitRADTRANS.physics import madhu_seager_2009 as MadhuTP
import torch
from sbi.inference import SNPE, DirectPosterior
from sbi.utils import BoxUniform
from sbi import utils as sbi_utils
from sbi import neural_nets as sbi_neural_nets
from sbi.analysis import pairplot
from astropy.convolution import convolve, Gaussian1DKernel

#wave322, width322, depth322, error322 = np.loadtxt('./GJ3470/GJ3470_F322W2_Bin36_TGB_250911.dat', unpack=True)
#wave444, width444, depth444, error444 = np.loadtxt('./GJ3470/GJ3470_F444W_Bin36_TGB_250911.dat', unpack=True)
wave322, width322, depth322, error322 = np.loadtxt('./GJ3470/GJ3470_F322W2_TGB_250911.dat', unpack=True)
wave444, width444, depth444, error444 = np.loadtxt('./GJ3470/GJ3470_F444W_TGB_250911.dat', unpack=True)
depth322 = depth322*1e6 - 103.
depth444 = depth444*1e6
wave = np.concatenate((wave322, wave444))
width = np.concatenate((width322, width444))
depth = np.concatenate((depth322, depth444))
error = np.concatenate((error322, error444))
error = error * 1e6

planet = Planet.get('GJ 3470 b')

atmosphere = Radtrans(
    pressures=np.logspace(-9, 2, 200),
    line_species=[
        'H2O',
        'CO-NatAbund',
        'CH4',
        'CO2',
        'SO2'
    ],
    rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=['H2--H2', 'H2--He'],
    wavelength_boundaries=[2.3, 5.2],
    scattering_in_emission=False
)

OtherParams = {
    'mean_molar_masses': 4.0,  # in g/mol
    'star_radius': 0.50, # in solar radii
    'planet_radius': 0.390 * cst.r_jup_mean,
    'planet_mass': 17.6 * cst.m_earth,
    'reference_gravity': planet.reference_gravity,  # in cm/s^2
    'reference_pressure': 10**-5., # in bar
    'observed_wavelengths': wave
}

# Initial guess for parameters
X_H2O = -1.08  # log10 of volume mixing ratio of H2O
X_CH4 = -4.05  # log10 of volume mixing ratio of CH4
X_CO2 = -2.47  # log10 of volume mixing ratio
X_CO = -1.96  # log10 of volume mixing ratio of CO
X_SO2 = -3.57  # log10 of volume mixing ratio of SO2
logP2 = -6.11  # log10 of pressure point 2 (bar); inversion if logP2 > logP1
logP1 = -2.54  # log10 of pressure point 1 (bar)
logP3 = 0.43   # log10 of pressure point 3 (bar)
logPref = -5.25 # log10 of reference pressure (bar)
Tref = 400.0 # Temperature at reference pressure (K)
alpha1 = 1.37  # Temperature gradient parameter 1
alpha2 = 1.11  # Temperature gradient parameter 2
logPcloud = -1.5  # log10 of cloud top pressure (bar)

ThetaTrans = np.asarray([X_H2O, X_CH4, X_CO, X_CO2, X_SO2, logP1, logP2, logP3, logPref, Tref, alpha1, alpha2, logPcloud])
nParams = len(ThetaTrans)

def ptr3transmission(ThetaTrans, AtmoObject, OtherParams, Resample=True):
    '''
    Transmission spectrum model for petitRADTRANS.

    Parameters
    ----------
    theta : list
        List of model parameters in the following order:
        - log10 of volume mixing ratio of H2O
        - log10 of volume mixing ratio of CH4
        - log10 of volume mixing ratio of CO2
        - log10 of volume mixing ratio of NH3
        - log10 of surface gravity (cm/s^2)
        - temperature profile parameters (T0, alpha1, alpha2, P1, P2, P3)
        - planetary radius at 10 bar (cm)
    '''

    if isinstance(ThetaTrans, np.ndarray):
        X_H2O, X_CH4, X_CO2, X_CO, X_SO2, logP1, logP2, logP3, logPref, Tref, alpha1, alpha2, logPcloud = ThetaTrans
    else:
        X_H2O, X_CH4, X_CO2, X_CO, X_SO2, logP1, logP2, logP3, logPref, Tref, alpha1, alpha2, logPcloud = ThetaTrans.cpu().numpy() # put these on the CPU for pRT

    temperatures = MadhuTP(
        pressures=AtmoObject.pressures * 1e-6, # convert to bar
        log_pressure_points=[np.log10(AtmoObject.pressures[0]*1e-6), logP1, logP2, logP3, logPref],
        T_set=Tref,
        alpha_points=[alpha1, alpha2],
        beta_points=[0.5, 0.5] # don't change these
    )

    # this is a very janky rejection of TP profiles with negative temperatures
    if np.any(np.isnan(temperatures)) or np.any(temperatures < 0):
        toolow = temperatures < 0
        temperatures[toolow] = 1.0

    # Plot the temperature-pressure profile for debugging
    #plt.plot(temperatures, np.log10(AtmoObject.pressures * 1e-6), 'b-')
    #plt.gca().invert_yaxis()
    #plt.xlabel('Temperature (K)')
    #plt.ylabel('Pressure (bar)')
    #plt.show()
    #plt.clf()

    mass_fractions = {
    'H2': 0.74 * np.ones_like(temperatures),
    'He': 0.24 * np.ones_like(temperatures),
    'H2O': 10**X_H2O * np.ones_like(temperatures),
    'CH4': 10**X_CH4 * np.ones_like(temperatures),
    'CO2': 10**X_CO2 * np.ones_like(temperatures),
    'CO-NatAbund': 10**X_CO * np.ones_like(temperatures),
    'SO2': 10**X_SO2 * np.ones_like(temperatures)
    }

    wavelengths, transit_radii, _ = AtmoObject.calculate_transit_radii(
    temperatures=temperatures,
    mass_fractions=mass_fractions,
    mean_molar_masses=OtherParams['mean_molar_masses'] * np.ones_like(temperatures),
    reference_gravity=OtherParams['reference_gravity'],
    planet_radius=OtherParams['planet_radius'],
    reference_pressure=OtherParams['reference_pressure'],
    opaque_cloud_top_pressure=10**logPcloud  # (bar)
    )
    
    transit_depths = (transit_radii / (OtherParams['star_radius']*cst.r_sun)) ** 2 * 1e6  # in ppm

    # Interpolate to observed wavelengths
    if Resample:
        transit_depths = spectres(OtherParams['observed_wavelengths'], wavelengths*1e4, transit_depths)
        return transit_depths
    else:
        return transit_depths, wavelengths*1e4

def z_to_theta_trans(z: torch.Tensor) -> torch.Tensor:
    """
    Map z (nParams,) or (N,nParams) to ThetaTrans (nParams,) or (N,nParams) with ordering enforced:
    ThetaTrans = [X_H2O, X_CH4, X_NH3, X_CO2,
                  logP1, logP2, logP3, logPref,
                  Tref, alpha1, alpha2, logPcloud]
    """
    z = torch.atleast_2d(z)

    X = z[:, 0:5]
    logP2 = z[:, 5]
    u21, u13, uref = z[:, 6], z[:, 7], z[:, 8]
    Tref = z[:, 9]
    alpha1 = z[:, 10]
    alpha2 = z[:, 11]
    logPcloud = z[:, 12]

    logP1 = logP2 + u21 * (LOGP_MAX - logP2)
    # logP3 in [max(logP1, 0+eps), LOGP_MAX]
    P3_EPS = 1e-6
    logP3_min = torch.maximum(logP1, torch.tensor(-1.0 + P3_EPS, dtype=logP1.dtype, device=logP1.device))
    logP3 = logP3_min + u13 * (LOGP_MAX - logP3_min)
    logPref = LOGP_MIN + uref * (LOGP_MAX - LOGP_MIN)

    theta = torch.stack([X[:,0], X[:,1], X[:,2], X[:,3], X[:,4],
                         logP1, logP2, logP3, logPref,
                         Tref, alpha1, alpha2, logPcloud], dim=1)
    return theta.squeeze(0) if theta.shape[0] == 1 else theta

def simulator(z: torch.Tensor, AtmoObject, OtherParams) -> torch.Tensor:
    """
    Input: z (nParams,) or (N,nParams)
    Output: whitened noisy spectrum x = (mu + N(0,sigma))/sigma  shape (nObs,) or (N,nObs)
    """
    z = torch.atleast_2d(z)
    theta = z_to_theta_trans(z)  # (N,nParams)

    xs = []
    for th in theta:
        # ptr3transmission must return transit depths in ppm
        mu_ppm = ptr3transmission(th, AtmoObject, OtherParams)
        mu_ppm = torch.as_tensor(mu_ppm, dtype=torch.float32)

        y_sim = mu_ppm + torch.randn_like(mu_ppm) * sig
        x_sim = y_sim / sig
        xs.append(x_sim)

    X = torch.stack(xs, dim=0)
    return X.squeeze(0) if X.shape[0] == 1 else X


lam_um   = wave
width_um = width
y_ppm    = depth
sig_ppm  = error

y_obs = torch.as_tensor(y_ppm, dtype=torch.float32)
sig   = torch.as_tensor(sig_ppm, dtype=torch.float32)
x_obs = y_obs / sig   # whitened observation (15,)

# Define the device here since we need it for the prior definition
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

# Prior bounds
# log10 VMR bounds for 5 species: H2O, CH4, SO2, CO2, CO
X_MIN, X_MAX = -10.0, -0.2

# pressure domain in log10(bar)
LOGP_MIN, LOGP_MAX = -6.0, 2.0

# temperature and MS09 alphas
TREF_MIN, TREF_MAX = 300.0, 800.0

# Use a tiny epsilon instead of exact 0 to avoid numerical edge cases
ALPHA_EPS = 0.02
ALPHA1_MIN, ALPHA1_MAX = ALPHA_EPS, 2.0
ALPHA2_MIN, ALPHA2_MAX = ALPHA_EPS, 2.0

low = torch.tensor([X_MIN, X_MIN, X_MIN, X_MIN, X_MIN,
                    LOGP_MIN, 0.0, 0.0, 0.0,
                    TREF_MIN, ALPHA1_MIN, ALPHA2_MIN, LOGP_MIN], dtype=torch.float32)

high = torch.tensor([X_MAX, X_MAX, X_MAX, X_MAX, X_MAX,
                     LOGP_MAX, 1.0, 1.0, 1.0,
                     TREF_MAX, ALPHA1_MAX, ALPHA2_MAX, LOGP_MAX], dtype=torch.float32)

prior_z = BoxUniform(low=low, high=high, device=device)


# Train sequential NPE (SNPE)
torch.manual_seed(0)

density_estimator = sbi_neural_nets.posterior_nn(
    model="nsf",            # strong default for low-D conditioning
    hidden_features=128,
    num_transforms=5
)

inference = SNPE(prior=prior_z, density_estimator=density_estimator, device=device, show_progress_bars=True)

round_sizes = [8000, 4000, 4000]

posterior = None
proposal = prior_z
for r, n_sim in enumerate(round_sizes, start=1):
    if posterior is None:
        z = prior_z.sample((n_sim,))
    else:
        z = posterior.sample((n_sim,), x=x_obs)

    x = simulator(z, atmosphere, OtherParams)
    density_estimator = inference.append_simulations(z, x, proposal=proposal, exclude_invalid_x=True).train(training_batch_size=256)
    posterior = inference.build_posterior(density_estimator)
    proposal = posterior.set_default_x(x_obs)
    print(f"Finished SNPE round {r} with {n_sim} simulations.")


# Draw posterior samples
with torch.no_grad():
    z_samps = posterior.sample((20000,), x=x_obs)      # z-space samples
    theta_samps = z_to_theta_trans(z_samps)            # ThetaTrans samples (N,12)

torch.save(posterior, 'GJ3470_posterior.pt')
torch.save(theta_samps, 'GJ3470_theta_samps.pt')

# Load previously saved posterior and samples
posterior = torch.load('GJ3470_posterior.pt', map_location=device)
theta_samps = torch.load('GJ3470_theta_samps.pt', map_location=device)

# theta_samps columns:
# 0 X_H2O, 1 X_CH4, 2 X_CO2, 3 X_CO, 4 X_SO2,
# 5 logP1, 6 logP2, 7 logP3, 8 logPref,
# 9 Tref, 10 alpha1, 11 alpha2, 12 logPcloud

# Quick summaries:
names = ["X_H2O","X_CH4","X_CO2","X_CO","X_SO2",
          "logP1","logP2","logP3","logPref",
          "Tref","alpha1","alpha2","logPcloud"]
qs = torch.quantile(theta_samps, torch.tensor([0.16, 0.5, 0.84]), dim=0).cpu().numpy()

for i, nm in enumerate(names):
    lo, med, hi = qs[0,i], qs[1,i], qs[2,i]
    print(f"{nm:8s}: {med: .4f}  (16–84%: {lo:.4f}, {hi:.4f})")

_ = pairplot(
    theta_samps,
    limits=[[X_MIN, X_MAX], [X_MIN, X_MAX], [X_MIN, X_MAX], [X_MIN, X_MAX], [X_MIN, X_MAX],
            [LOGP_MIN, LOGP_MAX], [LOGP_MIN, LOGP_MAX], [LOGP_MIN, LOGP_MAX], [LOGP_MIN, LOGP_MAX],
            [TREF_MIN, TREF_MAX], [ALPHA1_MIN, ALPHA1_MAX], [ALPHA2_MIN, ALPHA2_MAX], [LOGP_MIN, LOGP_MAX]],
    figsize=(10, 10),
    labels=[r"$X_{H_2O}$", r"$X_{CH_4}$", r"$X_{CO_2}$", r"$X_{CO}$", r"$X_{SO_2}$",
            r"$\log P_1$", r"$\log P_2$", r"$\log P_3$", r"$\log P_{ref}$",
            r"$T_{ref}$", r"$\alpha_1$", r"$\alpha_2$", r"$\log P_{cloud}$"]
)
plt.savefig('GJ3470_pairplot.png', dpi=300)
plt.clf()

bestfit_theta = qs[1,:]
bestfit_spectrum, wavelengths = ptr3transmission(bestfit_theta, atmosphere, OtherParams, Resample=False)
GaussKernel = Gaussian1DKernel(stddev=2)
DepthSmooth = convolve(bestfit_spectrum, GaussKernel)
plt.errorbar(lam_um, y_ppm, yerr=sig_ppm, xerr=width_um, fmt='o', color='black', ecolor='gray', capsize=0, label='Observed Data')
plt.plot(wavelengths, DepthSmooth, 'r-', label='Best-fit Model', lw=3)
plt.xlabel('Wavelength (um)')
plt.ylabel('Transit Depth (ppm)')

plt.xlim(2.35,5.1)
plt.ylim(5600,6300)

fig = plt.gcf()
fig.set_size_inches(10,6)
#plt.legend()
plt.savefig('GJ3470_bestfit_spectrum.png', dpi=300)
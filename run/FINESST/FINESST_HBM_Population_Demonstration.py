"""Create the FINESST synthetic hierarchical-population demonstration.

This module generates an explicitly synthetic gas-planet population, creates
non-Gaussian individual retrieval posteriors under a broad uniform prior, and
infers a mass-metallicity relation by posterior-sample reweighting. It is an
explanatory methods figure, not an analysis of observed planets.

The hierarchical construction follows the population-retrieval framework of
Lustig-Yaeger et al. (2022, AJ, 163, 140; doi:10.3847/1538-3881/ac5034), but
the population, variables, and numerical values here are newly simulated for
this proposal figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, norm, skewnorm


@dataclass
class PathConfig:
    """Input provenance and output locations."""

    data_directory: Path
    figure_directory: Path
    reference_paper: Path
    output_stem: str = "figure3_HBM_population_demo"


@dataclass
class SyntheticPopulationConfig:
    """All assumptions defining the synthetic planet population."""

    random_seed: int = 270107
    n_planets: int = 30
    mass_range_mj: Tuple[float, float] = (0.03, 10.0)
    true_intercept: float = 0.68
    true_slope: float = -0.62
    true_intrinsic_scatter: float = 0.22
    retrieval_prior_range: Tuple[float, float] = (-1.0, 2.5)
    posterior_samples_per_planet: int = 3000
    inference_samples_per_planet: int = 600
    n_tight: int = 8
    n_moderate: int = 11
    n_weak: int = 9
    n_upper_limit: int = 2
    tight_sigma_range: Tuple[float, float] = (0.09, 0.16)
    moderate_sigma_range: Tuple[float, float] = (0.20, 0.34)
    weak_sigma_range: Tuple[float, float] = (0.42, 0.68)
    skewed_fraction: float = 0.22
    mixture_fraction: float = 0.22
    upper_limit_offset_range: Tuple[float, float] = (0.22, 0.42)
    upper_limit_transition_width: float = 0.08


@dataclass
class GridConfig:
    """Hierarchical hyperparameter grids and posterior sampling controls."""

    alpha_range: Tuple[float, float] = (-0.20, 1.55)
    beta_range: Tuple[float, float] = (-1.45, 0.35)
    sigma_range: Tuple[float, float] = (0.05, 0.65)
    n_alpha: int = 31
    n_beta: int = 37
    n_sigma: int = 25
    evaluation_chunk_size: int = 256
    n_hyperparameter_samples: int = 30000
    relation_grid_size: int = 300
    run_grid_convergence_check: bool = True
    coarse_grid_stride: int = 2


@dataclass
class StyleConfig:
    """POSEIDON-aligned typography and color-blind-safe colors."""

    use_sans_serif: bool = False
    serif_font: str = "DejaVu Serif"
    sans_serif_font: str = "DejaVu Sans"
    axis_label_size: float = 12.5
    tick_label_size: float = 10.5
    legend_size: float = 8.5
    annotation_size: float = 8.5
    panel_label_size: float = 13.0
    axes_line_width: float = 0.8
    tick_length: float = 4.0
    minor_tick_length: float = 2.2
    line_width: float = 1.7
    posterior_color: str = "#0072B2"
    recovered_color: str = "#D55E00"
    truth_color: str = "#009E73"
    null_color: str = "#4D4D4D"
    upper_limit_color: str = "#6A3D9A"

    def rc_params(self) -> Dict[str, Any]:
        if self.use_sans_serif:
            font = {
                "font.family": "sans-serif",
                "font.sans-serif": [self.sans_serif_font],
                "mathtext.fontset": "dejavusans",
            }
        else:
            font = {
                "font.family": "serif",
                "font.serif": [self.serif_font],
                "mathtext.fontset": "dejavuserif",
            }
        return {
            **font,
            "axes.labelsize": self.axis_label_size,
            "xtick.labelsize": self.tick_label_size,
            "ytick.labelsize": self.tick_label_size,
            "legend.fontsize": self.legend_size,
            "axes.linewidth": self.axes_line_width,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": self.axes_line_width,
            "ytick.major.width": self.axes_line_width,
            "xtick.minor.width": self.axes_line_width,
            "ytick.minor.width": self.axes_line_width,
            "xtick.major.size": self.tick_length,
            "ytick.major.size": self.tick_length,
            "xtick.minor.size": self.minor_tick_length,
            "ytick.minor.size": self.minor_tick_length,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }


@dataclass
class LayoutConfig:
    """Wide two-panel layout controls."""

    figure_width_inches: float = 7.5
    figure_height_inches: float = 4.7
    width_ratios: Tuple[float, float] = (1.0, 1.0)
    horizontal_space: float = 0.19
    left_margin: float = 0.085
    right_margin: float = 0.985
    bottom_margin: float = 0.14
    top_margin: float = 0.955


@dataclass
class PanelConfig:
    """Axes, violin, bands, annotations, legends, and inset controls."""

    x_label: str = r"$\log_{10}(M_p/M_J)$"
    y_label: str = r"$\log_{10}(Z_{\rm atm}/Z_\star)$"
    x_limits: Optional[Tuple[float, float]] = None
    y_limits: Optional[Tuple[float, float]] = (-1.02, 2.52)
    show_minor_ticks: bool = True
    show_synthetic_truth_points: bool = True
    show_synthetic_truth_relation: bool = True
    show_synthetic_label_each_panel: bool = False
    synthetic_label: str = "Synthetic demonstration"
    synthetic_label_figure_xy: Tuple[float, float] = (0.50, 0.985)
    violin_grid_size: int = 220
    violin_width_fraction: float = 0.34
    violin_bandwidth: float = 0.22
    violin_density_floor_fraction: float = 0.012
    violin_alpha_by_class: Mapping[str, float] = field(
        default_factory=lambda: {
            "tight": 0.58,
            "moderate": 0.46,
            "weak": 0.30,
            "upper_limit": 0.25,
        }
    )
    upper_limit_hatch: str = "///"
    posterior_median_marker_size: float = 2.8
    truth_marker_size: float = 2.5
    truth_relation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": ":", "linewidth": 1.35, "alpha": 0.85}
    )
    input_relation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "--", "linewidth": 1.25, "alpha": 0.65}
    )
    recovered_relation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "-", "linewidth": 1.9, "zorder": 6}
    )
    null_relation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "--", "linewidth": 1.5, "zorder": 5}
    )
    mean_band_alpha: float = 0.30
    predictive_band_alpha: float = 0.13
    background_interval_alpha: float = 0.23
    panel_a_legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "lower left", "frameon": False, "ncol": 1,
            "handlelength": 1.8, "handletextpad": 0.5,
        }
    )
    panel_b_legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "lower left", "bbox_to_anchor": (0.01, 0.01),
            "frameon": False, "ncol": 1,
            "handlelength": 1.9, "handletextpad": 0.5,
        }
    )
    hyperparameter_annotation_axes_xy: Tuple[float, float] = (0.60, 0.045)
    hyperparameter_annotation_fontsize: float = 8.5
    inset_bounds: Tuple[float, float, float, float] = (0.52, 0.55, 0.44, 0.37)
    inset_x_label: str = r"Population slope, $\beta$"
    inset_tick_label_size: float = 7.5
    inset_axis_label_size: float = 8.0
    inset_density_color: str = "#D55E00"
    inset_interval_alpha: float = 0.30
    no_mass_dependence_label: str = "No mass\ndependence"
    no_mass_dependence_text_xy: Tuple[float, float] = (0.0, 0.94)
    no_mass_dependence_fontsize: float = 6.8


@dataclass
class FigureConfig:
    paths: PathConfig
    synthetic: SyntheticPopulationConfig = field(default_factory=SyntheticPopulationConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    panels: PanelConfig = field(default_factory=PanelConfig)


@dataclass
class SyntheticPopulation:
    names: np.ndarray
    masses_mj: np.ndarray
    log_masses: np.ndarray
    true_log_enrichment: np.ndarray
    information_class: np.ndarray
    posterior_form: np.ndarray
    posterior_scale: np.ndarray
    upper_limit: np.ndarray
    posterior_samples: np.ndarray
    prior_density: np.ndarray
    posterior_q16: np.ndarray
    posterior_median: np.ndarray
    posterior_q84: np.ndarray


@dataclass
class HierarchicalResult:
    alpha_grid: np.ndarray
    beta_grid: np.ndarray
    sigma_grid: np.ndarray
    log_posterior_grid: np.ndarray
    alpha_samples: np.ndarray
    beta_samples: np.ndarray
    sigma_samples: np.ndarray
    null_alpha_samples: np.ndarray
    null_sigma_samples: np.ndarray
    relation_log_mass: np.ndarray
    relation_median: np.ndarray
    relation_low: np.ndarray
    relation_high: np.ndarray
    predictive_low: np.ndarray
    predictive_high: np.ndarray
    null_relation_median: float
    quantiles: Mapping[str, Tuple[float, float, float]]
    log_evidence_full: float
    log_evidence_null: float
    log_bayes_factor_full_vs_null: float
    importance_ess: np.ndarray
    convergence: Mapping[str, Any]
    validation: Mapping[str, Any]


def default_config(run_directory: Path) -> FigureConfig:
    """Return portable defaults rooted at ``POSEIDON/run``."""

    run_directory = Path(run_directory).resolve()
    return FigureConfig(paths=PathConfig(
        data_directory=run_directory / "data" / "FINESST_26" / "Figure_3",
        figure_directory=(
            run_directory / "POSEIDON_output" / "WASP-107b" / "plots" / "FINESST"
        ),
        reference_paper=(
            run_directory / "POSEIDON_output" / "WASP-107b" / "plots" /
            "FINESST" / "Lustig-Yaeger_2022_AJ_163_140.pdf"
        ),
    ))


def _draw_truncated(
    draw_batch,
    n_samples: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Draw from ``draw_batch`` until enough samples lie inside the prior."""

    accepted = []
    n_accepted = 0
    for _ in range(100):
        batch = np.asarray(draw_batch(max(512, 2 * (n_samples - n_accepted))))
        valid = batch[(batch >= lower) & (batch <= upper)]
        if valid.size:
            accepted.append(valid)
            n_accepted += valid.size
        if n_accepted >= n_samples:
            return np.concatenate(accepted)[:n_samples]
    raise RuntimeError("Could not draw enough posterior samples inside the prior.")


def _draw_upper_limit_posterior(
    rng: np.random.Generator,
    n_samples: int,
    lower: float,
    upper: float,
    limit: float,
    transition_width: float,
) -> np.ndarray:
    """Sample a broad posterior with a soft upper cutoff."""

    accepted = []
    n_accepted = 0
    for _ in range(100):
        candidates = rng.uniform(lower, upper, max(1024, 4 * (n_samples - n_accepted)))
        probability = norm.cdf((limit - candidates) / transition_width)
        valid = candidates[rng.random(candidates.size) < probability]
        if valid.size:
            accepted.append(valid)
            n_accepted += valid.size
        if n_accepted >= n_samples:
            return np.concatenate(accepted)[:n_samples]
    raise RuntimeError("Could not draw enough upper-limit posterior samples.")


def generate_synthetic_population(config: FigureConfig) -> SyntheticPopulation:
    """Generate synthetic truths and full individual-planet posteriors."""

    options = config.synthetic
    if sum((options.n_tight, options.n_moderate, options.n_weak,
            options.n_upper_limit)) != options.n_planets:
        raise ValueError("Information-class counts must sum to n_planets.")
    lower, upper = options.retrieval_prior_range
    if not lower < upper:
        raise ValueError("retrieval_prior_range must be increasing.")

    rng = np.random.default_rng(options.random_seed)
    log_mass_bounds = np.log10(options.mass_range_mj)
    edges = np.linspace(log_mass_bounds[0], log_mass_bounds[1], options.n_planets + 1)
    log_masses = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(log_masses)
    log_masses.sort()
    masses = 10.0 ** log_masses

    population_mean = options.true_intercept + options.true_slope * log_masses
    true_y = population_mean + rng.normal(
        0.0, options.true_intrinsic_scatter, options.n_planets
    )
    true_y = np.clip(true_y, lower + 0.06, upper - 0.06)

    classes = np.array(
        ["tight"] * options.n_tight
        + ["moderate"] * options.n_moderate
        + ["weak"] * options.n_weak
        + ["upper_limit"] * options.n_upper_limit,
        dtype="U20",
    )
    rng.shuffle(classes)

    forms = np.full(options.n_planets, "gaussian", dtype="U20")
    eligible = np.where(classes != "upper_limit")[0]
    rng.shuffle(eligible)
    n_skew = int(round(options.skewed_fraction * eligible.size))
    n_mix = int(round(options.mixture_fraction * eligible.size))
    forms[eligible[:n_skew]] = "skew_normal"
    forms[eligible[n_skew:n_skew + n_mix]] = "mixture"
    forms[classes == "upper_limit"] = "upper_limit"

    samples = np.empty(
        (options.n_planets, options.posterior_samples_per_planet), dtype=np.float64
    )
    scales = np.empty(options.n_planets)
    upper_limits = np.full(options.n_planets, np.nan)

    sigma_ranges = {
        "tight": options.tight_sigma_range,
        "moderate": options.moderate_sigma_range,
        "weak": options.weak_sigma_range,
    }
    for i in range(options.n_planets):
        info = classes[i]
        if info == "upper_limit":
            scales[i] = options.upper_limit_transition_width
            upper_limits[i] = min(
                upper - 0.04,
                true_y[i] + rng.uniform(*options.upper_limit_offset_range),
            )
            samples[i] = _draw_upper_limit_posterior(
                rng,
                options.posterior_samples_per_planet,
                lower,
                upper,
                upper_limits[i],
                options.upper_limit_transition_width,
            )
            continue

        scale = rng.uniform(*sigma_ranges[info])
        scales[i] = scale
        observed_center = true_y[i] + rng.normal(0.0, 0.45 * scale)
        if forms[i] == "gaussian":
            samples[i] = _draw_truncated(
                lambda n, c=observed_center, s=scale: rng.normal(c, s, n),
                options.posterior_samples_per_planet, lower, upper,
            )
        elif forms[i] == "skew_normal":
            shape = rng.choice((-5.0, 5.0))
            mean_shift = (
                shape / np.sqrt(1.0 + shape**2) * np.sqrt(2.0 / np.pi) * scale
            )
            location = observed_center - mean_shift
            samples[i] = _draw_truncated(
                lambda n, a=shape, loc=location, s=scale: skewnorm.rvs(
                    a, loc=loc, scale=s, size=n, random_state=rng
                ),
                options.posterior_samples_per_planet, lower, upper,
            )
        else:
            weight_left = 0.72
            separation = 1.25 * scale
            left_center = observed_center - (1.0 - weight_left) * separation
            right_center = observed_center + weight_left * separation

            def draw_mixture(n, lc=left_center, rc=right_center, s=scale):
                choose_left = rng.random(n) < weight_left
                draw = np.empty(n)
                draw[choose_left] = rng.normal(lc, 0.72 * s, choose_left.sum())
                draw[~choose_left] = rng.normal(
                    rc, 0.58 * s, (~choose_left).sum()
                )
                return draw

            samples[i] = _draw_truncated(
                draw_mixture,
                options.posterior_samples_per_planet,
                lower,
                upper,
            )

    quantiles = np.quantile(samples, (0.16, 0.50, 0.84), axis=1)
    prior_density = np.full(options.n_planets, 1.0 / (upper - lower))
    names = np.array(
        [f"Synthetic planet {i + 1:02d}" for i in range(options.n_planets)],
        dtype="U24",
    )
    return SyntheticPopulation(
        names=names,
        masses_mj=masses,
        log_masses=log_masses,
        true_log_enrichment=true_y,
        information_class=classes,
        posterior_form=forms,
        posterior_scale=scales,
        upper_limit=upper_limits,
        posterior_samples=samples,
        prior_density=prior_density,
        posterior_q16=quantiles[0],
        posterior_median=quantiles[1],
        posterior_q84=quantiles[2],
    )


def _inference_samples(population: SyntheticPopulation, n_samples: int) -> np.ndarray:
    """Select deterministic, evenly spaced samples for grid integration."""

    total = population.posterior_samples.shape[1]
    if n_samples > total:
        raise ValueError("inference_samples_per_planet exceeds stored posterior samples.")
    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    return population.posterior_samples[:, indices]


def _hierarchical_log_likelihood(
    population: SyntheticPopulation,
    posterior_samples: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate the posterior-reweighting likelihood with stable log sums."""

    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    if not (alpha.size == beta.size == sigma.size):
        raise ValueError("alpha, beta, and sigma arrays must have equal size.")
    if np.any(sigma <= 0.0):
        raise ValueError("All intrinsic-scatter values must be positive.")

    result = np.zeros(alpha.size)
    normalization = 0.5 * np.log(2.0 * np.pi)
    log_n_samples = np.log(posterior_samples.shape[1])
    for start in range(0, alpha.size, chunk_size):
        stop = min(start + chunk_size, alpha.size)
        a = alpha[start:stop]
        b = beta[start:stop]
        s = sigma[start:stop]
        chunk_log_like = np.zeros(stop - start)
        for i, x_i in enumerate(population.log_masses):
            mu = a + b * x_i
            residual = (posterior_samples[i][None, :] - mu[:, None]) / s[:, None]
            log_population_density = (
                -0.5 * residual**2 - np.log(s[:, None]) - normalization
            )
            log_weights = log_population_density - np.log(population.prior_density[i])
            chunk_log_like += logsumexp(log_weights, axis=1) - log_n_samples
        result[start:stop] = chunk_log_like
    return result


def _normalize_log_prob(log_prob: np.ndarray) -> np.ndarray:
    shifted = np.asarray(log_prob) - np.max(log_prob)
    probability = np.exp(shifted)
    return probability / probability.sum()


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float] = (0.16, 0.50, 0.84),
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def _jitter_grid_samples(
    rng: np.random.Generator,
    values: np.ndarray,
    indices: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    if grid.size < 2:
        return values[indices]
    half_step = 0.5 * np.median(np.diff(grid))
    draws = values[indices] + rng.uniform(-half_step, half_step, indices.size)
    return np.clip(draws, grid[0], grid[-1])


def infer_hierarchical_population(
    population: SyntheticPopulation,
    config: FigureConfig,
) -> HierarchicalResult:
    """Infer the linear population relation and a flat null relation."""

    options = config.grid
    synthetic = config.synthetic
    alpha_grid = np.linspace(*options.alpha_range, options.n_alpha)
    beta_grid = np.linspace(*options.beta_range, options.n_beta)
    sigma_grid = np.linspace(*options.sigma_range, options.n_sigma)
    aa, bb, ss = np.meshgrid(alpha_grid, beta_grid, sigma_grid, indexing="ij")
    flat_alpha, flat_beta, flat_sigma = aa.ravel(), bb.ravel(), ss.ravel()
    samples = _inference_samples(
        population, synthetic.inference_samples_per_planet
    )
    log_like = _hierarchical_log_likelihood(
        population,
        samples,
        flat_alpha,
        flat_beta,
        flat_sigma,
        options.evaluation_chunk_size,
    )
    log_posterior_grid = log_like.reshape(aa.shape)
    probability = _normalize_log_prob(log_like)

    rng = np.random.default_rng(synthetic.random_seed + 991)
    indices = rng.choice(
        probability.size,
        size=options.n_hyperparameter_samples,
        replace=True,
        p=probability,
    )
    alpha_samples = _jitter_grid_samples(
        rng, flat_alpha, indices, alpha_grid
    )
    beta_samples = _jitter_grid_samples(rng, flat_beta, indices, beta_grid)
    sigma_samples = _jitter_grid_samples(rng, flat_sigma, indices, sigma_grid)

    null_aa, null_ss = np.meshgrid(alpha_grid, sigma_grid, indexing="ij")
    null_alpha = null_aa.ravel()
    null_sigma = null_ss.ravel()
    null_beta = np.zeros_like(null_alpha)
    null_log_like = _hierarchical_log_likelihood(
        population,
        samples,
        null_alpha,
        null_beta,
        null_sigma,
        options.evaluation_chunk_size,
    )
    null_probability = _normalize_log_prob(null_log_like)
    null_indices = rng.choice(
        null_probability.size,
        size=options.n_hyperparameter_samples,
        replace=True,
        p=null_probability,
    )
    null_alpha_samples = _jitter_grid_samples(
        rng, null_alpha, null_indices, alpha_grid
    )
    null_sigma_samples = _jitter_grid_samples(
        rng, null_sigma, null_indices, sigma_grid
    )

    quantiles = {
        "alpha": tuple(np.quantile(alpha_samples, (0.16, 0.50, 0.84))),
        "beta": tuple(np.quantile(beta_samples, (0.16, 0.50, 0.84))),
        "sigma_intrinsic": tuple(np.quantile(sigma_samples, (0.16, 0.50, 0.84))),
        "null_alpha": tuple(np.quantile(null_alpha_samples, (0.16, 0.50, 0.84))),
        "null_sigma_intrinsic": tuple(
            np.quantile(null_sigma_samples, (0.16, 0.50, 0.84))
        ),
    }

    x_relation = np.linspace(
        np.log10(synthetic.mass_range_mj[0]),
        np.log10(synthetic.mass_range_mj[1]),
        options.relation_grid_size,
    )
    relation_median = np.empty(x_relation.size)
    relation_low = np.empty(x_relation.size)
    relation_high = np.empty(x_relation.size)
    predictive_low = np.empty(x_relation.size)
    predictive_high = np.empty(x_relation.size)
    for k, x_value in enumerate(x_relation):
        mean_draws = alpha_samples + beta_samples * x_value
        relation_low[k], relation_median[k], relation_high[k] = np.quantile(
            mean_draws, (0.16, 0.50, 0.84)
        )
        predictive_draws = mean_draws + rng.normal(0.0, sigma_samples)
        predictive_low[k], predictive_high[k] = np.quantile(
            predictive_draws, (0.16, 0.84)
        )

    alpha_median = quantiles["alpha"][1]
    beta_median = quantiles["beta"][1]
    sigma_median = quantiles["sigma_intrinsic"][1]
    importance_ess = np.empty(synthetic.n_planets)
    for i, x_i in enumerate(population.log_masses):
        mu_i = alpha_median + beta_median * x_i
        weights = norm.pdf(
            population.posterior_samples[i], loc=mu_i, scale=sigma_median
        ) / population.prior_density[i]
        importance_ess[i] = weights.sum() ** 2 / np.sum(weights**2)

    # Uniform hyperpriors make the grid evidence the average likelihood.
    log_evidence_full = float(logsumexp(log_like) - np.log(log_like.size))
    log_evidence_null = float(
        logsumexp(null_log_like) - np.log(null_log_like.size)
    )

    convergence: Dict[str, Any] = {"performed": False}
    if options.run_grid_convergence_check:
        stride = options.coarse_grid_stride
        coarse_log = log_posterior_grid[::stride, ::stride, ::stride]
        coarse_prob = _normalize_log_prob(coarse_log.ravel())
        caa, cbb, css = np.meshgrid(
            alpha_grid[::stride], beta_grid[::stride], sigma_grid[::stride],
            indexing="ij",
        )
        full_q = {
            "alpha": _weighted_quantile(flat_alpha, probability),
            "beta": _weighted_quantile(flat_beta, probability),
            "sigma_intrinsic": _weighted_quantile(flat_sigma, probability),
        }
        coarse_q = {
            "alpha": _weighted_quantile(caa.ravel(), coarse_prob),
            "beta": _weighted_quantile(cbb.ravel(), coarse_prob),
            "sigma_intrinsic": _weighted_quantile(css.ravel(), coarse_prob),
        }
        convergence = {
            "performed": True,
            "coarse_grid_stride": stride,
            "fine_weighted_quantiles": {
                key: value.tolist() for key, value in full_q.items()
            },
            "coarse_weighted_quantiles": {
                key: value.tolist() for key, value in coarse_q.items()
            },
            "median_absolute_differences": {
                key: float(abs(full_q[key][1] - coarse_q[key][1]))
                for key in full_q
            },
        }

    validation = {
        "alpha_truth_inside_68pct": bool(
            quantiles["alpha"][0] <= synthetic.true_intercept <= quantiles["alpha"][2]
        ),
        "beta_truth_inside_68pct": bool(
            quantiles["beta"][0] <= synthetic.true_slope <= quantiles["beta"][2]
        ),
        "sigma_truth_inside_68pct": bool(
            quantiles["sigma_intrinsic"][0]
            <= synthetic.true_intrinsic_scatter
            <= quantiles["sigma_intrinsic"][2]
        ),
        "minimum_importance_ess": float(importance_ess.min()),
        "median_importance_ess": float(np.median(importance_ess)),
        "upper_limit_cases_retained": int(
            np.sum(population.information_class == "upper_limit")
        ),
    }

    return HierarchicalResult(
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        sigma_grid=sigma_grid,
        log_posterior_grid=log_posterior_grid,
        alpha_samples=alpha_samples,
        beta_samples=beta_samples,
        sigma_samples=sigma_samples,
        null_alpha_samples=null_alpha_samples,
        null_sigma_samples=null_sigma_samples,
        relation_log_mass=x_relation,
        relation_median=relation_median,
        relation_low=relation_low,
        relation_high=relation_high,
        predictive_low=predictive_low,
        predictive_high=predictive_high,
        null_relation_median=float(np.median(null_alpha_samples)),
        quantiles=quantiles,
        log_evidence_full=log_evidence_full,
        log_evidence_null=log_evidence_null,
        log_bayes_factor_full_vs_null=log_evidence_full - log_evidence_null,
        importance_ess=importance_ess,
        convergence=convergence,
        validation=validation,
    )


def _panel_label(axis: mpl.axes.Axes, label: str, config: FigureConfig) -> None:
    axis.text(
        0.018, 0.975, label,
        transform=axis.transAxes,
        ha="left", va="top",
        fontsize=config.style.panel_label_size,
        fontweight="bold",
        zorder=20,
    )


def _format_axis(axis: mpl.axes.Axes, config: FigureConfig) -> None:
    options = config.panels
    axis.set_xlabel(options.x_label)
    axis.set_ylabel(options.y_label)
    axis.set_xlim(options.x_limits)
    axis.set_ylim(options.y_limits)
    if options.show_minor_ticks:
        axis.minorticks_on()
    else:
        axis.minorticks_off()


def _violin_width(population: SyntheticPopulation, config: FigureConfig) -> float:
    spacing = np.median(np.diff(np.sort(population.log_masses)))
    return float(config.panels.violin_width_fraction * spacing)


def _draw_violin(
    axis: mpl.axes.Axes,
    x_value: float,
    samples: np.ndarray,
    information_class: str,
    config: FigureConfig,
    width: float,
    zorder: float = 3.0,
) -> None:
    options, style = config.panels, config.style
    lower, upper = config.synthetic.retrieval_prior_range
    y_grid = np.linspace(lower, upper, options.violin_grid_size)
    kde = gaussian_kde(samples, bw_method=options.violin_bandwidth)
    density = kde(y_grid)
    density /= max(density.max(), np.finfo(float).tiny)
    visible = density >= options.violin_density_floor_fraction
    if not np.any(visible):
        visible[np.argmax(density)] = True
    y_grid = y_grid[visible]
    density = density[visible]
    half_width = width * density
    is_limit = information_class == "upper_limit"
    color = style.upper_limit_color if is_limit else style.posterior_color
    alpha = options.violin_alpha_by_class[information_class]
    axis.fill_betweenx(
        y_grid,
        x_value - half_width,
        x_value + half_width,
        facecolor=color,
        edgecolor=color,
        linewidth=0.65,
        alpha=alpha,
        hatch=options.upper_limit_hatch if is_limit else None,
        zorder=zorder,
    )


def _format_interval(value: Tuple[float, float, float]) -> str:
    low, median, high = value
    return rf"${median:.2f}^{{+{high - median:.2f}}}_{{-{median - low:.2f}}}$"


def make_figure(
    population: SyntheticPopulation,
    result: HierarchicalResult,
    config: FigureConfig,
) -> Tuple[mpl.figure.Figure, Mapping[str, mpl.axes.Axes]]:
    """Create the two-panel synthetic HBM figure and slope inset."""

    style, layout, options = config.style, config.layout, config.panels
    with mpl.rc_context(style.rc_params()):
        fig, axes_array = plt.subplots(
            1, 2,
            figsize=(layout.figure_width_inches, layout.figure_height_inches),
            gridspec_kw={
                "width_ratios": layout.width_ratios,
                "wspace": layout.horizontal_space,
            },
        )
        axis_a, axis_b = axes_array
        fig.subplots_adjust(
            left=layout.left_margin,
            right=layout.right_margin,
            bottom=layout.bottom_margin,
            top=layout.top_margin,
        )
        if options.x_limits is None:
            mass_bounds = np.log10(config.synthetic.mass_range_mj)
            padding = 0.035 * (mass_bounds[1] - mass_bounds[0])
            x_limits = (mass_bounds[0] - padding, mass_bounds[1] + padding)
        else:
            x_limits = options.x_limits
        axis_a.set_xlim(x_limits)
        axis_b.set_xlim(x_limits)

        width = _violin_width(population, config)
        order = np.argsort(population.log_masses)
        for i in order:
            _draw_violin(
                axis_a,
                population.log_masses[i],
                population.posterior_samples[i],
                str(population.information_class[i]),
                config,
                width,
            )
            axis_a.plot(
                population.log_masses[i], population.posterior_median[i],
                marker="o", linestyle="none", color="black",
                markersize=options.posterior_median_marker_size, zorder=6,
            )
            if options.show_synthetic_truth_points:
                axis_a.plot(
                    population.log_masses[i], population.true_log_enrichment[i],
                    marker="x", linestyle="none", color=style.truth_color,
                    markersize=options.truth_marker_size, alpha=0.72, zorder=5,
                )

        x_relation = result.relation_log_mass
        input_relation = (
            config.synthetic.true_intercept + config.synthetic.true_slope * x_relation
        )
        axis_a.plot(
            x_relation, input_relation,
            color=style.truth_color,
            label="Synthetic input",
            **options.input_relation_kwargs,
        )
        panel_a_handles = [
            Patch(facecolor=style.posterior_color, edgecolor=style.posterior_color,
                  alpha=0.45, label="Individual retrieval posterior"),
            Line2D([0], [0], marker="o", color="black", linestyle="none",
                   markersize=options.posterior_median_marker_size + 0.7,
                   label="Posterior median"),
            Line2D([0], [0], color=style.truth_color,
                   label="Synthetic input", **options.input_relation_kwargs),
            Patch(facecolor=style.upper_limit_color,
                  edgecolor=style.upper_limit_color, alpha=0.25,
                  hatch=options.upper_limit_hatch, label="Upper-limit-like posterior"),
        ]
        axis_a.legend(handles=panel_a_handles, **options.panel_a_legend_kwargs)

        # Panel (b): keep every individual constraint visible in the background.
        for i in order:
            color = (
                style.upper_limit_color
                if population.information_class[i] == "upper_limit"
                else style.posterior_color
            )
            axis_b.vlines(
                population.log_masses[i],
                population.posterior_q16[i],
                population.posterior_q84[i],
                color=color,
                linewidth=0.8,
                alpha=options.background_interval_alpha,
                zorder=1,
            )
            axis_b.plot(
                population.log_masses[i], population.posterior_median[i],
                marker="v" if population.information_class[i] == "upper_limit" else "o",
                linestyle="none", color=color, markersize=2.3,
                alpha=0.45, zorder=2,
            )

        axis_b.fill_between(
            x_relation,
            result.predictive_low,
            result.predictive_high,
            color=style.recovered_color,
            alpha=options.predictive_band_alpha,
            linewidth=0.0,
            zorder=2,
        )
        axis_b.fill_between(
            x_relation,
            result.relation_low,
            result.relation_high,
            color=style.recovered_color,
            alpha=options.mean_band_alpha,
            linewidth=0.0,
            zorder=3,
        )
        axis_b.plot(
            x_relation,
            result.relation_median,
            color=style.recovered_color,
            label="Recovered relation",
            **options.recovered_relation_kwargs,
        )
        if options.show_synthetic_truth_relation:
            axis_b.plot(
                x_relation,
                input_relation,
                color=style.truth_color,
                label="Synthetic input",
                **options.truth_relation_kwargs,
            )
        axis_b.axhline(
            result.null_relation_median,
            color=style.null_color,
            label="Flat null relation",
            **options.null_relation_kwargs,
        )

        panel_b_handles = [
            Line2D([0], [0], color=style.recovered_color,
                   label="Recovered relation",
                   **options.recovered_relation_kwargs),
            Patch(facecolor=style.recovered_color, alpha=options.mean_band_alpha,
                  label="68% mean relation"),
            Patch(facecolor=style.recovered_color, alpha=options.predictive_band_alpha,
                  label="68% predictive"),
            Line2D([0], [0], color=style.truth_color,
                   label="Synthetic input", **options.truth_relation_kwargs),
            Line2D([0], [0], color=style.null_color,
                   label="Flat null relation", **options.null_relation_kwargs),
        ]
        axis_b.legend(handles=panel_b_handles, **options.panel_b_legend_kwargs)

        annotation = (
            rf"$\beta = ${_format_interval(result.quantiles['beta'])}" + "\n"
            + rf"$\sigma_{{\rm int}} = ${_format_interval(result.quantiles['sigma_intrinsic'])}"
        )
        axis_b.text(
            *options.hyperparameter_annotation_axes_xy,
            annotation,
            transform=axis_b.transAxes,
            ha="left", va="bottom",
            fontsize=options.hyperparameter_annotation_fontsize,
        )

        _format_axis(axis_a, config)
        _format_axis(axis_b, config)
        _panel_label(axis_a, "(a)", config)
        _panel_label(axis_b, "(b)", config)

        inset = axis_b.inset_axes(options.inset_bounds)
        beta_plot = np.linspace(config.grid.beta_range[0], config.grid.beta_range[1], 350)
        beta_kde = gaussian_kde(result.beta_samples)
        beta_density = beta_kde(beta_plot)
        beta_q16, beta_q50, beta_q84 = result.quantiles["beta"]
        inset.plot(beta_plot, beta_density, color=options.inset_density_color,
                   linewidth=1.25)
        inside = (beta_plot >= beta_q16) & (beta_plot <= beta_q84)
        inset.fill_between(
            beta_plot[inside], 0.0, beta_density[inside],
            color=options.inset_density_color,
            alpha=options.inset_interval_alpha,
            linewidth=0.0,
        )
        inset.axvline(beta_q50, color=options.inset_density_color,
                      linewidth=1.0)
        inset.axvline(0.0, color=style.null_color, linestyle="--", linewidth=1.0)
        inset.text(
            *options.no_mass_dependence_text_xy,
            options.no_mass_dependence_label,
            transform=inset.get_xaxis_transform(),
            ha="right", va="top",
            fontsize=options.no_mass_dependence_fontsize,
            color=style.null_color,
        )
        inset.set_xlabel(options.inset_x_label, fontsize=options.inset_axis_label_size,
                         labelpad=1.5)
        inset.set_yticks([])
        inset.tick_params(axis="x", labelsize=options.inset_tick_label_size,
                          top=True, direction="in", pad=1.5)
        inset.set_xlim(config.grid.beta_range)
        inset.set_ylim(bottom=0.0)

        if options.show_synthetic_label_each_panel:
            for axis in (axis_a, axis_b):
                axis.text(
                    0.5, 0.985, options.synthetic_label,
                    transform=axis.transAxes,
                    ha="center", va="top",
                    fontsize=style.annotation_size,
                    fontstyle="italic",
                )
        else:
            fig.text(
                *options.synthetic_label_figure_xy,
                options.synthetic_label,
                ha="center", va="top",
                fontsize=style.annotation_size,
                fontstyle="italic",
            )

        return fig, {"individual": axis_a, "population": axis_b, "slope": inset}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_products(
    population: SyntheticPopulation,
    result: HierarchicalResult,
    figure: mpl.figure.Figure,
    config: FigureConfig,
    script_path: Optional[Path] = None,
) -> Mapping[str, Path]:
    """Save synthetic data, HBM products, metadata, PDF, and PNG."""

    data_dir = config.paths.data_directory
    figure_dir = config.paths.figure_directory
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    population_csv = data_dir / "synthetic_planet_population.csv"
    with population_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "synthetic_planet_id", "mass_mj", "log10_mass_mj",
            "true_log10_zatm_over_zstar", "information_class",
            "posterior_form", "posterior_scale_dex",
            "upper_limit_log10_zatm_over_zstar", "posterior_q16",
            "posterior_median", "posterior_q84", "uniform_prior_density",
        ])
        for i in range(config.synthetic.n_planets):
            writer.writerow([
                population.names[i], population.masses_mj[i],
                population.log_masses[i], population.true_log_enrichment[i],
                population.information_class[i], population.posterior_form[i],
                population.posterior_scale[i], population.upper_limit[i],
                population.posterior_q16[i], population.posterior_median[i],
                population.posterior_q84[i], population.prior_density[i],
            ])

    posterior_path = data_dir / "synthetic_individual_posteriors.npz"
    np.savez_compressed(
        posterior_path,
        synthetic_planet_id=population.names,
        mass_mj=population.masses_mj,
        log10_mass_mj=population.log_masses,
        true_log10_zatm_over_zstar=population.true_log_enrichment,
        information_class=population.information_class,
        posterior_form=population.posterior_form,
        posterior_samples=population.posterior_samples,
        prior_density=population.prior_density,
    )

    hyper_path = data_dir / "hierarchical_posterior_products.npz"
    np.savez_compressed(
        hyper_path,
        alpha_grid=result.alpha_grid,
        beta_grid=result.beta_grid,
        sigma_grid=result.sigma_grid,
        log_posterior_grid=result.log_posterior_grid,
        alpha_samples=result.alpha_samples,
        beta_samples=result.beta_samples,
        sigma_intrinsic_samples=result.sigma_samples,
        null_alpha_samples=result.null_alpha_samples,
        null_sigma_intrinsic_samples=result.null_sigma_samples,
        relation_log10_mass_mj=result.relation_log_mass,
        relation_median=result.relation_median,
        relation_q16=result.relation_low,
        relation_q84=result.relation_high,
        predictive_q16=result.predictive_low,
        predictive_q84=result.predictive_high,
        importance_ess=result.importance_ess,
    )

    script_path = Path(script_path).resolve() if script_path is not None else None
    metadata = {
        "figure_type": "synthetic_explanatory_demonstration",
        "observational_result": False,
        "description": (
            "Synthetic demonstration of posterior-sample-reweighted hierarchical "
            "inference for a gas-planet atmospheric mass-metallicity relation."
        ),
        "relation": (
            "log10(Z_atm/Z_star) = alpha + beta*log10(M_p/M_J) + epsilon; "
            "epsilon ~ Normal(0, sigma_intrinsic)"
        ),
        "config": _jsonable(asdict(config)),
        "recovered_quantiles_16_50_84": _jsonable(result.quantiles),
        "log_evidence_full": result.log_evidence_full,
        "log_evidence_null": result.log_evidence_null,
        "log_bayes_factor_full_vs_null": result.log_bayes_factor_full_vs_null,
        "grid_convergence": _jsonable(result.convergence),
        "validation": _jsonable(result.validation),
        "method_reference": {
            "citation": "Lustig-Yaeger et al. 2022, AJ, 163, 140",
            "doi": "10.3847/1538-3881/ac5034",
            "arxiv": "2202.00701",
            "local_pdf": str(config.paths.reference_paper),
            "local_pdf_sha256": _sha256(config.paths.reference_paper),
            "note": (
                "The reference motivates HBAR; no values were digitized from it. "
                "All plotted planets and numerical values are synthetic."
            ),
        },
        "script": None if script_path is None else str(script_path),
        "script_sha256": None if script_path is None else _sha256(script_path),
    }
    metadata_path = data_dir / "synthetic_truth_and_recovery.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    pdf_path = figure_dir / f"{config.paths.output_stem}.pdf"
    png_path = figure_dir / f"{config.paths.output_stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=300)

    return {
        "population_csv": population_csv,
        "posterior_samples": posterior_path,
        "hierarchical_products": hyper_path,
        "metadata": metadata_path,
        "pdf": pdf_path,
        "png": png_path,
    }


def run_demo(
    config: FigureConfig,
    script_path: Optional[Path] = None,
) -> Tuple[SyntheticPopulation, HierarchicalResult, mpl.figure.Figure,
           Mapping[str, mpl.axes.Axes], Mapping[str, Path]]:
    """Generate, infer, plot, validate, and save the complete demonstration."""

    population = generate_synthetic_population(config)
    result = infer_hierarchical_population(population, config)
    figure, axes = make_figure(population, result, config)
    paths = save_products(population, result, figure, config, script_path)
    return population, result, figure, axes, paths


if __name__ == "__main__":
    run_dir = Path(__file__).resolve().parents[1]
    cfg = default_config(run_dir)
    pop, inferred, fig, _, outputs = run_demo(cfg, Path(__file__))
    print("Synthetic HBM demonstration complete.")
    print("Recovered beta (16/50/84):", inferred.quantiles["beta"])
    print("Recovered sigma_int (16/50/84):", inferred.quantiles["sigma_intrinsic"])
    print("Validation:", inferred.validation)
    for label, path in outputs.items():
        print(f"{label}: {path}")
    plt.close(fig)

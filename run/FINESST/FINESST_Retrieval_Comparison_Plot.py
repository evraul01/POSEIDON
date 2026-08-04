"""Build the FINESST TOI-270 d PyMultiNest versus SBI comparison figure.

This module reads completed POSEIDON products only. It does not run a
retrieval or evaluate the atmospheric forward model. The public entry points
are ``default_config()``, ``load_products()``, ``make_comparison_figure()``,
and ``save_comparison_figure()``.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from scipy.stats import gaussian_kde


@dataclass
class PathConfig:
    """Relative or absolute paths for every input and output product."""

    observation_paths: Sequence[Path]
    pmn_samples_path: Path
    sbi_samples_path: Path
    pmn_spectrum_path: Path
    sbi_spectrum_path: Path
    pmn_results_path: Path
    sbi_results_path: Path
    runtime_table_path: Path
    output_directory: Path
    sbi_log_path: Optional[Path] = None


@dataclass
class StyleConfig:
    """Centralized typography and color settings."""

    pmn_color: str = "#0072B2"
    sbi_color: str = "#D55E00"
    data_color: str = "#262626"
    projected_region_color: str = "#7A7A7A"
    use_sans_serif: bool = False
    serif_font: str = "DejaVu Serif"
    sans_serif_font: str = "DejaVu Sans"
    axis_label_size: float = 12.0
    tick_label_size: float = 9.5
    legend_size: float = 8.5
    annotation_size: float = 8.5
    panel_label_size: float = 13.0
    line_width: float = 1.7
    errorbar_line_width: float = 0.9
    tick_width: float = 0.8
    tick_length: float = 4.0
    minor_tick_length: float = 2.2

    def rc_params(self) -> Dict[str, Any]:
        if self.use_sans_serif:
            font_settings = {
                "font.family": "sans-serif",
                "font.sans-serif": [self.sans_serif_font],
                "mathtext.fontset": "dejavusans",
            }
        else:
            font_settings = {
                "font.family": "serif",
                "font.serif": [self.serif_font],
                "mathtext.fontset": "dejavuserif",
            }
        return {
            **font_settings,
            "axes.labelsize": self.axis_label_size,
            "xtick.labelsize": self.tick_label_size,
            "ytick.labelsize": self.tick_label_size,
            "legend.fontsize": self.legend_size,
            "axes.linewidth": self.tick_width,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": self.tick_width,
            "ytick.major.width": self.tick_width,
            "xtick.minor.width": self.tick_width,
            "ytick.minor.width": self.tick_width,
            "xtick.major.size": self.tick_length,
            "ytick.major.size": self.tick_length,
            "xtick.minor.size": self.minor_tick_length,
            "ytick.minor.size": self.minor_tick_length,
            "path.simplify": True,
            "path.simplify_threshold": 0.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }


@dataclass
class LayoutConfig:
    """Figure and subplot-mosaic controls."""

    mosaic: str = """
        AAAAABBB
        AAAAABBB
        AAAAABBB
        MMMMMMMM
        ccddeeff
        gghhiijj
    """
    spectrum_axis_key: str = "A"
    runtime_axis_key: str = "B"
    metrics_axis_key: str = "M"
    posterior_axis_keys: Sequence[str] = tuple("cdefghij")
    auto_posterior_layout: bool = True
    show_runtime_panel: bool = True
    posterior_columns: int = 4
    posterior_column_units: int = 2
    top_rows: int = 3
    top_width_fraction: float = 0.60
    top_row_height: float = 1.15
    metrics_row_height: float = 0.48
    posterior_row_height: float = 1.12
    top_panel_extra_horizontal_space: float = 0.025
    figure_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "figsize": (7.5, 6.6),
            "constrained_layout": False,
        }
    )
    subplot_mosaic_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "gridspec_kw": {
                "height_ratios": [1.15, 1.15, 1.15, 0.48, 1.12, 1.12],
                "wspace": 0.92,
                "hspace": 0.72,
            }
        }
    )
    subplots_adjust_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "left": 0.085,
            "right": 0.985,
            "bottom": 0.085,
            "top": 0.985,
        }
    )


@dataclass
class SpectrumConfig:
    """Panel (a) data, band, line, and legend controls."""

    observation_labels: Sequence[str] = (
        "NIRISS SOSS",
        "NIRISS SOSS",
        "NIRSpec G395H",
        "NIRSpec G395H",
    )
    observation_markers: Sequence[str] = ("o", "o", "D", "D")
    wavelength_scale: float = 1.0
    depth_scale: float = 1.0e6
    x_label: str = r"Wavelength ($\mu$m)"
    y_label: str = "Transit Depth (ppm)"
    x_limits: Optional[Tuple[float, float]] = (0.58, 5.30)
    y_limits: Optional[Tuple[float, float]] = None
    show_data_bin_width: bool = False
    data_n_wavelength_bins: Optional[int] = None
    show_minor_ticks: bool = True
    R_to_bin: Optional[float] = 100.0
    pmn_band_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"alpha": 0.15, "linewidth": 0.0, "zorder": 1}
    )
    sbi_band_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": 0.12,
            "linewidth": 0.4,
            "zorder": 2,
        }
    )
    pmn_line_kwargs: Dict[str, Any] = field(default_factory=dict)
    sbi_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "--"}
    )
    data_errorbar_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "linestyle": "none",
            "capsize": 0,
            "markerfacecolor": "white",
            "markeredgewidth": 0.8,
            "markersize": 3.5,
            "alpha": 0.92,
            "zorder": 6,
        }
    )
    legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "upper right",
            "frameon": False,
            "ncol": 2,
            "handlelength": 2.0,
            "columnspacing": 0.9,
            "handletextpad": 0.5,
        }
    )


@dataclass
class RuntimeConfig:
    """Panel (b) runtime columns, regimes, styling, and annotations."""

    parameter_column: str = "params"
    pmn_runtime_column: str = "PMN_runtime"
    sbi_runtime_column: str = "SBI_runtime"
    measured_max_parameters: int = 13
    benchmark_parameter_count: int = 13
    x_label: str = "Number of free parameters"
    y_label: str = "Wall-clock time (h)"
    x_label_size: float = 11.0
    y_label_size: float = 10.5
    y_label_pad: float = 1.0
    x_limits: Optional[Tuple[float, float]] = (0.5, 20.5)
    y_limits: Optional[Tuple[float, float]] = None
    y_scale: str = "log"
    measured_marker: str = "o"
    projected_marker: str = "o"
    show_markers: bool = True
    marker_size: float = 4.2
    line_width: float = 1.7
    measured_line_kwargs: Dict[str, Any] = field(default_factory=dict)
    projected_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "--", "markerfacecolor": "white"}
    )
    projected_between_curves_fill_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "facecolor": "none",
            "alpha": 0.22,
            "hatch": "////",
            "linewidth": 0.5,
            "zorder": 0,
        }
    )
    projected_below_sbi_fill_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": 0.055,
            "linewidth": 0.0,
            "zorder": -1,
        }
    )
    boundary_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": ":", "linewidth": 1.1, "alpha": 0.8}
    )
    show_grid: bool = True
    x_major_tick_interval: Optional[float] = 2.0
    show_x_minor_ticks: bool = True
    x_minor_tick_subdivisions: int = 2
    fill_full_x_range: bool = True
    grid_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "axis": "y",
            "which": "major",
            "alpha": 0.16,
            "linewidth": 0.6,
        }
    )
    benchmark_annotation_axes_xy: Tuple[float, float] = (0.60, 0.95)
    measured_label_axes_xy: Tuple[float, float] = (0.05, 0.06)
    projected_label_axes_xy: Tuple[float, float] = (0.69, 0.06)
    show_benchmark_annotation: bool = True
    show_regime_labels: bool = True
    benchmark_annotation_text: Optional[str] = None
    benchmark_annotation_fontsize: Optional[float] = None
    show_week_annotation: bool = True
    week_hours: float = 168.0
    week_annotation_text: str = "1 week"
    week_annotation_axes_x: float = 0.035
    week_tick_axes_width: float = 0.025
    week_annotation_fontsize: Optional[float] = None
    week_annotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "left", "va": "center", "color": "black"}
    )
    week_tick_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"color": "black", "linewidth": 0.8, "zorder": 7}
    )
    show_speedup_arrow: bool = True
    speedup_arrow_x: float = 12.6
    speedup_arrow_start_hours: Optional[float] = None
    speedup_arrow_end_hours: Optional[float] = None
    speedup_arrow_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "arrowstyle": "->",
            "color": "black",
            "linewidth": 0.9,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        }
    )
    show_speedup_annotation: bool = True
    speedup_annotation_text: Optional[str] = None
    speedup_annotation_xy: Tuple[float, float] = (12.25, 8.8)
    speedup_annotation_fontsize: Optional[float] = None
    speedup_annotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "right", "va": "center", "color": "black"}
    )
    measured_label_text: str = "Measured"
    projected_label_text: str = "Projected\n(Photochem)"
    regime_label_fontsize: Optional[float] = None
    benchmark_annotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "right", "va": "top"}
    )
    regime_label_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "left", "va": "bottom"}
    )
    measured_label_kwargs: Dict[str, Any] = field(default_factory=dict)
    projected_label_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "center", "multialignment": "center"}
    )
    show_regime_in_legend: bool = False
    legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "center left",
            "bbox_to_anchor": (0.01, 0.61),
            "frameon": False,
            "handlelength": 2.1,
            "borderaxespad": 0.3,
        }
    )


@dataclass
class PosteriorConfig:
    """Panel (c) parameter selection, density estimation, and styling."""

    parameters: Sequence[str] = (
        "R_p_ref",
        "T",
        "log_H2O",
        "log_CH4",
        "log_CO2",
        "log_SO2",
        "log_CO",
        "log_CS2",
    )
    labels: Mapping[str, str] = field(
        default_factory=lambda: {
            "R_p_ref": r"$R_{\mathrm{p,ref}}$ ($R_\oplus$)",
            "T": r"$T$ (K)",
            "log_H2O": r"$\log \, \mathrm{H_2O}$",
            "log_CH4": r"$\log \, \mathrm{CH_4}$",
            "log_CO2": r"$\log \, \mathrm{CO_2}$",
            "log_SO2": r"$\log \, \mathrm{SO_2}$",
            "log_CO": r"$\log \, \mathrm{CO}$",
            "log_CS2": r"$\log \, \mathrm{CS_2}$",
            "log_NH3": r"$\log \, \mathrm{NH_3}$",
            "log_N2": r"$\log \, \mathrm{N_2}$",
            "log_P_cloud": r"$\log \, P_{\mathrm{cloud}}$",
            "log_a": r"$\log \, a$",
            "gamma": r"$\gamma$",
        }
    )
    x_limits: Mapping[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    density_points: int = 400
    bandwidth: Any = "scott"
    max_kde_samples: int = 10000
    random_seed: int = 270
    density_line_kwargs: Dict[str, Any] = field(default_factory=dict)
    density_fill_alpha: float = 0.10
    median_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "-", "linewidth": 2.0, "alpha": 0.7}
    )
    interval_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "--", "linewidth": 1.0, "alpha": 0.65}
    )
    max_x_ticks: int = 4
    suppress_y_ticks: bool = True
    x_label_size: float = 9.5
    x_tick_size: float = 8.5
    shared_legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "lower right",
            "bbox_to_anchor": (1.0, -0.02),
            "frameon": False,
            "ncol": 2,
            "handlelength": 2.2,
            "columnspacing": 1.2,
        }
    )
    show_shared_legend: bool = True
    show_validation_metrics: bool = True
    metrics_text_xy: Tuple[float, float] = (0.055, 0.50)
    metrics_text_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "left", "va": "center"}
    )
    median_agreement_threshold_sigma: float = 0.25


@dataclass
class SaveConfig:
    """Output names and save options."""

    basename: str = "figure1_TOI270d_NPSE_validation"
    save_pdf: bool = True
    save_png: bool = True
    save_metadata: bool = True
    png_dpi: int = 300
    savefig_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"bbox_inches": "tight", "pad_inches": 0.03}
    )


@dataclass
class ComparisonConfig:
    """Complete user-editable configuration for the three-panel figure."""

    paths: PathConfig
    style: StyleConfig = field(default_factory=StyleConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    pmn_label: str = "PyMultiNest"
    sbi_label: str = "NPSE"
    panel_labels: Mapping[str, str] = field(
        default_factory=lambda: {"spectrum": "(a)", "runtime": "(b)", "posterior": "(c)"}
    )
    runtime_scope: str = (
        "All opacity interpolation and inference/training costs; same hardware for both methods."
    )
    model_comparison_note: str = (
        "Selected PyMultiNest run uses CLR priors and R=30000; selected NPSE run uses "
        "independent uniform log-abundance priors and R=20000."
    )


@dataclass
class PosteriorSamples:
    names: Tuple[str, ...]
    values: np.ndarray
    path: Path


@dataclass
class RetrievedSpectrum:
    wavelength_um: np.ndarray
    low2: np.ndarray
    low1: np.ndarray
    median: np.ndarray
    high1: np.ndarray
    high2: np.ndarray
    path: Path


@dataclass
class ObservationSegment:
    wavelength_um: np.ndarray
    half_width_um: np.ndarray
    depth: np.ndarray
    uncertainty: np.ndarray
    label: str
    marker: str
    path: Path


@dataclass
class RuntimeData:
    parameters: np.ndarray
    pmn_hours: np.ndarray
    sbi_hours: np.ndarray
    path: Path


@dataclass
class LoadedProducts:
    pmn_samples: PosteriorSamples
    sbi_samples: PosteriorSamples
    pmn_spectrum: RetrievedSpectrum
    sbi_spectrum: RetrievedSpectrum
    observations: List[ObservationSegment]
    runtime: RuntimeData
    pmn_summary: Dict[str, Any]
    sbi_summary: Dict[str, Any]


def locate_run_directory(start: Optional[Path] = None) -> Path:
    """Locate ``POSEIDON/run`` without embedding a machine-specific path."""

    candidates: List[Path] = []
    if start is not None:
        candidates.append(Path(start).expanduser().resolve())
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd().resolve())

    for candidate in candidates:
        if candidate.name == "run" and (candidate / "data").exists():
            return candidate
        if candidate.name in ("SBI", "HBM") and (candidate.parent / "data").exists():
            return candidate.parent
        if (candidate / "run" / "data").exists():
            return candidate / "run"
        for parent in candidate.parents:
            if parent.name == "run" and (parent / "data").exists():
                return parent
            if (parent / "run" / "data").exists():
                return parent / "run"

    raise FileNotFoundError("Could not locate the POSEIDON run directory from the current location.")


def default_config(run_directory: Optional[Path] = None) -> ComparisonConfig:
    """Return the configured TOI-270 d PMN/NPSE comparison used in the proposal."""

    run_dir = locate_run_directory(run_directory)
    pmn_root = (
        run_dir
        / "POSEIDON_output"
        / "TOI-270d_all"
        / "TOI-270d_final"
        / "retrievals"
    )
    sbi_root = run_dir / "POSEIDON_output" / "TOI-270d" / "retrievals"
    pmn_name = "multigas_CLR_HIGHR_2000_NIRISS_G395H_Tiberius"
    sbi_name = "NPSE_joint_multigas_64000_NIRISS_G395H_Tiberius"

    paths = PathConfig(
        observation_paths=(
            run_dir / "data" / "TOI-270d" / "TOI-270d_NIRISS_SOSS_Ord2_ExoTEP.dat",
            run_dir / "data" / "TOI-270d" / "TOI-270d_NIRISS_SOSS_Ord1_ExoTEP.dat",
            run_dir / "data" / "TOI-270d" / "TOI-270d_NIRSpec_G395H_NRS1_Tiberius.dat",
            run_dir / "data" / "TOI-270d" / "TOI-270d_NIRSpec_G395H_NRS2_Tiberius.dat",
        ),
        pmn_samples_path=pmn_root / "samples" / f"{pmn_name}_samples.txt",
        sbi_samples_path=sbi_root / "samples" / f"{sbi_name}_samples.txt",
        pmn_spectrum_path=pmn_root / "samples" / f"{pmn_name}_spectrum_retrieved.txt",
        sbi_spectrum_path=sbi_root / "samples" / f"{sbi_name}_spectrum_retrieved.txt",
        pmn_results_path=pmn_root / "results" / f"{pmn_name}_results.txt",
        sbi_results_path=sbi_root / "results" / f"{sbi_name}_results.txt",
        runtime_table_path=run_dir / "data" / "FINESST_26" / "Figure_1" / "runtime.csv",
        output_directory=run_dir / "POSEIDON_output" / "TOI-270d" / "plots" / "FINESST",
        sbi_log_path=sbi_root / "sbi-logs" / f"{sbi_name}.log",
    )
    return ComparisonConfig(paths=paths)


def configure_posterior_layout(
    config: ComparisonConfig,
    n_columns: Optional[int] = None,
) -> None:
    """Resize the posterior mosaic to the configured number of parameters.

    The validation strip and posterior axes are added automatically. When
    ``show_runtime_panel`` is false, the spectrum fills the complete top row;
    otherwise the original spectrum/runtime split is retained. Set
    ``config.layout.auto_posterior_layout = False`` to supply a completely
    custom mosaic and ``posterior_axis_keys`` instead.
    """

    n_parameters = len(config.posterior.parameters)
    if n_parameters < 1:
        raise ValueError("At least one posterior parameter must be requested.")
    columns_requested = n_columns or config.layout.posterior_columns
    if columns_requested < 1:
        raise ValueError("posterior_columns must be at least one.")
    n_columns_used = min(columns_requested, n_parameters)
    units = config.layout.posterior_column_units
    if units < 1:
        raise ValueError("posterior_column_units must be at least one.")

    available_keys = [
        key
        for key in "cdefghijklmnopqrstuvwxyzCDEFGHIJKLNOPQRSTUVWXYZ0123456789"
        if key not in {"A", "B", "M"}
    ]
    if n_parameters > len(available_keys):
        raise ValueError(f"Automatic mosaic supports at most {len(available_keys)} posterior axes.")
    posterior_keys = available_keys[:n_parameters]

    total_width = n_columns_used * units
    if config.layout.show_runtime_panel:
        spectrum_width = int(round(config.layout.top_width_fraction * total_width))
        spectrum_width = min(max(spectrum_width, 1), total_width - 1)
        top_line = "A" * spectrum_width + "B" * (total_width - spectrum_width)
    else:
        top_line = "A" * total_width
    lines = [top_line] * config.layout.top_rows
    lines.append("M" * total_width)

    n_rows = int(np.ceil(n_parameters / n_columns_used))
    for row in range(n_rows):
        row_keys = posterior_keys[row * n_columns_used : (row + 1) * n_columns_used]
        posterior_line = "".join(key * units for key in row_keys)
        posterior_line += "." * (total_width - len(posterior_line))
        lines.append(posterior_line)

    config.layout.mosaic = "\n".join(lines)
    config.layout.posterior_axis_keys = tuple(posterior_keys)
    config.layout.subplot_mosaic_kwargs.setdefault("gridspec_kw", {})["height_ratios"] = (
        [config.layout.top_row_height] * config.layout.top_rows
        + [config.layout.metrics_row_height]
        + [config.layout.posterior_row_height] * n_rows
    )


def _require_file(path: Path, product_name: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {product_name}: {path}")
    return path


def load_poseidon_samples(path: Path) -> PosteriorSamples:
    """Load a POSEIDON equal-weight ``*_samples.txt`` posterior."""

    path = _require_file(path, "posterior samples")
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    names = tuple(part.strip() for part in header.split("|") if part.strip())
    values = np.loadtxt(path, skiprows=1, ndmin=2)

    if not names:
        raise ValueError(f"No parameter names found in sample header: {path}")
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(
            f"Sample shape {values.shape} does not match {len(names)} parameter names in {path}."
        )
    if not np.isfinite(values).all():
        bad_rows = int(np.sum(~np.isfinite(values).all(axis=1)))
        raise ValueError(f"Posterior contains {bad_rows} non-finite rows: {path}")
    return PosteriorSamples(names=names, values=values, path=path)


def load_retrieved_spectrum(path: Path) -> RetrievedSpectrum:
    """Load and validate a POSEIDON ``*_spectrum_retrieved.txt`` product."""

    path = _require_file(path, "retrieved spectrum")
    array = np.loadtxt(path, skiprows=1, ndmin=2)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"Expected six spectrum columns, found shape {array.shape}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Retrieved spectrum contains NaN or Inf values: {path}")
    if np.any(array[:, 0] <= 0.0) or np.any(np.diff(array[:, 0]) <= 0.0):
        raise ValueError(f"Spectrum wavelength grid is not positive and strictly increasing: {path}")

    quantiles = array[:, 1:]
    ordered = np.all(np.diff(quantiles, axis=1) >= 0.0, axis=1)
    if not np.all(ordered):
        count = int(np.sum(~ordered))
        raise ValueError(
            f"Retrieved spectrum has {count}/{len(array)} rows with non-monotonic "
            f"credible quantiles: {path}"
        )

    return RetrievedSpectrum(
        wavelength_um=array[:, 0],
        low2=array[:, 1],
        low1=array[:, 2],
        median=array[:, 3],
        high1=array[:, 4],
        high2=array[:, 5],
        path=path,
    )


def load_observations(config: ComparisonConfig) -> List[ObservationSegment]:
    """Load each configured POSEIDON four-column observational segment."""

    paths = config.paths.observation_paths
    labels = config.spectrum.observation_labels
    markers = config.spectrum.observation_markers
    if not (len(paths) == len(labels) == len(markers)):
        raise ValueError("Observation paths, labels, and markers must have the same length.")

    segments: List[ObservationSegment] = []
    for path_in, label, marker in zip(paths, labels, markers):
        path = _require_file(path_in, "observational spectrum")
        array = np.loadtxt(path, ndmin=2)
        if array.ndim != 2 or array.shape[1] < 4:
            raise ValueError(f"Expected at least four observational columns: {path}")
        array = array[:, :4]
        if not np.isfinite(array).all():
            raise ValueError(f"Observational data contain NaN or Inf values: {path}")
        if np.any(array[:, 0] <= 0.0) or np.any(array[:, 3] <= 0.0):
            raise ValueError(f"Observation wavelengths and uncertainties must be positive: {path}")
        segments.append(
            ObservationSegment(
                wavelength_um=array[:, 0],
                half_width_um=array[:, 1],
                depth=array[:, 2],
                uncertainty=array[:, 3],
                label=label,
                marker=marker,
                path=path,
            )
        )
    return segments


def load_runtime_table(path: Path, config: RuntimeConfig) -> RuntimeData:
    """Load the configured runtime columns from a CSV file."""

    path = _require_file(path, "runtime table")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = (
        config.parameter_column,
        config.pmn_runtime_column,
        config.sbi_runtime_column,
    )
    if not rows:
        raise ValueError(f"Runtime table is empty: {path}")
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"Runtime table is missing columns {missing}: {path}")

    parameters = np.array([float(row[required[0]]) for row in rows])
    pmn_hours = np.array([float(row[required[1]]) for row in rows])
    sbi_hours = np.array([float(row[required[2]]) for row in rows])
    order = np.argsort(parameters)
    parameters = parameters[order]
    pmn_hours = pmn_hours[order]
    sbi_hours = sbi_hours[order]
    if not np.isfinite(np.column_stack((parameters, pmn_hours, sbi_hours))).all():
        raise ValueError(f"Runtime table contains NaN or Inf values: {path}")
    if np.any(pmn_hours <= 0.0) or np.any(sbi_hours <= 0.0):
        raise ValueError(f"Runtime values must be positive for a logarithmic axis: {path}")
    if len(np.unique(parameters)) != len(parameters):
        raise ValueError(f"Runtime table contains duplicate parameter counts: {path}")
    return RuntimeData(parameters, pmn_hours, sbi_hours, path)


def parse_poseidon_summary(path: Path) -> Dict[str, Any]:
    """Extract plotting provenance from a POSEIDON results summary."""

    path = _require_file(path, "retrieval summary")
    text = path.read_text(encoding="utf-8", errors="replace")

    def match(pattern: str, converter: Any = str) -> Any:
        result = re.search(pattern, text, flags=re.MULTILINE)
        return converter(result.group(1)) if result else None

    return {
        "path": str(path),
        "model": match(r"^Model:\s*(.+)$"),
        "algorithm": match(r"^Algorithm\s*=\s*(.+)$"),
        "n_parameters": match(r"^N_params\s*=\s*(\d+)", int),
        "n_live_or_simulations": match(r"^N_live\s*=\s*(\d+)", int),
        "reduced_chi_square": match(r"chi\^2_red\s*=\s*([-+0-9.eE]+)", float),
        "chi_square": match(r"^-> chi\^2\s*=\s*([-+0-9.eE]+)", float),
        "resolution": match(r"@ R\s*=\s*(\d+)", int),
    }


def load_products(config: ComparisonConfig, verbose: bool = True) -> LoadedProducts:
    """Load all configured products and perform cross-product validation."""

    pmn_samples = load_poseidon_samples(config.paths.pmn_samples_path)
    sbi_samples = load_poseidon_samples(config.paths.sbi_samples_path)
    if pmn_samples.names != sbi_samples.names:
        raise ValueError(
            "PyMultiNest and SBI parameter names differ.\n"
            f"PyMultiNest: {pmn_samples.names}\nSBI: {sbi_samples.names}"
        )

    requested = tuple(config.posterior.parameters)
    missing = [name for name in requested if name not in pmn_samples.names]
    if missing:
        raise ValueError(f"Requested posterior parameters are absent from both products: {missing}")

    pmn_spectrum = load_retrieved_spectrum(config.paths.pmn_spectrum_path)
    sbi_spectrum = load_retrieved_spectrum(config.paths.sbi_spectrum_path)
    overlap_low = max(pmn_spectrum.wavelength_um[0], sbi_spectrum.wavelength_um[0])
    overlap_high = min(pmn_spectrum.wavelength_um[-1], sbi_spectrum.wavelength_um[-1])
    if overlap_low >= overlap_high:
        raise ValueError("PyMultiNest and SBI retrieved spectra have no wavelength overlap.")

    observations = load_observations(config)
    observed_wavelengths = np.concatenate([segment.wavelength_um for segment in observations])
    if observed_wavelengths.min() < overlap_low or observed_wavelengths.max() > overlap_high:
        raise ValueError("Retrieved spectra do not span every observed wavelength.")

    runtime = load_runtime_table(config.paths.runtime_table_path, config.runtime)
    pmn_summary = parse_poseidon_summary(config.paths.pmn_results_path)
    sbi_summary = parse_poseidon_summary(config.paths.sbi_results_path)

    if verbose:
        print(f"PyMultiNest samples: {pmn_samples.values.shape} from {pmn_samples.path}")
        print(f"SBI samples:         {sbi_samples.values.shape} from {sbi_samples.path}")
        print(
            "PyMultiNest spectrum:",
            pmn_spectrum.wavelength_um.shape,
            f"[{pmn_spectrum.wavelength_um[0]:.3f}, {pmn_spectrum.wavelength_um[-1]:.3f}] um",
        )
        print(
            "SBI spectrum:        ",
            sbi_spectrum.wavelength_um.shape,
            f"[{sbi_spectrum.wavelength_um[0]:.3f}, {sbi_spectrum.wavelength_um[-1]:.3f}] um",
        )
        print(f"Observed bins: {sum(len(segment.wavelength_um) for segment in observations)}")
        print(f"Runtime rows: {len(runtime.parameters)} from {runtime.path}")
        print("Parameter names match exactly.")
        print("All arrays are finite and both spectral credible intervals are monotonic.")
        print("Model-comparison note:", config.model_comparison_note)

    return LoadedProducts(
        pmn_samples=pmn_samples,
        sbi_samples=sbi_samples,
        pmn_spectrum=pmn_spectrum,
        sbi_spectrum=sbi_spectrum,
        observations=observations,
        runtime=runtime,
        pmn_summary=pmn_summary,
        sbi_summary=sbi_summary,
    )


def _panel_label(ax: mpl.axes.Axes, label: str, style: StyleConfig) -> None:
    ax.text(
        0.02,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.panel_label_size,
        fontweight="bold",
        zorder=20,
    )


def _bin_spectrum_to_resolution(
    spectrum: RetrievedSpectrum,
    resolving_power: Optional[float],
) -> RetrievedSpectrum:
    """Average a retrieved spectrum into constant-resolution logarithmic bins."""

    if resolving_power is None:
        return spectrum
    if resolving_power <= 0.0:
        raise ValueError("R_to_bin must be positive or None.")

    log_min = np.log(spectrum.wavelength_um[0])
    log_max = np.log(spectrum.wavelength_um[-1])
    n_bins = max(2, int(np.ceil((log_max - log_min) * resolving_power)))
    edges = np.exp(np.linspace(log_min, log_max, n_bins + 1))
    bin_index = np.clip(np.digitize(spectrum.wavelength_um, edges) - 1, 0, n_bins - 1)

    columns = np.column_stack(
        (
            spectrum.wavelength_um,
            spectrum.low2,
            spectrum.low1,
            spectrum.median,
            spectrum.high1,
            spectrum.high2,
        )
    )
    binned = np.full((n_bins, columns.shape[1]), np.nan)
    for index in range(n_bins):
        selected = bin_index == index
        if np.any(selected):
            binned[index] = np.mean(columns[selected], axis=0)
    binned = binned[np.isfinite(binned).all(axis=1)]
    return RetrievedSpectrum(
        wavelength_um=binned[:, 0],
        low2=binned[:, 1],
        low1=binned[:, 2],
        median=binned[:, 3],
        high1=binned[:, 4],
        high2=binned[:, 5],
        path=spectrum.path,
    )


def _bin_observation_segments(
    segments: Sequence[ObservationSegment],
    n_wavelength_bins: Optional[int],
) -> List[ObservationSegment]:
    """Inverse-variance bin grouped observations on a common linear wavelength grid."""

    if n_wavelength_bins is None:
        return list(segments)
    if isinstance(n_wavelength_bins, bool) or int(n_wavelength_bins) != n_wavelength_bins:
        raise ValueError("data_n_wavelength_bins must be a positive integer or None.")
    n_bins = int(n_wavelength_bins)
    if n_bins < 1:
        raise ValueError("data_n_wavelength_bins must be a positive integer or None.")

    wavelength_min = min(float(segment.wavelength_um.min()) for segment in segments)
    wavelength_max = max(float(segment.wavelength_um.max()) for segment in segments)
    edges = np.linspace(wavelength_min, wavelength_max, n_bins + 1)

    grouped: Dict[str, List[ObservationSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.label, []).append(segment)

    binned_segments: List[ObservationSegment] = []
    for label, group in grouped.items():
        wavelength = np.concatenate([segment.wavelength_um for segment in group])
        depth = np.concatenate([segment.depth for segment in group])
        uncertainty = np.concatenate([segment.uncertainty for segment in group])
        bin_indices = np.clip(np.digitize(wavelength, edges) - 1, 0, n_bins - 1)

        wavelength_binned: List[float] = []
        half_width_binned: List[float] = []
        depth_binned: List[float] = []
        uncertainty_binned: List[float] = []
        for index in range(n_bins):
            selected = bin_indices == index
            if not np.any(selected):
                continue
            weights = 1.0 / np.square(uncertainty[selected])
            weight_sum = float(np.sum(weights))
            wavelength_binned.append(float(np.sum(weights * wavelength[selected]) / weight_sum))
            half_width_binned.append(float(0.5 * (edges[index + 1] - edges[index])))
            depth_binned.append(float(np.sum(weights * depth[selected]) / weight_sum))
            uncertainty_binned.append(float(np.sqrt(1.0 / weight_sum)))

        binned_segments.append(
            ObservationSegment(
                wavelength_um=np.asarray(wavelength_binned),
                half_width_um=np.asarray(half_width_binned),
                depth=np.asarray(depth_binned),
                uncertainty=np.asarray(uncertainty_binned),
                label=label,
                marker=group[0].marker,
                path=group[0].path,
            )
        )
    return binned_segments


def _plot_spectrum_panel(
    ax: mpl.axes.Axes,
    products: LoadedProducts,
    config: ComparisonConfig,
) -> None:
    style = config.style
    options = config.spectrum
    pmn = _bin_spectrum_to_resolution(products.pmn_spectrum, options.R_to_bin)
    sbi = _bin_spectrum_to_resolution(products.sbi_spectrum, options.R_to_bin)
    xscale = options.wavelength_scale
    yscale = options.depth_scale

    pmn_band_kwargs = {"color": style.pmn_color, **options.pmn_band_kwargs}
    sbi_band_kwargs = {
        "facecolor": style.sbi_color,
        "edgecolor": style.sbi_color,
        **options.sbi_band_kwargs,
    }
    ax.fill_between(
        pmn.wavelength_um * xscale,
        pmn.low1 * yscale,
        pmn.high1 * yscale,
        **pmn_band_kwargs,
    )
    ax.fill_between(
        sbi.wavelength_um * xscale,
        sbi.low1 * yscale,
        sbi.high1 * yscale,
        **sbi_band_kwargs,
    )

    pmn_line_kwargs = {
        "color": style.pmn_color,
        "linewidth": style.line_width,
        "label": config.pmn_label,
        "zorder": 4,
        **options.pmn_line_kwargs,
    }
    sbi_line_kwargs = {
        "color": style.sbi_color,
        "linewidth": style.line_width,
        "label": config.sbi_label,
        "zorder": 5,
        **options.sbi_line_kwargs,
    }
    ax.plot(pmn.wavelength_um * xscale, pmn.median * yscale, **pmn_line_kwargs)
    ax.plot(sbi.wavelength_um * xscale, sbi.median * yscale, **sbi_line_kwargs)

    display_segments = _bin_observation_segments(
        products.observations, options.data_n_wavelength_bins
    )
    for segment in display_segments:
        error_kwargs = {
            "color": style.data_color,
            "ecolor": style.data_color,
            "elinewidth": style.errorbar_line_width,
            "markeredgecolor": style.data_color,
            **options.data_errorbar_kwargs,
        }
        xerr = segment.half_width_um * xscale if options.show_data_bin_width else None
        ax.errorbar(
            segment.wavelength_um * xscale,
            segment.depth * yscale,
            xerr=xerr,
            yerr=segment.uncertainty * yscale,
            marker=segment.marker,
            label=segment.label,
            **error_kwargs,
        )

    ax.set_xlabel(options.x_label)
    ax.set_ylabel(options.y_label)
    if options.x_limits is not None:
        ax.set_xlim(*options.x_limits)
    if options.y_limits is not None:
        ax.set_ylim(*options.y_limits)
    if options.show_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    handles, labels = ax.get_legend_handles_labels()
    unique_handles: List[Any] = []
    unique_labels: List[str] = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, fontsize=style.legend_size, **options.legend_kwargs)
    _panel_label(ax, config.panel_labels["spectrum"], style)


def _runtime_value_at(runtime: RuntimeData, n_parameters: int) -> Tuple[float, float]:
    matches = np.where(np.isclose(runtime.parameters, n_parameters))[0]
    if len(matches) != 1:
        raise ValueError(f"Expected one runtime row at {n_parameters} parameters, found {len(matches)}.")
    index = int(matches[0])
    return float(runtime.pmn_hours[index]), float(runtime.sbi_hours[index])


def _plot_runtime_panel(
    ax: mpl.axes.Axes,
    products: LoadedProducts,
    config: ComparisonConfig,
) -> Dict[str, float]:
    style = config.style
    options = config.runtime
    runtime = products.runtime
    measured = runtime.parameters <= options.measured_max_parameters
    projected = runtime.parameters > options.measured_max_parameters

    measured_marker = options.measured_marker if options.show_markers else None
    projected_marker = options.projected_marker if options.show_markers else None
    measured_kwargs = {
        "linewidth": options.line_width,
        "marker": measured_marker,
        "markersize": options.marker_size,
        **options.measured_line_kwargs,
    }
    projected_kwargs = {
        "linewidth": options.line_width,
        "marker": projected_marker,
        "markersize": options.marker_size,
        **options.projected_line_kwargs,
    }
    ax.plot(
        runtime.parameters[measured],
        runtime.pmn_hours[measured],
        color=style.pmn_color,
        markerfacecolor=style.pmn_color,
        markeredgecolor=style.pmn_color,
        label=config.pmn_label,
        **measured_kwargs,
    )
    ax.plot(
        runtime.parameters[measured],
        runtime.sbi_hours[measured],
        color=style.sbi_color,
        markerfacecolor=style.sbi_color,
        markeredgecolor=style.sbi_color,
        label=config.sbi_label,
        **measured_kwargs,
    )

    projected_indices: Optional[np.ndarray] = None
    if np.any(projected):
        boundary_index = int(np.where(measured)[0][-1])
        projected_indices = np.concatenate(([boundary_index], np.where(projected)[0]))
        ax.plot(
            runtime.parameters[projected_indices],
            runtime.pmn_hours[projected_indices],
            color=style.pmn_color,
            markeredgecolor=style.pmn_color,
            **projected_kwargs,
        )
        ax.plot(
            runtime.parameters[projected_indices],
            runtime.sbi_hours[projected_indices],
            color=style.sbi_color,
            markeredgecolor=style.sbi_color,
            **projected_kwargs,
        )

    ax.set_yscale(options.y_scale)
    ax.set_xlabel(options.x_label, fontsize=options.x_label_size)
    ax.set_ylabel(options.y_label, fontsize=options.y_label_size, labelpad=options.y_label_pad)
    if options.x_limits is not None:
        ax.set_xlim(*options.x_limits)
    if options.y_limits is not None:
        ax.set_ylim(*options.y_limits)
    else:
        all_times = np.concatenate((runtime.pmn_hours, runtime.sbi_hours))
        ax.set_ylim(10 ** (np.log10(all_times.min()) - 0.20), 10 ** (np.log10(all_times.max()) + 0.20))

    boundary_x = float(options.measured_max_parameters)
    ax.axvline(boundary_x, color=style.data_color, **options.boundary_line_kwargs)
    fill_x: Optional[np.ndarray]
    pmn_fill: Optional[np.ndarray]
    sbi_fill: Optional[np.ndarray]
    if options.fill_full_x_range:
        x_axis_low, x_axis_high = ax.get_xlim()
        fill_x = runtime.parameters.copy()
        if x_axis_low < fill_x[0]:
            fill_x = np.insert(fill_x, 0, x_axis_low)
        if x_axis_high > fill_x[-1]:
            fill_x = np.append(fill_x, x_axis_high)
        pmn_fill = np.interp(fill_x, runtime.parameters, runtime.pmn_hours)
        sbi_fill = np.interp(fill_x, runtime.parameters, runtime.sbi_hours)
    elif projected_indices is not None:
        fill_x = runtime.parameters[projected_indices]
        pmn_fill = runtime.pmn_hours[projected_indices]
        sbi_fill = runtime.sbi_hours[projected_indices]
    else:
        fill_x, pmn_fill, sbi_fill = None, None, None
    if fill_x is not None and pmn_fill is not None and sbi_fill is not None:
        lower, _ = ax.get_ylim()
        below_sbi_kwargs = {
            "facecolor": style.projected_region_color,
            "edgecolor": style.projected_region_color,
            **options.projected_below_sbi_fill_kwargs,
        }
        between_curves_kwargs = {
            "edgecolor": style.projected_region_color,
            **options.projected_between_curves_fill_kwargs,
        }
        ax.fill_between(fill_x, lower, sbi_fill, **below_sbi_kwargs)
        ax.fill_between(fill_x, sbi_fill, pmn_fill, **between_curves_kwargs)

    if options.show_grid:
        ax.grid(**options.grid_kwargs)
    if options.x_major_tick_interval is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    else:
        if options.x_major_tick_interval <= 0.0:
            raise ValueError("x_major_tick_interval must be positive or None.")
        ax.xaxis.set_major_locator(MultipleLocator(options.x_major_tick_interval))
    if options.show_x_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(options.x_minor_tick_subdivisions))
    ax.tick_params(which="minor", labelsize=style.tick_label_size)

    pmn_at_benchmark, sbi_at_benchmark = _runtime_value_at(
        runtime, options.benchmark_parameter_count
    )
    speedup = pmn_at_benchmark / sbi_at_benchmark
    benchmark_text = options.benchmark_annotation_text
    if benchmark_text is None:
        benchmark_text = (
            "TOI-270 d\n"
            f"{pmn_at_benchmark:g} h $\\rightarrow$ {sbi_at_benchmark:g} h\n"
            f"$\\approx {speedup:.1f}\\times$ faster"
        )
    benchmark_fontsize = options.benchmark_annotation_fontsize or style.annotation_size
    week_fontsize = options.week_annotation_fontsize or style.annotation_size
    speedup_fontsize = options.speedup_annotation_fontsize or style.annotation_size
    regime_fontsize = options.regime_label_fontsize or style.annotation_size
    measured_label_kwargs = {
        **options.regime_label_kwargs,
        **options.measured_label_kwargs,
    }
    projected_label_kwargs = {
        **options.regime_label_kwargs,
        **options.projected_label_kwargs,
    }
    if options.show_benchmark_annotation:
        ax.text(
            *options.benchmark_annotation_axes_xy,
            benchmark_text,
            transform=ax.transAxes,
            fontsize=benchmark_fontsize,
            **options.benchmark_annotation_kwargs,
        )
    if options.show_week_annotation:
        if options.week_hours <= 0.0:
            raise ValueError("week_hours must be positive on the logarithmic runtime axis.")
        if options.week_tick_axes_width < 0.0:
            raise ValueError("week_tick_axes_width must be non-negative.")
        y_axis_transform = ax.get_yaxis_transform()
        ax.plot(
            [0.0, options.week_tick_axes_width],
            [options.week_hours, options.week_hours],
            transform=y_axis_transform,
            clip_on=True,
            **options.week_tick_kwargs,
        )
        ax.text(
            options.week_annotation_axes_x,
            options.week_hours,
            options.week_annotation_text,
            transform=y_axis_transform,
            fontsize=week_fontsize,
            **options.week_annotation_kwargs,
        )
    arrow_start = (
        pmn_at_benchmark
        if options.speedup_arrow_start_hours is None
        else options.speedup_arrow_start_hours
    )
    arrow_end = (
        sbi_at_benchmark
        if options.speedup_arrow_end_hours is None
        else options.speedup_arrow_end_hours
    )
    if options.show_speedup_arrow:
        if arrow_start <= 0.0 or arrow_end <= 0.0:
            raise ValueError(
                "speedup_arrow_start_hours and speedup_arrow_end_hours must be positive."
            )
        ax.annotate(
            "",
            xy=(options.speedup_arrow_x, arrow_end),
            xytext=(options.speedup_arrow_x, arrow_start),
            arrowprops=options.speedup_arrow_kwargs,
            annotation_clip=True,
        )
    if options.show_speedup_annotation:
        speedup_text = options.speedup_annotation_text
        if speedup_text is None:
            speedup_text = f"$\\approx {speedup:.1f}\\times$ faster"
        ax.text(
            *options.speedup_annotation_xy,
            speedup_text,
            fontsize=speedup_fontsize,
            **options.speedup_annotation_kwargs,
        )
    if options.show_regime_labels:
        ax.text(
            *options.measured_label_axes_xy,
            options.measured_label_text,
            transform=ax.transAxes,
            fontsize=regime_fontsize,
            **measured_label_kwargs,
        )
        ax.text(
            *options.projected_label_axes_xy,
            options.projected_label_text,
            transform=ax.transAxes,
            fontsize=regime_fontsize,
            **projected_label_kwargs,
        )

    method_handles = [
        Line2D([0], [0], color=style.pmn_color, lw=options.line_width, marker=measured_marker, label=config.pmn_label),
        Line2D([0], [0], color=style.sbi_color, lw=options.line_width, marker=measured_marker, label=config.sbi_label),
    ]
    if options.show_regime_in_legend:
        method_handles.extend(
            [
                Line2D([0], [0], color=style.data_color, lw=1.2, linestyle="-", label="Measured"),
                Line2D([0], [0], color=style.data_color, lw=1.2, linestyle="--", label="Projected"),
            ]
        )
    ax.legend(handles=method_handles, fontsize=style.legend_size, **options.legend_kwargs)
    _panel_label(ax, config.panel_labels["runtime"], style)
    return {
        "parameter_count": options.benchmark_parameter_count,
        "pmn_hours": pmn_at_benchmark,
        "sbi_hours": sbi_at_benchmark,
        "speedup": speedup,
    }


def _parameter_values(samples: PosteriorSamples, parameter: str) -> np.ndarray:
    return samples.values[:, samples.names.index(parameter)]


def _normalized_kde(
    values: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: Any,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    kde_values = values
    if len(kde_values) > max_samples:
        kde_values = rng.choice(kde_values, size=max_samples, replace=False)
    if np.ptp(kde_values) == 0.0:
        density = np.zeros_like(x_grid)
        density[np.argmin(np.abs(x_grid - kde_values[0]))] = 1.0
    else:
        density = gaussian_kde(kde_values, bw_method=bandwidth)(x_grid)
    trapezoid = getattr(np, "trapezoid", None)
    area = trapezoid(density, x_grid) if trapezoid is not None else np.trapz(density, x_grid)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("KDE normalization failed.")
    return density / area


def compute_validation_metrics(
    products: LoadedProducts,
    config: ComparisonConfig,
) -> Dict[str, Any]:
    """Compute posterior and predictive agreement from the loaded products."""

    parameter_metrics: Dict[str, Dict[str, Any]] = {}
    normalized_offsets: List[float] = []
    interval_overlaps: List[bool] = []
    threshold = config.posterior.median_agreement_threshold_sigma

    for parameter in products.pmn_samples.names:
        pmn_values = _parameter_values(products.pmn_samples, parameter)
        sbi_values = _parameter_values(products.sbi_samples, parameter)
        pmn_q16, pmn_median, pmn_q84 = np.quantile(pmn_values, [0.16, 0.50, 0.84])
        sbi_q16, sbi_median, sbi_q84 = np.quantile(sbi_values, [0.16, 0.50, 0.84])
        pmn_sigma = 0.5 * (pmn_q84 - pmn_q16)
        normalized_offset = (
            abs(sbi_median - pmn_median) / pmn_sigma if pmn_sigma > 0.0 else np.inf
        )
        overlap = max(pmn_q16, sbi_q16) <= min(pmn_q84, sbi_q84)
        normalized_offsets.append(float(normalized_offset))
        interval_overlaps.append(bool(overlap))
        parameter_metrics[parameter] = {
            "pmn_q16": float(pmn_q16),
            "pmn_median": float(pmn_median),
            "pmn_q84": float(pmn_q84),
            "sbi_q16": float(sbi_q16),
            "sbi_median": float(sbi_median),
            "sbi_q84": float(sbi_q84),
            "median_offset_in_pmn_sigma": float(normalized_offset),
            "intervals_68_overlap": bool(overlap),
        }

    predictive_differences: List[np.ndarray] = []
    for segment in products.observations:
        pmn_model = np.interp(
            segment.wavelength_um,
            products.pmn_spectrum.wavelength_um,
            products.pmn_spectrum.median,
        )
        sbi_model = np.interp(
            segment.wavelength_um,
            products.sbi_spectrum.wavelength_um,
            products.sbi_spectrum.median,
        )
        predictive_differences.append(np.abs(pmn_model - sbi_model) / segment.uncertainty)
    predictive = np.concatenate(predictive_differences)

    return {
        "definitions": {
            "median_offset": (
                "Absolute PMN-SBI median difference divided by the symmetric PMN "
                "68% half-width, summarized by the median over all common parameters."
            ),
            "interval_overlap": "Non-empty intersection of PMN and SBI [q16, q84] intervals.",
            "predictive_difference": (
                "Mean absolute difference between native-grid posterior-predictive medians "
                "interpolated to observed wavelengths and divided by each observed uncertainty."
            ),
        },
        "parameter_metrics": parameter_metrics,
        "n_parameters": len(normalized_offsets),
        "median_posterior_offset_sigma": float(np.median(normalized_offsets)),
        "n_medians_within_threshold": int(np.sum(np.array(normalized_offsets) <= threshold)),
        "median_threshold_sigma": float(threshold),
        "n_intervals_68_overlap": int(np.sum(interval_overlaps)),
        "mean_predictive_difference_sigma_data": float(np.mean(predictive)),
        "max_predictive_difference_sigma_data": float(np.max(predictive)),
        "coverage": None,
        "coverage_note": "Omitted: coverage requires a synthetic calibration ensemble with known truths.",
    }


def _plot_posterior_panels(
    axes: Sequence[mpl.axes.Axes],
    products: LoadedProducts,
    config: ComparisonConfig,
) -> None:
    style = config.style
    options = config.posterior
    rng = np.random.default_rng(options.random_seed)
    if len(options.parameters) > len(axes):
        raise ValueError(
            f"The mosaic provides {len(axes)} posterior axes for {len(options.parameters)} parameters."
        )

    for axis, parameter in zip(axes, options.parameters):
        pmn_values = _parameter_values(products.pmn_samples, parameter)
        sbi_values = _parameter_values(products.sbi_samples, parameter)
        custom_x_limits = options.x_limits.get(parameter)
        if custom_x_limits is not None:
            if len(custom_x_limits) != 2:
                raise ValueError(
                    f"Custom x-axis range for {parameter} must be (xmin, xmax) or None."
                )
            x_low, x_high = map(float, custom_x_limits)
            if not np.isfinite([x_low, x_high]).all() or x_low >= x_high:
                raise ValueError(
                    f"Custom x-axis range for {parameter} must be finite with xmin < xmax."
                )
        else:
            x_low = float(min(pmn_values.min(), sbi_values.min()))
            x_high = float(max(pmn_values.max(), sbi_values.max()))
            padding = 0.04 * (x_high - x_low) if x_high > x_low else 1.0
            x_low -= padding
            x_high += padding
        x_grid = np.linspace(x_low, x_high, options.density_points)
        pmn_density = _normalized_kde(
            pmn_values, x_grid, options.bandwidth, options.max_kde_samples, rng
        )
        sbi_density = _normalized_kde(
            sbi_values, x_grid, options.bandwidth, options.max_kde_samples, rng
        )

        line_kwargs = {"linewidth": style.line_width, **options.density_line_kwargs}
        axis.plot(x_grid, pmn_density, color=style.pmn_color, **line_kwargs)
        axis.plot(x_grid, sbi_density, color=style.sbi_color, **line_kwargs)
        axis.fill_between(x_grid, 0.0, pmn_density, color=style.pmn_color, alpha=options.density_fill_alpha)
        axis.fill_between(x_grid, 0.0, sbi_density, color=style.sbi_color, alpha=options.density_fill_alpha)

        for values, color in ((pmn_values, style.pmn_color), (sbi_values, style.sbi_color)):
            q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])
            axis.axvline(median, color=color, **options.median_line_kwargs)
            axis.axvline(q16, color=color, **options.interval_line_kwargs)
            axis.axvline(q84, color=color, **options.interval_line_kwargs)

        axis.set_xlim(x_low, x_high)
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel(options.labels.get(parameter, parameter))
        axis.xaxis.label.set_size(options.x_label_size)
        axis.xaxis.set_major_locator(MaxNLocator(options.max_x_ticks))
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        if options.suppress_y_ticks:
            axis.set_yticks([])
        axis.tick_params(axis="x", labelsize=options.x_tick_size)

    for axis in axes[len(options.parameters) :]:
        axis.set_visible(False)


def _plot_metrics_strip(
    ax: mpl.axes.Axes,
    metrics: Mapping[str, Any],
    config: ComparisonConfig,
) -> None:
    style = config.style
    options = config.posterior
    ax.set_axis_off()
    text = (
        f"Median posterior offset: {metrics['median_posterior_offset_sigma']:.2f}$\\sigma$   |   "
        f"{metrics['n_medians_within_threshold']}/{metrics['n_parameters']} medians within "
        f"{metrics['median_threshold_sigma']:.2f}$\\sigma$\n"
        f"{metrics['n_intervals_68_overlap']}/{metrics['n_parameters']} 68% intervals overlap   |   "
        f"Mean predictive difference: {metrics['mean_predictive_difference_sigma_data']:.2f}"
        r"$\sigma_{\rm data}$"
    )
    if options.show_validation_metrics:
        ax.text(
            *options.metrics_text_xy,
            text,
            transform=ax.transAxes,
            fontsize=style.annotation_size,
            **options.metrics_text_kwargs,
        )
    handles = [
        Line2D([0], [0], color=style.pmn_color, lw=style.line_width, label=config.pmn_label),
        Line2D([0], [0], color=style.sbi_color, lw=style.line_width, label=config.sbi_label),
    ]
    if options.show_shared_legend:
        ax.legend(handles=handles, fontsize=style.legend_size, **options.shared_legend_kwargs)
    ax.text(
        0.0,
        0.92,
        config.panel_labels["posterior"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.panel_label_size,
        fontweight="bold",
    )


def _adjust_top_panel_horizontal_space(
    left_axis: mpl.axes.Axes,
    right_axis: mpl.axes.Axes,
    extra_space: float,
) -> None:
    """Change only the Panel (a)/(b) gap in figure-coordinate units."""

    if extra_space == 0.0:
        return
    left_position = left_axis.get_position()
    right_position = right_axis.get_position()
    half_space = 0.5 * extra_space
    left_width = left_position.width - half_space
    right_width = right_position.width - half_space
    new_gap = (right_position.x0 + half_space) - (left_position.x1 - half_space)
    if left_width <= 0.0 or right_width <= 0.0 or new_gap < 0.0:
        raise ValueError(
            "top_panel_extra_horizontal_space is too large in magnitude for the current layout."
        )
    left_axis.set_position(
        [left_position.x0, left_position.y0, left_width, left_position.height]
    )
    right_axis.set_position(
        [
            right_position.x0 + half_space,
            right_position.y0,
            right_width,
            right_position.height,
        ]
    )


def make_comparison_figure(
    products: LoadedProducts,
    config: ComparisonConfig,
) -> Tuple[mpl.figure.Figure, Dict[str, mpl.axes.Axes], Dict[str, Any]]:
    """Create the complete three-part figure without saving it."""

    with mpl.rc_context(config.style.rc_params()):
        if config.layout.auto_posterior_layout:
            configure_posterior_layout(config)
        fig = plt.figure(**config.layout.figure_kwargs)
        axes = fig.subplot_mosaic(config.layout.mosaic, **config.layout.subplot_mosaic_kwargs)
        fig.subplots_adjust(**config.layout.subplots_adjust_kwargs)

        required_keys = {
            config.layout.spectrum_axis_key,
            config.layout.metrics_axis_key,
            *config.layout.posterior_axis_keys,
        }
        if config.layout.show_runtime_panel:
            required_keys.add(config.layout.runtime_axis_key)
        missing = required_keys.difference(axes)
        if missing:
            raise ValueError(f"The configured subplot mosaic is missing axes: {sorted(missing)}")

        if config.layout.show_runtime_panel:
            _adjust_top_panel_horizontal_space(
                axes[config.layout.spectrum_axis_key],
                axes[config.layout.runtime_axis_key],
                config.layout.top_panel_extra_horizontal_space,
            )

        metrics = compute_validation_metrics(products, config)
        _plot_spectrum_panel(axes[config.layout.spectrum_axis_key], products, config)
        runtime_benchmark: Mapping[str, float] = {}
        if config.layout.show_runtime_panel:
            runtime_benchmark = _plot_runtime_panel(
                axes[config.layout.runtime_axis_key], products, config
            )
        posterior_axes = [axes[key] for key in config.layout.posterior_axis_keys]
        _plot_posterior_panels(posterior_axes, products, config)
        _plot_metrics_strip(axes[config.layout.metrics_axis_key], metrics, config)

        metadata = build_metadata(products, config, metrics, runtime_benchmark)
        return fig, axes, metadata


def build_metadata(
    products: LoadedProducts,
    config: ComparisonConfig,
    metrics: Mapping[str, Any],
    runtime_benchmark: Mapping[str, float],
) -> Dict[str, Any]:
    """Build the JSON-serializable provenance and annotation summary."""

    sbi_log_runtime = None
    if config.paths.sbi_log_path is not None and Path(config.paths.sbi_log_path).is_file():
        text = Path(config.paths.sbi_log_path).read_text(encoding="utf-8", errors="replace")
        match = re.search(r"POSEIDON SBI retrieval finished in\s+([-+0-9.eE]+)\s+hours", text)
        if match:
            sbi_log_runtime = float(match.group(1))

    return {
        "figure": {
            "basename": config.save.basename,
            "pmn_label": config.pmn_label,
            "sbi_label": config.sbi_label,
            "parameters_plotted": list(config.posterior.parameters),
            "runtime_panel_shown": config.layout.show_runtime_panel,
            "panel_labels": dict(config.panel_labels),
        },
        "inputs": {
            "observation_paths": [str(segment.path) for segment in products.observations],
            "pmn_samples_path": str(products.pmn_samples.path),
            "sbi_samples_path": str(products.sbi_samples.path),
            "pmn_spectrum_path": str(products.pmn_spectrum.path),
            "sbi_spectrum_path": str(products.sbi_spectrum.path),
            "runtime_table_path": str(products.runtime.path),
        },
        "validation": {
            "pmn_sample_shape": list(products.pmn_samples.values.shape),
            "sbi_sample_shape": list(products.sbi_samples.values.shape),
            "parameter_names_match": products.pmn_samples.names == products.sbi_samples.names,
            "parameter_names": list(products.pmn_samples.names),
            "pmn_spectrum_shape": [len(products.pmn_spectrum.wavelength_um), 6],
            "sbi_spectrum_shape": [len(products.sbi_spectrum.wavelength_um), 6],
            "spectral_quantiles_monotonic": True,
            "observed_bin_count": int(sum(len(segment.wavelength_um) for segment in products.observations)),
        },
        "retrieval_summaries": {
            "pmn": products.pmn_summary,
            "sbi": products.sbi_summary,
            "comparison_note": config.model_comparison_note,
        },
        "runtime": {
            "source": str(products.runtime.path),
            "measured_max_parameters": config.runtime.measured_max_parameters,
            "projected_values_have_uncertainties": False,
            "projected_hatching_denotes_regime_not_uncertainty": True,
            "scope": config.runtime_scope,
            "hardware": "Same hardware for both methods; exact hardware specification not supplied.",
            "benchmark": dict(runtime_benchmark),
            "sbi_runtime_from_retrieval_log_hours": sbi_log_runtime,
            "runtime_source_note": (
                "The runtime CSV is authoritative for the plotted values. The retrieval log value "
                "is retained here as provenance when available."
            ),
        },
        "agreement_metrics": dict(metrics),
    }


def save_comparison_figure(
    fig: mpl.figure.Figure,
    metadata: Mapping[str, Any],
    config: ComparisonConfig,
) -> Dict[str, Path]:
    """Save vector PDF, 300 dpi PNG, and JSON annotation/provenance metadata."""

    output_directory = Path(config.paths.output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    basename = config.save.basename
    outputs: Dict[str, Path] = {}

    if config.save.save_pdf:
        pdf_path = output_directory / f"{basename}.pdf"
        fig.savefig(pdf_path, **config.save.savefig_kwargs)
        outputs["pdf"] = pdf_path
    if config.save.save_png:
        png_path = output_directory / f"{basename}.png"
        fig.savefig(png_path, dpi=config.save.png_dpi, **config.save.savefig_kwargs)
        outputs["png"] = png_path
    if config.save.save_metadata:
        metadata_path = output_directory / f"{basename}_metadata.json"
        metadata_to_write = dict(metadata)
        metadata_to_write["outputs"] = {key: str(value) for key, value in outputs.items()}
        metadata_path.write_text(
            json.dumps(metadata_to_write, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs["metadata"] = metadata_path

    for product, path in outputs.items():
        print(f"Saved {product}: {path}")
    return outputs


def run_comparison(
    config: Optional[ComparisonConfig] = None,
    show: bool = True,
) -> Tuple[mpl.figure.Figure, Dict[str, mpl.axes.Axes], Dict[str, Any], Dict[str, Path]]:
    """Load, validate, plot, and save the configured comparison."""

    if config is None:
        config = default_config()
    products = load_products(config)
    fig, axes, metadata = make_comparison_figure(products, config)
    outputs = save_comparison_figure(fig, metadata, config)
    if show:
        plt.show()
    return fig, axes, metadata, outputs


def main() -> None:
    run_comparison(default_config(), show=False)


if __name__ == "__main__":
    main()

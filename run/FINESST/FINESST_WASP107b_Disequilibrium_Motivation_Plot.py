"""Create the FINESST WASP-107 b disequilibrium-chemistry motivation figure.

All plotted numerical products are loaded from published Sing et al. (2024)
source data, a documented digitization of their published ATMO spectrum, or
the paper's reported free-retrieval constraints. No retrieval, radiative
transfer calculation, equilibrium calculation, or photochemical model is run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


@dataclass
class PathConfig:
    """Input and output paths for the published products."""

    observed_spectrum_path: Path
    digitized_model_path: Path
    digitization_metadata_path: Path
    disequilibrium_profiles_path: Path
    equilibrium_profiles_path: Path
    jwst_pressure_constraints_path: Path
    paper_path: Path
    output_directory: Path


@dataclass
class StyleConfig:
    """Centralized POSEIDON/Figure-1 typography and color controls."""

    observed_color: str = "#262626"
    model_color: str = "#0B1DE3"
    equilibrium_color: str = "#605D5D"
    disequilibrium_color: str = "#E3770B"
    retrieved_color: str = "#000000"
    ch4_band_color: str = "#E3770B"
    so2_band_color: str = "#0B1DE3"
    h2o_band_color: str = "#0072B2"
    co2_band_color: str = "#CC79A7"
    co_band_color: str = "#6A3D9A"
    use_sans_serif: bool = False
    serif_font: str = "DejaVu Serif"
    sans_serif_font: str = "DejaVu Sans"
    # Figure 1 typography scaled to the requested 7.5 x 4.8 inch canvas.
    axis_label_size: float = 10.5
    tick_label_size: float = 7.5
    legend_size: float = 7.4
    annotation_size: float = 6.4
    panel_label_size: float = 9.5
    line_width: float = 0.95
    errorbar_line_width: float = 0.9
    tick_width: float = 0.8
    tick_length: float = 4.0
    minor_tick_length: float = 2.2

    def rc_params(self) -> Dict[str, Any]:
        family = "sans-serif" if self.use_sans_serif else "serif"
        font = self.sans_serif_font if self.use_sans_serif else self.serif_font
        return {
            "font.family": family,
            "font.serif" if family == "serif" else "font.sans-serif": [font],
            "mathtext.fontset": "dejavusans" if self.use_sans_serif else "dejavuserif",
            "axes.labelsize": self.axis_label_size,
            "axes.linewidth": self.tick_width,
            "xtick.labelsize": self.tick_label_size,
            "ytick.labelsize": self.tick_label_size,
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
            "legend.fontsize": self.legend_size,
        }


@dataclass
class LayoutConfig:
    """Two-column GridSpec dimensions and spacing."""

    figure_width_inches: float = 7.5
    figure_height_inches: float = 4.8
    width_ratios: Sequence[float] = (0.63, 0.37)
    height_ratios: Sequence[float] = (1.0, 1.0)
    horizontal_space: float = 0.36
    vertical_space: float = 0.34
    left_margin: float = 0.085
    right_margin: float = 0.985
    bottom_margin: float = 0.13
    top_margin: float = 0.975


@dataclass
class SpectrumConfig:
    """Panel (a) full-spectrum, inset, molecular-band, and legend controls."""

    x_label: str = r"Wavelength ($\mu$m)"
    y_label: str = r"Planet-to-star radius ratio ($R_p/R_s$)"
    x_limits: Optional[Tuple[float, float]] = (2.70, 5.18)
    y_limits: Optional[Tuple[float, float]] = None
    overall_axis_label_fontsize: Optional[float] = 11.0
    overall_tick_label_fontsize: Optional[float] = 8.5
    use_shared_axis_labels: bool = True
    shared_x_label_fontsize: Optional[float] = 11.0
    shared_y_label_fontsize: Optional[float] = 11.0
    shared_x_label_pad_figure: float = 0.065
    shared_y_label_pad_figure: float = 0.055
    shared_x_label_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "center", "va": "top"}
    )
    shared_y_label_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "center", "va": "center", "rotation": 90}
    )
    show_inset: bool = True
    shrink_overall_for_external_inset: bool = True
    overall_axis_height_fraction: float = 0.44
    inset_bounds: Tuple[float, float, float, float] = (0.0, -1.18, 1.0, 1.0)
    inset_x_limits: Optional[Tuple[float, float]] = (3.10, 4.20)
    inset_y_limits: Optional[Tuple[float, float]] = None
    auto_inset_y_limits: bool = True
    inset_y_padding_fraction: float = 0.05
    inset_facecolor: str = "white"
    inset_facecolor_alpha: float = 0.96
    inset_zorder: float = 10.0
    inset_axis_label_fontsize: Optional[float] = 11.0
    inset_tick_label_fontsize: Optional[float] = 8.5
    inset_band_label_fontsize: Optional[float] = 8.0
    show_inset_zoom_indicator: bool = True
    show_all_inset_zoom_connectors: bool = True
    inset_zoom_indicator_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "edgecolor": "#555555", "linewidth": 0.8, "alpha": 0.75,
            "zorder": 9,
        }
    )
    show_minor_ticks: bool = True
    show_bin_widths: bool = False
    data_n_wavelength_bins: Optional[int] = None
    data_label: str = "JWST/NIRSpec"
    model_label: str = "Sing et al. (2024) ATMO fit"
    data_marker: str = "o"
    data_errorbar_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "markersize": 2.4,
            "markerfacecolor": "white",
            "markeredgewidth": 0.7,
            "capsize": 0,
            "linestyle": "none",
            "alpha": 0.90,
            "zorder": 5,
        }
    )
    model_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "-", "linewidth": 0.95, "zorder": 4}
    )
    molecular_bands: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: {"CH4": (3.285, 3.355), "SO2": (3.95, 4.10)}
    )
    molecular_labels: Mapping[str, str] = field(
        default_factory=lambda: {"CH4": r"CH$_4$", "SO2": r"SO$_2$"}
    )
    band_alpha: float = 0.16
    band_alpha_by_species: Mapping[str, float] = field(
        default_factory=lambda: {"CH4": 0.16, "SO2": 0.16}
    )
    band_zorder: float = 0.0
    band_label_y_axes: float = 0.965
    band_label_fontsize: Optional[float] = 10.0
    band_label_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "center", "va": "top", "fontweight": "bold"}
    )
    overall_molecular_bands: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "H2O": (2.70, 3.18),
            "CH4": (3.285, 3.355),
            "SO2": (3.95, 4.10),
            "CO2": (4.22, 4.43),
            "CO": (4.55, 4.95),
        }
    )
    overall_molecular_labels: Mapping[str, str] = field(
        default_factory=lambda: {
            "H2O": r"H$_2$O", "CH4": r"CH$_4$", "SO2": r"SO$_2$",
            "CO2": r"CO$_2$", "CO": "CO",
        }
    )
    overall_band_alpha: float = 0.11
    overall_band_alpha_by_species: Mapping[str, float] = field(
        default_factory=lambda: {
            "H2O": 0.11, "CH4": 0.11, "SO2": 0.11, "CO2": 0.11, "CO": 0.11,
        }
    )
    overall_band_zorder: float = 0.0
    overall_band_label_y_axes: float = 0.965
    overall_band_label_y_axes_by_species: Mapping[str, float] = field(default_factory=dict)
    overall_band_label_fontsize: Optional[float] = 10.0
    overall_band_label_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "center", "va": "top", "fontweight": "bold"}
    )
    show_optional_context_bands: bool = False
    optional_context_bands: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: {"CO2": (4.22, 4.43), "CO": (4.55, 4.95)}
    )
    optional_context_labels: Mapping[str, str] = field(
        default_factory=lambda: {"CO2": r"CO$_2$", "CO": "CO"}
    )
    show_overall_legend: bool = False
    show_inset_legend: bool = True
    overall_legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "upper right", "frameon": False, "ncol": 1,
            "handlelength": 1.8, "handletextpad": 0.5,
        }
    )
    legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "upper right", "frameon": False, "ncol": 1,
            "handlelength": 1.8, "handletextpad": 0.5,
        }
    )


@dataclass
class RetrievalConstraint:
    """A pressure-independent free-retrieval log10 VMR constraint."""

    median_log10_vmr: float
    minus_error_dex: float
    plus_error_dex: float


@dataclass
class ChemistryConfig:
    """Panels (b)/(c) profile, retrieval-point, annotation, and legend controls."""

    pressure_label: str = "Pressure (bar)"
    ch4_x_label: str = r"CH$_4$ volume mixing ratio"
    so2_x_label: str = r"SO$_2$ volume mixing ratio"
    ch4_x_label_fontsize: Optional[float] = None
    ch4_y_label_fontsize: Optional[float] = None
    so2_x_label_fontsize: Optional[float] = None
    so2_y_label_fontsize: Optional[float] = None
    pressure_log10_limits: Tuple[float, float] = (-5.0, 0.704)
    ch4_x_limits: Tuple[float, float] = (1.0e-8, 2.0e-2)
    so2_x_limits: Tuple[float, float] = (1.0e-8, 1.0e-4)
    equilibrium_label: str = "Equilibrium"
    disequilibrium_label: str = "Disequilibrium"
    retrieved_label: str = "ATMO retrieval"
    equilibrium_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "linestyle": "-.", "linewidth": 1.25, "alpha": 0.75, "zorder": 2,
        }
    )
    disequilibrium_line_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"linestyle": "-", "linewidth": 1.55, "zorder": 3}
    )
    retrieved_errorbar_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "fmt": "o", "markersize": 4.0, "capsize": 2.0,
            "elinewidth": 0.9, "markeredgewidth": 0.8, "zorder": 5,
        }
    )
    show_retrieved_points: bool = True
    show_pressure_errorbars: bool = True
    show_ch4_depletion_arrow: bool = True
    ch4_depletion_log_pressure_bar: float = -2.6
    depletion_arrow_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "arrowstyle": "<->", "color": "black", "linewidth": 0.8,
            "shrinkA": 0.0, "shrinkB": 0.0,
        }
    )
    depletion_annotation_text: str = r"$\sim10^3\!\times$ depleted"
    depletion_annotation_log_pressure_offset: float = -0.20
    depletion_annotation_fontsize: Optional[float] = None
    depletion_annotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "center", "va": "bottom"}
    )
    ch4_process_text: str = "Vertical mixing +\nhot deep atmosphere"
    ch4_process_axes_xy: Tuple[float, float] = (0.05, 0.08)
    so2_process_text: str = "Photochemical\nproduction"
    so2_process_axes_xy: Tuple[float, float] = (0.58, 0.64)
    equilibrium_so2_text: str = "Equilibrium SO$_2$ below\npublished plotting range"
    show_equilibrium_so2_note: bool = False
    equilibrium_so2_axes_xy: Tuple[float, float] = (0.06, 0.88)
    show_so2_enhancement_arrow: bool = True
    so2_figure3_floor_log10_vmr: float = -8.0
    so2_enhancement_arrow_origin_log10_vmr: Optional[float] = None
    so2_enhancement_arrow_end_log10_vmr: Optional[float] = None
    so2_enhancement_log_pressure_bar: float = -3.0
    so2_enhancement_annotation_text: Optional[str] = None
    so2_enhancement_annotation_log10_vmr: Optional[float] = None
    so2_enhancement_annotation_log_pressure_offset: float = -0.20
    so2_enhancement_annotation_fontsize: Optional[float] = None
    so2_enhancement_annotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "center", "va": "bottom"}
    )
    so2_enhancement_arrow_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "arrowstyle": "->", "color": "black", "linewidth": 0.8,
            "shrinkA": 0.0, "shrinkB": 0.0, "zorder": 8,
        }
    )
    process_annotation_fontsize: Optional[float] = None
    process_annotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "left", "va": "bottom"}
    )
    equilibrium_so2_annotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"ha": "left", "va": "top"}
    )
    show_shared_legend: bool = True
    shared_legend_line_color: str = "#666666"
    shared_legend_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "loc": "lower right", "bbox_to_anchor": (0.98, 0.03),
            "ncol": 1, "frameon": False, "handlelength": 2.2,
            "labelspacing": 0.20, "borderaxespad": 0.0,
        }
    )
    show_shared_process_note: bool = False
    shared_process_note: str = (
        "Deep thermal chemistry + vertical mixing $\\rightarrow$ quenched CH$_4$\n"
        "Stellar UV + chemical kinetics $\\rightarrow$ SO$_2$"
    )
    shared_process_note_xy: Tuple[float, float] = (0.78, 0.015)


@dataclass
class SaveConfig:
    """Output names and rendering controls."""

    basename: str = "figure2_WASP107b_disequilibrium_motivation"
    save_pdf: bool = True
    save_png: bool = True
    save_metadata: bool = True
    png_dpi: int = 300
    savefig_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"bbox_inches": "tight", "pad_inches": 0.03}
    )


@dataclass
class FigureConfig:
    paths: PathConfig
    style: StyleConfig = field(default_factory=StyleConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    chemistry: ChemistryConfig = field(default_factory=ChemistryConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    panel_labels: Mapping[str, str] = field(
        default_factory=lambda: {
            "spectrum": "(a)", "spectrum_inset": "(b)",
            "ch4": "(c)", "so2": "(d)",
        }
    )
    retrieval_constraints: Mapping[str, RetrievalConstraint] = field(
        default_factory=lambda: {
            "CH4": RetrievalConstraint(-6.03, 0.20, 0.22),
            "SO2": RetrievalConstraint(-5.06, 0.15, 0.14),
        }
    )


@dataclass
class SpectrumData:
    wavelength_um: np.ndarray
    half_width_um: np.ndarray
    radius_ratio: np.ndarray
    uncertainty: np.ndarray


@dataclass
class ModelSpectrum:
    wavelength_um: np.ndarray
    radius_ratio: np.ndarray


@dataclass
class ChemistryData:
    log_pressure_bar: np.ndarray
    equilibrium_ch4_log_vmr: np.ndarray
    disequilibrium_ch4_log_vmr: np.ndarray
    disequilibrium_so2_log_vmr: np.ndarray
    pressure_constraints: Mapping[str, Mapping[str, float]]


@dataclass
class LoadedProducts:
    observations: SpectrumData
    model: ModelSpectrum
    chemistry: ChemistryData
    input_hashes: Mapping[str, str]
    digitization_metadata: Mapping[str, Any]


def locate_run_directory(start: Optional[Path] = None) -> Path:
    """Locate ``POSEIDON/run`` without a machine-specific path."""

    candidates = [Path(start).resolve()] if start is not None else []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd().resolve())
    for candidate in candidates:
        if candidate.name == "run" and (candidate / "data").is_dir():
            return candidate
        if candidate.name in ("FINESST", "SBI", "HBM") and (candidate.parent / "data").is_dir():
            return candidate.parent
        if (candidate / "run" / "data").is_dir():
            return candidate / "run"
        for parent in candidate.parents:
            if parent.name == "run" and (parent / "data").is_dir():
                return parent
            if (parent / "run" / "data").is_dir():
                return parent / "run"
    raise FileNotFoundError("Could not locate the POSEIDON run directory.")


def default_config(run_directory: Optional[Path] = None) -> FigureConfig:
    """Return the proposal Figure 2 configuration."""

    run_dir = locate_run_directory(run_directory)
    data_dir = run_dir / "data" / "FINESST_26" / "Figure_2"
    paths = PathConfig(
        observed_spectrum_path=data_dir / "Sing_2024_Fig2_WASP107b_transit_spec_data.csv",
        digitized_model_path=data_dir / "Sing_2024_Fig2_ATMO_full_model_digitized.csv",
        digitization_metadata_path=data_dir / "Sing_2024_Fig2_ATMO_full_model_digitization_metadata.json",
        disequilibrium_profiles_path=data_dir / "Sing_2024_Fig3_neq_chem_vmr.csv",
        equilibrium_profiles_path=data_dir / "equilibrium_SO2_CH4_values.csv",
        jwst_pressure_constraints_path=data_dir / "SO2_CH4_JWST_pressure_constraints_on_data_points.csv",
        paper_path=data_dir / "Sing_2024_WASP107b_paper.pdf",
        output_directory=run_dir / "POSEIDON_output" / "WASP-107b" / "plots" / "FINESST",
    )
    return FigureConfig(paths=paths)


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _finite(*arrays: np.ndarray, label: str) -> None:
    if not all(np.isfinite(np.asarray(array, dtype=float)).all() for array in arrays):
        raise ValueError(f"{label} contains non-finite values.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_products(config: FigureConfig, verbose: bool = True) -> LoadedProducts:
    """Load and validate every numerical product used by the figure."""

    p = config.paths
    observed_path = _require_file(p.observed_spectrum_path, "published spectrum")
    model_path = _require_file(p.digitized_model_path, "digitized ATMO spectrum")
    digitization_path = _require_file(p.digitization_metadata_path, "digitization metadata")
    diseq_path = _require_file(p.disequilibrium_profiles_path, "disequilibrium profiles")
    equilibrium_path = _require_file(p.equilibrium_profiles_path, "equilibrium profiles")
    pressure_path = _require_file(p.jwst_pressure_constraints_path, "JWST pressure constraints")
    paper_path = _require_file(p.paper_path, "Sing et al. paper")

    observed = pd.read_csv(observed_path)
    _require_columns(
        observed,
        ["Wavelength_center(microns)", "Wavelength_half_width(microns)",
         "Rplanet/Rstar", "Rplanet/Rstar_error"],
        "published spectrum",
    )
    observations = SpectrumData(
        wavelength_um=observed["Wavelength_center(microns)"].to_numpy(float),
        half_width_um=observed["Wavelength_half_width(microns)"].to_numpy(float),
        radius_ratio=observed["Rplanet/Rstar"].to_numpy(float),
        uncertainty=observed["Rplanet/Rstar_error"].to_numpy(float),
    )
    _finite(
        observations.wavelength_um, observations.half_width_um,
        observations.radius_ratio, observations.uncertainty,
        label="published spectrum",
    )
    if np.any(np.diff(observations.wavelength_um) <= 0.0):
        raise ValueError("Published wavelengths must be strictly increasing.")
    if np.any(observations.half_width_um <= 0.0) or np.any(observations.uncertainty <= 0.0):
        raise ValueError("Published bin half-widths and uncertainties must be positive.")

    model_frame = pd.read_csv(model_path)
    _require_columns(model_frame, ["Wavelength(microns)", "Rplanet/Rstar"], "digitized model")
    model = ModelSpectrum(
        wavelength_um=model_frame["Wavelength(microns)"].to_numpy(float),
        radius_ratio=model_frame["Rplanet/Rstar"].to_numpy(float),
    )
    _finite(model.wavelength_um, model.radius_ratio, label="digitized model")
    if np.any(np.diff(model.wavelength_um) <= 0.0):
        raise ValueError("Digitized model wavelengths must be strictly increasing.")
    if model.wavelength_um[0] > observations.wavelength_um[0] or model.wavelength_um[-1] < observations.wavelength_um[-1]:
        raise ValueError("Digitized model does not cover all published measurements.")

    diseq = pd.read_csv(diseq_path)
    _require_columns(
        diseq, ["log Pressure(bar)", "log VMR CH_4", "log VMR SO_2"],
        "disequilibrium profiles",
    )
    equilibrium = pd.read_csv(equilibrium_path)
    _require_columns(
        equilibrium, ["log_pressure_bar", "equilibrium_CH4_log10_volume_mixing_ratio"],
        "equilibrium profiles",
    )
    log_pressure = diseq["log Pressure(bar)"].to_numpy(float)
    equilibrium_pressure = equilibrium["log_pressure_bar"].to_numpy(float)
    if log_pressure.shape != equilibrium_pressure.shape or not np.allclose(
        log_pressure, equilibrium_pressure, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("Equilibrium and disequilibrium pressure grids do not match.")
    chemistry_arrays = (
        log_pressure,
        equilibrium["equilibrium_CH4_log10_volume_mixing_ratio"].to_numpy(float),
        diseq["log VMR CH_4"].to_numpy(float),
        diseq["log VMR SO_2"].to_numpy(float),
    )
    _finite(*chemistry_arrays, label="chemistry profiles")
    if np.any(np.diff(log_pressure) <= 0.0):
        raise ValueError("Chemistry pressure grid must be strictly increasing in log pressure.")

    pressure_frame = pd.read_csv(pressure_path)
    _require_columns(
        pressure_frame,
        ["species", "log_pressure_central_bar", "log_pressure_lower_bound_bar",
         "log_pressure_upper_bound_bar"],
        "JWST pressure constraints",
    )
    pressure_constraints: Dict[str, Dict[str, float]] = {}
    for species in ("CH4", "SO2"):
        rows = pressure_frame.loc[pressure_frame["species"].astype(str).str.upper() == species]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one pressure-constraint row for {species}.")
        row = rows.iloc[0]
        pressure_constraints[species] = {
            "central": float(row["log_pressure_central_bar"]),
            "lower": float(row["log_pressure_lower_bound_bar"]),
            "upper": float(row["log_pressure_upper_bound_bar"]),
        }
        values = pressure_constraints[species]
        if not values["lower"] < values["central"] < values["upper"]:
            raise ValueError(f"Invalid pressure bounds for {species}: {values}")

    chemistry = ChemistryData(
        log_pressure_bar=log_pressure,
        equilibrium_ch4_log_vmr=chemistry_arrays[1],
        disequilibrium_ch4_log_vmr=chemistry_arrays[2],
        disequilibrium_so2_log_vmr=chemistry_arrays[3],
        pressure_constraints=pressure_constraints,
    )
    digitization_metadata = json.loads(digitization_path.read_text(encoding="utf-8"))
    input_paths = (
        observed_path, model_path, digitization_path, diseq_path,
        equilibrium_path, pressure_path, paper_path,
    )
    hashes = {str(path): _sha256(path) for path in input_paths}
    if verbose:
        print(f"Loaded {len(observations.wavelength_um)} published spectral points.")
        print(f"Loaded {len(model.wavelength_um)} digitized ATMO model points.")
        print(f"Loaded {len(log_pressure)} chemistry pressure levels.")
        print("Equilibrium SO2 is intentionally omitted: no curve/data are reported in range.")
    return LoadedProducts(observations, model, chemistry, hashes, digitization_metadata)


def _bin_observations(data: SpectrumData, n_bins: Optional[int]) -> SpectrumData:
    if n_bins is None:
        return data
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("data_n_wavelength_bins must be a positive integer or None.")
    if n_bins >= len(data.wavelength_um):
        return data
    edges = np.linspace(data.wavelength_um.min(), data.wavelength_um.max(), n_bins + 1)
    indices = np.clip(np.digitize(data.wavelength_um, edges) - 1, 0, n_bins - 1)
    wavelength = []
    half_width = []
    radius_ratio = []
    uncertainty = []
    for index in range(n_bins):
        selected = indices == index
        if not np.any(selected):
            continue
        weights = 1.0 / data.uncertainty[selected] ** 2
        wavelength.append(np.average(data.wavelength_um[selected], weights=weights))
        radius_ratio.append(np.average(data.radius_ratio[selected], weights=weights))
        uncertainty.append(np.sqrt(1.0 / weights.sum()))
        half_width.append(0.5 * (edges[index + 1] - edges[index]))
    return SpectrumData(
        np.asarray(wavelength), np.asarray(half_width),
        np.asarray(radius_ratio), np.asarray(uncertainty),
    )


def _panel_label(axis: mpl.axes.Axes, text: str, style: StyleConfig) -> None:
    axis.text(
        0.015, 0.985, text, transform=axis.transAxes,
        ha="left", va="top", fontsize=style.panel_label_size,
        fontweight="bold", zorder=20,
    )


def _spectrum_band_colors(style: StyleConfig) -> Mapping[str, str]:
    return {
        "H2O": style.h2o_band_color,
        "CH4": style.ch4_band_color,
        "SO2": style.so2_band_color,
        "CO2": style.co2_band_color,
        "CO": style.co_band_color,
    }


def _draw_spectrum_bands(
    axis: mpl.axes.Axes,
    bands: Mapping[str, Tuple[float, float]],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    alpha_by_species: Mapping[str, float],
    fallback_alpha: float,
    zorder: float,
    label_y_axes: float,
    label_y_axes_by_species: Mapping[str, float],
    label_fontsize: float,
    label_kwargs: Mapping[str, Any],
    fallback_color: str,
) -> None:
    """Draw independently configurable molecular feature regions on one axis."""

    for species, (lower, upper) in bands.items():
        if lower >= upper:
            raise ValueError(
                f"Molecular band for {species} must have lower < upper wavelength."
            )
        color = colors.get(species, fallback_color)
        opacity = float(alpha_by_species.get(species, fallback_alpha))
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(
                f"Molecular-band opacity for {species} must lie between 0 and 1."
            )
        axis.axvspan(lower, upper, color=color, alpha=opacity, zorder=zorder)
        axis.text(
            0.5 * (lower + upper),
            label_y_axes_by_species.get(species, label_y_axes),
            labels.get(species, species),
            transform=axis.get_xaxis_transform(),
            color=color,
            fontsize=label_fontsize,
            **label_kwargs,
        )


def _draw_spectrum_traces(
    axis: mpl.axes.Axes,
    data: SpectrumData,
    products: LoadedProducts,
    config: FigureConfig,
) -> None:
    style, options = config.style, config.spectrum
    xerr = data.half_width_um if options.show_bin_widths else None
    error_kwargs = {
        "color": style.observed_color,
        "ecolor": style.observed_color,
        "markeredgecolor": style.observed_color,
        "elinewidth": style.errorbar_line_width,
        "label": options.data_label,
        **options.data_errorbar_kwargs,
    }
    axis.errorbar(
        data.wavelength_um, data.radius_ratio, xerr=xerr,
        yerr=data.uncertainty, marker=options.data_marker, **error_kwargs,
    )
    axis.plot(
        products.model.wavelength_um, products.model.radius_ratio,
        color=style.model_color, label=options.model_label,
        **options.model_line_kwargs,
    )


def _format_spectrum_axis(
    axis: mpl.axes.Axes,
    config: FigureConfig,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    axis_label_fontsize: Optional[float] = None,
    tick_label_fontsize: Optional[float] = None,
    show_x_label: bool = True,
    show_y_label: bool = True,
) -> None:
    options = config.spectrum
    axis.set_xlabel(options.x_label if show_x_label else "", fontsize=axis_label_fontsize)
    axis.set_ylabel(options.y_label if show_y_label else "", fontsize=axis_label_fontsize)
    if x_limits is not None:
        axis.set_xlim(*x_limits)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    if options.show_minor_ticks:
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    if tick_label_fontsize is not None:
        axis.tick_params(axis="both", which="both", labelsize=tick_label_fontsize)


def _focused_spectrum_y_limits(
    data: SpectrumData,
    model: ModelSpectrum,
    x_limits: Tuple[float, float],
    padding_fraction: float,
) -> Tuple[float, float]:
    """Return a padded y-domain containing the focused data errors and model."""

    if padding_fraction < 0.0:
        raise ValueError("inset_y_padding_fraction must be non-negative.")
    lower_x, upper_x = x_limits
    data_selected = (
        (data.wavelength_um >= lower_x) & (data.wavelength_um <= upper_x)
    )
    model_selected = (
        (model.wavelength_um >= lower_x) & (model.wavelength_um <= upper_x)
    )
    if not np.any(data_selected) or not np.any(model_selected):
        raise ValueError("The inset x-limits must contain both data and model samples.")
    lower = min(
        np.min(data.radius_ratio[data_selected] - data.uncertainty[data_selected]),
        np.min(model.radius_ratio[model_selected]),
    )
    upper = max(
        np.max(data.radius_ratio[data_selected] + data.uncertainty[data_selected]),
        np.max(model.radius_ratio[model_selected]),
    )
    padding = padding_fraction * (upper - lower)
    return float(lower - padding), float(upper + padding)


def _plot_spectrum(
    axis: mpl.axes.Axes,
    products: LoadedProducts,
    config: FigureConfig,
    aligned_row_positions: Optional[Tuple[mpl.transforms.Bbox, mpl.transforms.Bbox]] = None,
) -> Optional[mpl.axes.Axes]:
    """Draw the full spectrum and the prior chemistry-focus view as an inset."""

    style, options = config.style, config.spectrum
    data = _bin_observations(products.observations, options.data_n_wavelength_bins)
    band_colors = _spectrum_band_colors(style)

    inset_bounds = options.inset_bounds
    if options.show_inset and len(inset_bounds) != 4:
        raise ValueError("inset_bounds must contain (left, bottom, width, height).")

    if options.show_inset and options.shrink_overall_for_external_inset:
        original_position = axis.get_position()
        if aligned_row_positions is not None:
            upper_row, lower_row = aligned_row_positions
            axis.set_position([
                original_position.x0, upper_row.y0,
                original_position.width, upper_row.height,
            ])
            parent_position = axis.get_position()
            inset_bounds = (
                inset_bounds[0],
                (lower_row.y0 - parent_position.y0) / parent_position.height,
                inset_bounds[2],
                lower_row.height / parent_position.height,
            )
        else:
            if not 0.0 < options.overall_axis_height_fraction <= 1.0:
                raise ValueError("overall_axis_height_fraction must lie in (0, 1].")
            reduced_height = (
                original_position.height * options.overall_axis_height_fraction
            )
            axis.set_position([
                original_position.x0,
                original_position.y1 - reduced_height,
                original_position.width,
                reduced_height,
            ])

    _draw_spectrum_bands(
        axis,
        options.overall_molecular_bands,
        options.overall_molecular_labels,
        band_colors,
        options.overall_band_alpha_by_species,
        options.overall_band_alpha,
        options.overall_band_zorder,
        options.overall_band_label_y_axes,
        options.overall_band_label_y_axes_by_species,
        options.overall_band_label_fontsize or style.annotation_size,
        options.overall_band_label_kwargs,
        style.equilibrium_color,
    )
    _draw_spectrum_traces(axis, data, products, config)
    _format_spectrum_axis(
        axis,
        config,
        options.x_limits,
        options.y_limits,
        axis_label_fontsize=options.overall_axis_label_fontsize,
        tick_label_fontsize=options.overall_tick_label_fontsize,
        show_x_label=not options.use_shared_axis_labels,
        show_y_label=not options.use_shared_axis_labels,
    )
    if options.show_overall_legend:
        axis.legend(fontsize=style.legend_size, **options.overall_legend_kwargs)
    _panel_label(axis, config.panel_labels["spectrum"], style)

    if not options.show_inset:
        return None

    if not 0.0 <= options.inset_facecolor_alpha <= 1.0:
        raise ValueError("inset_facecolor_alpha must lie between 0 and 1.")

    inset_axis = axis.inset_axes(inset_bounds, zorder=options.inset_zorder)
    inset_axis.set_facecolor(options.inset_facecolor)
    inset_axis.patch.set_alpha(options.inset_facecolor_alpha)

    inset_bands = dict(options.molecular_bands)
    inset_labels = dict(options.molecular_labels)
    if options.show_optional_context_bands:
        inset_bands.update(options.optional_context_bands)
        inset_labels.update(options.optional_context_labels)
    _draw_spectrum_bands(
        inset_axis,
        inset_bands,
        inset_labels,
        band_colors,
        options.band_alpha_by_species,
        options.band_alpha,
        options.band_zorder,
        options.band_label_y_axes,
        {},
        options.inset_band_label_fontsize
        or options.band_label_fontsize
        or style.annotation_size,
        options.band_label_kwargs,
        style.equilibrium_color,
    )
    _draw_spectrum_traces(inset_axis, data, products, config)
    inset_y_limits = options.inset_y_limits
    if (
        options.auto_inset_y_limits
        and inset_y_limits is None
        and options.inset_x_limits is not None
    ):
        inset_y_limits = _focused_spectrum_y_limits(
            data,
            products.model,
            options.inset_x_limits,
            options.inset_y_padding_fraction,
        )
    _format_spectrum_axis(
        inset_axis,
        config,
        options.inset_x_limits,
        inset_y_limits,
        axis_label_fontsize=options.inset_axis_label_fontsize,
        tick_label_fontsize=options.inset_tick_label_fontsize,
        show_x_label=not options.use_shared_axis_labels,
        show_y_label=not options.use_shared_axis_labels,
    )
    if options.show_inset_legend:
        inset_axis.legend(fontsize=style.legend_size, **options.legend_kwargs)
    if options.show_inset_zoom_indicator:
        indicator = axis.indicate_inset_zoom(
            inset_axis, **options.inset_zoom_indicator_kwargs
        )
        if options.show_all_inset_zoom_connectors:
            for connector in indicator.connectors:
                connector.set_visible(False)
            connector_kwargs = dict(options.inset_zoom_indicator_kwargs)
            connector_kwargs.pop("facecolor", None)
            connector_color = connector_kwargs.pop(
                "color", connector_kwargs.pop("edgecolor", "#555555")
            )
            connector_kwargs["zorder"] = min(
                float(connector_kwargs.get("zorder", options.inset_zorder - 1.0)),
                options.inset_zorder - 1.0,
            )
            x_min, x_max = inset_axis.get_xlim()
            y_min, y_max = inset_axis.get_ylim()
            for parent_corner, inset_corner in (
                ((x_min, y_min), (0.0, 0.0)),
                ((x_min, y_max), (0.0, 1.0)),
                ((x_max, y_min), (1.0, 0.0)),
                ((x_max, y_max), (1.0, 1.0)),
            ):
                # Keep connectors in the parent axes artist tree. Figure-level
                # artists are drawn after the complete axes, which places them
                # over the external inset regardless of the inset z-order.
                axis.add_artist(ConnectionPatch(
                    xyA=parent_corner,
                    coordsA=axis.transData,
                    xyB=inset_corner,
                    coordsB=inset_axis.transAxes,
                    axesA=axis,
                    axesB=inset_axis,
                    color=connector_color,
                    clip_on=False,
                    **connector_kwargs,
                ))
    if options.use_shared_axis_labels:
        parent_position = axis.get_position()
        inset_left, inset_bottom, inset_width, inset_height = inset_bounds
        inset_x0 = parent_position.x0 + inset_left * parent_position.width
        inset_y0 = parent_position.y0 + inset_bottom * parent_position.height
        inset_width_figure = inset_width * parent_position.width
        combined_y_center = 0.5 * (inset_y0 + parent_position.y1)
        axis.figure.text(
            inset_x0 + 0.5 * inset_width_figure,
            inset_y0 - options.shared_x_label_pad_figure,
            options.x_label,
            fontsize=(
                options.shared_x_label_fontsize
                or options.overall_axis_label_fontsize
                or style.axis_label_size
            ),
            **options.shared_x_label_kwargs,
        )
        axis.figure.text(
            min(parent_position.x0, inset_x0) - options.shared_y_label_pad_figure,
            combined_y_center,
            options.y_label,
            fontsize=(
                options.shared_y_label_fontsize
                or options.overall_axis_label_fontsize
                or style.axis_label_size
            ),
            **options.shared_y_label_kwargs,
        )
    _panel_label(inset_axis, config.panel_labels["spectrum_inset"], style)
    return inset_axis

def _retrieval_errorbar(
    axis: mpl.axes.Axes,
    species: str,
    products: LoadedProducts,
    config: FigureConfig,
) -> None:
    options, style = config.chemistry, config.style
    constraint = config.retrieval_constraints[species]
    pressure = products.chemistry.pressure_constraints[species]
    x = 10.0 ** constraint.median_log10_vmr
    xerr = np.array(
        [[x - 10.0 ** (constraint.median_log10_vmr - constraint.minus_error_dex)],
         [10.0 ** (constraint.median_log10_vmr + constraint.plus_error_dex) - x]]
    )
    y = 10.0 ** pressure["central"]
    yerr = None
    if options.show_pressure_errorbars:
        yerr = np.array(
            [[y - 10.0 ** pressure["lower"]], [10.0 ** pressure["upper"] - y]]
        )
    axis.errorbar(
        [x], [y], xerr=xerr, yerr=yerr,
        color=style.retrieved_color, ecolor=style.retrieved_color,
        markerfacecolor="white", markeredgecolor=style.retrieved_color,
        **options.retrieved_errorbar_kwargs,
    )


def _format_chemistry_axis(
    axis: mpl.axes.Axes,
    x_label: str,
    x_limits: Tuple[float, float],
    config: FigureConfig,
    x_label_fontsize: Optional[float] = None,
    y_label_fontsize: Optional[float] = None,
) -> None:
    options = config.chemistry
    pressure_min, pressure_max = options.pressure_log10_limits
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*x_limits)
    axis.set_ylim(10.0**pressure_max, 10.0**pressure_min)
    axis.set_xlabel(x_label, fontsize=x_label_fontsize)
    axis.set_ylabel(options.pressure_label, fontsize=y_label_fontsize)
    axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _plot_ch4(axis: mpl.axes.Axes, products: LoadedProducts, config: FigureConfig) -> float:
    style, options, chemistry = config.style, config.chemistry, products.chemistry
    pressure = 10.0 ** chemistry.log_pressure_bar
    axis.plot(
        10.0 ** chemistry.equilibrium_ch4_log_vmr, pressure,
        color=style.ch4_band_color, **options.equilibrium_line_kwargs,
    )
    axis.plot(
        10.0 ** chemistry.disequilibrium_ch4_log_vmr, pressure,
        color=style.ch4_band_color, **options.disequilibrium_line_kwargs,
    )
    if options.show_retrieved_points:
        _retrieval_errorbar(axis, "CH4", products, config)
    _format_chemistry_axis(
        axis, options.ch4_x_label, options.ch4_x_limits, config,
        x_label_fontsize=options.ch4_x_label_fontsize,
        y_label_fontsize=options.ch4_y_label_fontsize,
    )

    arrow_log_pressure = options.ch4_depletion_log_pressure_bar
    equilibrium_log = float(np.interp(
        arrow_log_pressure, chemistry.log_pressure_bar,
        chemistry.equilibrium_ch4_log_vmr,
    ))
    retrieved_log = config.retrieval_constraints["CH4"].median_log10_vmr
    depletion_ratio = 10.0 ** (equilibrium_log - retrieved_log)
    if options.show_ch4_depletion_arrow:
        y = 10.0 ** arrow_log_pressure
        x_equilibrium = 10.0 ** equilibrium_log
        x_retrieved = 10.0 ** retrieved_log
        axis.annotate(
            "", xy=(x_retrieved, y), xytext=(x_equilibrium, y),
            arrowprops=options.depletion_arrow_kwargs,
        )
        label_y = 10.0 ** (
            arrow_log_pressure + options.depletion_annotation_log_pressure_offset
        )
        axis.text(
            np.sqrt(x_retrieved * x_equilibrium), label_y,
            options.depletion_annotation_text,
            fontsize=options.depletion_annotation_fontsize or style.annotation_size,
            **options.depletion_annotation_kwargs,
        )
    if options.ch4_process_text:
        axis.text(
            *options.ch4_process_axes_xy, options.ch4_process_text,
            transform=axis.transAxes,
            fontsize=options.process_annotation_fontsize or style.annotation_size,
            **options.process_annotation_kwargs,
        )
    _panel_label(axis, config.panel_labels["ch4"], style)
    return depletion_ratio


def _plot_so2(
    axis: mpl.axes.Axes,
    products: LoadedProducts,
    config: FigureConfig,
) -> float:
    style, options, chemistry = config.style, config.chemistry, products.chemistry
    pressure = 10.0 ** chemistry.log_pressure_bar
    axis.plot(
        10.0 ** chemistry.disequilibrium_so2_log_vmr, pressure,
        color=style.so2_band_color, **options.disequilibrium_line_kwargs,
    )
    if options.show_retrieved_points:
        _retrieval_errorbar(axis, "SO2", products, config)
    _format_chemistry_axis(
        axis, options.so2_x_label, options.so2_x_limits, config,
        x_label_fontsize=options.so2_x_label_fontsize,
        y_label_fontsize=options.so2_y_label_fontsize,
    )
    fontsize = options.process_annotation_fontsize or style.annotation_size
    if options.so2_process_text:
        axis.text(
            *options.so2_process_axes_xy, options.so2_process_text,
            transform=axis.transAxes, fontsize=fontsize,
            **options.process_annotation_kwargs,
        )
    if options.show_equilibrium_so2_note and options.equilibrium_so2_text:
        axis.text(
            *options.equilibrium_so2_axes_xy, options.equilibrium_so2_text,
            transform=axis.transAxes, fontsize=fontsize,
            **options.equilibrium_so2_annotation_kwargs,
        )

    retrieved_log_vmr = config.retrieval_constraints["SO2"].median_log10_vmr
    enhancement_lower_bound = 10.0 ** (
        retrieved_log_vmr - options.so2_figure3_floor_log10_vmr
    )
    if options.show_so2_enhancement_arrow:
        y = 10.0 ** options.so2_enhancement_log_pressure_bar
        model_log_vmr = float(np.interp(
            options.so2_enhancement_log_pressure_bar,
            chemistry.log_pressure_bar,
            chemistry.disequilibrium_so2_log_vmr,
        ))
        origin_log_vmr = options.so2_enhancement_arrow_origin_log10_vmr
        if origin_log_vmr is None:
            origin_log_vmr = model_log_vmr
        end_log_vmr = options.so2_enhancement_arrow_end_log10_vmr
        if end_log_vmr is None:
            x_end = axis.get_xlim()[0]
            endpoint = (0.0, y)
            endpoint_coordinates = ("axes fraction", "data")
        else:
            x_end = 10.0 ** end_log_vmr
            endpoint = (x_end, y)
            endpoint_coordinates = "data"
        x_origin = 10.0 ** origin_log_vmr
        axis.annotate(
            "", xy=endpoint, xycoords=endpoint_coordinates,
            xytext=(x_origin, y), textcoords="data",
            arrowprops=options.so2_enhancement_arrow_kwargs,
        )
        annotation = options.so2_enhancement_annotation_text
        if annotation is not None:
            label_x = (
                np.sqrt(x_end * x_origin)
                if options.so2_enhancement_annotation_log10_vmr is None
                else 10.0 ** options.so2_enhancement_annotation_log10_vmr
            )
            label_y = 10.0 ** (
                options.so2_enhancement_log_pressure_bar
                + options.so2_enhancement_annotation_log_pressure_offset
            )
            axis.text(
                label_x, label_y, annotation,
                fontsize=(
                    options.so2_enhancement_annotation_fontsize
                    or style.annotation_size
                ),
                **options.so2_enhancement_annotation_kwargs,
            )
    _panel_label(axis, config.panel_labels["so2"], style)
    return enhancement_lower_bound


def _shared_legend(axis: mpl.axes.Axes, config: FigureConfig) -> None:
    options, style = config.chemistry, config.style
    if not options.show_shared_legend:
        return
    handles = [
        Line2D([0], [0], color=options.shared_legend_line_color,
               label=options.equilibrium_label, **options.equilibrium_line_kwargs),
        Line2D([0], [0], color=options.shared_legend_line_color,
               label=options.disequilibrium_label, **options.disequilibrium_line_kwargs),
        Line2D([0], [0], color=style.retrieved_color, marker="o",
               markerfacecolor="white", linestyle="none", label=options.retrieved_label),
    ]
    axis.legend(handles=handles, fontsize=style.legend_size, **options.shared_legend_kwargs)


def make_figure(
    products: LoadedProducts,
    config: FigureConfig,
) -> Tuple[mpl.figure.Figure, Mapping[str, mpl.axes.Axes], Mapping[str, Any]]:
    """Create the three-panel figure without saving it."""

    layout = config.layout
    with mpl.rc_context(config.style.rc_params()):
        fig = plt.figure(figsize=(layout.figure_width_inches, layout.figure_height_inches))
        grid = fig.add_gridspec(
            2, 2, width_ratios=layout.width_ratios, height_ratios=layout.height_ratios,
            wspace=layout.horizontal_space, hspace=layout.vertical_space,
        )
        axes = {
            "spectrum": fig.add_subplot(grid[:, 0]),
            "ch4": fig.add_subplot(grid[0, 1]),
            "so2": fig.add_subplot(grid[1, 1]),
        }
        fig.subplots_adjust(
            left=layout.left_margin, right=layout.right_margin,
            bottom=layout.bottom_margin, top=layout.top_margin,
        )
        spectrum_inset = _plot_spectrum(
            axes["spectrum"], products, config,
            aligned_row_positions=(
                axes["ch4"].get_position(), axes["so2"].get_position()
            ),
        )
        if spectrum_inset is not None:
            axes["spectrum_inset"] = spectrum_inset
        depletion_ratio = _plot_ch4(axes["ch4"], products, config)
        so2_enhancement_lower_bound = _plot_so2(axes["so2"], products, config)
        _shared_legend(axes["so2"], config)
        if config.chemistry.show_shared_process_note:
            fig.text(
                *config.chemistry.shared_process_note_xy,
                config.chemistry.shared_process_note,
                ha="center", va="bottom", fontsize=config.style.annotation_size,
            )
        metadata = build_metadata(
            products, config, depletion_ratio, so2_enhancement_lower_bound
        )
        return fig, axes, metadata


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_metadata(
    products: LoadedProducts,
    config: FigureConfig,
    depletion_ratio: float,
    so2_enhancement_lower_bound: float,
) -> Mapping[str, Any]:
    """Return a complete provenance and scientific-guardrail summary."""

    return {
        "figure": {
            "basename": config.save.basename,
            "purpose": "Motivate chemically self-consistent disequilibrium retrievals",
            "no_new_model_calculation": True,
        },
        "source": {
            "citation": "Sing et al. (2024), Nature 630, 831-835",
            "doi": "10.1038/s41586-024-07395-z",
            "input_sha256": dict(products.input_hashes),
        },
        "column_mapping": {
            "observed_wavelength": "Wavelength_center(microns), linear microns",
            "observed_half_width": "Wavelength_half_width(microns), linear microns",
            "observed_radius_ratio": "Rplanet/Rstar, linear dimensionless ratio",
            "observed_uncertainty": "Rplanet/Rstar_error, linear 1-sigma uncertainty",
            "pressure": "log Pressure(bar), base-10 logarithm of pressure in bar",
            "disequilibrium_CH4": "log VMR CH_4, base-10 logarithm of dimensionless VMR",
            "disequilibrium_SO2": "log VMR SO_2, base-10 logarithm of dimensionless VMR",
            "equilibrium_CH4": (
                "equilibrium_CH4_log10_volume_mixing_ratio, base-10 logarithm of VMR"
            ),
        },
        "equilibrium_so2": {
            "plotted": False,
            "reason": (
                "No equilibrium SO2 curve is visible within the published Figure 3 "
                "log10(VMR) range [-8, -1], and no machine-readable equilibrium SO2 "
                "profile was supplied. The similarly named supplied column was confirmed "
                "by the user to be equilibrium CO2 and is not used."
            ),
            "interpretation": (
                "The omission is consistent with the paper's statement that photochemistry "
                "is needed to account for SO2. The optional enhancement annotation is a "
                "lower bound relative to the published Figure 3 plotting floor, not a "
                "tabulated equilibrium SO2 abundance."
            ),
            "published_figure3_log10_vmr_floor": (
                config.chemistry.so2_figure3_floor_log10_vmr
            ),
            "median_retrieved_to_figure_floor_ratio": so2_enhancement_lower_bound,
            "enhancement_arrow_shown": config.chemistry.show_so2_enhancement_arrow,
            "enhancement_arrow_origin": (
                "disequilibrium model interpolated at the arrow pressure"
                if config.chemistry.so2_enhancement_arrow_origin_log10_vmr is None
                else "user-specified log10 VMR"
            ),
            "enhancement_arrow_origin_log10_vmr": (
                float(np.interp(
                    config.chemistry.so2_enhancement_log_pressure_bar,
                    products.chemistry.log_pressure_bar,
                    products.chemistry.disequilibrium_so2_log_vmr,
                ))
                if config.chemistry.so2_enhancement_arrow_origin_log10_vmr is None
                else config.chemistry.so2_enhancement_arrow_origin_log10_vmr
            ),
            "enhancement_arrow_end_log10_vmr": (
                float(np.log10(config.chemistry.so2_x_limits[0]))
                if config.chemistry.so2_enhancement_arrow_end_log10_vmr is None
                else config.chemistry.so2_enhancement_arrow_end_log10_vmr
            ),
            "enhancement_arrow_origin_to_figure_floor_ratio": (
                10.0 ** (
                    (
                        float(np.interp(
                            config.chemistry.so2_enhancement_log_pressure_bar,
                            products.chemistry.log_pressure_bar,
                            products.chemistry.disequilibrium_so2_log_vmr,
                        ))
                        if config.chemistry.so2_enhancement_arrow_origin_log10_vmr
                        is None
                        else config.chemistry.so2_enhancement_arrow_origin_log10_vmr
                    )
                    - config.chemistry.so2_figure3_floor_log10_vmr
                )
            ),
        },
        "free_retrieval_constraints_log10_vmr": {
            species: asdict(value) for species, value in config.retrieval_constraints.items()
        },
        "ch4_equilibrium_to_retrieved_ratio_at_annotation_pressure": {
            "log10_pressure_bar": config.chemistry.ch4_depletion_log_pressure_bar,
            "ratio": depletion_ratio,
        },
        "digitization": products.digitization_metadata,
        "configuration": _json_ready(asdict(config)),
    }


def save_figure(
    fig: mpl.figure.Figure,
    metadata: Mapping[str, Any],
    config: FigureConfig,
) -> Mapping[str, Path]:
    """Save the configured PDF, PNG, and JSON products."""

    output_dir = Path(config.paths.output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    base = output_dir / config.save.basename
    if config.save.save_pdf:
        outputs["pdf"] = base.with_suffix(".pdf")
        fig.savefig(outputs["pdf"], **config.save.savefig_kwargs)
    if config.save.save_png:
        outputs["png"] = base.with_suffix(".png")
        fig.savefig(outputs["png"], dpi=config.save.png_dpi, **config.save.savefig_kwargs)
    if config.save.save_metadata:
        outputs["metadata"] = base.with_suffix(".json")
        outputs["metadata"].write_text(
            json.dumps(_json_ready(metadata), indent=2) + "\n", encoding="utf-8"
        )
    return outputs


if __name__ == "__main__":
    figure_config = default_config()
    loaded_products = load_products(figure_config)
    figure, figure_axes, figure_metadata = make_figure(loaded_products, figure_config)
    saved_outputs = save_figure(figure, figure_metadata, figure_config)
    for output_type, output_path in saved_outputs.items():
        print(f"{output_type.upper()}: {output_path}")

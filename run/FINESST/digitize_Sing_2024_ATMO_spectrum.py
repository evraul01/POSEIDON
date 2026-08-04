"""Digitize the published Sing et al. (2024) ATMO spectrum in Figure 2.

This script performs image-coordinate extraction only. It does not run ATMO,
POSEIDON, a retrieval, or a chemistry model. The official full-resolution
Nature figure is cached locally, its printed ticks define the two linear axis
calibrations, and the native dark-red ATMO stroke is isolated by RGB distance.

Outputs
-------
* ``Sing_2024_Fig2_ATMO_full_model_digitized.csv``
* ``Sing_2024_Fig2_ATMO_full_model_digitization_metadata.json``
* ``Sing_2024_Fig2_ATMO_digitization_validation.png``
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


FIGURE_URL = (
    "https://media.springernature.com/full/springer-static/image/"
    "art%3A10.1038%2Fs41586-024-07395-z/MediaObjects/"
    "41586_2024_7395_Fig2_HTML.png"
)
FIGURE_SHA256 = "6ac30dd3a3bb6a479a1ee07814381d32c90681b94f11aaed2185b70ca138d1aa"


@dataclass
class DigitizationConfig:
    """Coordinates and thresholds for the official 2086 x 1143 PNG."""

    model_rgb: Tuple[int, int, int] = (153, 51, 51)
    rgb_distance_tolerance: float = 10.0
    extraction_x_pixels: Tuple[int, int] = (159, 2066)
    extraction_y_pixels: Tuple[int, int] = (40, 599)
    diagnostic_dpi: int = 180


def locate_run_directory(start: Optional[Path] = None) -> Path:
    """Locate ``POSEIDON/run`` without embedding a machine-specific path."""

    candidates = [Path(start).resolve()] if start is not None else []
    candidates.extend((Path(__file__).resolve().parent, Path.cwd().resolve()))
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_official_figure(path: Path) -> None:
    """Download the official panel only when a verified copy is unavailable."""

    if path.is_file() and _sha256(path) == FIGURE_SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(FIGURE_URL, timeout=60) as response:
        path.write_bytes(response.read())
    digest = _sha256(path)
    if digest != FIGURE_SHA256:
        path.unlink(missing_ok=True)
        raise ValueError(
            "The downloaded official Figure 2 image checksum differs from the "
            f"validated source: expected {FIGURE_SHA256}, received {digest}."
        )


def axis_calibration() -> dict:
    """Fit the two linear axes from centers of every printed major tick."""

    x_pixels = np.array(
        [
            159.0, 232.5, 305.5, 379.0, 452.5, 525.5, 599.0,
            672.5, 745.5, 819.0, 892.5, 965.5, 1039.0, 1112.5,
            1185.5, 1259.0, 1332.5, 1405.5, 1479.0, 1552.5,
            1625.5, 1699.0, 1772.5, 1845.5, 1919.0, 1992.5, 2065.5,
        ]
    )
    x_values = np.arange(2.6, 5.2001, 0.1)
    y_pixels = np.array(
        [40.5, 124.5, 207.5, 290.5, 374.5, 457.5, 540.5,
         624.5, 707.5, 790.5, 874.5, 957.5, 1040.5]
    )
    y_values = np.arange(0.149, 0.1369, -0.001)

    x_slope, x_intercept = np.polyfit(x_pixels, x_values, 1)
    y_slope, y_intercept = np.polyfit(y_pixels, y_values, 1)
    return {
        "x_tick_pixels": x_pixels,
        "x_tick_values_microns": x_values,
        "y_tick_pixels": y_pixels,
        "y_tick_values_radius_ratio": y_values,
        "x_slope_microns_per_pixel": float(x_slope),
        "x_intercept_microns": float(x_intercept),
        "y_slope_radius_ratio_per_pixel": float(y_slope),
        "y_intercept_radius_ratio": float(y_intercept),
        "x_max_calibration_residual_microns": float(
            np.max(np.abs(x_slope * x_pixels + x_intercept - x_values))
        ),
        "y_max_calibration_residual_radius_ratio": float(
            np.max(np.abs(y_slope * y_pixels + y_intercept - y_values))
        ),
    }


def extract_model(image: np.ndarray, config: DigitizationConfig, calibration: dict) -> pd.DataFrame:
    """Extract the median visible red-stroke coordinate in each image column."""

    target = np.asarray(config.model_rgb, dtype=float)
    rgb_distance = np.linalg.norm(image.astype(float) - target, axis=2)
    model_mask = rgb_distance <= config.rgb_distance_tolerance

    x_first, x_last = config.extraction_x_pixels
    y_first, y_last = config.extraction_y_pixels
    x_pixels = np.arange(x_first, x_last + 1)
    y_pixels = np.full(x_pixels.shape, np.nan, dtype=float)

    for index, x_pixel in enumerate(x_pixels):
        candidates = np.flatnonzero(model_mask[y_first : y_last + 1, x_pixel]) + y_first
        if candidates.size:
            y_pixels[index] = np.median(candidates)

    visible = np.isfinite(y_pixels)
    if visible.sum() < 0.5 * len(visible):
        raise ValueError("Fewer than half of image columns contain a visible model stroke.")
    y_filled = np.interp(x_pixels, x_pixels[visible], y_pixels[visible])

    wavelength = (
        calibration["x_slope_microns_per_pixel"] * x_pixels
        + calibration["x_intercept_microns"]
    )
    radius_ratio = (
        calibration["y_slope_radius_ratio_per_pixel"] * y_filled
        + calibration["y_intercept_radius_ratio"]
    )
    return pd.DataFrame(
        {
            "Wavelength(microns)": wavelength,
            "Rplanet/Rstar": radius_ratio,
            "source_x_pixel": x_pixels,
            "source_y_pixel": y_filled,
            "pixel_trace_interpolated": ~visible,
        }
    )


def _maximum_missing_run(interpolated: np.ndarray) -> int:
    longest = current = 0
    for missing in interpolated:
        current = current + 1 if missing else 0
        longest = max(longest, current)
    return longest


def validate_against_data(model: pd.DataFrame, observations: pd.DataFrame) -> dict:
    """Compare the fixed digitized trace to the published measurements."""

    predicted = np.interp(
        observations["Wavelength_center(microns)"],
        model["Wavelength(microns)"],
        model["Rplanet/Rstar"],
    )
    standardized = (
        observations["Rplanet/Rstar"].to_numpy() - predicted
    ) / observations["Rplanet/Rstar_error"].to_numpy()
    return {
        "n_observations": int(len(observations)),
        "mean_squared_standardized_residual": float(np.mean(standardized**2)),
        "median_absolute_standardized_residual": float(np.median(np.abs(standardized))),
        "note": "Diagnostic only; no fit or adjustment was applied to the digitized trace.",
    }


def save_diagnostic(
    source_image: np.ndarray,
    model: pd.DataFrame,
    observations: pd.DataFrame,
    output_path: Path,
    config: DigitizationConfig,
) -> None:
    """Save source-pixel and physical-coordinate validation panels."""

    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    ):
        fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), constrained_layout=True)
        axes[0].imshow(source_image)
        axes[0].plot(
            model["source_x_pixel"], model["source_y_pixel"],
            color="cyan", linewidth=0.65, label="Digitized trace",
        )
        axes[0].set_xlim(140, 2080)
        axes[0].set_ylim(620, 10)
        axes[0].legend(frameon=False, loc="lower right")
        axes[0].set_title("Digitized trace over official Sing et al. (2024) Figure 2")

        axes[1].errorbar(
            observations["Wavelength_center(microns)"],
            observations["Rplanet/Rstar"],
            yerr=observations["Rplanet/Rstar_error"],
            color="#262626", ecolor="#777777", marker="o", markersize=1.6,
            linewidth=0.0, elinewidth=0.45, alpha=0.55, label="Published data",
        )
        axes[1].plot(
            model["Wavelength(microns)"], model["Rplanet/Rstar"],
            color="#993333", linewidth=0.75, label="Digitized ATMO full model",
        )
        axes[1].set(xlabel=r"Wavelength ($\mu$m)", ylabel=r"$R_p/R_s$", xlim=(2.6, 5.2))
        axes[1].legend(frameon=False, loc="upper right")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.diagnostic_dpi, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    run_dir = locate_run_directory()
    data_dir = run_dir / "data" / "FINESST_26" / "Figure_2"
    output_dir = run_dir / "POSEIDON_output" / "WASP-107b" / "plots" / "FINESST"
    source_image_path = data_dir / "Sing_2024_Fig2_official_full.png"
    observation_path = data_dir / "Sing_2024_Fig2_WASP107b_transit_spec_data.csv"
    model_path = data_dir / "Sing_2024_Fig2_ATMO_full_model_digitized.csv"
    metadata_path = data_dir / "Sing_2024_Fig2_ATMO_full_model_digitization_metadata.json"
    diagnostic_path = output_dir / "Sing_2024_Fig2_ATMO_digitization_validation.png"

    config = DigitizationConfig()
    cache_official_figure(source_image_path)
    source_image = np.asarray(Image.open(source_image_path).convert("RGB"))
    if source_image.shape != (1143, 2086, 3):
        raise ValueError(f"Unexpected official image shape: {source_image.shape}")

    observations = pd.read_csv(observation_path)
    calibration = axis_calibration()
    model = extract_model(source_image, config, calibration)
    validation = validate_against_data(model, observations)
    model.to_csv(model_path, index=False, float_format="%.10g")
    save_diagnostic(source_image, model, observations, diagnostic_path, config)

    interpolated = model["pixel_trace_interpolated"].to_numpy(dtype=bool)
    metadata = {
        "source": {
            "article": "Sing et al. (2024), Nature 630, 831-835",
            "doi": "10.1038/s41586-024-07395-z",
            "figure": 2,
            "url": FIGURE_URL,
            "cached_image": str(source_image_path),
            "sha256": _sha256(source_image_path),
            "image_shape_pixels": list(source_image.shape),
        },
        "method": {
            "description": "RGB stroke extraction with printed-tick axis calibration",
            "config": asdict(config),
            "axis_calibration": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in calibration.items()
            },
            "visible_source_pixel_fraction": float(np.mean(~interpolated)),
            "interpolated_source_pixel_fraction": float(np.mean(interpolated)),
            "maximum_consecutive_interpolated_pixels": _maximum_missing_run(interpolated),
            "one_pixel_wavelength_microns": abs(calibration["x_slope_microns_per_pixel"]),
            "one_pixel_radius_ratio": abs(calibration["y_slope_radius_ratio_per_pixel"]),
            "guardrail": "This is digitization of a published curve, not a new model calculation.",
        },
        "validation_against_published_data": validation,
        "outputs": {
            "digitized_model_csv": str(model_path),
            "diagnostic_overlay_png": str(diagnostic_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote digitized model: {model_path}")
    print(f"Wrote digitization metadata: {metadata_path}")
    print(f"Wrote validation overlay: {diagnostic_path}")
    print(
        "Mean squared standardized residual (diagnostic only): "
        f"{validation['mean_squared_standardized_residual']:.3f}"
    )


if __name__ == "__main__":
    main()

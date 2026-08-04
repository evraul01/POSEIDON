#!/usr/bin/env python3
"""Apply the learned 5→5.2 / 6→6.2 rank-based distribution warp.

Example:
    python quantile_distribution_warp.py \
      --reference-base 'Pasted text(19).txt' \
      --reference-shifted 'Pasted text (2)(2).txt' \
      --inputs input_1.csv input_2.csv \
      --output-dir transformed

By default, every numeric input column is transformed. Each column is
classified automatically as 5-like, 6-like, or intermediate, and receives a
blend of the two learned quantile-shift curves. Original files are untouched.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def names(value: str | None) -> set[str]:
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def detect_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        first = next((x for x in sample.splitlines() if x.strip()), "")
        if "\t" in first:
            return "\t"
        if "," in first:
            return ","
        return r"\s+"


def read_table(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df, sep


def finite_values(series: pd.Series, label: str) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        raise ValueError(f"Column {label!r} has fewer than two finite values")
    return arr


def qcurve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.quantile(values, grid, method="linear")


def build_calibration(
    base: pd.DataFrame,
    shifted: pd.DataFrame,
    mild_source: str,
    strong_source: str,
    mild_shifted: str,
    strong_shifted: str,
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    for col in (mild_source, strong_source):
        if col not in base:
            raise KeyError(f"Missing {col!r} in reference-base; found {list(base.columns)}")
    for col in (mild_shifted, strong_shifted):
        if col not in shifted:
            raise KeyError(f"Missing {col!r} in reference-shifted; found {list(shifted.columns)}")

    q5 = qcurve(finite_values(base[mild_source], mild_source), grid)
    q6 = qcurve(finite_values(base[strong_source], strong_source), grid)
    q52 = qcurve(finite_values(shifted[mild_shifted], mild_shifted), grid)
    q62 = qcurve(finite_values(shifted[strong_shifted], strong_shifted), grid)
    return {
        "q5": q5,
        "q6": q6,
        "d5": q52 - q5,
        "d6": q62 - q6,
    }


def auto_alpha(values: np.ndarray, grid: np.ndarray, cal: dict[str, np.ndarray]) -> float:
    """0 = fully 5-like; 1 = fully 6-like; intermediate values blend both."""
    qin = qcurve(values, grid)
    direction = cal["q6"] - cal["q5"]
    denom = float(direction @ direction)
    if denom <= np.finfo(float).eps:
        return 0.5
    alpha = float(((qin - cal["q5"]) @ direction) / denom)
    return float(np.clip(alpha, 0.0, 1.0))


def warp_series(
    series: pd.Series,
    grid: np.ndarray,
    cal: dict[str, np.ndarray],
    alpha: float,
    strength: float,
    tail_clip: float,
    preserve_rank: bool,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    all_values = numeric.to_numpy(float)
    valid = np.isfinite(all_values)
    x = all_values[valid]
    if len(x) < 2:
        return series.copy()

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = (np.arange(len(xs), dtype=float) + 0.5) / len(xs)
    ranks = np.clip(ranks, tail_clip, 1.0 - tail_clip)

    delta = (1.0 - alpha) * cal["d5"] + alpha * cal["d6"]
    ys = xs + strength * np.interp(ranks, grid, delta)
    if preserve_rank:
        ys = np.maximum.accumulate(ys)

    y = np.empty_like(x)
    y[order] = ys
    result = series.copy()
    result.loc[valid] = y
    return result


def numeric_columns(df: pd.DataFrame, selected: set[str]) -> list[str]:
    if selected:
        missing = selected - set(df.columns)
        if missing:
            raise KeyError(f"Input column(s) not found: {sorted(missing)}")
        return [c for c in df.columns if c in selected]
    return [
        c for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
    ]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Learn 5→5.2 and 6→6.2 quantile shifts and apply them to CSV columns."
    )
    p.add_argument("--reference-base", required=True, type=Path)
    p.add_argument("--reference-shifted", required=True, type=Path)
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--suffix", default="_warped")
    p.add_argument("--columns", help="Comma-separated columns to transform; default: all numeric")
    p.add_argument("--mild-columns", help="Comma-separated columns forced to use 5→5.2")
    p.add_argument("--strong-columns", help="Comma-separated columns forced to use 6→6.2")
    p.add_argument("--mild-source-column", default="5")
    p.add_argument("--strong-source-column", default="6")
    p.add_argument("--mild-shifted-column", default="5.2")
    p.add_argument("--strong-shifted-column", default="6.2")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--random-strength-min", type=float, default=1.0)
    p.add_argument("--random-strength-max", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--quantile-points", type=int, default=2001)
    p.add_argument("--tail-clip", type=float, default=0.001)
    p.add_argument("--no-preserve-rank-order", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    a = parser().parse_args()
    if not (0 <= a.tail_clip < 0.5):
        raise ValueError("--tail-clip must be in [0, 0.5)")
    if a.quantile_points < 21:
        raise ValueError("--quantile-points must be at least 21")
    if not (0 < a.random_strength_min <= a.random_strength_max):
        raise ValueError("Random-strength bounds must be positive and ordered")

    base, _ = read_table(a.reference_base)
    shifted, _ = read_table(a.reference_shifted)
    grid = np.linspace(a.tail_clip, 1.0 - a.tail_clip, a.quantile_points)
    cal = build_calibration(
        base, shifted,
        a.mild_source_column, a.strong_source_column,
        a.mild_shifted_column, a.strong_shifted_column,
        grid,
    )

    selected = names(a.columns)
    forced_mild = names(a.mild_columns)
    forced_strong = names(a.strong_columns)
    overlap = forced_mild & forced_strong
    if overlap:
        raise ValueError(f"Columns forced to both modes: {sorted(overlap)}")

    rng = np.random.default_rng(a.seed)
    diagnostics: list[dict[str, object]] = []

    for input_path in a.inputs:
        df, sep = read_table(input_path)
        out = df.copy()
        cols = numeric_columns(df, selected)
        if not cols:
            raise ValueError(f"No numeric columns found in {input_path}")

        for col in cols:
            x = finite_values(df[col], col)
            if col in forced_mild:
                alpha, mode = 0.0, "forced_mild"
            elif col in forced_strong:
                alpha, mode = 1.0, "forced_strong"
            else:
                alpha = auto_alpha(x, grid, cal)
                mode = "auto_mild" if alpha < 0.25 else "auto_strong" if alpha > 0.75 else "auto_blend"

            random_factor = float(rng.uniform(a.random_strength_min, a.random_strength_max))
            effective_strength = a.strength * random_factor
            out[col] = warp_series(
                df[col], grid, cal, alpha, effective_strength,
                a.tail_clip, not a.no_preserve_rank_order,
            )

            before = pd.to_numeric(df[col], errors="coerce")
            after = pd.to_numeric(out[col], errors="coerce")
            valid = before.notna() & after.notna()
            shift_values = (after[valid] - before[valid]).to_numpy(float)
            diagnostics.append({
                "input_file": str(input_path),
                "column": col,
                "mode": mode,
                "strong_blend_weight_alpha": alpha,
                "effective_strength": effective_strength,
                "finite_values": int(valid.sum()),
                "mean_original": float(before[valid].mean()),
                "mean_transformed": float(after[valid].mean()),
                "median_original": float(before[valid].median()),
                "median_transformed": float(after[valid].median()),
                "mean_shift": float(np.mean(shift_values)),
                "median_shift": float(np.median(shift_values)),
                "min_shift": float(np.min(shift_values)),
                "max_shift": float(np.max(shift_values)),
            })

        output_dir = a.output_dir or input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}{a.suffix}{input_path.suffix}"
        if output_path.exists() and not a.overwrite:
            raise FileExistsError(f"{output_path} exists; use --overwrite or change --suffix")
        write_sep = "," if sep == r"\s+" else sep
        out.to_csv(output_path, index=False, sep=write_sep, float_format="%.18e")
        print(f"Wrote: {output_path}")

    diag_dir = a.output_dir or a.inputs[0].parent
    diag_path = diag_dir / f"quantile_warp_diagnostics{a.suffix}.csv"
    if diag_path.exists() and not a.overwrite:
        raise FileExistsError(f"{diag_path} exists; use --overwrite or change --suffix")
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False, float_format="%.10g")
    print(f"Wrote: {diag_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

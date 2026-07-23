"""
Full Dataset Audit Utility

Analyzes raw and processed agricultural data files:
1. Validates raw schema, missing value counts, and initial samples.
2. Inspects cleaned dataset dimensions, temporal span, and categorical diversity.
3. Computes statistical yield distributions by crop and state.
4. Identifies sparse (state, crop, season) data combinations.
5. Verifies log-transform distribution metrics.
"""

import os
import numpy as np
import pandas as pd

from scripts.config import CLEANED_DATA_PATH, RAW_DATA_PATH


def print_section(title: str) -> None:
    """Print a visually distinct section banner for clear terminal reading."""
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def audit_raw_dataset(raw_path: str) -> None:
    """Inspect raw input file schema, data types, nulls, and head samples."""
    print_section("Raw Dataset Audit")
    if not os.path.exists(raw_path):
        print(f"Warning: Raw dataset not found at '{raw_path}'")
        return

    df_raw = pd.read_csv(raw_path)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    print(f"Shape        : {df_raw.shape}")
    print(f"Columns      : {list(df_raw.columns)}")
    print(f"Dtypes:\n{df_raw.dtypes}")
    print(f"\nNull counts:\n{df_raw.isnull().sum()}")
    print("\nSample rows:")
    print(df_raw.head(5).to_string())


def audit_cleaned_dataset(cleaned_path: str) -> None:
    """Audit processed dataset structure, categories, and target statistics."""
    print_section("Cleaned Dataset Audit")
    if not os.path.exists(cleaned_path):
        print(f"Error: Cleaned dataset not found at '{cleaned_path}'")
        return

    df = pd.read_csv(cleaned_path)
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"Year range   : {df['crop_year'].min()} – {df['crop_year'].max()}")
    print(f"Unique states: {df['state'].nunique()} -> {sorted(df['state'].unique())}")
    print(f"Unique crops : {df['crop'].nunique()} -> {sorted(df['crop'].unique())}")
    print(f"Unique seasons: {sorted(df['season'].unique())}")

    # Yield stats per Crop
    print_section("Yield Stats per Crop (kg/ha)")
    crop_stats = df.groupby("crop")["yield_kg_ha"].agg(["count", "min", "mean", "median", "max", "std"])
    crop_stats = crop_stats.sort_values("mean", ascending=False)
    print(crop_stats.round(1).to_string())

    # Yield stats per State
    print_section("Yield Stats per State (kg/ha)")
    state_stats = df.groupby("state")["yield_kg_ha"].agg(["count", "min", "mean", "median", "max"])
    state_stats = state_stats.sort_values("mean", ascending=False)
    print(state_stats.round(1).to_string())

    # Coverage matrix analysis
    print_section("Coverage: state x crop x season combos")
    combos = df.groupby(["state", "crop", "season"])["yield_kg_ha"].agg(["count", "mean"]).reset_index()
    combos.columns = ["state", "crop", "season", "n_records", "mean_yield"]
    print(f"Total unique (state, crop, season) combos: {len(combos)}")
    print(f"Combos with < 3 records (sparse): {(combos['n_records'] < 3).sum()}")
    print(f"Combos with 1 record only       : {(combos['n_records'] == 1).sum()}")
    print("\nTop 20 sparsest combos:")
    print(combos.nsmallest(20, "n_records").to_string(index=False))

    # Year coverage analysis
    print_section("Year Coverage per Crop")
    year_crop = df.groupby("crop")["crop_year"].agg(["min", "max", "nunique"])
    year_crop.columns = ["first_year", "last_year", "n_years"]
    year_crop = year_crop.sort_values("n_years", ascending=True)
    print(year_crop.to_string())

    # Future / Extrapolated years analysis
    print_section("Data for Years > 2020 (extrapolated?)")
    future_df = df[df["crop_year"] > 2020]
    print(f"Records with year > 2020: {len(future_df)}")
    if not future_df.empty:
        print(future_df.groupby(["crop_year", "crop"])["yield_kg_ha"].mean().unstack().round(1).to_string())

    # Target Distribution & Skewness Analysis
    print_section("Target Variable Distribution (yield_kg_ha)")
    y = df["yield_kg_ha"]
    print(f"  Count  : {len(y):,}")
    print(f"  Min    : {y.min():,.1f}")
    print(f"  P5     : {y.quantile(0.05):,.1f}")
    print(f"  P25    : {y.quantile(0.25):,.1f}")
    print(f"  Median : {y.median():,.1f}")
    print(f"  Mean   : {y.mean():,.1f}")
    print(f"  P75    : {y.quantile(0.75):,.1f}")
    print(f"  P95    : {y.quantile(0.95):,.1f}")
    print(f"  Max    : {y.max():,.1f}")
    print(f"  Std    : {y.std():,.1f}")
    print(f"  Skew   : {y.skew():.3f}")

    y_log = np.log1p(y)
    print("\n  log1p target stats:")
    print(f"  Min log1p: {y_log.min():.3f}")
    print(f"  Max log1p: {y_log.max():.3f}")
    print(f"  Mean log1p: {y_log.mean():.3f}")
    print(f"  Skew log1p: {y_log.skew():.3f}")


def run_audit() -> None:
    """Execute complete dataset audit pipeline."""
    audit_raw_dataset(RAW_DATA_PATH)
    audit_cleaned_dataset(CLEANED_DATA_PATH)
    print("\n=== AUDIT COMPLETE ===")


if __name__ == "__main__":
    run_audit()

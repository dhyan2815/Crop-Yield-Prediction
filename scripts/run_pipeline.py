"""
End-to-End Crop Yield ML Training Pipeline

Executes the 3-step pipeline:
1. Data Cleaning: Normalizes raw schemas, converts metrics, filters physical yield bounds (10-100k kg/ha).
2. Feature Engineering: Normalizes years, one-hot encodes state/crop/season, exports feature contract.
3. Model Training: Trains Random Forest Regressor on log1p(yield), evaluates on real scale, saves compressed model.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from scripts.config import (
    CLEANED_DATA_PATH,
    CONTRACT_PATH,
    FEATURES_DATA_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
)


def find_col(df: pd.DataFrame, keyword: str) -> str | None:
    """Find the first column matching a substring keyword (case-insensitive)."""
    matches = [col for col in df.columns if keyword in col.lower()]
    return matches[0] if matches else None


def clean_raw_data() -> pd.DataFrame:
    """Load raw dataset, standardize columns, calculate yield, and filter extreme outliers."""
    print("1. CLEANING DATA...")
    raw_path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else "india_crop_yield.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at '{RAW_DATA_PATH}' or fallback 'india_crop_yield.csv'.")

    # Read raw dataset and normalize column names
    df = pd.read_csv(raw_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "", regex=True)

    # Locate required columns dynamically
    state_col = find_col(df, "state")
    crop_col = find_col(df, "crop")
    year_col = find_col(df, "year")
    season_col = find_col(df, "season")
    production_col = find_col(df, "production")
    area_col = find_col(df, "area")

    df = df.rename(
        columns={
            state_col: "state",
            crop_col: "crop",
            year_col: "crop_year",
            season_col: "season",
            production_col: "production",
            area_col: "area",
        }
    )

    # Convert numeric fields & compute yield in kg/ha (production is in tonnes)
    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df.dropna(subset=["production", "area"])
    df = df[df["area"] > 0]

    df["yield_kg_ha"] = (df["production"] * 1000) / df["area"]
    # Filter realistic physical limits (10 to 100,000 kg/ha)
    df = df[(df["yield_kg_ha"] > 10) & (df["yield_kg_ha"] < 100000)]
    df = df[["state", "crop_year", "season", "crop", "yield_kg_ha"]]

    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"   Saved cleaned data to '{CLEANED_DATA_PATH}' ({len(df):,} rows).")
    return df


def engineer_features() -> None:
    """Encode categorical features, normalize year, and generate feature contract."""
    print("2. FEATURE ENGINEERING...")
    if not os.path.exists(CLEANED_DATA_PATH):
        clean_raw_data()

    df = pd.read_csv(CLEANED_DATA_PATH)

    # Min-max normalize year to [0, 1] range for smooth temporal scaling
    year_min, year_max = df["crop_year"].min(), df["crop_year"].max()
    df["year_normalized"] = (df["crop_year"] - year_min) / (year_max - year_min)

    # One-hot encode state, crop, and season categories
    df_encoded = pd.get_dummies(df, columns=["state", "crop", "season"], prefix=["state", "crop", "season"])
    y = df_encoded["yield_kg_ha"]
    X = df_encoded.drop(columns=["yield_kg_ha", "crop_year"])

    feature_cols = X.columns.tolist()

    # Save contract metadata defining feature ordering and target transformation
    contract = {"features": feature_cols, "target_transform": "log1p"}
    os.makedirs(os.path.dirname(CONTRACT_PATH), exist_ok=True)
    with open(CONTRACT_PATH, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    os.makedirs(os.path.dirname(FEATURES_DATA_PATH), exist_ok=True)
    df_final = pd.concat([X, y], axis=1)
    df_final.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"   Saved features to '{FEATURES_DATA_PATH}' & contract to '{CONTRACT_PATH}' ({len(feature_cols)} features).")


def train_and_evaluate_model() -> None:
    """Train Random Forest Regressor on log1p-transformed target and export model artifact."""
    print("3. MODEL TRAINING (Log-Scale)...")
    if not os.path.exists(FEATURES_DATA_PATH):
        engineer_features()

    df = pd.read_csv(FEATURES_DATA_PATH)
    X = df.drop(columns=["yield_kg_ha"])
    y = np.log1p(df["yield_kg_ha"])  # Log transform handles right-skewed target distribution

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training Random Forest on {len(X_train):,} samples...")

    model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate predictions converted back to original kg/ha scale
    preds_log = model.predict(X_test)
    preds_actual = np.expm1(preds_log)
    y_actual = np.expm1(y_test)

    r2 = r2_score(y_actual, preds_actual)
    mae = mean_absolute_error(y_actual, preds_actual)

    print("--- MODEL PERFORMANCE (Original Scale) ---")
    print(f"   R-squared Score     : {r2:.4f}")
    print(f"   Mean Absolute Error : {mae:,.2f} kg/ha")

    # Persist compressed model (compress=3 ensures size stays under Git limits)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"   Saved compressed model artifact to '{MODEL_PATH}'.")


def run_pipeline() -> None:
    """Execute complete ML pipeline from cleaning to model persistence."""
    print("--- STARTING DATA PIPELINE ---")
    clean_raw_data()
    engineer_features()
    train_and_evaluate_model()
    print("--- PIPELINE COMPLETE ---")


if __name__ == "__main__":
    run_pipeline()

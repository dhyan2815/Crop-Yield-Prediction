# scripts/train_models.py
"""Model training module for crop yield prediction."""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .config import (
    FEATURES_DATA_PATH,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    CORE_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_COLUMN
)


def load_training_data(filepath: str = FEATURES_DATA_PATH,
                       use_engineered: bool = True) -> tuple:
    """
    Load training data with specified features.

    Args:
        filepath: Path to the feature-engineered CSV
        use_engineered: Whether to include engineered features

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature names used
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Clean crop names (consistency with training)
    if 'Item' in df.columns:
        df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)

    # Select features
    if use_engineered:
        feature_cols = CORE_FEATURES + ENGINEERED_FEATURES + ['year_normalized']
    else:
        feature_cols = CORE_FEATURES

    # One-hot encode crop (Item) column if present
    if 'Item' in df.columns:
        df = pd.get_dummies(df, columns=['Item'])
        crop_cols = [col for col in df.columns if col.startswith('Item_')]
        feature_cols = feature_cols + crop_cols

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    return X, y, feature_cols


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """Calculate and return evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")

    return metrics


def train_models(X_train, X_test, y_train, y_test,
                n_estimators: int = 200,
                random_state: int = 42) -> tuple:
    """
    Train Linear Regression and Random Forest models.

    Returns:
        lr_model, rf_model, metrics_lr, metrics_rf
    """
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    metrics_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")

    # Cross-validation for LR
    cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='r2')
    print(f"  CV R² scores: {cv_scores_lr}")
    print(f"  Mean CV R²: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

    # Cross-validation for RF
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    print(f"  CV R² scores: {cv_scores_rf}")
    print(f"  Mean CV R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

    return lr_model, rf_model, metrics_lr, metrics_rf


def save_models(lr_model, rf_model,
                lr_path: str = LR_MODEL_PATH,
                rf_path: str = RF_MODEL_PATH):
    """Save trained models to disk."""
    joblib.dump(lr_model, lr_path)
    joblib.dump(rf_model, rf_path)
    print(f"\nModels saved:")
    print(f"  Linear Regression: {lr_path}")
    print(f"  Random Forest: {rf_path}")


def train_pipeline(use_engineered: bool = True,
                   test_size: float = 0.2,
                   random_state: int = 42) -> dict:
    """
    Complete training pipeline.

    Returns dictionary with models, metrics, and feature names.
    """
    print("=" * 60)
    print("CROP YIELD PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    X, y, feature_names = load_training_data(use_engineered=use_engineered)
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Dataset shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models
    lr_model, rf_model, metrics_lr, metrics_rf = train_models(
        X_train, X_test, y_train, y_test
    )

    # Save models
    save_models(lr_model, rf_model)

    # Return results
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'feature_names': feature_names,
        'metrics_lr': metrics_lr,
        'metrics_rf': metrics_rf
    }


if __name__ == "__main__":
    results = train_pipeline(use_engineered=True)
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

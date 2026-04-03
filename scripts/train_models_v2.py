"""
Next-Gen Model Training (v2) for Project 2026.
Features: XGBoost, Advanced Features, and SHAP readiness.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Check for XGBoost, fallback to RF if not available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not found. Falling back to Random Forest for v2 Champion model.")

from scripts.config import (
    FEATURES_DATA_PATH,
    MODEL_DIR,
    TARGET_COLUMN
)
from scripts.feature_engineer_v2 import engineer_features_v2

# New Model Paths
V2_MODEL_PATH = f"{MODEL_DIR}/champion_model_v2.pkl"

def train_v2_models():
    """Train advanced models for Project 2026."""
    print("🧠 Starting Project 2026 Model Training Pipeline...")
    
    # 1. Load data
    if not os.path.exists(FEATURES_DATA_PATH):
        print(f"❌ Data not found at {FEATURES_DATA_PATH}. Please run feature engineering first.")
        return

    df = pd.read_csv(FEATURES_DATA_PATH)
    
    # 2. Upgrade features to v2
    print("✨ Upgrading features to 2026 Professional standard...")
    df = engineer_features_v2(df)
    
    # 3. Prepare features
    # Drop non-numeric columns and target
    X = df.drop(columns=[TARGET_COLUMN, 'Item', 'Country', 'date'], errors='ignore')
    # Filter for numeric columns only (handle one-hot encoded crops)
    X = X.select_dtypes(include=[np.number])
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Champion Model (XGBoost)
    if HAS_XGB:
        print("🚀 Training XGBoost Champion Model...")
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            n_jobs=-1,
            random_state=42
        )
    else:
        print("🌲 Training Advanced Random Forest Champion Model...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"✅ Training Complete!")
    print(f"📊 Model Performance (v2): R2={r2:.4f}, MAE={mae:.2f}")
    
    # 6. Save Model
    joblib.dump(model, V2_MODEL_PATH)
    print(f"💾 Champion Model saved to: {V2_MODEL_PATH}")
    
    # Save feature names for inference alignment
    feature_names_path = f"{MODEL_DIR}/v2_features.joblib"
    joblib.dump(list(X.columns), feature_names_path)
    
    return model

if __name__ == "__main__":
    train_v2_models()

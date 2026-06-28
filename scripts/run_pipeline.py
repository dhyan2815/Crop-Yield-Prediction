import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Find source columns by keyword so the pipeline can adapt to minor schema variations.
def find_col(df, keyword):
    matches = [col for col in df.columns if keyword in col.lower()]
    return matches[0] if matches else None

print("--- STARTING DATA PIPELINE ---")

print("1. CLEANING DATA...")
raw_path = 'data/raw/india_crop_yield.csv'
if not os.path.exists(raw_path):
    # Fall back to the working directory copy if the repository data folder has not been populated yet.
    raw_path = 'india_crop_yield.csv'

# Normalize column names and map the source schema onto the project's canonical field names.
df = pd.read_csv(raw_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)

state_col = find_col(df, 'state')
crop_col = find_col(df, 'crop')
year_col = find_col(df, 'year')
season_col = find_col(df, 'season')
production_col = find_col(df, 'production')
area_col = find_col(df, 'area')

# Rename the discovered fields so downstream cleaning and training can assume one schema.
df = df.rename(columns={
    state_col: 'state',
    crop_col: 'crop',
    year_col: 'crop_year',
    season_col: 'season',
    production_col: 'production',
    area_col: 'area'
})

# Convert the raw production and area fields into numeric values before yield calculation.
df['production'] = pd.to_numeric(df['production'], errors='coerce')
df['area'] = pd.to_numeric(df['area'], errors='coerce')
df = df.dropna(subset=['production', 'area'])
df = df[df['area'] > 0]
# Compute yield in kg/ha and filter out obviously invalid extremes.
df['yield_kg_ha'] = (df['production'] * 1000) / df['area']
df = df[(df['yield_kg_ha'] > 10) & (df['yield_kg_ha'] < 100000)]
df = df[['state', 'crop_year', 'season', 'crop', 'yield_kg_ha']]

# Save the cleaned dataset as the shared input for later feature engineering.
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/cleaned.csv', index=False)
print(f"Saved cleaned.csv with {len(df)} rows.")

print("2. FEATURE ENGINEERING...")
# Re-read the cleaned dataset so feature engineering starts from the persisted canonical file.
df = pd.read_csv('data/processed/cleaned.csv')
year_min, year_max = df['crop_year'].min(), df['crop_year'].max()
# Normalize year into [0, 1] so the model sees a consistent scale.
df['year_normalized'] = (df['crop_year'] - year_min) / (year_max - year_min)

# Expand the categorical fields into the one-hot columns expected by contract-based inference.
df_encoded = pd.get_dummies(df, columns=['state', 'crop', 'season'], prefix=['state', 'crop', 'season'])
y = df_encoded['yield_kg_ha']
X = df_encoded.drop(columns=['yield_kg_ha', 'crop_year'])

feature_cols = X.columns.tolist()
os.makedirs('models', exist_ok=True)
# Save the ordered feature list plus target transform metadata for inference.
contract = {
    "features": feature_cols,
    "target_transform": "log1p"
}
with open('models/feature_columns.json', 'w') as f:
    json.dump(contract, f)

os.makedirs('data/features', exist_ok=True)
df_final = pd.concat([X, y], axis=1)
df_final.to_csv('data/features/features.csv', index=False)
print(f"Saved features.csv & feature_columns.json ({len(feature_cols)} features)")

print("3. MODEL TRAINING (Log-Scale)...")
# Train on log1p(yield) so the forest learns relative differences more smoothly.
df = pd.read_csv('data/features/features.csv')
X = df.drop(columns=['yield_kg_ha'])
y = np.log1p(df['yield_kg_ha']) # Applying Log-Transform for robustness

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples...")
# Random Forest gives a strong baseline without requiring much feature scaling or tuning.
model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Convert predictions back to the original kg/ha scale before evaluating performance.
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
y_actual = np.expm1(y_test)

print("--- MODEL PERFORMANCE (Original Scale) ---")
print(f"R-squared Score: {r2_score(y_actual, preds):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_actual, preds):.2f} kg/ha")

# Persist the trained estimator in the canonical model path used by the app.
joblib.dump(model, 'models/model.pkl', compress=3)
print("Saved models/model.pkl (compressed)")

print("--- PIPELINE COMPLETE ---")

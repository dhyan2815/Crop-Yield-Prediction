import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Helper to find column by keyword
def find_col(df, keyword):
    matches = [col for col in df.columns if keyword in col.lower()]
    return matches[0] if matches else None

print("--- STARTING DATA PIPELINE ---")

print("1. CLEANING DATA...")
raw_path = 'data/raw/india_crop_yield.csv'
if not os.path.exists(raw_path):
    # Fallback to current directory for debugging
    raw_path = 'india_crop_yield.csv'

df = pd.read_csv(raw_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)

state_col = find_col(df, 'state')
crop_col = find_col(df, 'crop')
year_col = find_col(df, 'year')
season_col = find_col(df, 'season')
production_col = find_col(df, 'production')
area_col = find_col(df, 'area')

df = df.rename(columns={
    state_col: 'state',
    crop_col: 'crop',
    year_col: 'crop_year',
    season_col: 'season',
    production_col: 'production',
    area_col: 'area'
})

df['production'] = pd.to_numeric(df['production'], errors='coerce')
df['area'] = pd.to_numeric(df['area'], errors='coerce')
df = df.dropna(subset=['production', 'area'])
df = df[df['area'] > 0]
df['yield_kg_ha'] = (df['production'] * 1000) / df['area']
df = df[(df['yield_kg_ha'] > 10) & (df['yield_kg_ha'] < 100000)]
df = df[['state', 'crop_year', 'season', 'crop', 'yield_kg_ha']]

os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/cleaned.csv', index=False)
print(f"Saved cleaned.csv with {len(df)} rows.")

print("2. FEATURE ENGINEERING...")
df = pd.read_csv('data/processed/cleaned.csv')
year_min, year_max = df['crop_year'].min(), df['crop_year'].max()
df['year_normalized'] = (df['crop_year'] - year_min) / (year_max - year_min)

df_encoded = pd.get_dummies(df, columns=['state', 'crop', 'season'], prefix=['state', 'crop', 'season'])
y = df_encoded['yield_kg_ha']
X = df_encoded.drop(columns=['yield_kg_ha', 'crop_year'])

feature_cols = X.columns.tolist()
os.makedirs('models', exist_ok=True)
# Save metadata-enhanced contract
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
df = pd.read_csv('data/features/features.csv')
X = df.drop(columns=['yield_kg_ha'])
y = np.log1p(df['yield_kg_ha']) # Applying Log-Transform for robustness

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples...")
model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluate on original scale
preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
y_actual = np.expm1(y_test)

print("--- MODEL PERFORMANCE (Original Scale) ---")
print(f"R-squared Score: {r2_score(y_actual, preds):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_actual, preds):.2f} kg/ha")

joblib.dump(model, 'models/model.pkl', compress=3)
print("Saved models/model.pkl (compressed)")

print("--- PIPELINE COMPLETE ---")

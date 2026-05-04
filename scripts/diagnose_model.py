import joblib, json, numpy as np, pandas as pd, sys, os

# Make sure we run from project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load model and contract
model = joblib.load('models/model.pkl')
with open('models/feature_columns.json') as f:
    contract = json.load(f)

features = contract['features']
transform = contract.get('target_transform')

print(f'=== CONTRACT INFO ===')
print(f'Transform: {transform}')
print(f'Num features: {len(features)}')
print(f'First 5 features: {features[:5]}')
print(f'Season features: {[f for f in features if f.startswith("season_")]}')

# ---- YEAR NORMALIZATION CHECK ----
YEAR_MIN = 2000
YEAR_MAX = 2026
print(f'\n=== YEAR NORMALIZATION CHECK ===')
for yr in [2000, 2010, 2020, 2026]:
    norm = (yr - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    print(f'  Year {yr} -> year_normalized = {norm:.4f}')

# ---- TEST PREDICTION FUNCTION (mirroring predictor.py exactly) ----
def predict_yield(inputs, verbose=True):
    row = {col: 0 for col in features}
    if 'year_normalized' in row:
        year = inputs.get('year', YEAR_MAX)
        row['year_normalized'] = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)

    state_col = f"state_{inputs.get('state')}"
    crop_col  = f"crop_{inputs.get('crop')}"
    season_col = f"season_{inputs.get('season')}"

    state_hit  = state_col  in row
    crop_hit   = crop_col   in row
    season_hit = season_col in row

    if state_col  in row: row[state_col]  = 1
    if crop_col   in row: row[crop_col]   = 1
    if season_col in row: row[season_col] = 1

    input_df = pd.DataFrame([row])[features]
    raw_pred = model.predict(input_df)[0]

    if transform == 'log1p':
        inv_pred = np.expm1(raw_pred)
    else:
        inv_pred = raw_pred

    safe_pred = np.nan_to_num(inv_pred, nan=0.0, posinf=100000.0)
    final = max(0, float(safe_pred))

    if verbose:
        print(f'  state_col  = {state_col!r:40s}  hit={state_hit}')
        print(f'  crop_col   = {crop_col!r:40s}  hit={crop_hit}')
        print(f'  season_col = {season_col!r:40s}  hit={season_hit}')
        print(f'  raw_pred (log-space) = {raw_pred:.6f}')
        print(f'  inv_pred (expm1)     = {inv_pred:.4f}')
        print(f'  safe_pred            = {safe_pred:.4f}')
        print(f'  FINAL                = {final:,.2f} kg/ha')
    return final

# ---- TEST CASES ----
print('\n=== TEST CASE 1: Rice / West Bengal / Kharif / 2020 ===')
predict_yield({'state': 'West Bengal', 'crop': 'Rice', 'season': 'Kharif', 'year': 2020})

print('\n=== TEST CASE 2: Wheat / Punjab / Rabi / 2020 ===')
predict_yield({'state': 'Punjab', 'crop': 'Wheat', 'season': 'Rabi', 'year': 2020})

print('\n=== TEST CASE 3: Sugarcane / Uttar Pradesh / Whole Year / 2020 ===')
predict_yield({'state': 'Uttar Pradesh', 'crop': 'Sugarcane', 'season': 'Whole Year', 'year': 2020})

print('\n=== TEST CASE 4: INVALID crop (MISMATCH SIMULATION) ===')
predict_yield({'state': 'West Bengal', 'crop': 'FAKECROP', 'season': 'Kharif', 'year': 2020})

print('\n=== TEST CASE 5: ALL ZEROS (all identifiers miss) ===')
predict_yield({'state': 'FAKESTATE', 'crop': 'FAKECROP', 'season': 'FAKESEASON', 'year': 2020})

print('\n=== YEAR SENSITIVITY: Rice / West Bengal / Kharif across years ===')
for yr in [2000, 2005, 2010, 2015, 2020, 2026]:
    row = {col: 0 for col in features}
    row['year_normalized'] = (yr - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    if 'state_West Bengal' in row: row['state_West Bengal'] = 1
    if 'crop_Rice' in row: row['crop_Rice'] = 1
    if 'season_Kharif' in row: row['season_Kharif'] = 1
    input_df = pd.DataFrame([row])[features]
    raw = model.predict(input_df)[0]
    final = np.expm1(raw)
    print(f'  Year {yr}: {final:,.2f} kg/ha')

print('\n=== WHAT DOES THE MODEL PREDICT ON AN ALL-ZERO ROW? ===')
zero_row = pd.DataFrame([{col: 0 for col in features}])[features]
raw_zero = model.predict(zero_row)[0]
print(f'  raw log-space: {raw_zero:.6f}')
print(f'  expm1: {np.expm1(raw_zero):,.4f} kg/ha')
print()
print('=== IMPORTANT: Is 100000 the posinf clip value? ===')
print(f'  np.nan_to_num(np.inf,  nan=0.0, posinf=100000.0) = {np.nan_to_num(np.inf,  nan=0.0, posinf=100000.0)}')
print(f'  np.nan_to_num(np.nan, nan=0.0, posinf=100000.0) = {np.nan_to_num(np.nan, nan=0.0, posinf=100000.0)}')
print()
print('=== FEATURE IMPORTANCE: Top 20 features ===')
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print(feat_imp.head(20).to_string())
print()
print('=== ZERO IMPORTANCE features (potential dead features) ===')
zero_imp = feat_imp[feat_imp == 0]
print(f'  Count of zero-importance features: {len(zero_imp)}')
print(zero_imp.to_string())

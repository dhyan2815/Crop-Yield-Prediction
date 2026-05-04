"""
Full dataset audit script.
Answers: what is in the raw CSV, what are the yield ranges per crop,
how many unique state/crop/season combos exist, and what is the
distribution of the target variable.
"""
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = 'data/raw/india_crop_yield.csv'
CLEANED = 'data/processed/cleaned.csv'

# ──────────────────────────────────────────────────────────
# 1. RAW FILE
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("RAW DATASET AUDIT")
print("=" * 60)
df_raw = pd.read_csv(RAW)
df_raw.columns = df_raw.columns.str.strip().str.lower()
print(f"Shape        : {df_raw.shape}")
print(f"Columns      : {list(df_raw.columns)}")
print(f"Dtypes:\n{df_raw.dtypes}")
print(f"\nNull counts:\n{df_raw.isnull().sum()}")
print(f"\nSample rows:")
print(df_raw.head(5).to_string())

# ──────────────────────────────────────────────────────────
# 2. CLEANED FILE (post-pipeline)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANED DATASET AUDIT")
print("=" * 60)
df = pd.read_csv(CLEANED)
print(f"Shape        : {df.shape}")
print(f"Columns      : {list(df.columns)}")
print(f"Year range   : {df['crop_year'].min()} – {df['crop_year'].max()}")
print(f"Unique states: {df['state'].nunique()} → {sorted(df['state'].unique())}")
print(f"Unique crops : {df['crop'].nunique()} → {sorted(df['crop'].unique())}")
print(f"Unique seasons: {sorted(df['season'].unique())}")

# ──────────────────────────────────────────────────────────
# 3. YIELD DISTRIBUTION per CROP
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("YIELD STATS PER CROP (kg/ha)")
print("=" * 60)
crop_stats = df.groupby('crop')['yield_kg_ha'].agg(['count','min','mean','median','max','std'])
crop_stats.columns = ['count','min','mean','median','max','std']
crop_stats = crop_stats.sort_values('mean', ascending=False)
print(crop_stats.round(1).to_string())

# ──────────────────────────────────────────────────────────
# 4. YIELD DISTRIBUTION per STATE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("YIELD STATS PER STATE (kg/ha)")
print("=" * 60)
state_stats = df.groupby('state')['yield_kg_ha'].agg(['count','min','mean','median','max'])
state_stats.columns = ['count','min','mean','median','max']
state_stats = state_stats.sort_values('mean', ascending=False)
print(state_stats.round(1).to_string())

# ──────────────────────────────────────────────────────────
# 5. COMBO COVERAGE: how many (state, crop) pairs exist?
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COVERAGE: state × crop × season combos")
print("=" * 60)
combos = df.groupby(['state','crop','season'])['yield_kg_ha'].agg(['count','mean']).reset_index()
combos.columns = ['state','crop','season','n_records','mean_yield']
print(f"Total unique (state, crop, season) combos: {len(combos)}")
print(f"Combos with < 3 records (sparse): {(combos['n_records'] < 3).sum()}")
print(f"Combos with 1 record only       : {(combos['n_records'] == 1).sum()}")
# Show top sparse
print("\nTop 20 sparsest combos:")
print(combos.nsmallest(20, 'n_records').to_string(index=False))

# ──────────────────────────────────────────────────────────
# 6. YEAR x CROP: which crops have data in which years?
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("YEAR COVERAGE per CROP")
print("=" * 60)
year_crop = df.groupby('crop')['crop_year'].agg(['min','max','nunique'])
year_crop.columns = ['first_year','last_year','n_years']
year_crop = year_crop.sort_values('n_years', ascending=True)
print(year_crop.to_string())

# ──────────────────────────────────────────────────────────
# 7. EXTRAPOLATED YEARS (2021-2026): are these real or synthetic?
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA FOR YEARS > 2020 (extrapolated?)")
print("=" * 60)
future_df = df[df['crop_year'] > 2020]
print(f"Records with year > 2020: {len(future_df)}")
if len(future_df) > 0:
    print(future_df.groupby(['crop_year','crop'])['yield_kg_ha'].mean().unstack().round(1).to_string())

# ──────────────────────────────────────────────────────────
# 8. TARGET VARIABLE DISTRIBUTION
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TARGET VARIABLE DISTRIBUTION (yield_kg_ha)")
print("=" * 60)
y = df['yield_kg_ha']
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
print(f"\n  log1p stats:")
print(f"  Min log1p: {y_log.min():.3f}")
print(f"  Max log1p: {y_log.max():.3f}")
print(f"  Mean log1p: {y_log.mean():.3f}")
print(f"  Skew log1p: {y_log.skew():.3f}")

print("\n\n=== AUDIT COMPLETE ===")

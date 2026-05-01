# Dataset Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the new 2000-2026 dataset (62 crops) into the crop yield prediction model, update the feature contract, and adjust the UI configurations to support the expanded data.

**Architecture:** 
1. Create a script to download the Hugging Face dataset and replace the raw CSV.
2. Update the `final_data_cleaning.ipynb` to remove the 15-crop restriction.
3. Update `config.py` with the new year boundaries and datasets.
4. Run the data processing and training pipelines to generate a new model and feature contract.
5. Move the generated artifacts to their proper directories.

**Tech Stack:** Python, Pandas, Scikit-Learn, Jupyter

---

### Task 1: Download New Dataset

**Files:**
- Create: `scripts/download_hf_dataset.py`
- Modify: `data/raw/india_crop_yield.csv`

- [ ] **Step 1: Write dataset download script**

```python
# scripts/download_hf_dataset.py
import pandas as pd
import os

url = "hf://datasets/dhyann2815/india-crop-yield-prediction/data/train-00000-of-00001.parquet"
print(f"Downloading dataset from Hugging Face...")
df = pd.read_parquet(url)

# Save to data/raw
output_path = os.path.join("data", "raw", "india_crop_yield.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Dataset successfully saved to {output_path} with {len(df)} rows.")
```

- [ ] **Step 2: Execute script**

Run: `python scripts/download_hf_dataset.py`
Expected: Output showing successful save to `data/raw/india_crop_yield.csv`.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_hf_dataset.py
git commit -m "feat: add script to download Hugging Face dataset"
```

### Task 2: Remove Crop Filter in Data Cleaning Notebook

**Files:**
- Modify: `notebook/final_data_cleaning.ipynb`

- [ ] **Step 1: Edit Notebook Content**

Open `notebook/final_data_cleaning.ipynb` and remove the filtering logic for the top 15 crops. Replace the cell under `# 4. FILTER TOP 15 CROPS` with just the printing logic so we don't drop any crops:

```python
# ==============================
# 4. REVIEW UNIQUE CROPS
# ==============================
print("Initial unique crops:", df['crop'].nunique())
```

- [ ] **Step 2: Commit**

```bash
git add notebook/final_data_cleaning.ipynb
git commit -m "chore: remove top 15 crop filter to support 62 crops"
```

### Task 3: Update Configuration

**Files:**
- Modify: `scripts/config.py`

- [ ] **Step 1: Update Configuration Boundaries**

Modify `scripts/config.py` to change `YEAR_MIN` and `YEAR_MAX`, update `APP_DATA_SOURCES`, and remove `TOP_15_CROPS`.

```python
# In scripts/config.py, update these variables:

# Target variable definition
TARGET_COLUMN = "yield_kg_ha"

# Metadata for UI
# (Remove TOP_15_CROPS list entirely, as it is no longer needed with dynamic loading)

# Year range (based on the new dataset)
YEAR_MIN = 2000
YEAR_MAX = 2026

# Data sourcing references for transparency
APP_DATA_SOURCES = {
    "Primary Dataset": "https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction"
}
```

- [ ] **Step 2: Commit**

```bash
git add scripts/config.py
git commit -m "config: update year boundaries and data source for new dataset"
```

### Task 4: Run Pipeline and Generate New Artifacts

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `data/processed/cleaned.csv`
- Modify: `data/features/features.csv`
- Modify: `models/feature_columns.json`
- Modify: `models/model.pkl`

- [ ] **Step 1: Execute Cleaning, Engineering, and Training**

Create a quick `run_pipeline.py` script to run the required logic from the notebooks headlessly.

```python
# scripts/run_pipeline.py
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

print("1. CLEANING DATA...")
df = pd.read_csv('data/raw/india_crop_yield.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)

df = df.rename(columns={
    'state': 'state', 'crop': 'crop', 'year': 'crop_year',
    'season': 'season', 'production': 'production', 'area': 'area'
})

df['production'] = pd.to_numeric(df['production'], errors='coerce')
df['area'] = pd.to_numeric(df['area'], errors='coerce')
df = df.dropna(subset=['production', 'area'])
df = df[df['area'] > 0]
df['yield_kg_ha'] = (df['production'] * 1000) / df['area']
df = df[(df['yield_kg_ha'] > 10) & (df['yield_kg_ha'] < 100000)]
df = df[['state', 'crop_year', 'season', 'crop', 'yield_kg_ha']]

df.to_csv('data/processed/cleaned.csv', index=False)
print("Saved cleaned.csv")

print("2. FEATURE ENGINEERING...")
df = pd.read_csv('data/processed/cleaned.csv')
year_min, year_max = df['crop_year'].min(), df['crop_year'].max()
df['year_normalized'] = (df['crop_year'] - year_min) / (year_max - year_min)

df_encoded = pd.get_dummies(df, columns=['state', 'crop', 'season'], prefix=['state', 'crop', 'season'])
y = df_encoded['yield_kg_ha']
X = df_encoded.drop(columns=['yield_kg_ha', 'crop_year'])

feature_cols = X.columns.tolist()
with open('models/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)

df_final = pd.concat([X, y], axis=1)
df_final.to_csv('data/features/features.csv', index=False)
print(f"Saved features.csv & feature_columns.json ({len(feature_cols)} features)")

print("3. MODEL TRAINING...")
df = pd.read_csv('data/features/features.csv')
X = df.drop(columns=['yield_kg_ha'])
y = df['yield_kg_ha']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"R2: {r2_score(y_test, preds):.4f}, MAE: {mean_absolute_error(y_test, preds):.2f}")

joblib.dump(model, 'models/model.pkl')
print("Saved model.pkl")
```

- [ ] **Step 2: Run Pipeline Script**

Run: `python scripts/run_pipeline.py`
Expected: Passes cleaning, feature engineering, and training, updating all CSVs and the model/contract JSON.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_pipeline.py data/processed/cleaned.csv data/features/features.csv models/feature_columns.json models/model.pkl
git commit -m "feat: re-run data processing and train new model with expanded dataset"
```

### Task 5: Verify App

**Files:**
- None modified, manual testing.

- [ ] **Step 1: Run the Streamlit App**

Run: `streamlit run app.py`

- [ ] **Step 2: Verify Results**
Open the browser (usually `http://localhost:8501`).
Check that the year slider now goes up to 2026.
Check that the crop dropdown contains 62 options instead of 15.
Ensure predictions generate successfully when selecting one of the newly added crops.

# Codebase Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `app.py` into a modular structure to improve the logic flow, structure, and readability, without breaking existing logic.

**Architecture:** Break down the monolithic `app.py` by extracting data loading, predicting, visualizing, and UI styling into a new `utils/` package. `app.py` will serve solely as the Streamlit orchestration entry point.

**Tech Stack:** Python, Streamlit, Pandas, Matplotlib

---

### Task 1: Setup Utils Package & Extract Data Loading

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/data_loader.py`

- [ ] **Step 1: Create `utils` package**

```bash
mkdir -p utils
```

- [ ] **Step 2: Create `utils/__init__.py`**

```bash
touch utils/__init__.py
```

- [ ] **Step 3: Create `utils/data_loader.py`**

```python
import streamlit as st
import pandas as pd
import joblib
from scripts.config import LR_MODEL_PATH, RF_MODEL_PATH, CHAMPION_MODEL_PATH, FEATURES_DATA_PATH

@st.cache_resource
def load_models():
    """Load trained ML models."""
    models = {}
    try:
        models['champion'] = joblib.load(CHAMPION_MODEL_PATH)
    except Exception:
        models['champion'] = None
    try:
        models['lr'] = joblib.load(LR_MODEL_PATH)
    except Exception:
        models['lr'] = None
    try:
        models['rf'] = joblib.load(RF_MODEL_PATH)
    except Exception:
        models['rf'] = None
    return models

@st.cache_data
def load_features_data():
    """Load processed features data."""
    try:
        df = pd.read_csv(FEATURES_DATA_PATH)
        df.columns = df.columns.str.strip()
        if 'Item' in df.columns:
            df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_available_options():
    """Get available crops and year range."""
    df = load_features_data()
    if df.empty:
        return [], 1990, 2013
    crops = sorted(df['Item'].unique())
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    return crops, min_year, max_year

@st.cache_data
def get_dataset_stats():
    """Compute pesticide statistics."""
    df = load_features_data()
    if df.empty:
        return {'pesticide_min': 0, 'pesticide_max': 100000, 'pesticide_median': 5000}
    return {
        'pesticide_min': float(df['pesticides_tonnes'].min()),
        'pesticide_max': float(df['pesticides_tonnes'].max()),
        'pesticide_median': float(df['pesticides_tonnes'].median()),
    }
```

### Task 2: Extract Prediction & Feature Logic

**Files:**
- Create: `utils/predictor.py`

- [ ] **Step 1: Create `utils/predictor.py`**

```python
import pandas as pd
import numpy as np
from scripts.feature_engineer_v2 import (
    DEFAULT_NDVI, DEFAULT_SOIL_PH, DEFAULT_SOIL_NITROGEN, DEFAULT_SOIL_ORGANIC_CARBON
)

def get_crop_columns(available_crops: list) -> list:
    return [f'Item_{c}' for c in available_crops]

def build_prediction_features(crop, year, pesticides, rainfall, temp, crop_columns):
    """
    Build feature dict for model prediction using the v2 feature pipeline.
    All v2 features are constructed here, using defaults for satellite/soil
    when external APIs are unavailable.
    """
    features = {
        'average_rain_fall_mm_per_year': rainfall,
        'avg_temp': temp,
        'pesticides_tonnes': pesticides,
    }

    features['temp_rainfall_interaction'] = temp * rainfall
    features['rainfall_squared'] = rainfall ** 2
    features['temp_squared'] = temp ** 2
    features['pesticide_per_rainfall'] = pesticides / (rainfall + 1)
    features['rainfall_deviation'] = rainfall - 1083

    year_min, year_max = 1990, 2013
    if year_min != year_max:
        features['year_normalized'] = (year - year_min) / (year_max - year_min)
    else:
        features['year_normalized'] = 1.0

    features['heat_stress_degreedays'] = float(max(0.0, temp - 35.0))
    features['drought_intensity'] = float(max(0.0, 1.0 - (rainfall / 500.0)) if rainfall < 500 else 0.0)

    features['ndvi'] = float(DEFAULT_NDVI)
    ndvi_adj = DEFAULT_NDVI + (rainfall / 2000.0) - (features['heat_stress_degreedays'] / 10.0)
    features['ndvi_adjusted'] = float(np.clip(ndvi_adj, 0, 1))
    features['soil_ph'] = DEFAULT_SOIL_PH
    features['soil_nitrogen'] = DEFAULT_SOIL_NITROGEN
    features['soil_organic_carbon'] = DEFAULT_SOIL_ORGANIC_CARBON

    base_year = 1990
    features['msp_trend'] = 1.0 + (year - base_year) * 0.03

    for col in crop_columns:
        features[col] = 0
    sel = f'Item_{crop}'
    if sel in features:
        features[sel] = 1

    return pd.DataFrame([features])

def get_feature_importance(model, crop_columns):
    """Extract feature importance, grouping crop one-hot encodings into 'Crop Type'."""
    importances = model.feature_importances_

    if hasattr(model, 'feature_names_in_'):
        names = model.feature_names_in_
    else:
        names = [
            'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes',
            'heat_stress_degreedays', 'drought_intensity',
            'ndvi', 'ndvi_adjusted',
            'soil_ph', 'soil_nitrogen', 'soil_organic_carbon',
            'msp_trend',
            'temp_rainfall_interaction', 'rainfall_deviation',
            'rainfall_squared', 'temp_squared', 'pesticide_per_rainfall',
            'year_normalized'
        ] + crop_columns

    imp_dict = dict(zip(names, importances))

    crop_imp = sum(imp_dict.get(c, 0) for c in crop_columns)

    display = {
        'Crop Type': crop_imp,
        'Rainfall': imp_dict.get('average_rain_fall_mm_per_year', 0),
        'Temperature': imp_dict.get('avg_temp', 0),
        'Pesticides': imp_dict.get('pesticides_tonnes', 0),
        'Year': imp_dict.get('year_normalized', 0),
    }
    total = sum(display.values())
    if total > 0:
        display = {k: v / total for k, v in display.items()}
    return display
```

### Task 3: Extract Visualization Logic

**Files:**
- Create: `utils/visualizations.py`

- [ ] **Step 1: Create `utils/visualizations.py`**

```python
import streamlit as st
import matplotlib.pyplot as plt

def display_results_table(y_v2, y_rf, y_lr=None):
    """Display predictions in a clean table."""
    rows = []
    rows.append(('Champion Forecast (v2)', f'{y_v2:,.0f}'))
    if y_lr is not None:
        rows.append(('Linear Regression (v1)', f'{y_lr:,.0f}'))
    if y_rf is not None:
        rows.append(('Random Forest (v1)', f'{y_rf:,.0f}'))

    table_rows = ''
    for model_name, value in rows:
        is_first = (model_name == rows[0][0])
        row_class = ' class="highlight-row"' if is_first else ''
        table_rows += f'<tr{row_class}><td>{model_name}</td><td>{value} kg/ha</td></tr>\\n'

    table_html = f"""
    <table class="data-table">
        <thead><tr><th>Model</th><th>Prediction</th></tr></thead>
        <tbody>{{table_rows}}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

def create_area_chart(df, crop):
    """Modern area chart with green gradient fill for historical yield."""
    crop_df = df[df['Item'] == crop].copy()
    avg_yield = crop_df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()
    years = avg_yield['Year'].values
    yields = avg_yield['kg_per_ha_yield'].values

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('transparent')
    ax.set_facecolor('transparent')

    ax.fill_between(years, yields, alpha=0.35, color='#2D5A27')
    ax.plot(years, yields, color='#2D5A27', linewidth=3, marker='o', markersize=6, markerfacecolor='#FFFFFF', markeredgecolor='#2D5A27', markeredgewidth=2, zorder=3)

    ax.set_title(f'{crop} — Yield Trajectory Over Time', fontsize=15, fontweight='700', color='#1A1A2E', pad=18, loc='left')
    ax.set_xlabel('Year', fontsize=12, color='#6B7280', fontweight='500')
    ax.set_ylabel('Average Yield (kg/ha)', fontsize=12, color='#6B7280', fontweight='500')

    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#D1D5DB')
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#E5E7EB')
        ax.spines[spine].set_linewidth(1)

    ax.tick_params(colors='#6B7280', labelsize=11)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig

def create_importance_chart(importance_dict):
    """Horizontal bar chart for feature importance with earthy green palette."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('transparent')
    ax.set_facecolor('transparent')

    palette = ['#1B4332', '#2D5A27', '#388E3C', '#4CAF50', '#66BB6A']
    colors = palette[:len(features)]

    bars = ax.barh(list(features), list(importances), color=colors, height=0.55, edgecolor='none', alpha=0.9)

    for bar, imp in zip(bars, importances):
        ax.text(imp + 0.02, bar.get_y() + bar.get_height() / 2, f'{imp * 100:.1f}%', va='center', fontsize=10.5, color='#1A1A2E', fontweight='500')

    ax.set_title('What Drives the Prediction?', fontsize=15, fontweight='700', color='#1A1A2E', pad=18, loc='left')
    ax.set_xlabel('Relative Contribution', fontsize=12, color='#6B7280', fontweight='500')

    xlim = max(importances) * 1.2
    ax.set_xlim(0, min(xlim, 0.6))

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis='y', labelsize=11, length=0)
    ax.set_yticklabels(list(features), color='#1A1A2E')
    ax.invert_yaxis()
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig
```

### Task 4: Refactor `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace all extracted code in `app.py` with imports**

Since replacing `app.py` is large, we extract the UI styling into a local string variable in `app.py` or keep it but remove the rest. Actually, it's safer to just import the utilities and rewrite `app.py`'s imports to be lean:

Remove all functions starting from `load_models()` up to the `main()` function in `app.py` and add these imports at the top:

```python
from utils.data_loader import load_models, load_features_data, get_available_options, get_dataset_stats
from utils.predictor import get_crop_columns, build_prediction_features, get_feature_importance
from utils.visualizations import display_results_table, create_area_chart, create_importance_chart
```

Note: Make sure to keep `CROP_DATA_SOURCES` and the massive `st.markdown(...)` CSS block in `app.py` OR move them to `utils/ui_components.py` as well. For safety, keep them in `app.py` to minimize breakage, but delete the 10+ helper functions that we just extracted.

- [ ] **Step 2: Commit**

```bash
git add utils/ app.py
git commit -m "refactor: optimize codebase structure by extracting helper modules"
```

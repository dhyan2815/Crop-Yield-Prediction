# 2026 UPGRADE PLAN

## 1. Purpose
Transform the existing Crop Yield Prediction project into a professional, AI-powered intelligence system that delivers accurate, explainable, and actionable yield forecasts for India up to 2026.

## 2. Scope Overview
- **Temporal Scope:** Extend historical data from 1990‑2013 to 1990‑2026 using latest government releases.
- **Feature Expansion:** Add soil health, satellite vegetation indices, economic market data, and climate resilience metrics.
- **Model Elevation:** Replace basic Random Forest with Gradient Boosting (XGBoost/LightGBM) plus ensemble voting and SHAP explainability.
- **UI Revolution:** Build interactive geospatial dashboard with scenario simulation and automated reporting.
- **Export Capability:** Enable CSV and PDF exports for downstream analysis and farmer use.

## 3. Target Files & Responsibilities

| Step | Task | Target File(s) | Key Operations |
|------|------|----------------|----------------|
| 1 | **Data Ingestion Pipeline** | `scripts/data_ingestor.py` | - Use UPAg API for crop yield data<br>- Pull historical weather from Open-Meteo<br>- Integrate Digital Crop Survey (DCS) district data<br>- Store raw CSVs in `data/raw/` |
| 2 | **Remote Sensing & Economic API Integration** | `scripts/api_connectors.py` | - Connect to Sentinel‑2/Landsat for NDVI/EVI<br>- Retrieve MSP and Mandi price data via Agmarknet API<br>- Cache results locally for reproducibility |
| 3 | **Advanced Feature Engineering** | `scripts/feature_engineer.py` | - Combine yield, soil NPK, weather, satellite indices into unified feature set<br>- Compute Climate Stress Degrees Days and Drought Index<br>- Standardize all yields to kg/ha, drop hg/ha column |
| 4 | **Next‑Gen Model Training** | `scripts/train_models_v2.py` | - Build XGBoost/LightGBM pipelines<br>- Implement VotingRegressor ensemble<br>- Generate SHAP explanations per prediction |
| 5 | **Dashboard & UI Refactor** | `app.py`, `scripts/feature_engineer_v2.py` | - Rebuild Streamlit UI with 3‑column layout<br>- Add interactive map (Folium/Streamlit‑Geo)<br>- Implement Scenario Builder sliders<br>- Add CSV download endpoint and PDF export module |
| 6 | **Testing & Validation** | `tests/` (new test modules) | - Backtest on 2021‑2023 historical data<br>- Time‑Series cross‑validation<br>- Validate scenario plausibility |
| 7 | **Documentation & Rollout** | `docs/2026_UPGRADE_PLAN.md`, `docs/Upgrade_Report.pdf` | - Update upgrade plan with completion checklist<br>- Generate user guide for scenario builder and export features |

## 4. Detailed Task Breakdown

### Task 1: Set Up Isolated Worktree
- [ ] **Step 1.1:** Create isolated worktree `worktree-modernize`  
  ```bash
  git worktree add ../worktree-modernize enhancements
  ```
- [ ] **Step 1.2:** Activate virtual environment  
  ```bash
  source ../.venv/bin/activate   # Linux/macOS
  .\\.venv\\Scripts\\activate   # Windows
  ```

### Task 2: Implement Data Ingestion (data_ingestor.py)
- [ ] **Step 2.1:** Write API client to fetch UPAg yield data (CSV)  
  ```python
  import requests, pandas as pd

  def fetch_upag_yields(year: int) -> pd.DataFrame:
      url = f"https://api.upag.gov.in/v1/yield/{year}"
      resp = requests.get(url)
      resp.raise_for_status()
      return pd.DataFrame(resp.json())
  ```
- [ ] **Step 2.2:** Cache response to `data/raw/upag_{year}.csv`  
- [ ] **Step 2.3:** Add CLI wrapper (`python -m scripts.data_ingestor --year 2025`)  
- [ ] **Step 2.3:** Verify schema matches existing `yield_df.csv` columns

### Task 3: Implement Remote Sensing & Economic Connectors
- [ ] **Step 3.1:** Add Sentinel‑2 NDVI retrieval (`scripts/api_connectors.py`)  
- [ ] **Step 3.2:** Integrate Agmarknet MSP price feed  
- [ ] **Step 3.3:** Cache raw JSON payloads in `data/external/`  
- [ ] **Step 3.4:** Write unit tests (`tests/test_api_connectors.py`) confirming 200 responses

### Task 4: Build Unified Feature Engineering Pipeline
- [ ] **Step 4.1:** Merge `feature_engineer.py` and `feature_engineer_v2.py` into `scripts/feature_engineer_v2.py`  
- [ ] **Step 4.2:** Implement functions:  
  - `load_soil_data()` (ICAR SoilGrids API)  
  - `calculate_nvdvi()` (Sentinel‑2 processing)  
  - `encode_stress_indices()` (heat‑stress, drought)  
- [ ] **Step 4.3:** Standardize yield column: `df['yield_kg_ha'] = df['hg/ha_yield'].div(10)`; drop `hg/ha_yield`  
- [ ] **Step 4.4:** Ensure deterministic ordering of feature columns (`feature_order = [...]`)  
- [ ] **Step 4.5:** Write tests (`tests/test_feature_engineer.py`) for shape and null‑check

### Task 5: Train Next‑Gen Model
- [ ] **Step 5.1:** In `scripts/train_models_v2.py` import XGBoost, LightGBM  
- [ ] **Step 5.2:** Build `XGBRegressor` and `LGBMRegressor` pipelines with proper preprocessing (OneHotEncoder for categorical, StandardScaler for numeric)  
- [ ] **Step 5.3:** Create `VotingRegressor` ensemble with the two models + baseline RandomForest (import from sklearn)  
- [ ] **Step 5.4:** Fit model on `processed_data/merged.csv` and serialize `model.joblib` to `models/champion_model_v2.pkl`  
- [ ] **Step 5.5:** Add SHAP explainer block:  
  ```python
  import shap
  explainer = shap.Explainer(best_model)
  shap_values = explainer(X_test)
  shap.summary_plot(shap_values, X_test)
  ```  
- [ ] **Step 5.6:** Save SHAP explanations as PNG assets (`assets/shap_{crop}.png`)

### Task 6: Dashboard Refactor (app.py)
- [ ] **Step 6.1:** Adopt 3‑column layout using Streamlit columns API  
- [ ] **Step 6.2:** Add interactive map (folium) displaying district‑level predictions  
- [ ] **Step 6.3:** Implement Scenario Builder: slider inputs for rainfall, temperature, fertilizer usage; re‑run prediction pipeline on altered inputs  
- [ ] **Step 6.4:** Add CSV export endpoint:  
  ```python
  import streamlit as st
  csv = df.to_csv(index=False).encode('utf-8')
  st.download_button("Download CSV", data=csv, file_name="yield_export.csv")
  ```
- [ ] **Step 6.5:** Implement PDF export using `reportlab` or `weasyprint` → generate `assets/report_{crop}.pdf`  
- [ ] **Step 6.6:** Integrate feature importance chart (log‑scaled) from `scripts/feature_engineer_v2.py` results  
- [ ] **Step 6.7:** Add URL display component with dropdown for crop selection and filter button

### Task 7: Testing & Validation Suite
- [ ] **Step 7.1:** Create `tests/test_backtest.py` using 2021‑2023 held‑out data  
- [ ] **Step 7.2:** Implement time‑series CV (`sklearn.model_selection.TimeSeriesSplit`)  
- [ ] **Step 7.3:** Validate SHAP explanations against domain experts (store feedback)  
- [ ] **Step 7.4:** Run unit tests for each new script (`pytest tests/`)

### Task 8: Documentation & Release Prep
- [ ] **Step 8.1:** Populate `docs/Upgrade_Report.pdf` with screenshots of map, scenario builder, and PDF report  
- [ ] **Step 8.2:** Update `README.md` with instructions:  
  ```markdown
  ## Running the Modernized Pipeline
  1. Data ingestion: `python -m scripts.data_ingestor --year 2025`
  2. Feature engineering: `python scripts/feature_engineer_v2.py`
  3. Model training: `python scripts/train_models_v2.py`
  4. Dashboard: `streamlit run app.py`
  ```
- [ ] **Step 8.3:** Tag GitHub issue #4 with label `modernization-complete`  
- [ ] **Step 8.4:** Publish release notes summarizing key upgrades (unit standardization, SHAP explanations, interactive map)

## 5. Execution Timeline (Milestones)

| Week | Milestone |
|------|-----------|
| 1 | Worktree setup, data ingestion pipelines functional |
| 2 | API connectors operational; raw external data cached |
| 3 | Feature engineering pipeline stable; unit tests passing |
| 4 | Model training completed; champion model serialized |
| 4 | UI prototype with map and scenario sliders functional |
| 4 | CSV & PDF export working; end‑to‑end testing passed |
| 5 | Documentation finalized; release notes drafted; issue #4 labeled complete |

## 6. Success Criteria
- All target files (`data_ingestor.py`, `api_connectors.py`, `feature_engineer_v2.py`, `train_models_v2.py`, `app.py`) exist and are importable.
- Data processing produces a standardized dataset with `yield_kg_ha` column and stress indices.
- Model achieves ≥ 0.85 R² on backtest set and SHAP explanations are generated per prediction.
- UI renders interactive map, scenario builder, CSV/PDF export without errors.
- All unit and integration tests pass with ≥ 90 % coverage.

## 7. Dependencies
- `pandas`, `numpy`, `requests`, `pandas-datareader`, `folium`, `streamlit`, `xgboost`, `lightgbm`, `shap`, `reportlab`, `weasyprint`, `scikit-learn`.

All right - now we have a comprehensive plan. Shall I proceed to begin executing these tasks using the executing-plans skill? We'll need your confirmation to start the first task (worktree setup).
# 📊 Dataset Comparison & Integration Report

This report analyzes the differences between the existing project dataset and the new high-granularity dataset identified in **Issue #8**.

## 🔍 Dataset Overview

| Feature | Existing Dataset (`data/raw/india_crop_yield.csv`) | New Dataset (Hugging Face) |
| :--- | :--- | :--- |
| **Source** | Kaggle (processed) | [Hugging Face (dhyann2815/india-crop-yield-prediction)](https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction) |
| **Total Records** | ~19,690 | 21,750 |
| **Temporal Range** | 1997 – 2020 | 2000 – 2026 (Extrapolated 2021-2026) |
| **Crop Coverage** | 15 Major Crops | 62 Crop Types (Fruits, Veg, Cereals, etc.) |
| **State Coverage** | ~30 States | 31 States/UTs |
| **Missing Values** | Negligible (Cleaned) | 0 (Fully Populated) |

## 🛠️ Structural Differences

### 1. Column Mapping
The new dataset uses slightly different naming conventions for key columns:
- **Existing**: `Crop_Year`
- **New**: `Year`
- **Integration Impact**: High. The preprocessing scripts and the `predict_yield` logic in `utils/predictor.py` expect `Crop_Year`.

### 2. Feature Contract
- The current **Feature Contract** (`models/feature_columns.json`) is strictly limited to 15 crops and specific seasons.
- **Integration Impact**: Very High. Adding 47 new crop types will increase the input vector size significantly (One-Hot Encoding). The model **must** be re-trained.

### 3. Data Granularity
- The existing dataset row count (~19.6k) is very similar to the new one (~21.7k), suggesting they both operate at a **State-level** aggregation rather than District-level.
- This makes the transition smoother as the "Scenario Simulator" logic remains valid.

## 🚀 Integration Criteria

To successfully integrate the new dataset while preserving the "Yield Metrics" core logic, the following steps are required:

### Phase 1: Data Preparation
- [ ] **Download & Replace**: Fetch the dataset from Hugging Face and replace `data/raw/india_crop_yield.csv`.
- [ ] **Standardization**: Rename `Year` to `Crop_Year` in the CSV to maintain compatibility with existing `scripts/config.py`.

### Phase 2: Configuration Update
- [ ] **Update `scripts/config.py`**:
    - Change `YEAR_MIN` to `2000`.
    - Change `YEAR_MAX` to `2026`.
    - Expand `TOP_15_CROPS` to a more comprehensive list (or rename it to `AVAILABLE_CROPS`).

### Phase 3: Model & Contract Refresh
- [ ] **Re-run Training**: Use the new dataset to train a new `RandomForestRegressor`.
- [ ] **Export Contract**: Generate a new `models/feature_columns.json` reflecting all 62 crops and 31 states.
- [ ] **Validation**: Ensure the new model's R-squared score meets or exceeds the current baseline (~0.98).

### Phase 4: UI/UX Alignment
- [ ] **UI Options**: The `utils/data_loader.py` logic (specifically `get_ui_options`) will automatically pick up new crops/states if it reads from the updated features file.
- [ ] **Risk Benchmarks**: Historical yield benchmarks for the 47 new crops must be calculated to support the "Risk Assessment" card.

## ⚠️ Key Observation
The new dataset includes **extrapolated data (2021-2026)**. While this is excellent for forecasting, it should be clearly communicated to users in the UI that predictions for these years are based on trend-augmented models rather than historical census data.

---

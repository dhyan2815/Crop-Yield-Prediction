# Project Enhancement & Modernization Strategy for Crop Yield Prediction

## 1. Executive Summary

The Crop Yield Prediction project provides a solid foundation for agricultural forecasting, with a data processing pipeline, model training notebooks, and a user-friendly Streamlit interface. However, its core limitation is its reliance on a dataset that ends in 2013, which significantly reduces its practical utility for current and future predictions.

This document outlines a strategic plan to modernize the application. The proposed enhancements focus on automating data ingestion, integrating real-time data, expanding the feature set with modern data sources like geospatial and satellite imagery, and creating a more robust, automated MLOps pipeline. Implementing these suggestions will transform the project from a historical analysis tool into a powerful, real-time decision-support system for agriculture.

## 2. Analysis of the Current System

### Architecture Overview

The current system is structured as follows:
-   **Data Processing (`notebooks/`):** A series of Jupyter notebooks handle data ingestion from a raw CSV, filtering for data from India, feature engineering, and cleaning.
-   **Model Training (`scripts/`):** Modular Python scripts handle the full pipeline:
    - `scripts/config.py` – shared configuration constants and data paths
    - `scripts/data_cleaner.py` – data loading, filtering, and cleaning
    - `scripts/feature_engineer.py` – interaction features, squared terms, and year normalization
    - `scripts/train_models.py` – unified training with all engineered features and crop one-hot encoding
-   **Web Application (`app.py`):** A refactored Streamlit application that imports from the modular scripts, uses all engineered features in its predictions, and serves crop-specific yield predictions based on user inputs for crop, year, and pesticide usage.

### Identified Limitations

1.  **Outdated Core Dataset:** The primary dataset (`yield_df.csv`) only contains data up to 2013. This is the single most critical limitation, making predictions for recent or future years highly unreliable.
2.  **Static Prediction Environment:** The application predicts yields using historical average weather data for the selected year. While a function exists to call a live weather API (`get_temperature`), it is not used in the final prediction logic, which instead relies on the static `avg_temp` from the dataset.
3.  **Limited Feature Set (Historical):** ❌ **RESOLVED** – Models are now trained with 17 features including crop one-hot encoding and all engineered interaction features. Predictions correctly vary by crop.
4.  **Lack of Location Specificity:** The model predicts a single yield value for a crop across all of India for a given year. It does not account for the vast regional variations in climate, soil, and farming practices within the country.
5.  **Manual Model Retraining:** ❌ **RESOLVED** – A modular training pipeline (`scripts/`) now exists. The training can be run with `python -m scripts.train_models`. Full CI/CD automation (scheduled runs on data update) remains a future goal.
6.  **Underutilized Engineered Features:** ❌ **RESOLVED** – All engineered features from `feature_engineering.ipynb` (`temp_rainfall_interaction`, `rainfall_deviation`, `rainfall_squared`, `temp_squared`, `pesticide_per_rainfall`, `year_normalized`) are now included in the training pipeline and used in predictions.
7.  **Pipeline Inconsistencies:** ❌ **RESOLVED** – The data pipeline is now consolidated into modular scripts under `scripts/`. All scripts import from a shared `config.py` with consistent paths.

## 3. Proposed Feature Enhancements

Here are six strategic enhancements to address the identified limitations and modernize the project.

---

### 1. Automated Data Ingestion Pipeline for Up-to-Date Data

-   **Suggestion:** Build a data ingestion pipeline to automatically download and integrate recent agricultural and climate data.
-   **Description:** To make relevant predictions, the model must be trained on recent data. This pipeline would replace the static CSV file with a system that regularly fetches new data from reliable sources.
-   **Implementation Steps:**
    1.  Identify new data sources (e.g., India's [Directorate of Economics and Statistics](https://eands.dacnet.nic.in/), [Open-Meteo](https://open-meteo.com/) for historical weather, or other government/research APIs).
    2.  Write Python scripts using libraries like `requests` or `pandas` to fetch and parse data from these APIs.
    3.  Automate these scripts to run on a schedule (e.g., monthly or quarterly) using a cron job or a workflow orchestrator like GitHub Actions.
    4.  Store the fetched data in a structured format (e.g., a local database or partitioned data files in the `data/raw` directory).
-   **Potential Hurdles & Solutions:**
    -   **Hurdle:** API rate limits or costs.
    -   **Solution:** Implement caching and request throttling. Start with free APIs and only move to paid ones if necessary.
    -   **Hurdle:** Inconsistent data formats from different sources.
    -   **Solution:** Develop robust data cleaning and validation functions to standardize all incoming data into a single, clean schema.

---

### 2. Integration of Real-Time & Forecasted Weather Data

-   **Suggestion:** Modify the Streamlit app to use live, location-specific weather data for current-year predictions.
-   **Description:** Instead of relying on historical averages, the app should fetch current weather data for a user-specified location to make its predictions far more accurate and relevant. The existing `get_temperature` function is a great starting point.
-   **Implementation Steps:**
    1.  Add location inputs (e.g., latitude and longitude, or City/State dropdowns) to the Streamlit UI.
    2.  Modify the prediction logic in `app.py` to call a weather API (like the existing OpenWeatherMap one) using the user's location input.
    3.  Incorporate this real-time data (temperature, rainfall, humidity) into the feature vector that is fed to the model.
-   **Potential Hurdles & Solutions:**
    -   **Hurdle:** The model was not trained on real-time features.
    -   **Solution:** Retrain the model using a new dataset that includes features similar to what the live API will provide. The feature set must be consistent between training and inference.

---

### 3. Location-Based Predictions with Geospatial Features

-   **Suggestion:** Incorporate geospatial data to enable region-specific predictions.
-   **Description:** Agriculture is highly dependent on local conditions. By adding features like soil type, elevation, and proximity to water sources, the model can move beyond a single national estimate to provide tailored predictions for specific agricultural zones.
-   **Implementation Steps:**
    1.  Obtain geospatial datasets (e.g., soil maps from [ISRIC](https://www.isric.org/), elevation data from a Digital Elevation Model like SRTM).
    2.  Create a feature engineering step that maps a given latitude/longitude to these geospatial features.
    3.  Incorporate these new features into the model training dataset.
    4.  The Streamlit app will use the user's location input to look up these features for prediction.
-   **Potential Hurdles & Solutions:**
    -   **Hurdle:** Handling large geospatial datasets can be complex.
    -   **Solution:** Use libraries like `geopandas`, `rasterio`, and `shapely` to efficiently process this data. Pre-process the data into a more accessible format if needed.

---

### 4. Expanded Feature Set with Satellite Imagery

-   **Suggestion:** Use satellite imagery to derive vegetation indices as powerful predictive features.
-   **Description:** Satellite data provides a direct measure of crop health. The Normalized Difference Vegetation Index (NDVI) is a standard metric that correlates strongly with crop yield.
-   **Implementation Steps:**
    1.  Use a platform like Google Earth Engine or Planetary Computer to access satellite imagery archives (e.g., Landsat, Sentinel-2).
    2.  Write scripts to calculate average NDVI for a given region and time period during the growing season.
    3.  Integrate these NDVI features into the training dataset.
-   **Potential Hurdles & Solutions:**
    -   **Hurdle:** Processing satellite imagery can be computationally intensive.
    -   **Solution:** Leverage cloud-based platforms like Google Earth Engine, which perform the computation in the cloud and only require you to download the final results.

---

### 5. Automated Model Retraining Pipeline

-   **Suggestion:** Convert the manual notebook-based training process into an automated pipeline.
-   **Description:** As new data is ingested (from Suggestion #1), the model should be automatically retrained to ensure it remains accurate and up-to-date.
-   **Implementation Steps:**
    1.  Refactor the code from the `data_processing`, `feature_engineering`, and `model_training` notebooks into modular Python scripts.
    2.  Create a master script that runs these steps in sequence: data cleaning -> feature engineering -> model training -> evaluation -> saving the new model (if performance improves).
    3.  Schedule this master script to run automatically whenever the raw data is updated.
-   **Potential Hurdles & Solutions:**
    -   **Hurdle:** Ensuring the stability and reliability of the automated pipeline.
    -   **Solution:** Implement comprehensive logging, error handling, and model versioning. Use a simple orchestrator like GitHub Actions to manage the workflow.

---

### 6. Improve Model Architecture and Feature Utilization

-   **Suggestion:** Utilize the existing engineered features and experiment with more powerful models.
-   **Description:** The current project has untapped potential. The feature engineering is not fully utilized, and more advanced models could likely provide a significant accuracy boost.
-   **Implementation Steps:**
    1.  Fix the data pipeline inconsistency to ensure the features created in `feature_engineering.ipynb` are actually used by `model_training.ipynb`.
    2.  Experiment with gradient boosting models like XGBoost or LightGBM, as they often outperform Random Forest on tabular data.
    3.  Perform hyperparameter tuning to optimize the model's performance.
-   **Potential Hurdles & Solutions:**
    -   **Hurdle:** More complex models can be harder to interpret.
    -   **Solution:** Use SHAP (SHapley Additive exPlanations) plots to understand feature importance and model predictions, ensuring the model's behavior remains logical.

## 4. Recommended Roadmap

The roadmap below reflects the current implementation status:

1.  **Automated Data Ingestion:** This is the highest priority, as all other enhancements depend on having current data. *(Pending)*
2.  **Automated Model Retraining:** ✅ **COMPLETED** – Modular scripts under `scripts/` provide a clean, reproducible pipeline. Run `python -m scripts.train_models` to retrain.
3.  **Improve Model & Features:** ✅ **COMPLETED** – All 17 features are now wired into training and predictions. Crop encoding, interaction features, and year normalization are all active.
4.  **Integrate Real-Time Weather:** With a solid model in place, connect it to live data for the front-end application. *(Pending)*
5.  **Add Geospatial Features:** Begin adding location-specific data to improve prediction granularity. *(Pending)*
6.  **Incorporate Satellite Imagery:** Finally, add the most advanced data sources to further refine accuracy. *(Pending)*

By following this roadmap, the Crop Yield Prediction project can evolve into a sophisticated, valuable, and modern tool for agricultural forecasting.

---

## 5. Current Implementation Results

### Model Performance (After Enhancement)

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | 1,817 | 1,170 | 0.966 |
| Random Forest | 2.13 | 0.40 | 0.9999 |

### Features Used in Training (17 total)

**Core Features (3):**
- `average_rain_fall_mm_per_year`
- `avg_temp`
- `pesticides_tonnes`

**Engineered Features (6):**
- `temp_rainfall_interaction`
- `rainfall_deviation`
- `rainfall_squared`
- `temp_squared`
- `pesticide_per_rainfall`
- `year_normalized`

**Crop Encoding (8):**
- `Item_Cassava`, `Item_Maize`, `Item_Potatoes`, `Item_Rice Paddy`
- `Item_Sorghum`, `Item_Soybeans`, `Item_Sweet Potatoes`, `Item_Wheat`

### How to Run the Pipeline

```bash
# Run data cleaning (if needed)
python -c "from scripts.data_cleaner import clean_data; clean_data()"

# Run feature engineering (if needed)
python -c "from scripts.feature_engineer import engineer_features; engineer_features()"

# Train models
python -m scripts.train_models

# Run the web app
streamlit run app.py
```

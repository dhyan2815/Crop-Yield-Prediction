# 🌾 Yield Metrics

A Streamlit app to predict crop yields in India using historical agricultural data and climate features (rainfall, temperature, pesticides).

---

## 🚀 Project Overview

This project provides crop yield forecasts to assist farmers, researchers, and policymakers in making data-driven agricultural decisions. The UI is built with Streamlit and loads trained models from `models/`.

---

## 🧱 Tech Stack

| Layer       | Tools/Frameworks                                |
|-------------|--------------------------------------------------|
| Language    | **Python**                                      |
| UI          | **Streamlit**                                   |
| ML          | **Scikit-learn, Linear Regression, Random Forest** |
| Data        | **pandas, numpy**                               |
| Viz         | **Matplotlib, Seaborn**                         |

---

## 📦 Setup

```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔑 API Key (Optional)

Temperature can be fetched from OpenWeatherMap. If unavailable, the app falls back to dataset average temperature.

- Preferred: add `.streamlit/secrets.toml`:
```toml
OPENWEATHERMAP_API_KEY = "YOUR_KEY"
```
- Or paste the key in the sidebar input at runtime.

---

## 🧮 Inputs & Outputs

- Inputs (UI):
  - **Crop**: populated dynamically from the dataset
  - **Year**: bounded to min/max years available in the dataset
  - **Pesticide Usage (tonnes)**: defaults to dataset median; shows dataset min/median/max for guidance
  - **Location**: used to fetch live temperature (if API key provided); otherwise, dataset temperature is used

- Outputs:
  - Predictions from **Linear Regression** and **Random Forest** in kg/ha
  - Bar plot comparing model predictions
  - Historical average yield trend line chart

---

## 📁 Data

- Processed CSV: `data/processed/CLEANED_Processed_India_Crop_Yield_Data.csv`
- Expected columns (trimmed by the app):
  - `Item`, `Year`, `average_rain_fall_mm_per_year`, `avg_temp`, `pesticides_tonnes`, `kg_per_ha_yield`

The app aggregates per (`Item`, `Year`) to derive rainfall and temperature for predictions.

---

## 🔧 Notes

- Models are loaded from `models/linear_regression_model.pkl` and `models/random_forest_model.pkl`.
- `requirements.txt` pins `scikit-learn==1.7.0` to ensure pickle compatibility.
- The repository contains Flask template artifacts (`templates/`, `static/`) which are not used by this Streamlit app.

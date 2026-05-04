---
title: Yield Metrics
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.48.1
app_file: app.py
pinned: false
license: mit
---

# 🌾 Yield Metrics
**Indian Crop Yield Forecasting System**

Yield Metrics is a professional-grade web application designed to predict and simulate crop yields across Indian states. Rebuilt from the ground up in April 2026, it features a "Contract-Based" machine learning architecture that guarantees zero interface mismatches and high-precision forecasting.

## 🌟 Key Features
-   **Contract-Based Inference**: Uses a JSON-based "Feature Contract" (`models/feature_columns.json`) to perfectly align frontend inputs with ML model requirements.
-   **Scenario Simulator**: Interactive sliders and dropdowns to simulate yield forecasts based on State, Crop, Year, and Season.
-   **Yield Performance Indexing**: Intelligent benchmarking that converts forecasts into a "Performance Score" (100 = National Average) for more intuitive and positive insight analysis.
-   **Premium Dashboard**: A modern, theme-aware UI (Light/Dark mode) with interactive Plotly visualizations.
-   **Risk Assessment**: Real-time evaluation of yield health (Healthy, Moderate, or Critical) based on historical benchmarks.
-   **Historical Data**: Deep-dive into 27 years of Indian agricultural statistics (2000–2026).

## 🛠️ Technology Stack
-   **Frontend**: Streamlit with custom CSS (Glassmorphism & Interactive Cards)
-   **Machine Learning**: Random Forest Regressor (Scikit-Learn 1.7.0)
-   **Visualizations**: Plotly (Interactive Charts) & HTML5/CSS3
-   **Data Processing**: Pandas & NumPy

## 📂 Project Architecture
The project follows a modular 2026-standard structure:
-   `app.py`: Main entry point and Dashboard UI.
-   `models/`: Contains the Champion Model (`model.pkl`) and the Feature Contract (`feature_columns.json`).
-   `utils/`: Refactored modules for Prediction engine, Data Loading, and UI Components.
-   `scripts/`: Configuration and project metadata (including automated data downloaders).
-   `data/`: Version-controlled datasets (Raw & Processed).

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.10+ installed, then run:
```powershell
pip install -r requirements.txt
```

### 2. Launch the Dashboard
Start the Streamlit application:
```powershell
streamlit run app.py
```

## 📊 The "Contract" Pattern
This project implements a unique architectural pattern to eliminate common "shape mismatch" errors in ML deployments. The `models/feature_columns.json` file serves as a single source of truth. The prediction engine dynamically builds input vectors to match this contract, allowing for a flexible and crash-proof user experience.

## 📈 Data Sources
Our intelligence is powered by high-granularity Indian agricultural data:
-   **Hugging Face India Crop Yield Dataset** (2000–2026) - [dhyann2815/india-crop-yield-prediction](https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction)
-   **Agricultural Statistics at a Glance**

---
© 2026 Crop Yield Prediction Project | Designed for Accuracy & Professionalism
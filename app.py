# app.py
"""
Crop Yield Prediction Web Application
Streamlit interface for predicting crop yields based on historical and real-time data.
"""

import streamlit as st
import pandas as pd
import time

from scripts.config import (
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    CHAMPION_MODEL_PATH,
    FEATURES_DATA_PATH,
    CORE_FEATURES,
    ENGINEERED_FEATURES,
    CHAMPION_FEATURES,
)
from scripts.feature_engineer_v2 import (
    FEATURE_COLUMNS as V2_FEATURE_COLUMNS,
    DEFAULT_NDVI,
    DEFAULT_SOIL_PH,
    DEFAULT_SOIL_NITROGEN,
    DEFAULT_SOIL_ORGANIC_CARBON,
    calculate_interaction_features,
    add_year_based_features,
)

from utils.data_loader import load_models, load_features_data, get_available_options, get_dataset_stats
from utils.predictor import get_crop_columns, build_prediction_features, get_feature_importance
from utils.visualizations import display_results_table, create_area_chart, create_importance_chart

# =============================================================================
# DATA SOURCE MAPPINGS (URL transparency per crop)
# =============================================================================

CROP_DATA_SOURCES = {
    "Rice": {
        "UPAg Yield Statistics": "https://api.upag.gov.in/v1/yield",
        "FAOSTAT Rice Data": "https://www.fao.org/faostat/en/#data/QC",
        "Sentinel-2 NDVI": "https://services.sentinel-hub.com/ogc/wms",
        "SoilGrids India": "https://rest.isric.org/soilgrids/v2.0/properties/query",
    },
    "Wheat": {
        "UPAg Yield Statistics": "https://api.upag.gov.in/v1/yield",
        "Agmarknet MSP": "https://api.data.gov.in/resource/9ef273ef-a641-4de2-a243-a04145617300",
        "Open-Meteo Weather": "https://archive-api.open-meteo.com/v1/archive",
    },
    "Maize": {
        "UPAg Yield Statistics": "https://api.upag.gov.in/v1/yield",
        "ICAR Research": "https://icar.org.in/technical-documents",
    },
    "Sugar Cane": {
        "Agmarknet MSP": "https://api.data.gov.in/resource/9ef273ef-a641-4de2-a243-a04145617300",
        "FAOSTAT Sugar Cane Data": "https://www.fao.org/faostat/en/#data/QC",
    },
    "default": {
        "UPAg API Documentation": "https://api.upag.gov.in/docs",
        "FAOSTAT Data Portal": "https://www.fao.org/faostat/en/#data/QC",
        "India Data Portal": "https://data.gov.in",
    },
}

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Yield Metrics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Modern Minimal / Auto-Detecting Theme (Light/Dark)
# =============================================================================

st.markdown("""
<style>
/* ================================================
   THEME SYSTEM - Auto Light/Dark Mode Detection
   ================================================ */
:root {
    /* Light theme (default) */
    --bg-primary: #FAFAFA;
    --bg-secondary: #f8faf7;
    --text-primary: #1A1A2E;
    --text-secondary: #4B5563;
    --card-bg: #ffffff;
    --card-border: #E5E7EB;
    --accent-green: #2D5A27;
    --accent-green-light: #E8F5E9;
    --accent-blue: #1976D2;
    --accent-blue-light: #E3F2FD;
    --accent-orange: #F59E0B;
    --accent-orange-light: #FFF8E1;
    --table-header-bg: #E8F5E9;
    --table-row-even: #FAFAFA;
    --table-row-hover: #F0F7EF;
    --link-color: #2D5A27;
    --link-hover: #1B4332;
    --button-primary: #2D5A27;
    --button-primary-hover: #1B4332;
    --button-secondary: #1976D2;
    --error-bg: #450a0a;
    --error-border: #dc2626;
    --error-text: #fca5a5;
}

/* Dark theme overrides - auto detected from system/Streamlight config */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1a2332;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --card-bg: #1e293b;
        --card-border: #334155;
        --accent-green: #4ade80;
        --accent-green-light: #14532d;
        --accent-blue: #60a5fa;
        --accent-blue-light: #1e3a5f;
        --accent-orange: #fbbf24;
        --accent-orange-light: #451a03;
        --table-header-bg: #14532d;
        --table-row-even: #0f172a;
        --table-row-hover: #1a2332;
        --link-color: #4ade80;
        --link-hover: #22c55e;
        --button-primary: #4ade80;
        --button-primary-hover: #22c55e;
        --button-secondary: #60a5fa;
        --error-bg: #450a0a;
        --error-border: #dc2626;
        --error-text: #fca5a5;
    }
}

/* Apply global background */
.stApp {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    color: var(--text-primary);
}

/* Headers - always high contrast */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Text content */
p, span, div, label {
    color: var(--text-primary) !important;
}

/* Streamlit alert components */
.stSuccess {
    background-color: var(--accent-green-light) !important;
    border-left: 4px solid var(--accent-green) !important;
    color: var(--text-primary) !important;
    border-radius: 0;
}

.stInfo {
    background-color: var(--accent-blue-light) !important;
    border-left: 4px solid var(--accent-blue) !important;
    color: var(--text-primary) !important;
}

.stWarning {
    background-color: var(--accent-orange-light) !important;
    border-left: 4px solid var(--accent-orange) !important;
    color: var(--text-primary) !important;
}

.stError {
    background-color: var(--error-bg) !important;
    border-left: 4px solid var(--error-border) !important;
    color: var(--error-text) !important;
}

/* Metrics - Modern Cards */
div[data-testid="stMetric"] {
    background-color: var(--card-bg) !important;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--card-border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    color: var(--text-primary) !important;
}

/* Metric label and value colors */
div[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* Tables */
.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.95rem;
    border-radius: 8px;
    overflow: hidden;
    color: var(--text-primary);
}
.data-table th {
    background-color: var(--table-header-bg);
    color: var(--accent-green) !important;
    font-weight: 600;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 2px solid var(--accent-green);
}
.data-table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--card-border);
    color: var(--text-primary);
}
.data-table tr:nth-child(even) {
    background-color: var(--table-row-even);
}
.data-table tr:hover {
    background-color: var(--table-row-hover);
}

/* Highlight row */
.highlight-row {
    background-color: var(--accent-green-light) !important;
}
.highlight-row td {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* Data Source Links */
a[href] {
    color: var(--link-color) !important;
    text-decoration: none;
    font-weight: 500;
}
a[href]:hover {
    text-decoration: underline;
    color: var(--link-hover) !important;
}

/* Footer */
.app-footer {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--card-border);
}

/* Section dividers */
hr {
    border: none;
    border-top: 1px solid var(--card-border);
    margin: 2rem 0;
}

/* Inputs */
input, textarea, select, div[role="combobox"] input {
    color: var(--text-primary) !important;
    background-color: var(--card-bg) !important;
    border-color: var(--card-border) !important;
}

/* Buttons */
button {
    background-color: var(--button-primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
button:hover {
    background-color: var(--button-primary-hover) !important;
}

/* Streamlit button specific */
.stButton button {
    background-color: var(--button-primary) !important;
    color: #ffffff !important;
    border-radius: 0.5rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
}
.stButton button:hover {
    background-color: var(--button-primary-hover) !important;
}

/* Secondary button variant */
.stButton button[kind="secondary"] {
    background-color: var(--card-bg) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--card-border) !important;
}

/* Download button */
.stDownloadButton button {
    background-color: var(--button-secondary) !important;
    color: #ffffff !important;
}
.stDownloadButton button:hover {
    background-color: var(--accent-green) !important;
}

/* Selectbox / Multiselect */
[data-baseweb="select"] {
    background-color: var(--card-bg) !important;
    color: var(--text-primary) !important;
    border-color: var(--card-border) !important;
}

/* Number input */
[data-baseweb="input"] input {
    color: var(--text-primary) !important;
    background-color: var(--card-bg) !important;
    caret-color: var(--accent-green) !important;
}

/* Slider */
[data-baseweb="slider"] {
    background-color: var(--card-border) !important;
}

/* Checkbox */
[data-baseweb="checkbox"] {
    color: var(--text-primary) !important;
}
[data-baseweb="checkbox"] input:checked + div {
    background-color: var(--accent-green) !important;
}

/* Expander */
details {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
}
details summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.title("Yield Metrics")
    st.caption("Crop yield prediction for Indian agriculture, based on historical weather patterns and agronomic inputs.")

    # Disclaimer
    st.warning("Predictions are based on historical data analysis (1990–2013). Results should not be used for critical agricultural decisions without further validation.")

    # Load models
    models = load_models()
    if models['champion'] is None:
        st.error("Champion model not found. Please train the model first.")
        return

    # Load data & options
    df = load_features_data()
    if df.empty:
        st.error("Dataset failed to load. Check data/processed/Feature_Engineered_Crop_Yield_Data.csv")
        return

    available_crops, min_year, max_year = get_available_options()
    dataset_stats = get_dataset_stats()
    crop_columns = get_crop_columns(available_crops)

    # Feature extraction from champion model
    if models['champion']:
        feature_importance_dict = get_feature_importance(models['champion'], crop_columns)
    elif models['rf']:
        feature_importance_dict = get_feature_importance(models['rf'], crop_columns)
    else:
        st.error("No trained model available for feature importance.")
        return

    st.divider()

    # ==========================================================================
    # INPUT SECTION
    # ==========================================================================
    st.header("Input Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        crop = st.selectbox("Crop", available_crops, help="Select target crop for prediction")

    with col2:
        year = st.number_input(
            "Forecast Year",
            min_value=min_year,
            max_value=max_year,
            value=max_year,
            step=1,
            help=f"Year to predict (dataset range: {min_year}–{max_year})"
        )

    with col3:
        default_pest = float(dataset_stats.get('pesticide_median', 5000))
        min_pest = float(dataset_stats.get('pesticide_min', 0))
        max_pest = float(dataset_stats.get('pesticide_max', default_pest * 10))
        step_pest = float(max(1, round((max_pest - min_pest) / 200)))
        pesticides = st.number_input(
            "Pesticide Usage (tonnes)",
            min_value=min_pest,
            max_value=max_pest,
            value=default_pest,
            step=step_pest,
        )

    predict_btn = st.button("Predict Yield", type="primary", use_container_width=True)

    # ==========================================================================
    # PREDICTION SECTION
    # ==========================================================================
    if predict_btn:
        # Validate
        if year < min_year or year > max_year:
            st.error(f"Year must be between {min_year} and {max_year}")
            st.stop()

        # Fetch historical data for crop-year to get rainfall & temp
        match = df[(df['Item'] == crop) & (df['Year'] == int(year))]
        if match.empty:
            st.error(f"No historical data for {crop} in {year}. The dataset only covers years {min_year}–{max_year}.")
            st.stop()

        rainfall = float(match['average_rain_fall_mm_per_year'].iloc[0])
        temp = float(match['avg_temp'].iloc[0])

        st.info(f"Historical context for {crop} in {year}: Rainfall ≈ {rainfall:.0f} mm, Temperature ≈ {temp:.1f}°C")

        with st.spinner("Analyzing..."):
            try:
                # Build feature DataFrame using v2 pipeline (includes all V2 features)
                input_df = build_prediction_features(crop, year, pesticides, rainfall, temp, crop_columns)

                # Align to full v2 feature set expected by champion model
                # Note: input_df already contains all V2_FEATURE_COLUMNS plus crop one-hots
                # but we ensure correct column order and presence
                missing_cols = [c for c in V2_FEATURE_COLUMNS if c not in input_df.columns]
                for col in missing_cols:
                    input_df[col] = 0  # Should not happen, but safeguard

                # Final feature order for champion model
                X_champion = input_df[V2_FEATURE_COLUMNS + crop_columns]

                # Predict with champion model
                y_champion = float(models['champion'].predict(X_champion)[0])

                # Predict with legacy models for comparison (if available)
                y_lr = None
                y_rf = None
                if models['lr'] and models['rf']:
                    # Build v1 features (without v2 additions)
                    df_v1 = input_df.copy()
                    df_v1 = calculate_interaction_features(df_v1)
                    df_v1 = add_year_based_features(df_v1)
                    v1_feats = CORE_FEATURES + ENGINEERED_FEATURES + ['year_normalized'] + crop_columns
                    for col in v1_feats:
                        if col not in df_v1.columns:
                            df_v1[col] = 0
                    X_v1 = df_v1[v1_feats]
                    y_lr = float(models['lr'].predict(X_v1)[0])
                    y_rf = float(models['rf'].predict(X_v1)[0])

                # Results
                st.success("Prediction Complete")

                # Display table
                display_results_table(y_champion, y_rf, y_lr)

                # Charts - now two columns (historical trend, feature importance)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Historical Trend")
                    st.pyplot(create_area_chart(df, crop))
                with col_b:
                    st.subheader("Feature Importance")
                    st.pyplot(create_importance_chart(feature_importance_dict))

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ==========================================================================
    # DATA SOURCES TRANSPARENCY SECTION
    # ==========================================================================
    st.divider()
    st.header("📊 Data Sources & References")
    st.markdown(f"Transparent sourcing for **{crop}** prediction:")

    sources = CROP_DATA_SOURCES.get(crop, CROP_DATA_SOURCES["default"])
    for source_name, url in sources.items():
        st.markdown(f"- [{source_name}]({url})")

    # ==========================================================================
    # CSV DOWNLOAD SECTION
    # ==========================================================================
    st.header("📥 Download Data")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if st.button("📊 Download Full Dataset", type="secondary", use_container_width=True):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Confirm Download",
                data=csv,
                file_name="crop_yield_full_features.csv",
                mime="text/csv",
            )

    with col_dl2:
        crop_filter = st.checkbox("Filter to selected crop only?", value=False, key="dfilter")
        if crop_filter:
            filtered_df = df[df['Item'] == crop]
            csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"⬇️ Download {crop} Data",
                data=csv_filtered,
                file_name=f"crop_yield_{crop.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )

    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.divider()
    st.markdown("""
    <div class="app-footer">
        <p>Yield Metrics — Built with Streamlit</p>
        <p style="font-size: 0.75rem; margin-top: 0.25rem; opacity: 0.7;">Data: India Crop Yield (1990–2013) | Models: v1 (Linear Regression & Random Forest)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

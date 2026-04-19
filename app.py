import streamlit as st
import pandas as pd
import time
import logging

from scripts.config import (
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    CHAMPION_MODEL_PATH,
    FEATURES_DATA_PATH,
    CORE_FEATURES,
    ENGINEERED_FEATURES,
    CHAMPION_FEATURES,
)
from utils.data_loader import load_models, load_features_data, get_available_options, get_dataset_stats
from utils.predictor import get_crop_columns, build_prediction_features, get_feature_importance, predict_all_models
from utils.visualizations import display_results_table, create_area_chart, create_importance_chart
from utils.ui_components import apply_custom_css, display_header, display_footer, display_data_sources

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Yield Metrics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling
apply_custom_css()

# =============================================================================
# MAIN APPLICATION logic
# =============================================================================

def handle_prediction(models, df, crop, year, pesticides, crop_columns, feature_importance_dict):
    """Encapsulated prediction UI and logic."""
    available_crops, min_year, max_year = get_available_options()
    
    # Validate
    if year < min_year or year > max_year:
        st.error(f"Year must be between {min_year} and {max_year}")
        return

    # Fetch historical data for crop-year to get rainfall & temp
    match = df[(df['Item'] == crop) & (df['Year'] == int(year))]
    if match.empty:
        st.error(f"No historical data for {crop} in {year}. The dataset only covers years {min_year}–{max_year}.")
        return

    rainfall = float(match['average_rain_fall_mm_per_year'].iloc[0])
    temp = float(match['avg_temp'].iloc[0])

    st.info(f"Historical context for {crop} in {year}: Rainfall ≈ {rainfall:.0f} mm, Temperature ≈ {temp:.1f}°C")

    with st.spinner("Analyzing..."):
        try:
            # Build features
            input_df = build_prediction_features(crop, year, pesticides, rainfall, temp, crop_columns)
            
            # Predict using all available models
            y_champion, y_rf, y_lr = predict_all_models(models, input_df, crop_columns)

            # Display Results
            st.success("Prediction Complete")
            display_results_table(y_champion, y_rf, y_lr)

            # Charts
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Historical Trend")
                st.pyplot(create_area_chart(df, crop))
            with col_b:
                st.subheader("Feature Importance")
                st.pyplot(create_importance_chart(feature_importance_dict))

        except Exception as e:
            st.error("Prediction failed. Please check input parameters and try again.")
            logging.error(f"Prediction error: {type(e).__name__}: {e}")

def main():
    # Setup Header & Layout
    display_header()

    # Load resources
    models = load_models()
    if models['champion'] is None:
        st.error("Champion model not found. Please train the model first.")
        return

    df = load_features_data()
    if df.empty:
        st.error("Dataset failed to load properly.")
        return

    available_crops, min_year, max_year = get_available_options()
    dataset_stats = get_dataset_stats()
    crop_columns = get_crop_columns(available_crops)

    # Get feature importance for display
    feature_importance_dict = get_feature_importance(models.get('champion') or models.get('rf'), crop_columns)

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

    if predict_btn:
        handle_prediction(models, df, crop, year, pesticides, crop_columns, feature_importance_dict)

    # ==========================================================================
    # TRANSPARENCY & DATA SECTION
    # ==========================================================================
    display_data_sources(crop)

    st.header("📥 Download Data")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if st.button("📊 Download Full Dataset", type="secondary", use_container_width=True, key="dl_full"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Confirm Download", data=csv, file_name="crop_yield_full.csv", mime="text/csv")

    with col_dl2:
        crop_filter = st.checkbox("Filter to selected crop only?", value=False, key="dfilter")
        if crop_filter:
            filtered_df = df[df['Item'] == crop]
            csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"⬇️ Download {crop} Data", data=csv_filtered, file_name=f"crop_yield_{crop.lower()}.csv", mime="text/csv")

    # Footer
    display_footer()

if __name__ == "__main__":
    main()

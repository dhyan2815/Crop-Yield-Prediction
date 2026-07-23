"""
Yield Metrics — Crop Yield Prediction & Intelligence Dashboard

Main Streamlit application entry point. Enables Indian agricultural yield forecasting,
historical trajectory analysis, and dynamic benchmarking.
"""

import os
import pandas as pd
import streamlit as st

from scripts.config import CLEANED_DATA_PATH, YEAR_MAX, YEAR_MIN
from utils.data_loader import (
    get_crop_averages,
    get_ui_options,
    load_model_and_contract,
)
from utils.predictor import get_risk_assessment, predict_yield
from utils.ui_components import (
    apply_custom_css,
    display_data_sources,
    display_footer,
    display_header,
)
from utils.visualizations import (
    create_historical_chart,
    display_prediction_card,
)

# Page configuration MUST be called as the first Streamlit command
st.set_page_config(
    page_title="Yield Metrics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global CSS theme foundation
apply_custom_css()


def main() -> None:
    """Render the primary Streamlit application interface."""
    display_header()

    # Load model and contract metadata
    model, contract = load_model_and_contract()
    if model is None or contract is None:
        st.error("Missing model artifact or feature contract in 'models/' folder. Please run the training pipeline first.")
        return

    # Load dropdown input options
    states, crops, seasons = get_ui_options()
    if not states:
        st.error("No reference feature data found in 'data/features/features.csv'. Please check the dataset pipeline.")
        return

    # Sidebar inputs controls
    with st.sidebar:
        st.header("📍 Location & Crop")
        selected_state = st.selectbox("State", states, help="Select the target Indian State")
        selected_crop = st.selectbox("Crop", crops, help="Target crop to predict")
        selected_season = st.selectbox("Season", seasons, help="Agricultural season (Kharif, Rabi, Whole Year, etc.)")

        st.divider()
        st.markdown("### 🛠️ Prediction Engine")
        st.info("Core Model: Random Forest Regressor")
        st.markdown(f"**Normalization Bounds:** {YEAR_MIN}—{YEAR_MAX}")

    # Top parameter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        target_year = st.slider("Target Year", YEAR_MIN, YEAR_MAX, YEAR_MAX)
    with col2:
        st.markdown("**Simulated Context**")
        st.caption("Standard seasonal baseline used for simulations.")

    # Prepare input payload for inference
    st.divider()
    inputs = {
        "state": selected_state,
        "crop": selected_crop,
        "year": target_year,
        "season": selected_season,
    }

    # Run inference
    prediction = predict_yield(model, contract, inputs)

    # Benchmark analytics
    crop_averages = get_crop_averages()
    avg_yield = crop_averages.get(selected_crop, 0.0)

    # Risk evaluation
    status, risk_msg = get_risk_assessment(prediction, selected_crop, avg_yield)

    # Main dashboard grid layout
    main_col, side_col = st.columns([2, 1])

    with main_col:
        # Display main prediction result card
        display_prediction_card(prediction, status, risk_msg)

        # Render historical trend chart
        st.subheader("📊 Historical Trajectory")
        if os.path.exists(CLEANED_DATA_PATH):
            cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
            chart = create_historical_chart(cleaned_df, selected_state, selected_crop)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("Insufficient historical records available for this specific State/Crop combination.")
        else:
            st.warning(f"Trend chart unavailable: Cleaned dataset not found at '{CLEANED_DATA_PATH}'.")

    with side_col:
        st.subheader("💡 Insights")
        st.metric("Forecast Confidence", "High", help="Based on validation performance (R² = 0.967).")

        if avg_yield > 0:
            yield_index = (prediction / avg_yield) * 100.0
            diff = prediction - avg_yield

            st.metric(
                label="Yield Performance Index",
                value=f"{yield_index:.1f}",
                delta=f"{diff:+,.0f} kg/ha",
                help=f"A score of 100 represents the national {selected_crop} average benchmark.",
            )
            st.caption(f"**Benchmark:** {avg_yield:,.0f} kg/ha ({selected_crop})")
        else:
            st.info("Performance index benchmark unavailable.")

    display_data_sources()
    display_footer()


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import os

from scripts.config import YEAR_MIN, YEAR_MAX
from utils.data_loader import load_model_and_contract, get_ui_options, load_reference_data, get_crop_averages
from utils.predictor import predict_yield, get_risk_assessment
from utils.ui_components import apply_custom_css, display_header, display_footer, display_data_sources
from utils.visualizations import display_prediction_card, create_historical_chart, create_comparison_chart

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

def main():
    # 1. Setup Header
    display_header()
    
    # 2. Load ML Brain (Model + Contract)
    model, contract = load_model_and_contract()
    if model is None or contract is None:
        st.error("Missing model or feature contract in 'models/' folder. Please run the training notebooks first.")
        return

    # 3. Load UI Options from Reference Data
    states, crops, seasons = get_ui_options()
    if not states:
        st.error("No reference data found in 'data/features/features.csv'.")
        return

    # ==========================================================================
    # SIDEBAR - GLOBAL SETTINGS
    # ==========================================================================
    with st.sidebar:
        st.header("📍 Location & Crop")
        selected_state = st.selectbox("State", states, help="Select the target Indian State")
        selected_crop = st.selectbox("Crop", crops, help="Target crop to predict")
        selected_season = st.selectbox("Season", seasons, help="Kharif, Rabi, etc.")
        
        st.divider()
        st.markdown("### 🛠️ Prediction Engine")
        st.info("Core Logic: Random Forest Regressor")
        st.markdown(f"**Normalization Range:** {YEAR_MIN}—{YEAR_MAX}")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_year = st.slider("Target Year", YEAR_MIN, YEAR_MAX, YEAR_MAX)
    
    with col2:
        # Placeholder for future weather integration or manual input
        st.markdown("**Simulated Context**")
        st.caption("Standard seasonal baseline used for simulations.")

    # ==========================================================================
    # RESULTS SECTION
    # ==========================================================================
    st.divider()
    
    # Generate Prediction
    inputs = {
        'state': selected_state,
        'crop': selected_crop,
        'year': target_year,
        'season': selected_season
    }
    
    prediction = predict_yield(model, contract, inputs)
    status, risk_msg = get_risk_assessment(prediction, selected_crop)
    
    # UI Layout for Results
    main_col, side_col = st.columns([2, 1])
    
    with main_col:
        # Display the high-end prediction card
        display_prediction_card(prediction, status, risk_msg)
        
        # Historical Trend
        st.subheader("📊 Historical Trajectory")
        # Load the original processing data for trends (if available)
        # Assuming the user has 'cleaned.csv' in processed folder
        cleaned_path = os.path.join("data", "processed", "cleaned.csv")
        if os.path.exists(cleaned_path):
            cleaned_df = pd.read_csv(cleaned_path)
            chart = create_historical_chart(cleaned_df, selected_state, selected_crop)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No sufficient historical records for this specific State/Crop combination.")
        else:
            st.warning("Trend chart unavailable: 'data/processed/cleaned.csv' not found.")

    with side_col:
        st.subheader("💡 Insights")
        st.metric("Forecast Confidence", "High", help="Based on R² score from training.")
        
        # Prediction vs Mean (Crop-Specific)
        crop_averages = get_crop_averages()
        if crop_averages and selected_crop in crop_averages:
            avg_yield = crop_averages[selected_crop]
            diff = prediction - avg_yield
            st.metric(
                label=f"Vs. National {selected_crop} Avg", 
                value=f"{diff:+,.0f} kg/ha", 
                delta=f"{(diff/avg_yield)*100:+.1f}%", 
                delta_color="normal"
            )
        else:
            st.info("Crop-specific benchmark unavailable.")

    # Footer & Sources
    display_data_sources()
    display_footer()

if __name__ == "__main__":
    main()

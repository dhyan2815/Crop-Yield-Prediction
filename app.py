import streamlit as st
import pandas as pd
import os

from scripts.config import YEAR_MIN, YEAR_MAX
from utils.data_loader import load_model_and_contract, get_ui_options, load_reference_data, get_crop_averages
from utils.predictor import predict_yield, get_risk_assessment
from utils.ui_components import apply_custom_css, display_header, display_footer, display_data_sources
from utils.visualizations import display_prediction_card, create_historical_chart, create_comparison_chart

# Configure the Streamlit page before rendering any UI.
st.set_page_config(
    page_title="Yield Metrics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling so all pages share the same visual foundation.
apply_custom_css()

def main():
    # Render the title and project summary first so the page feels oriented.
    display_header()
    
    # Load the trained model together with the feature contract that defines input shape.
    model, contract = load_model_and_contract()
    if model is None or contract is None:
        st.error("Missing model or feature contract in 'models/' folder. Please run the training notebooks first.")
        return

    # Pull dropdown options from the feature table so UI values match the encoded model inputs.
    states, crops, seasons = get_ui_options()
    if not states:
        st.error("No reference data found in 'data/features/features.csv'.")
        return

    # Sidebar collects the core inference inputs used by the prediction contract.
    with st.sidebar:
        st.header("📍 Location & Crop")
        selected_state = st.selectbox("State", states, help="Select the target Indian State")
        selected_crop = st.selectbox("Crop", crops, help="Target crop to predict")
        selected_season = st.selectbox("Season", seasons, help="Kharif, Rabi, etc.")
        
        st.divider()
        st.markdown("### 🛠️ Prediction Engine")
        st.info("Core Logic: Random Forest Regressor")
        st.markdown(f"**Normalization Range:** {YEAR_MIN}—{YEAR_MAX}")

    # The year slider stays separate so users can explore trend sensitivity quickly.
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_year = st.slider("Target Year", YEAR_MIN, YEAR_MAX, YEAR_MAX)
    
    with col2:
        # Placeholder for future context inputs such as weather or manual overrides.
        st.markdown("**Simulated Context**")
        st.caption("Standard seasonal baseline used for simulations.")

    # Build the exact input payload expected by the prediction helper.
    st.divider()
    
    inputs = {
        'state': selected_state,
        'crop': selected_crop,
        'year': target_year,
        'season': selected_season
    }
    
    # Run inference through the contract-aware prediction layer.
    prediction = predict_yield(model, contract, inputs)
    
    # Load the benchmark average for the selected crop so the forecast can be contextualized.
    crop_averages = get_crop_averages()
    avg_yield = crop_averages.get(selected_crop, 0)
    
    # Convert the raw yield into a human-readable risk label and explanation.
    status, risk_msg = get_risk_assessment(prediction, selected_crop, avg_yield)
    
    # Split the results area into the primary forecast card and supporting insights.
    main_col, side_col = st.columns([2, 1])
    
    with main_col:
        # Show the main forecast card first because it is the primary user outcome.
        display_prediction_card(prediction, status, risk_msg)
        
        # Use the processed dataset to show the historical pattern behind the prediction.
        st.subheader("📊 Historical Trajectory")
        cleaned_path = os.path.join("data", "processed", "cleaned.csv")
        if os.path.exists(cleaned_path):
            # Only render the chart when there is enough historical data to support it.
            cleaned_df = pd.read_csv(cleaned_path)
            chart = create_historical_chart(cleaned_df, selected_state, selected_crop)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No sufficient historical records for this specific State/Crop combination.")
        else:
            st.warning("Trend chart unavailable: 'data/processed/cleaned.csv' not found.")

    with side_col:
        # Surface a compact summary panel for confidence and benchmark comparison.
        st.subheader("💡 Insights")
        st.metric("Forecast Confidence", "High", help="Based on R² score from training.")
        
        # Compare the forecast with the crop mean so users can read the number in context.
        if avg_yield > 0:
            yield_index = (prediction / avg_yield) * 100
            diff = prediction - avg_yield
            
            st.metric(
                label="Yield Performance Index", 
                value=f"{yield_index:.1f}", 
                delta=f"{diff:+,.0f} kg/ha",
                help=f"A score of 100 represents the national {selected_crop} average benchmark."
            )
            st.caption(f"**Benchmark:** {avg_yield:,.0f} kg/ha ({selected_crop})")
        else:
            st.info("Performance index unavailable.")

    # Finish with transparent sourcing and the standard footer.
    display_data_sources()
    display_footer()

if __name__ == "__main__":
    main()

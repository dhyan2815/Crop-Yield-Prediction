# app.py
"""
Crop Yield Prediction Web Application
Streamlit interface for predicting crop yields based on historical and real-time data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import from modular scripts
from scripts.config import (
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    FEATURES_DATA_PATH,
    CORE_FEATURES,
    ENGINEERED_FEATURES
)
from scripts.feature_engineer import get_feature_names

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

st.set_page_config(
    page_title="Yield Metrics",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #FAFAFA;
    }

    /* Section headers */
    h2 {
        color: #1A1A2E !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-size: 1.25rem !important;
    }

    h3 {
        color: #1A1A2E !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Success message */
    .stSuccess {
        background-color: #E8F5E9 !important;
        border-left: 4px solid #2D5A27 !important;
    }

    /* Info message */
    .stInfo {
        background-color: #E3F2FD !important;
        border-left: 4px solid #1976D2 !important;
    }

    /* Warning banner */
    .stWarning {
        background-color: #FFF8E1 !important;
        border-left: 4px solid #F59E0B !important;
    }

    /* Metric styling */
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Custom table styling */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.95rem;
    }

    .custom-table th {
        background-color: #E8F5E9;
        color: #1A1A2E;
        font-weight: 600;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 2px solid #2D5A27;
    }

    .custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #E5E7EB;
    }

    .custom-table tr:nth-child(even) {
        background-color: #FAFAFA;
    }

    .custom-table tr:hover {
        background-color: #F5F5F5;
    }

    /* Number cells right-aligned */
    .custom-table td:nth-child(2),
    .custom-table td:nth-child(3) {
        text-align: right;
        font-family: 'SF Mono', Monaco, 'Courier New', monospace;
    }

    /* Best model highlight */
    .best-model {
        background-color: #E8F5E9 !important;
        font-weight: 600;
        color: #2D5A27;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-color: #2D5A27 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING (Cached)
# =============================================================================

@st.cache_resource
def load_models():
    """Load trained ML models."""
    try:
        lr_model = joblib.load(LR_MODEL_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)
        return lr_model, rf_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None


@st.cache_data
def load_features_data():
    """Load processed features data."""
    try:
        df = pd.read_csv(FEATURES_DATA_PATH)
        df.columns = df.columns.str.strip()
        # Clean crop names
        if 'Item' in df.columns:
            df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


@st.cache_data
def get_available_options():
    """Get available crops and year range from data."""
    df = load_features_data()
    if df.empty:
        return [], 1990, 2013

    crops = sorted(df['Item'].unique())
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())

    return crops, min_year, max_year


@st.cache_data
def get_dataset_stats():
    """Compute pesticide statistics from data."""
    df = load_features_data()
    if df.empty:
        return {'pesticide_min': 0.0, 'pesticide_max': 100000.0, 'pesticide_median': 5000.0}

    return {
        'pesticide_min': float(df['pesticides_tonnes'].min()),
        'pesticide_max': float(df['pesticides_tonnes'].max()),
        'pesticide_median': float(df['pesticides_tonnes'].median()),
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_crop_columns(available_crops: list) -> list:
    """Generate one-hot encoded column names for crops."""
    return [f'Item_{crop}' for crop in available_crops]


def build_prediction_features(crop: str, year: int, pesticides: float,
                             rainfall: float, temp: float,
                             crop_columns: list) -> pd.DataFrame:
    """Build feature DataFrame for model prediction."""
    feature_dict = {
        'average_rain_fall_mm_per_year': rainfall,
        'avg_temp': temp,
        'pesticides_tonnes': pesticides,
    }

    # Add engineered features (using values from input)
    feature_dict.update({
        'temp_rainfall_interaction': temp * rainfall,
        'rainfall_deviation': 0,  # Will be computed based on training mean
        'rainfall_squared': rainfall ** 2,
        'temp_squared': temp ** 2,
        'pesticide_per_rainfall': pesticides / (rainfall + 1),
        'year_normalized': 0  # Will be computed based on year range
    })

    # Compute rainfall_deviation based on approximate mean (from training)
    approx_mean_rainfall = 1083  # Approximate mean from data
    feature_dict['rainfall_deviation'] = rainfall - approx_mean_rainfall

    # Compute year_normalized based on dataset range (1990-2013)
    year_min, year_max = 1990, 2013
    if year_min != year_max:
        feature_dict['year_normalized'] = (year - year_min) / (year_max - year_min)
    else:
        feature_dict['year_normalized'] = 0

    # One-hot encode crop
    for col in crop_columns:
        feature_dict[col] = 0
    selected_col = f'Item_{crop}'
    if selected_col in feature_dict:
        feature_dict[selected_col] = 1

    return pd.DataFrame([feature_dict])


def get_feature_importance(rf_model, crop_columns: list) -> dict:
    """
    Extract and normalize feature importance from Random Forest model.

    Returns a dictionary of feature names to importance values.
    """
    # Get feature importances from RF model
    importances = rf_model.feature_importances_
    feature_names = rf_model.feature_names_in_

    # Create importance dictionary
    importance_dict = dict(zip(feature_names, importances))

    # Group crop columns into single "Crop Type" importance
    crop_importance = sum(
        importance_dict.get(col, 0) for col in crop_columns
    )

    # Build display-friendly importance dict
    display_importance = {
        'Crop Type': crop_importance,
        'Rainfall': importance_dict.get('average_rain_fall_mm_per_year', 0),
        'Temperature': importance_dict.get('avg_temp', 0),
        'Pesticides': importance_dict.get('pesticides_tonnes', 0),
        'Year': importance_dict.get('year_normalized', 0),
    }

    # Normalize to percentages
    total = sum(display_importance.values())
    if total > 0:
        display_importance = {
            k: v / total for k, v in display_importance.items()
        }

    return display_importance


def display_prediction_table(y_lr: float, y_rf: float):
    """
    Display prediction results in a clean, styled table.

    Shows both model predictions with percentage comparison to RF.
    """
    # Calculate max value for percentage comparison
    max_pred = max(y_lr, y_rf)
    lr_pct = (y_lr / max_pred) * 100 if max_pred > 0 else 0
    rf_pct = (y_rf / max_pred) * 100 if max_pred > 0 else 0

    # Build HTML table
    table_html = """
    <table class="custom-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Prediction (kg/ha)</th>
                <th>Relative Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Linear Regression</td>
                <td>{:.0f}</td>
                <td>{:.1f}%</td>
            </tr>
            <tr class="best-model">
                <td>Random Forest</td>
                <td>{:.0f}</td>
                <td>{:.1f}%</td>
            </tr>
        </tbody>
    </table>
    """.format(y_lr, lr_pct, y_rf, rf_pct)

    st.markdown(table_html, unsafe_allow_html=True)


def create_trend_plot(df: pd.DataFrame, crop: str):
    """Create line plot showing historical yield trend for a crop."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter for selected crop
    crop_df = df[df['Item'] == crop].copy()
    avg_yield = crop_df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()

    sns.lineplot(x='Year', y='kg_per_ha_yield', data=avg_yield,
                ax=ax, linewidth=2, color='blue', marker='o')

    ax.set_title(f'{crop} - Historical Yield Trend in India', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Yield (kg/ha)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def validate_inputs(crop: str, year: int, pesticides: float,
                   min_year: int, max_year: int) -> tuple:
    """Validate user inputs. Returns (is_valid, errors_list)."""
    errors = []

    if not crop or crop.strip() == "":
        errors.append("Please select a crop")

    if year < min_year or year > max_year:
        errors.append(f"Year must be between {min_year} and {max_year}")

    if pesticides < 0:
        errors.append("Pesticide usage cannot be negative")

    return len(errors) == 0, errors


# =============================================================================
# MAIN APPLICATION UI
# =============================================================================

def main():
    """Main application function."""

    # Header
    st.title("🌾 Yield Metrics")
    st.markdown("""
    Welcome to **Yield Metrics** – a crop yield prediction app for India.
    """)

    # Disclaimer
    st.warning("⚠️ **Predictions** are based on historical data and may not reflect current conditions. Use results for guidance only.")

    # Load models
    lr_model, rf_model = load_models()
    if lr_model is None or rf_model is None:
        st.error("Failed to load models. Please ensure training has been completed.")
        return

    # Get available options
    available_crops, min_year, max_year = get_available_options()
    if not available_crops:
        st.error("No data available. Please check data files.")
        return

    dataset_stats = get_dataset_stats()
    crop_columns = get_crop_columns(available_crops)

    # Main content area
    st.header("📝 Input Parameters")
    st.caption(f"Available years: {min_year} - {max_year} | Crops: {len(available_crops)}")

    # Input widgets
    crop = st.selectbox("🌱 Select Crop", available_crops, help="Choose the crop for yield prediction")
    year = st.number_input("📅 Select Year", min_value=min_year, max_value=max_year,
                          value=max_year, step=1)

    # Pesticide usage
    default_pest = float(dataset_stats.get('pesticide_median', 5000.0))
    min_pest = float(dataset_stats.get('pesticide_min', 0.0))
    max_pest = float(dataset_stats.get('pesticide_max', default_pest * 10))
    step_pest = float(max(1.0, round((max_pest - min_pest) / 200.0)))
    pesticides = st.number_input(
        "🧪 Pesticide Usage (tonnes)",
        min_value=min_pest,
        max_value=max_pest,
        value=default_pest,
        step=step_pest,
        help=f"Based on dataset: min={min_pest:.0f}, median={default_pest:.0f}, max={max_pest:.0f}"
    )

    st.markdown("---")

    # Prediction button
    if st.button("🚀 Predict Yield", type="primary", use_container_width=True):

        # Validate inputs
        is_valid, errors = validate_inputs(crop, year, pesticides, min_year, max_year)
        if not is_valid:
            for error in errors:
                st.error(f"❌ {error}")
            return

        # Show loading state
        with st.spinner("🔄 Making predictions..."):

            # Get historical data for crop-year
            df = load_features_data()
            match = df[(df['Item'] == crop) & (df['Year'] == int(year))]

            if match.empty:
                st.error(f"No data available for {crop} in {year}. Please try a different year.")
                return

            # Use historical rainfall and temperature
            rainfall = float(match['average_rain_fall_mm_per_year'].iloc[0])
            temp = float(match['avg_temp'].iloc[0])

            st.info(f"📊 Using historical data: Rainfall={rainfall:.0f}mm, Temp={temp:.1f}°C")

            # Build features and predict
            try:
                input_features = build_prediction_features(
                    crop, year, pesticides, rainfall, temp, crop_columns
                )

                # Get predictions
                yield_lr = lr_model.predict(input_features)[0]
                yield_rf = rf_model.predict(input_features)[0]

                # Display results
                st.success("✅ Prediction Complete!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Linear Regression", f"{yield_lr:.0f} kg/ha")
                with col2:
                    st.metric("Random Forest", f"{yield_rf:.0f} kg/ha")

                # Plots
                st.markdown("### 📊 Model Comparison")
                pred_fig = create_prediction_plot(yield_lr, yield_rf, crop, year)
                st.pyplot(pred_fig)

                st.markdown("### 📈 Historical Trend")
                trend_fig = create_trend_plot(df, crop)
                st.pyplot(trend_fig)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>🌾 Yield Metrics | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

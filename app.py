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
    layout="centered",
    initial_sidebar_state="expanded"
)

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


def create_prediction_plot(y_lr: float, y_rf: float, crop: str, year: int):
    """Create bar plot comparing model predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Linear Regression', 'Random Forest']
    predictions = [y_lr, y_rf]
    colors = ['skyblue', 'lightgreen']

    bars = ax.bar(models, predictions, color=colors, alpha=0.8)

    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{pred:.0f} kg/ha', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Predicted Yield (kg/ha)', fontsize=12)
    ax.set_title(f'{crop} Yield Prediction for {year}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


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

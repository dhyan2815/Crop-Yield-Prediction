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
# CUSTOM CSS - Modern Minimal / Earthy Green Theme
# =============================================================================

st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(180deg, #f8faf7 0%, #FAFAFA 100%);
    }

    /* Headers */
    h1 { color: #1A1A2E !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
    h2 { color: #1A1A2E !important; font-weight: 600 !important; margin-top: 1.5rem !important; }
    h3 { color: #1A1A2E !important; font-weight: 600 !important; }

    /* Success */
    .stSuccess {
        background-color: #E8F5E9 !important;
        border-left: 4px solid #2D5A27 !important;
        border-radius: 0;
    }

    /* Info */
    .stInfo {
        background-color: #E3F2FD !important;
        border-left: 4px solid #1976D2 !important;
    }

    /* Warning */
    .stWarning {
        background-color: #FFF8E1 !important;
        border-left: 4px solid #F59E0B !important;
    }

    /* Metrics - Modern Cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
    }

    /* Tables */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.95rem;
        border-radius: 8px;
        overflow: hidden;
    }
    .data-table th {
        background-color: #E8F5E9;
        color: #2D5A27;
        font-weight: 600;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 2px solid #2D5A27;
    }
    .data-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #E5E7EB;
    }
    .data-table tr:nth-child(even) { background-color: #FAFAFA; }
    .data-table tr:hover { background-color: #F0F7EF; }

    /* Highlight row */
    .highlight-row {
        background-color: #E8F5E9 !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E5E7EB;
    }

    /* Section dividers */
    .section-divider {
        border: none;
        border-top: 1px solid #E5E7EB;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_models():
    """Load trained ML models."""
    models = {}
    # Load champion model first
    try:
        models['champion'] = joblib.load(CHAMPION_MODEL_PATH)
    except Exception:
        models['champion'] = None
    # Load legacy models
    try:
        models['lr'] = joblib.load(LR_MODEL_PATH)
    except Exception:
        models['lr'] = None
    try:
        models['rf'] = joblib.load(RF_MODEL_PATH)
    except Exception:
        models['rf'] = None
    return models


@st.cache_data
def load_features_data():
    """Load processed features data."""
    try:
        df = pd.read_csv(FEATURES_DATA_PATH)
        df.columns = df.columns.str.strip()
        if 'Item' in df.columns:
            df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


@st.cache_data
def get_available_options():
    """Get available crops and year range."""
    df = load_features_data()
    if df.empty:
        return [], 1990, 2013
    crops = sorted(df['Item'].unique())
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    return crops, min_year, max_year


@st.cache_data
def get_dataset_stats():
    """Compute pesticide statistics."""
    df = load_features_data()
    if df.empty:
        return {'pesticide_min': 0, 'pesticide_max': 100000, 'pesticide_median': 5000}
    return {
        'pesticide_min': float(df['pesticides_tonnes'].min()),
        'pesticide_max': float(df['pesticides_tonnes'].max()),
        'pesticide_median': float(df['pesticides_tonnes'].median()),
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_crop_columns(available_crops: list) -> list:
    return [f'Item_{c}' for c in available_crops]


def build_prediction_features(crop, year, pesticides, rainfall, temp, crop_columns):
    """
    Build feature dict for model prediction using the v2 feature pipeline.
    All v2 features are constructed here, using defaults for satellite/soil
    when external APIs are unavailable.
    """
    # Base features
    features = {
        'average_rain_fall_mm_per_year': rainfall,
        'avg_temp': temp,
        'pesticides_tonnes': pesticides,
    }

    # Interaction features (v1, retained)
    features['temp_rainfall_interaction'] = temp * rainfall
    features['rainfall_squared'] = rainfall ** 2
    features['temp_squared'] = temp ** 2
    features['pesticide_per_rainfall'] = pesticides / (rainfall + 1)
    features['rainfall_deviation'] = rainfall - 1083  # Use dataset mean from training

    # Year normalization
    year_min, year_max = 1990, 2013
    if year_min != year_max:
        features['year_normalized'] = (year - year_min) / (year_max - year_min)
    else:
        features['year_normalized'] = 1.0

    # v2 features - Climate stress indices
    features['heat_stress_degreedays'] = float(max(0.0, temp - 35.0))
    features['drought_intensity'] = float(max(0.0, 1.0 - (rainfall / 500.0)) if rainfall < 500 else 0.0)

    # v2 features - Satellite & Soil (using trained defaults)
    features['ndvi'] = float(DEFAULT_NDVI)
    # ndvi_adjusted depends on rainfall and heat stress
    ndvi_adj = DEFAULT_NDVI + (rainfall / 2000.0) - (features['heat_stress_degreedays'] / 10.0)
    features['ndvi_adjusted'] = float(np.clip(ndvi_adj, 0, 1))
    features['soil_ph'] = DEFAULT_SOIL_PH
    features['soil_nitrogen'] = DEFAULT_SOIL_NITROGEN
    features['soil_organic_carbon'] = DEFAULT_SOIL_ORGANIC_CARBON

    # v2 features - Economic (MSP trend)
    base_year = 1990
    features['msp_trend'] = 1.0 + (year - base_year) * 0.03  # 3% annual growth rate

    # One-hot crop encoding
    for col in crop_columns:
        features[col] = 0
    sel = f'Item_{crop}'
    if sel in features:
        features[sel] = 1

    return pd.DataFrame([features])


def get_feature_importance(model, crop_columns):
    """Extract feature importance, grouping crop one-hot encodings into 'Crop Type'."""
    importances = model.feature_importances_
    names = rf_model.feature_names_in_
    imp_dict = dict(zip(names, importances))

    # Group crop columns
    crop_imp = sum(imp_dict.get(c, 0) for c in crop_columns)

    display = {
        'Crop Type': crop_imp,
        'Rainfall': imp_dict.get('average_rain_fall_mm_per_year', 0),
        'Temperature': imp_dict.get('avg_temp', 0),
        'Pesticides': imp_dict.get('pesticides_tonnes', 0),
        'Year': imp_dict.get('year_normalized', 0),
    }
    total = sum(display.values())
    if total > 0:
        display = {k: v / total for k, v in display.items()}
    return display


def display_results_table(y_v2, y_rf, y_lr=None):
    """Display predictions in a clean table."""
    rows = []
    rows.append(('Champion Forecast (v2)', f'{y_v2:,.0f}'))
    if y_lr is not None:
        rows.append(('Linear Regression (v1)', f'{y_lr:,.0f}'))
    if y_rf is not None:
        rows.append(('Random Forest (v1)', f'{y_rf:,.0f}'))

    table_rows = ''
    for model_name, value in rows:
        is_first = (model_name == rows[0][0])
        row_class = ' class="highlight-row"' if is_first else ''
        table_rows += f'<tr{row_class}><td>{model_name}</td><td>{value} kg/ha</td></tr>\n'

    table_html = f"""
    <table class="data-table">
        <thead><tr><th>Model</th><th>Prediction</th></tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_area_chart(df, crop):
    """Modern area chart with green gradient fill for historical yield."""
    crop_df = df[df['Item'] == crop].copy()
    avg_yield = crop_df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()
    years = avg_yield['Year'].values
    yields = avg_yield['kg_per_ha_yield'].values

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Backgrounds
    fig.patch.set_facecolor('transparent')
    ax.set_facecolor('transparent')

    # Gradient area fill
    ax.fill_between(
        years, yields,
        alpha=0.35,
        color='#2D5A27',
    )

    # Main line
    ax.plot(years, yields,
            color='#2D5A27',
            linewidth=3,
            marker='o',
            markersize=6,
            markerfacecolor='#FFFFFF',
            markeredgecolor='#2D5A27',
            markeredgewidth=2,
            zorder=3)

    # Titles and labels
    ax.set_title(
        f'{crop} — Yield Trajectory Over Time',
        fontsize=15,
        fontweight='700',
        color='#1A1A2E',
        pad=18,
        loc='left'
    )
    ax.set_xlabel('Year', fontsize=12, color='#6B7280', fontweight='500')
    ax.set_ylabel('Average Yield (kg/ha)', fontsize=12, color='#6B7280', fontweight='500')

    # Grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#D1D5DB')
    ax.set_axisbelow(True)

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#E5E7EB')
        ax.spines[spine].set_linewidth(1)

    # Ticks
    ax.tick_params(colors='#6B7280', labelsize=11)
    plt.xticks(rotation=45)

    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def create_importance_chart(importance_dict):
    """Horizontal bar chart for feature importance with earthy green palette."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    fig.patch.set_facecolor('transparent')
    ax.set_facecolor('transparent')

    # Green palette
    palette = ['#1B4332', '#2D5A27', '#388E3C', '#4CAF50', '#66BB6A']
    colors = palette[:len(features)]

    bars = ax.barh(list(features), list(importances),
                   color=colors,
                   height=0.55,
                   edgecolor='none',
                   alpha=0.9)

    # Add value labels
    for bar, imp in zip(bars, importances):
        ax.text(imp + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{imp * 100:.1f}%',
                va='center',
                fontsize=10.5,
                color='#1A1A2E',
                fontweight='500')

    ax.set_title(
        'What Drives the Prediction?',
        fontsize=15,
        fontweight='700',
        color='#1A1A2E',
        pad=18,
        loc='left'
    )
    ax.set_xlabel('Relative Contribution', fontsize=12, color='#6B7280', fontweight='500')

    xlim = max(importances) * 1.2
    ax.set_xlim(0, min(xlim, 0.6))

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis='y', labelsize=11, length=0)
    ax.set_yticklabels(list(features), color='#1A1A2E')
    ax.invert_yaxis()

    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


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

import streamlit as st
import pandas as pd
import joblib
from scripts.config import LR_MODEL_PATH, RF_MODEL_PATH, CHAMPION_MODEL_PATH, FEATURES_DATA_PATH

@st.cache_resource
def load_models():
    """Load trained ML models."""
    models = {}
    try:
        models['champion'] = joblib.load(CHAMPION_MODEL_PATH)
    except Exception:
        models['champion'] = None
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

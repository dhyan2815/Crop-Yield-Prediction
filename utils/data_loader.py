import streamlit as st
import pandas as pd
import joblib
import json
import os
from scripts.config import MODEL_PATH, CONTRACT_PATH, FEATURES_DATA_PATH

@st.cache_resource
def load_model_and_contract():
    """Load the trained model and the feature contract JSON."""
    try:
        model = joblib.load(MODEL_PATH)
        with open(CONTRACT_PATH, 'r') as f:
            contract = json.load(f)
        return model, contract
    except Exception as e:
        st.error(f"Failed to load AI model or contract: {e}")
        return None, None

@st.cache_data
def load_reference_data():
    """Load the processed features data to extract UI options (States, Crops, Seasons)."""
    try:
        if not os.path.exists(FEATURES_DATA_PATH):
            return pd.DataFrame()
        df = pd.read_csv(FEATURES_DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Failed to load reference data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_ui_options():
    """Extract states, crops, and seasons from the data for the UI dropdowns."""
    df = load_reference_data()
    if df.empty:
        return [], [], []
    
    # We find columns starting with state_, crop_, season_
    states = sorted([c.replace('state_', '') for c in df.columns if c.startswith('state_')])
    crops = sorted([c.replace('crop_', '') for c in df.columns if c.startswith('crop_')])
    seasons = sorted([c.replace('season_', '') for c in df.columns if c.startswith('season_')])
    
    return states, crops, seasons

@st.cache_data
def get_crop_averages():
    """Calculate the national average yield for every crop type."""
    try:
        cleaned_path = os.path.join("data", "processed", "cleaned.csv")
        if not os.path.exists(cleaned_path):
            return {}
        df = pd.read_csv(cleaned_path)
        return df.groupby('crop')['yield_kg_ha'].mean().to_dict()
    except Exception as e:
        st.error(f"Failed to calculate crop averages: {e}")
        return {}

"""
Data and Model Loading Utilities for Streamlit Application

Centralizes cached data loading, reference dataset parsing, and feature contract retrieval.
Uses Streamlit caching decorators to optimize performance and prevent repeated file IO.
"""

import json
import os
from typing import Any
import joblib
import pandas as pd
import streamlit as st

from scripts.config import (
    CLEANED_DATA_PATH,
    CONTRACT_PATH,
    FEATURES_DATA_PATH,
    MODEL_PATH,
)


@st.cache_resource
def load_model_and_contract() -> tuple[Any, dict[str, Any] | None]:
    """Load the trained machine learning model and feature contract.

    Uses @st.cache_resource to cache heavy ML model objects in memory across user sessions.
    """
    try:
        model = joblib.load(MODEL_PATH)
        with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
            contract = json.load(f)

        # Ensure backward compatibility with legacy list-based contracts
        if isinstance(contract, list):
            contract = {"features": contract, "target_transform": None}

        return model, contract
    except Exception as e:
        st.error(f"Failed to load model or contract artifact: {e}")
        return None, None


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Load feature matrix used to extract valid UI options.

    Uses @st.cache_data to cache dataframes efficiently across reruns.
    """
    try:
        if not os.path.exists(FEATURES_DATA_PATH):
            return pd.DataFrame()
        return pd.read_csv(FEATURES_DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load reference data matrix: {e}")
        return pd.DataFrame()


def _extract_category_levels(columns: list[str], prefix: str) -> list[str]:
    """Helper function to extract sorted category names from one-hot column headers."""
    prefix_str = f"{prefix}_"
    return sorted([col.replace(prefix_str, "") for col in columns if col.startswith(prefix_str)])


@st.cache_data
def get_ui_options() -> tuple[list[str], list[str], list[str]]:
    """Extract distinct States, Crops, and Seasons available in the feature dataset."""
    df = load_reference_data()
    if df.empty:
        return [], [], []

    cols = df.columns.tolist()
    states = _extract_category_levels(cols, "state")
    crops = _extract_category_levels(cols, "crop")
    seasons = _extract_category_levels(cols, "season")

    return states, crops, seasons


@st.cache_data
def get_crop_averages() -> dict[str, float]:
    """Compute historical national average yield (kg/ha) for each crop."""
    try:
        if not os.path.exists(CLEANED_DATA_PATH):
            return {}
        df = pd.read_csv(CLEANED_DATA_PATH)
        return df.groupby("crop")["yield_kg_ha"].mean().to_dict()
    except Exception as e:
        st.error(f"Failed to calculate historical crop benchmarks: {e}")
        return {}

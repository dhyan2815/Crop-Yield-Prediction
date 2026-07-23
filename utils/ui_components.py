"""
UI Components & Styling Module for Streamlit Application

Provides modular UI layout components:
1. CSS custom style injection matching Streamlit's native theme.
2. Main header display with application branding.
3. Page footer with data provenance and technology stack summary.
4. Transparent dataset sources list.
"""

import streamlit as st
from scripts.config import APP_DATA_SOURCES


def apply_custom_css() -> None:
    """Inject minimal custom CSS to style prediction cards and metrics cleanly."""
    st.markdown(
        """
        <style>
        .stApp {
            padding: 0;
        }

        /* Card styling using Streamlit native theme variables */
        div[data-testid="stMetric"] {
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(128, 128, 128, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }

        .prediction-card {
            background-color: rgba(128, 128, 128, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
        }

        .app-footer {
            text-align: center;
            opacity: 0.7;
            font-size: 0.875rem;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(128, 128, 128, 0.1);
        }

        h1, h2, h3 {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_header() -> None:
    """Display the application title and introductory overview banner."""
    st.title("🌾 Yield Metrics")
    st.caption("Crop Yield Forecasting for Indian States using Machine Learning.")
    st.info("This system uses a ground-up rebuild with a strict feature contract to ensure zero-mismatch predictions.")


def display_footer() -> None:
    """Display standard page footer with framework and dataset details."""
    st.divider()
    st.markdown(
        """
        <div class="app-footer">
            <p>Built with Streamlit & Random Forest</p>
            <p style="font-size: 0.75rem; margin-top: 0.25rem;">Data: India Agricultural Statistics (1997–2020)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_data_sources() -> None:
    """Render data sourcing links for dataset transparency."""
    st.divider()
    st.header("📊 Data Sourcing")
    st.markdown("This project is powered by open agricultural datasets:")
    for name, url in APP_DATA_SOURCES.items():
        st.markdown(f"- [{name}]({url})")

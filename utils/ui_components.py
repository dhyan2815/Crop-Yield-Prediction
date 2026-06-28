import streamlit as st
from scripts.config import APP_DATA_SOURCES

def apply_custom_css():
    """Apply a truly minimal CSS that lets Streamlit's native theme handle colors."""
    # Inject a small styling layer so metrics and cards stay readable without replacing the theme.
    st.markdown("""
    <style>
    /* Let Streamlit handle the main app background and base text colors */
    
    .stApp {
        padding: 0;
    }

    /* Professional Card styling using native Streamlit theme variables */
    div[data-testid="stMetric"] {
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Custom classes for our HTML cards in visualizations.py */
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
    
    /* Ensure markdown headers use the brand primary color */
    h1, h2, h3 {
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header."""
    # Lead with the project title and a one-line explanation of what the app does.
    st.title("🌾 Yield Metrics")
    st.caption("Crop Yield Forecasting for Indian States using Machine Learning.")
    st.info("This system uses a ground-up rebuild with a strict feature contract to ensure zero-mismatch predictions.")

def display_footer():
    """Display the application footer."""
    # End the page with a concise provenance block.
    st.divider()
    st.markdown("""
    <div class="app-footer">
        <p>Built with Streamlit & Random Forest</p>
        <p style="font-size: 0.75rem; margin-top: 0.25rem;">Data: India Agricultural Statistics (1997–2020)</p>
    </div>
    """, unsafe_allow_html=True)

def display_data_sources():
    """Display transparent data sourcing."""
    # Show the data sources in one place so users can trace the dataset provenance.
    st.divider()
    st.header("📊 Data Sourcing")
    st.markdown("This project is powered by open agricultural datasets:")
    for name, url in APP_DATA_SOURCES.items():
        st.markdown(f"- [{name}]({url})")

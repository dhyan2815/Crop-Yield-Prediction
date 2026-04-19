import streamlit as st
from scripts.config import APP_DATA_SOURCES

def apply_custom_css():
    """Apply modern minimal CSS for both Light and Dark modes."""
    st.markdown("""
    <style>
    :root {
        --bg-primary: #FAFAFA;
        --bg-secondary: #f8faf7;
        --text-primary: #1A1A2E;
        --text-secondary: #4B5563;
        --card-bg: #ffffff;
        --card-border: #E5E7EB;
        --accent-green: #2D5A27;
        --accent-green-light: #E8F5E9;
        --accent-blue: #1976D2;
        --accent-blue-light: #E3F2FD;
        --accent-orange: #F59E0B;
        --accent-orange-light: #FFF8E1;
        --button-primary: #2D5A27;
        --button-primary-hover: #1B4332;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1a2332;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --card-bg: #1e293b;
            --card-border: #334155;
            --accent-green: #4ade80;
            --accent-green-light: #14532d;
            --accent-blue: #60a5fa;
            --accent-blue-light: #1e3a5f;
            --accent-orange: #fbbf24;
            --accent-orange-light: #451a03;
            --button-primary: #4ade80;
            --button-primary-hover: #22c55e;
        }
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        color: var(--text-primary);
    }

    /* Fix for Sidebar Background and Text */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--card-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* Fix for Header Background */
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* Input Styling Fixes */
    .stSelectbox div[data-baseweb="select"] {
        background-color: var(--card-bg) !important;
        border-radius: 8px;
        border: 1px solid var(--card-border);
    }
    
    .stSelectbox div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }

    .stSlider div[data-baseweb="slider"] {
        background-color: transparent !important;
    }

    /* Target the dropdown menu itself */
    div[data-baseweb="menu"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
    }
    
    div[data-baseweb="menu"] li {
        color: var(--text-primary) !important;
    }

    h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; font-weight: 600 !important; }
    p, span, div, label { color: var(--text-primary) !important; }

    div[data-testid="stMetric"] {
        background-color: var(--card-bg) !important;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--card-border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .app-footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--card-border);
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header."""
    st.title("🌾 Yield Metrics")
    st.caption("Next-Gen Crop Yield Forecasting for Indian States using Machine Learning.")
    st.info("This system uses a ground-up rebuild with a strict feature contract to ensure zero-mismatch predictions.")

def display_footer():
    """Display the application footer."""
    st.divider()
    st.markdown("""
    <div class="app-footer">
        <p>Built with Streamlit & Random Forest</p>
        <p style="font-size: 0.75rem; margin-top: 0.25rem; opacity: 0.7;">Data: India Agricultural Statistics (1997–2020)</p>
    </div>
    """, unsafe_allow_html=True)

def display_data_sources():
    """Display transparent data sourcing."""
    st.divider()
    st.header("📊 Data Sourcing")
    st.markdown("This project is powered by open agricultural datasets:")
    for name, url in APP_DATA_SOURCES.items():
        st.markdown(f"- [{name}]({url})")

import streamlit as st
from scripts.config import CROP_DATA_SOURCES

def apply_custom_css():
    """Apply modern minimal CSS for both Light and Dark modes."""
    st.markdown("""
    <style>
    /* ================================================
       THEME SYSTEM - Auto Light/Dark Mode Detection
       ================================================ */
    :root {
        /* Light theme (default) */
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
        --table-header-bg: #E8F5E9;
        --table-row-even: #FAFAFA;
        --table-row-hover: #F0F7EF;
        --link-color: #2D5A27;
        --link-hover: #1B4332;
        --button-primary: #2D5A27;
        --button-primary-hover: #1B4332;
        --button-secondary: #1976D2;
        --error-bg: #450a0a;
        --error-border: #dc2626;
        --error-text: #fca5a5;
    }

    /* Dark theme overrides - auto detected from system/Streamlight config */
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
            --table-header-bg: #14532d;
            --table-row-even: #0f172a;
            --table-row-hover: #1a2332;
            --link-color: #4ade80;
            --link-hover: #22c55e;
            --button-primary: #4ade80;
            --button-primary-hover: #22c55e;
            --button-secondary: #60a5fa;
            --error-bg: #450a0a;
            --error-border: #dc2626;
            --error-text: #fca5a5;
        }
    }

    /* Apply global background */
    .stApp {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        color: var(--text-primary);
    }

    /* Headers - always high contrast */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Text content */
    p, span, div, label {
        color: var(--text-primary) !important;
    }

    /* Streamlit alert components */
    .stSuccess {
        background-color: var(--accent-green-light) !important;
        border-left: 4px solid var(--accent-green) !important;
        color: var(--text-primary) !important;
        border-radius: 0;
    }

    .stInfo {
        background-color: var(--accent-blue-light) !important;
        border-left: 4px solid var(--accent-blue) !important;
        color: var(--text-primary) !important;
    }

    .stWarning {
        background-color: var(--accent-orange-light) !important;
        border-left: 4px solid var(--accent-orange) !important;
        color: var(--text-primary) !important;
    }

    .stError {
        background-color: var(--error-bg) !important;
        border-left: 4px solid var(--error-border) !important;
        color: var(--error-text) !important;
    }

    /* Metrics - Modern Cards */
    div[data-testid="stMetric"] {
        background-color: var(--card-bg) !important;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid var(--card-border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        color: var(--text-primary) !important;
    }

    /* Metric label and value colors */
    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }

    /* Tables */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.95rem;
        border-radius: 8px;
        overflow: hidden;
        color: var(--text-primary);
    }
    .data-table th {
        background-color: var(--table-header-bg);
        color: var(--accent-green) !important;
        font-weight: 600;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 2px solid var(--accent-green);
    }
    .data-table td {
        padding: 12px 16px;
        border-bottom: 1px solid var(--card-border);
        color: var(--text-primary);
    }
    .data-table tr:nth-child(even) {
        background-color: var(--table-row-even);
    }
    .data-table tr:hover {
        background-color: var(--table-row-hover);
    }

    /* Highlight row */
    .highlight-row {
        background-color: var(--accent-green-light) !important;
    }
    .highlight-row td {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Data Source Links */
    a[href] {
        color: var(--link-color) !important;
        text-decoration: none;
        font-weight: 500;
    }
    a[href]:hover {
        text-decoration: underline;
        color: var(--link-hover) !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--card-border);
    }

    /* Section dividers */
    hr {
        border: none;
        border-top: 1px solid var(--card-border);
        margin: 2rem 0;
    }

    /* Inputs */
    input, textarea, select, div[role="combobox"] input {
        color: var(--text-primary) !important;
        background-color: var(--card-bg) !important;
        border-color: var(--card-border) !important;
    }

    /* Buttons */
    button {
        background-color: var(--button-primary) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button:hover {
        background-color: var(--button-primary-hover) !important;
    }

    /* Streamlit button specific */
    .stButton button {
        background-color: var(--button-primary) !important;
        color: #ffffff !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton button:hover {
        background-color: var(--button-primary-hover) !important;
    }

    /* Secondary button variant */
    .stButton button[kind="secondary"] {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
    }

    /* Download button */
    .stDownloadButton button {
        background-color: var(--button-secondary) !important;
        color: #ffffff !important;
    }
    .stDownloadButton button:hover {
        background-color: var(--accent-green) !important;
    }

    /* Selectbox / Multiselect */
    [data-baseweb="select"] {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border-color: var(--card-border) !important;
    }

    /* Number input */
    [data-baseweb="input"] input {
        color: var(--text-primary) !important;
        background-color: var(--card-bg) !important;
        caret-color: var(--accent-green) !important;
    }

    /* Slider */
    [data-baseweb="slider"] {
        background-color: var(--card-border) !important;
    }

    /* Checkbox */
    [data-baseweb="checkbox"] {
        color: var(--text-primary) !important;
    }
    [data-baseweb="checkbox"] input:checked + div {
        background-color: var(--accent-green) !important;
    }

    /* Expander */
    details {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
    }
    details summary {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the application header and disclaimer."""
    st.title("Yield Metrics")
    st.caption("Crop yield prediction for Indian agriculture, based on historical weather patterns and agronomic inputs.")
    st.warning("Predictions are based on historical data analysis (1990–2013). Results should not be used for critical agricultural decisions without further validation.")

def display_footer():
    """Display the application footer."""
    st.divider()
    st.markdown("""
    <div class="app-footer">
        <p>Yield Metrics — Built with Streamlit</p>
        <p style="font-size: 0.75rem; margin-top: 0.25rem; opacity: 0.7;">Data: India Crop Yield (1990–2013) | Models: v2 (Champion) & v1 (Legacy)</p>
    </div>
    """, unsafe_allow_html=True)

def display_data_sources(crop):
    """Display transparent data sourcing for the selected crop."""
    st.divider()
    st.header("📊 Data Sources & References")
    st.markdown(f"Transparent sourcing for **{crop}** prediction:")

    sources = CROP_DATA_SOURCES.get(crop, CROP_DATA_SOURCES["default"])
    for source_name, url in sources.items():
        st.markdown(f"- [{source_name}]({url})")

"""
Data Visualization Utilities for Streamlit Application

Provides Plotly interactive chart builders and HTML prediction card renderer:
1. Custom HTML card rendering with color-coded risk status accents.
2. Historical yield trend area charts (Plotly Express).
3. Baseline vs Simulated yield comparison bar charts (Plotly Graph Objects).
"""

from typing import Any
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def display_prediction_card(yield_val: float, status: str, message: str) -> None:
    """Render a dynamic HTML card displaying predicted yield and color-coded risk condition."""
    # Map status labels to intuitive accent colors
    color_map = {
        "Optimal": "#10b981",       # Green
        "Healthy": "#2D5A27",       # Deep Green
        "Stable": "#3b82f6",        # Blue
        "Moderate": "#F59E0B",      # Amber
        "Critical Low": "#dc2626",  # Red
        "Unknown": "#6b7280",       # Gray
    }
    status_color = color_map.get(status, "#6b7280")

    st.markdown(
        f"""
        <div class="prediction-card" style="border-left: 8px solid {status_color};">
            <h3 style="margin: 0; opacity: 0.8; font-size: 0.9rem; text-transform: uppercase;">Predicted Yield</h3>
            <div style="display: flex; align-items: baseline; gap: 10px; margin: 10px 0;">
                <span style="font-size: 3.5rem; font-weight: 800;">{yield_val:,.0f}</span>
                <span style="font-size: 1.2rem; opacity: 0.8;">kg/ha</span>
            </div>
            <div style="background-color: {status_color}20; color: {status_color}; padding: 6px 14px; border-radius: 20px; display: inline-block; font-weight: 600; font-size: 0.85rem;">
                ● {status} Condition
            </div>
            <p style="margin-top: 15px; opacity: 0.9; font-size: 1rem; line-height: 1.5;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_historical_chart(df: pd.DataFrame, state: str, crop: str) -> Any | None:
    """Create a Plotly area chart representing historical yield trajectory for a state and crop."""
    if not HAS_PLOTLY:
        st.warning("Plotly is required for historical charts. Please install using: pip install plotly")
        return None

    mask = (df["state"] == state) & (df["crop"] == crop)
    filtered_df = df[mask].sort_values("crop_year")

    if filtered_df.empty:
        return None

    fig = px.area(
        filtered_df,
        x="crop_year",
        y="yield_kg_ha",
        title=f"Yield Trend: {crop} in {state}",
        labels={"crop_year": "Year", "yield_kg_ha": "Yield (kg/ha)"},
        color_discrete_sequence=["#2D5A27"],
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False},
        yaxis={"gridcolor": "rgba(128,128,128,0.2)"},
        title_font={"size": 18, "family": "Outfit, sans-serif"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    return fig


def create_comparison_chart(yield_current: float, yield_simulated: float) -> Any | None:
    """Create a Plotly bar chart comparing baseline forecast against simulated yield."""
    if not HAS_PLOTLY:
        return None

    fig = go.Figure(
        go.Bar(
            x=["Baseline", "Simulated"],
            y=[yield_current, yield_simulated],
            marker_color=["#1976D2", "#2D5A27"],
            text=[f"{yield_current:,.0f}", f"{yield_simulated:,.0f}"],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Baseline vs Simulation",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )

    return fig

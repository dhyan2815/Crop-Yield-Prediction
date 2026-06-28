import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    # Fall back gracefully when Plotly is not installed so the app can still load.
    HAS_PLOTLY = False


def display_prediction_card(yield_val, status, message):
    """Display a professional card for the prediction result using native theme colors."""
    # Map status labels to a stable accent color so the card can communicate risk at a glance.
    color_map = {
        "Optimal": "#10b981", # Green
        "Healthy": "#2D5A27", # Deep Green
        "Stable": "#3b82f6",  # Blue
        "Moderate": "#F59E0B", # Amber
        "Critical Low": "#dc2626", # Red
        "Unknown": "#6b7280"   # Gray
    }
    status_color = color_map.get(status, "#6b7280")
    
    # Render the prediction as a compact HTML card for stronger visual hierarchy.
    st.markdown(f"""
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
    """, unsafe_allow_html=True)


def create_historical_chart(df, state, crop):
    """Plot historical yield trend for the selected state and crop."""
    if not HAS_PLOTLY:
        st.warning("Plotly is required for historical charts. Please install it using: pip install plotly")
        return None
        
    # Filter to the selected state/crop pair and sort chronologically before plotting.
    mask = (df['state'] == state) & (df['crop'] == crop)
    filtered_df = df[mask].sort_values('crop_year')
    
    if filtered_df.empty:
        return None
        
    fig = px.area(
        filtered_df, 
        x='crop_year', 
        y='yield_kg_ha',
        title=f"Yield Trend: {crop} in {state}",
        labels={'crop_year': 'Year', 'yield_kg_ha': 'Yield (kg/ha)'},
        color_discrete_sequence=['#2D5A27']
    )
    
    # Keep the chart theme-neutral so it works in both light and dark layouts.
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        title_font=dict(size=18, family="Outfit, sans-serif"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_comparison_chart(yield_current, yield_simulated):
    """Compare baseline vs simulated scenario yield."""
    if not HAS_PLOTLY:
        return None
        
    # Show the baseline and simulated cases side by side for quick scenario comparison.
    fig = go.Figure(go.Bar(
        x=['Baseline', 'Simulated'],
        y=[yield_current, yield_simulated],
        marker_color=['#1976D2', '#2D5A27'],
        text=[f"{yield_current:,.0f}", f"{yield_simulated:,.0f}"],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Baseline vs Simulation",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

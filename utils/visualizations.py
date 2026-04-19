import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def display_prediction_card(yield_val, status, message):
    """Display a professional card for the prediction result."""
    color = "#2D5A27" if status == "Healthy" else "#F59E0B" if status == "Moderate" else "#dc2626"
    
    st.markdown(f"""
    <div style="background-color: #ffffff; border-radius: 12px; padding: 25px; border-left: 8px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;">
        <h3 style="margin: 0; color: #6B7280; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Predicted Yield</h3>
        <div style="display: flex; align-items: baseline; gap: 10px; margin: 10px 0;">
            <span style="font-size: 3rem; font-weight: 800; color: #1A1A2E;">{yield_val:,.0f}</span>
            <span style="font-size: 1.2rem; color: #4B5563;">kg/ha</span>
        </div>
        <div style="background-color: {color}20; color: {color}; padding: 6px 12px; border-radius: 20px; display: inline-block; font-weight: 600; font-size: 0.85rem;">
            ● {status} Condition
        </div>
        <p style="margin-top: 15px; color: #4B5563; font-size: 0.95rem; line-height: 1.5;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def create_historical_chart(df, state, crop):
    """Plot historical yield trend for the selected state and crop."""
    if not HAS_PLOTLY:
        st.warning("Plotly is required for historical charts. Please install it using: pip install plotly")
        return None
        
    # Note: df here should be the 'cleaned.csv' or equivalent with readable names
    # Filter by state and crop
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
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#E5E7EB'),
        title_font=dict(size=18, family="Outfit, sans-serif"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_comparison_chart(yield_current, yield_simulated):
    """Compare baseline vs simulated scenario yield."""
    if not HAS_PLOTLY:
        return None
        
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

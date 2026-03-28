import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Set page configuration
st.set_page_config(
    page_title="Yield Metrics",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODEL LOADING
# =============================================================================

import time

@st.cache_resource
def load_models():
    """
    Load the trained machine learning models.
    Returns: tuple of (linear_regression_model, random_forest_model)
    """
    try:
        lr_model = joblib.load('models/linear_regression_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        msg = st.empty()
        msg.success("✅ Models loaded successfully!")
        time.sleep(4)
        msg.empty()
        return lr_model, rf_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

# Load models
lr_model, rf_model = load_models()

# HELPER FUNCTIONS

def get_temperature(lat, lon, api_key):
    """
    Fetch current temperature for given coordinates using OpenWeatherMap API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        api_key (str): OpenWeatherMap API key
    
    Returns:
        float: Temperature in Celsius, or None if failed
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        data = response.json()
        temperature = data['main']['temp']
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            st.write(f"🌡️ API Response: {data}")
            st.write(f"🌡️ Extracted Temperature: {temperature}°C")
        
        return temperature
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error fetching temperature: {e}")
        return None
    except KeyError as e:
        st.error(f"❌ Invalid API response format: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error fetching temperature: {e}")
        return None


@st.cache_data
def get_rainfall_data():
    """
    Load and cache rainfall data from the processed CSV file.
    
    Returns:
        pandas.DataFrame: Grouped rainfall data by Item and Year
    """
    try:
        # Load the cleaned CSV file
        df = pd.read_csv("data/processed/CLEANED_Processed_India_Crop_Yield_Data.csv")
        
        # Clean column names by stripping whitespace
        df.columns = df.columns.str.strip()
        
        # Group by Item and Year to get average rainfall
        rainfall_df = df.groupby(['Item', 'Year'])['average_rain_fall_mm_per_year'].mean().reset_index()
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            st.write(f"📊 Rainfall data shape: {rainfall_df.shape}")
            st.write(f"📊 Available crops: {rainfall_df['Item'].unique()}")
        
        return rainfall_df
        
    except FileNotFoundError:
        st.error("❌ Cleaned CSV file not found. Please run the data cleaning notebook first.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading rainfall data: {e}")
        return pd.DataFrame()


@st.cache_data
def get_features_by_crop_year():
    """
    Load and cache crop-year level features (rainfall and temperature) from the processed CSV.
    Returns:
        pandas.DataFrame: columns ['Item', 'Year', 'average_rain_fall_mm_per_year', 'avg_temp']
    """
    try:
        df = pd.read_csv("data/processed/CLEANED_Processed_India_Crop_Yield_Data.csv")
        df.columns = df.columns.str.strip()
        features_df = (
            df.groupby(['Item', 'Year'])
              [['average_rain_fall_mm_per_year', 'avg_temp']]
              .mean()
              .reset_index()
        )
        return features_df
    except FileNotFoundError:
        st.error("❌ Cleaned CSV file not found. Please run the data cleaning notebook first.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading features: {e}")
        return pd.DataFrame()


@st.cache_data
def get_dataset_stats():
    """
    Compute basic dataset statistics used for sensible UI defaults and bounds.
    Returns:
        dict: {'pesticide_min': float, 'pesticide_max': float, 'pesticide_median': float}
    """
    try:
        df = pd.read_csv("data/processed/CLEANED_Processed_India_Crop_Yield_Data.csv")
        df.columns = df.columns.str.strip()
        stats = {
            'pesticide_min': float(df['pesticides_tonnes'].min()),
            'pesticide_max': float(df['pesticides_tonnes'].max()),
            'pesticide_median': float(df['pesticides_tonnes'].median()),
        }
        return stats
    except Exception as e:
        st.warning(f"⚠️ Could not compute dataset stats: {e}")
        return {}

def get_average_rainfall(crop, year, rainfall_df):
    """
    Get average rainfall for a specific crop and year.
    
    Args:
        crop (str): Crop name
        year (int): Year
        rainfall_df (pandas.DataFrame): Rainfall data
    
    Returns:
        float: Average rainfall in mm, or None if not found
    """
    try:
        # Search for matching crop and year
        match = rainfall_df[(rainfall_df['Item'] == crop) & (rainfall_df['Year'] == int(year))]
        
        # Debug information
        if st.session_state.get('debug_mode', False):
            st.write(f"🔍 Searching for Crop: '{crop}' and Year: {year}")
            st.write(f"🔍 Found {len(match)} matching records")
            if not match.empty:
                st.write(f"🔍 Rainfall value: {match['average_rain_fall_mm_per_year'].values[0]} mm")
        
        # Return rainfall value if found, otherwise None
        if not match.empty:
            return match['average_rain_fall_mm_per_year'].values[0]
        else:
            st.warning(f"⚠️ No rainfall data found for {crop} in {year}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error getting rainfall data: {e}")
        return None


def create_prediction_plot(y_lr, y_rf, crop, year):
    """
    Create a bar plot comparing predictions from both models.
    
    Args:
        y_lr (float): Linear Regression prediction
        y_rf (float): Random Forest prediction
        crop (str): Crop name
        year (int): Year
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        models = ['Linear Regression', 'Random Forest']
        predictions = [y_lr, y_rf]
        colors = ['skyblue', 'lightgreen']
        
        bars = ax.bar(models, predictions, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, pred in zip(bars, predictions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{pred:.0f} kg/ha', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        ax.set_ylabel('Predicted Yield (kg/ha)', fontsize=12)
        ax.set_title(f'{crop} Yield Prediction for {year}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating prediction plot: {e}")
        return None


def create_trend_plot():
    """
    Create a line plot showing historical yield trends.
    
    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    try:
        # Load data
        df = pd.read_csv("data/processed/CLEANED_Processed_India_Crop_Yield_Data.csv")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Calculate average yield by year
        avg_yield_by_year = df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot line
        sns.lineplot(x='Year', y='kg_per_ha_yield', data=avg_yield_by_year, 
                    ax=ax, linewidth=2, color='blue')
        
        # Customize plot
        ax.set_title('Average Crop Yield in India Over Years', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Average Yield (kg/ha)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating trend plot: {e}")
        return None


def validate_inputs(crop, year, pesticides, min_year, max_year):
    """
    Validate user inputs before making predictions.
    
    Args:
        crop (str): Selected crop
        year (int): Selected year
        pesticides (float): Pesticide usage
        lat (float): Latitude
        lon (float): Longitude
    
    Returns:
        bool: True if all inputs are valid, False otherwise
    """
    errors = []
    
    # Validate crop
    if not crop or crop.strip() == "":
        errors.append("Please select a crop")
    
    # Validate year
    if year < min_year or year > max_year:
        errors.append(f"Year must be between {min_year} and {max_year}")
    
    # Validate pesticides
    if pesticides < 0:
        errors.append("Pesticide usage cannot be negative")
    
    # Display errors if any
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        return False
    
    return True


# =============================================================================
# MAIN APPLICATION UI
# =============================================================================

def main():
    """
    Main application function that handles the Streamlit UI and logic.
    """
    
    # Header
    st.title("🌾 Yield Metrics")
    st.markdown("""
    Welcome to **Yield Metrics** – a crop yield prediction app for India.
    """)
    
    # Disclaimer
    st.warning("⚠️ **Predictions** are based on historical data and may not reflect current conditions. Use results for guidance only, not for critical decisions.")
    
    # Check if models are loaded
    if lr_model is None or rf_model is None:
        st.error("❌ Failed to load models. Please check if model files exist.")
        return
    
    # Load features once and derive dynamic crop list and year bounds
    features_df = get_features_by_crop_year()
    if features_df.empty:
        st.error("❌ Features could not be loaded from dataset.")
        return
    available_crops = sorted(features_df['Item'].unique())
    min_year = int(features_df['Year'].min())
    max_year = int(features_df['Year'].max())
    dataset_stats = get_dataset_stats()

    # Main content area
    st.header("📝 Input Parameters")
    st.caption(f"Available years in dataset: {min_year} - {max_year}")

    # Crop selection
    crop = st.selectbox("🌱 Select Crop", available_crops, help="Choose the crop for yield prediction")

    # Year selection
    year = st.number_input(
        "📅 Select Year", 
        min_value=min_year, 
        max_value=max_year, 
        value=max_year, 
        step=1,
        help=f"Select a year between {min_year} and {max_year} available in the dataset"
    )

    # Pesticide usage
    default_pest = float(dataset_stats.get('pesticide_median', 5000.0))
    min_pest = float(dataset_stats.get('pesticide_min', 0.0))
    max_pest = float(dataset_stats.get('pesticide_max', default_pest * 10))
    step_pest = float(max(1.0, round((max_pest - min_pest) / 200.0)))
    pesticides = st.number_input(
        "🧪 Pesticide Usage (tonnes)", 
        min_value=min_pest, 
        max_value=max_pest,
        value=default_pest, 
        step=step_pest,
        help=f"Based on dataset: min={min_pest:.0f}, median={default_pest:.0f}, max={max_pest:.0f}"
    ) 
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🚀 Predict Yield", type="primary", use_container_width=True):
        
        # Validate inputs
        if not validate_inputs(crop, year, pesticides, min_year, max_year):
            st.stop()
        
        # Show progress
        with st.spinner("🔄 Fetching weather data and making predictions..."):
            
            # Get dataset features for the selected crop-year
            match = features_df[(features_df['Item'] == crop) & (features_df['Year'] == int(year))]
            if match.empty:
                st.error("❌ No dataset features found for the selected crop and year.")
                st.stop()
            rainfall = float(match['average_rain_fall_mm_per_year'].values[0])
            dataset_temp = float(match['avg_temp'].values[0])

            # Use dataset temperature directly for consistency with training data
            temp = dataset_temp
            st.info("ℹ️ Using dataset average temperature for the selected crop and year.")
            
            # Debug information removed in simplified UI
            
            # Make predictions
            try:
                # Construct feature DataFrame for prediction with explicit column names
                input_features = pd.DataFrame([{
                    'average_rain_fall_mm_per_year': rainfall,
                    'avg_temp': temp,
                    'pesticides_tonnes': pesticides
                }])
                
                # Get predictions from both models
                yield_lr = lr_model.predict(input_features)[0]
                yield_rf = rf_model.predict(input_features)[0]
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                # Results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Linear Regression Prediction",
                        value=f"{yield_lr:.0f} kg/ha",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Random Forest Prediction",
                        value=f"{yield_rf:.0f} kg/ha",
                        delta=None
                    )
                
                # Create and display prediction plot
                prediction_fig = create_prediction_plot(yield_lr, yield_rf, crop, year)
                if prediction_fig:
                    st.pyplot(prediction_fig)
                
                # Display historical trend
                st.markdown("### 📈 Historical Yield Trend")
                trend_fig = create_trend_plot()
                if trend_fig:
                    st.pyplot(trend_fig)
                
            except Exception as e:
                st.error(f"❌ Error making predictions: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🌾 Yield Metrics</p>
        <p style='font-size: 0.9em;'>Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
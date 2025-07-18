import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Load Models
lr_model = joblib.load('/models/linear_regression_model.pkl')
rf_model = joblib.load('/models/random_forest_model.pkl')

# Helper Functions
def get_temperature(api_key):
    api_key = "60d79498a70e5a4e54bc7620b6914ee0"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data['main']['temp']
    except Exception as e:
        st.error(f"Temperature fetch failed: {e}")
        return None

@st.cache_data
def get_rainfall_data():
    df = pd.read_csv("data/processed/Processed_India_Crop_Yield_Data.csv")
    return df.groupby(['Item', 'Year'])['average_rain_fall_mm_per_year'].mean().reset_index()

def get_average_rainfall(crop, year, rainfall_df):
    match = rainfall_df[(rainfall_df['Item'] == crop) & (rainfall_df['Year'] == int(year))]
    return match['average_rain_fall_mm_per_year'].values[0] if not match.empty else None

def create_prediction_plot(y_lr, y_rf, crop, year):
    fig, ax = plt.subplots()
    ax.bar(['Linear Regression', 'Random Forest'], [y_lr, y_rf], color=['skyblue', 'lightgreen'])
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title(f'{crop} Yield Prediction for {year}')
    return fig

def create_trend_plot():
    df = pd.read_csv("data/processed/Processed_India_Crop_Yield_Data.csv")
    avg_yield_by_year = df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x='Year', y='kg_per_ha_yield', data=avg_yield_by_year, ax=ax)
    ax.set_title('Average Crop Yield in India Over Years')
    ax.set_ylabel('Yield (kg/ha)')
    ax.grid(True)
    return fig

# Streamlit UI
st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌾 Crop Yield Prediction App")

# Inputs
crop_list = ['Rice', 'Wheat', 'Maize', 'Barley', 'Pulses']  # Add as needed
crop = st.selectbox("Select Crop", crop_list)
year = st.number_input("Select Year", min_value=2000, max_value=2050, value=2025, step=1)
pesticides = st.number_input("Enter Pesticide Usage (tonnes)", min_value=0.0, step=0.1)

# Location Inputs (Latitude & Longitude)
st.markdown("##### 🌡️ Allow location access or enter manually")
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=20.5937)
with col2:
    lon = st.number_input("Longitude", value=78.9629)

# Predict Button
if st.button("Predict Yield"):
    with st.spinner("Fetching temperature and rainfall..."):
        temp = get_temperature(lat, lon, api_key="60d79498a70e5a4e54bc7620b6914ee0")
        rainfall_df = get_rainfall_data()
        rainfall = get_average_rainfall(crop, year, rainfall_df)

    if temp is None or rainfall is None:
        st.error("Could not fetch necessary weather data.")
    else:
        # Construct feature vector for prediction
        input_features = np.array([[rainfall, temp, pesticides]])  # Based on original training features
        yield_lr = lr_model.predict(input_features)[0]
        yield_rf = rf_model.predict(input_features)[0]

        # Display Predictions
        st.success("✅ Prediction Complete")
        st.write(f"**Predicted Yield (Linear Regression):** {yield_lr:.2f} kg/ha")
        st.write(f"**Predicted Yield (Random Forest):** {yield_rf:.2f} kg/ha")

        # Yield Prediction Plot
        st.pyplot(create_prediction_plot(yield_lr, yield_rf, crop, year))

        # Historical Trend Plot
        st.markdown("### 📈 Historical Yield Trend")
        st.pyplot(create_trend_plot())


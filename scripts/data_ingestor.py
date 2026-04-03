"""
Data Ingestion Module for Project 2026.
Handles fetching latest yield, weather, and economic data from various APIs.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from typing import Optional, Dict, Any

class OpenMeteoClient:
    """Client for Open-Meteo API to fetch historical and forecast weather data."""
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def get_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather for a specific location.
        Dates format: YYYY-MM-DD
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto"
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        df = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time")),
            "avg_temp": daily.get("temperature_2m_mean"),
            "precipitation": daily.get("precipitation_sum")
        })
        return df

class FAOSTATClient:
    """Client for FAOSTAT API to fetch latest crop yield data."""
    
    # FAOSTAT API endpoint for India (Area Code 100)
    # This is a simplified interface for demonstration; 
    # professional use often involves complex DMX queries.
    BASE_URL = "https://fenixservices.fao.org/faostat/api/v1/en/data/QC"

    def get_india_yield(self, item_code: str, year_start: int, year_end: int) -> pd.DataFrame:
        """
        Fetch yield data for India for a specific crop.
        """
        # Parameters for India (100), Yield (5419), and specified Crop and Years
        params = {
            "area": "100", 
            "element": "5419", # Yield
            "item": item_code,
            "year": ",".join(map(str, range(year_start, year_end + 1))),
            "format": "json"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # FAOSTAT response parsing
            records = data.get("data", [])
            if not records:
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            # Standardizing columns to match existing project
            return df[['Year', 'Value', 'Item']].rename(columns={'Value': 'kg_per_ha_yield'})
        except Exception as e:
            print(f"Error fetching from FAOSTAT: {e}")
            return pd.DataFrame()

def main():
    """Example usage for testing data ingestion."""
    print("🚀 Initializing Project 2026 Data Ingestor...")
    
    # 1. Fetching recent weather for a sample location (New Delhi)
    weather_client = OpenMeteoClient()
    print("Fetching weather for New Delhi (2024-2025)...")
    weather_df = weather_client.get_historical_weather(28.61, 77.21, "2024-01-01", "2025-01-01")
    print(f"Weather data fetched: {len(weather_df)} days.")
    
    # 2. Fetching Wheat yield from FAOSTAT
    # item_code '15' is Wheat in FAOSTAT
    faostat = FAOSTATClient()
    print("Fetching Wheat yield for India from FAOSTAT...")
    yield_df = faostat.get_india_yield("15", 2014, 2024)
    
    if not yield_df.empty:
        print(f"Newer yield data found (2014-2024):")
        print(yield_df.tail())
    else:
        print("No newer yield data available via API yet. May require manual UPAg sync.")

if __name__ == "__main__":
    main()

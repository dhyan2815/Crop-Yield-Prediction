"""
Data Ingestion Module for Project 2026.
Handles fetching latest yield, weather, and economic data from various APIs.
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

class UPAgClient:
    """Client for UPAg (Unified Portal for Agricultural Statistics) API."""

    BASE_URL = "https://api.upag.gov.in/v1"

    def fetch_yield(self, year: int, crop_code: Optional[str] = None) -> pd.DataFrame:
        """Fetch crop yield data for a specific year."""
        url = f"{self.BASE_URL}/yield"
        params = {"year": year}
        if crop_code:
            params["crop"] = crop_code

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        return df

    def fetch_and_cache(self, year: int, crop_code: Optional[str] = None,
                       cache_dir: str = "data/raw") -> pd.DataFrame:
        """Fetch yield data and cache to CSV."""
        os.makedirs(cache_dir, exist_ok=True)
        df = self.fetch_yield(year, crop_code)
        suffix = f"{crop_code}_{year}" if crop_code else str(year)
        path = os.path.join(cache_dir, f"upag_{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"[UPAg] Cached {len(df)} records to {path}")
        return df


class DCSClient:
    """Client for Digital Crop Survey (DCS) district-level data."""

    BASE_URL = "https://api.dcs.gov.in/v1"

    def fetch_district_data(self, year: int, state: Optional[str] = None) -> pd.DataFrame:
        """Fetch district-level crop production data."""
        url = f"{self.BASE_URL}/district-crop"
        params = {"year": year}
        if state:
            params["state"] = state

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())

    def fetch_and_cache(self, year: int, state: Optional[str] = None,
                       cache_dir: str = "data/raw") -> pd.DataFrame:
        """Fetch district data and cache to CSV."""
        os.makedirs(cache_dir, exist_ok=True)
        df = self.fetch_district_data(year, state)
        suffix = f"{state}_{year}" if state else str(year)
        path = os.path.join(cache_dir, f"dcs_district_{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"[DCS] Cached {len(df)} records to {path}")
        return df


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

        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        df = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time")),
            "avg_temp": daily.get("temperature_2m_mean"),
            "precipitation": daily.get("precipitation_sum")
        })
        return df

    def get_annual_agg(self, lat: float, lon: float, year: int) -> pd.DataFrame:
        """Fetch aggregated annual weather metrics for a location."""
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        daily = self.get_historical_weather(lat, lon, start, end)
        daily["year"] = pd.to_datetime(daily["date"]).dt.year
        annual = daily.groupby("year").agg(
            avg_temp=("avg_temp", "mean"),
            total_rainfall=("precipitation", "sum")
        ).reset_index()
        return annual

    def fetch_and_cache(self, lat: float, lon: float, year: int,
                       cache_dir: str = "data/raw") -> pd.DataFrame:
        """Fetch annual weather data and cache to CSV."""
        os.makedirs(cache_dir, exist_ok=True)
        annual = self.get_annual_agg(lat, lon, year)
        path = os.path.join(cache_dir, f"openmeteo_{lat}_{lon}_{year}.csv")
        annual.to_csv(path, index=False)
        print(f"[OpenMeteo] Cached annual metrics to {path}")
        return annual

class FAOSTATClient:
    """Client for FAOSTAT API to fetch latest crop yield data."""

    BASE_URL = "https://fenixservices.fao.org/faostat/api/v1/en/data/QC"

    def get_india_yield(self, item_code: str, year_start: int, year_end: int) -> pd.DataFrame:
        """Fetch yield data for India for a specific crop."""
        params = {
            "area": "100",
            "element": "5419",
            "item": item_code,
            "year": ",".join(map(str, range(year_start, year_end + 1))),
            "format": "json"
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            records = data.get("data", [])
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            return df[['Year', 'Value', 'Item']].rename(columns={'Value': 'kg_per_ha_yield'})
        except Exception as e:
            print(f"Error fetching from FAOSTAT: {e}")
            return pd.DataFrame()

    def fetch_and_cache(self, item_code: str, year_start: int, year_end: int,
                       cache_dir: str = "data/raw") -> pd.DataFrame:
        """Fetch FAOSTAT yield data and cache to CSV."""
        os.makedirs(cache_dir, exist_ok=True)
        df = self.get_india_yield(item_code, year_start, year_end)
        if not df.empty:
            path = os.path.join(cache_dir, f"faostat_yield_{item_code}_{year_start}-{year_end}.csv")
            df.to_csv(path, index=False)
            print(f"[FAOSTAT] Cached {len(df)} records to {path}")
        return df


def main():
    """CLI entry point for data ingestion."""
    parser = argparse.ArgumentParser(description="Fetch and cache latest agricultural data")
    parser.add_argument("--year", type=int, default=2025, help="Target year for data")
    parser.add_argument("--crop", type=str, default=None, help="Specific crop code")
    parser.add_argument("--state", type=str, default=None, help="Filter district data by state")
    args = parser.parse_args()

    year = args.year
    cache_dir = "data/raw"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[Ingestor] Fetching data for year={year}, crop={args.crop}")

    # 1. UPAg yield data
    upag = UPAgClient()
    try:
        upag.fetch_and_cache(year, args.crop, cache_dir=cache_dir)
    except Exception as e:
        print(f"[UPAg] Warning: {e}")

    # 2. DCS district data
    dcs = DCSClient()
    try:
        dcs.fetch_and_cache(year, args.state, cache_dir=cache_dir)
    except Exception as e:
        print(f"[DCS] Warning: {e}")

    # 3. Open-Meteo weather (New Delhi sample location)
    weather = OpenMeteoClient()
    try:
        weather.fetch_and_cache(28.61, 77.21, year, cache_dir=cache_dir)
    except Exception as e:
        print(f"[OpenMeteo] Warning: {e}")

    # 4. FAOSTAT as fallback for newer year ranges
    faostat = FAOSTATClient()
    try:
        faostat.fetch_and_cache("15", max(2014, year - 5), year, cache_dir=cache_dir)
    except Exception as e:
        print(f"[FAOSTAT] Warning: {e}")

    print("[Ingestor] Data fetch complete. All cached files in", cache_dir)


if __name__ == "__main__":
    main()

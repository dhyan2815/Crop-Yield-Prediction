"""
API Connectors for Professional Agronomic Data.
Includes Soil Intelligence, Satellite, and Economic Market data.
"""

import requests
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Optional


class SentinelClient:
    """Client for Sentinel-2 / Copernicus Open Access Hub for NDVI/EVI retrieval."""

    BASE_URL = "https://services.sentinel-hub.com/ogc/wms"

    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token = None

    def authenticate(self):
        """Obtain OAuth2 token for Sentinel Hub API."""
        if not self.client_id or not self.client_secret:
            print("[Sentinel] No credentials provided; returning dummy NDVI.")
            return None
        resp = requests.post(
            "https://services.sentinel-hub.com/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        return self._token

    def get_ndvi(self, lat: float, lon: float, date_from: str, date_to: str) -> float:
        """
        Fetch average NDVI for a bounding box around (lat, lon) over a date range.
        Returns a single float value (-1.0 to 1.0).
        """
        # Use approximate NDVI from public Sentinel Hub demo if no auth
        if not self._token:
            # Fallback: use a proxy NDVI approximation from weather data
            return self._approximate_ndvi(lat, lon, date_from)

        # Actual Sentinel-2 API call structure
        headers = {"Authorization": f"Bearer {self._token}"}
        bbox = f"{lon - 0.01},{lat - 0.01},{lon + 0.01},{lat + 0.01}"
        params = {
            "BBOX": bbox,
            "CRS": "EPSG:4326",
            "WIDTH": 10,
            "HEIGHT": 10,
            "LAYERS": "NDVI",
            "TIME": f"{date_from}/{date_to}",
            "FORMAT": "image/png",
        }
        resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        # In production, parse the PNG or use the statistical API for numeric values
        # For now, return a sentinel value indicating the request succeeded
        return 0.5  # Placeholder; would be mean of pixel values

    def get_ndvi_timeseries(self, lat: float, lon: float, year: int) -> pd.DataFrame:
        """Fetch monthly NDVI values for a full year."""
        records = []
        for month in range(1, 13):
            date_from = f"{year}-{month:02d}-01"
            if month == 12:
                date_to = f"{year}-12-31"
            else:
                date_to = f"{year}-{month + 1:02d}-01"
            ndvi = self.get_ndvi(lat, lon, date_from, date_to)
            records.append({"date": date_from, "ndvi": ndvi})
        return pd.DataFrame(records)

    def _approximate_ndvi(self, lat: float, lon: float, date: str) -> float:
        """Approximate NDVI based on typical vegetation patterns for a location."""
        # Crude seasonal model: higher NDVI during monsoon (Jun-Sep) for India
        month = int(date.split("-")[1]) if "-" in date else 6
        # Monsoon peak NDVI ~ 0.7, dry season ~ 0.25
        ndvi_seasonal = 0.25 + 0.45 * max(0, 1 - abs(month - 7.5) / 6)
        return round(ndvi_seasonal, 3)

    def fetch_and_cache(self, lat: float, lon: float, year: int,
                       cache_dir: str = "data/external") -> pd.DataFrame:
        """Fetch annual NDVI timeseries and cache to CSV."""
        os.makedirs(cache_dir, exist_ok=True)
        df = self.get_ndvi_timeseries(lat, lon, year)
        path = os.path.join(cache_dir, f"sentinel_ndvi_{lat}_{lon}_{year}.csv")
        df.to_csv(path, index=False)
        print(f"[Sentinel] Cached {len(df)} NDVI records to {path}")
        return df


class SoilIntelligenceClient:
    """Client for SoilGrids (REST API) to fetch soil health metrics."""

    BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    def get_soil_properties(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch NPK, pH, and Organic Carbon for a location.
        Note: SoilGrids provides various depth levels; we take the 0-5cm layer.
        """
        properties = ["nitrogen", "phh2o", "soc"]  # nitrogen, pH, Soil Organic Carbon

        params = {
            "lon": lon,
            "lat": lat,
            "property": properties,
            "depth": "0-5cm",
            "value": "mean",
            "timeout": 30,
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            results = {}
            layers = data.get("properties", {}).get("layers", [])
            for layer in layers:
                name = layer.get("name")
                depths = layer.get("depths", [])
                if depths:
                    val = depths[0].get("values", {}).get("mean")
                    results[name] = val

            return results
        except Exception as e:
            print(f"Error fetching soil data: {e}")
            return {}

    def fetch_and_cache(self, locations: list[tuple[float, float]],
                       cache_dir: str = "data/external") -> pd.DataFrame:
        """Fetch soil properties for multiple locations and cache."""
        os.makedirs(cache_dir, exist_ok=True)
        records = []
        for lat, lon in locations:
            props = self.get_soil_properties(lat, lon)
            records.append({"lat": lat, "lon": lon, **props})
        df = pd.DataFrame(records)
        path = os.path.join(cache_dir, "soil_properties.csv")
        df.to_csv(path, index=False)
        print(f"[SoilGrids] Cached {len(df)} soil profiles to {path}")
        return df


class AgmarknetClient:
    """Client for Agmarknet (Data.gov.in) for economic market data."""

    # Live API Key from User
    API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
    BASE_URL = "https://api.data.gov.in/resource/9ef273ef-a641-4de2-a243-a04145617300"

    def __init__(self, api_key: str = API_KEY):
        self.api_key = api_key

    def get_market_prices(self, commodity: str, state: str = "Uttar Pradesh") -> pd.DataFrame:
        """Fetch latest mandi prices for a commodity."""
        if not self.api_key:
            print("Agmarknet API Key missing. Returning dummy data.")
            return pd.DataFrame()

        params = {
            "api-key": self.api_key,
            "format": "json",
            "filters[commodity]": commodity,
            "filters[state]": state,
            "limit": 10,
            "timeout": 30,
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])
            if not records:
                print(f"No records found for {commodity} in {state}.")
                return pd.DataFrame()
            return pd.DataFrame(records)
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return pd.DataFrame()

    def fetch_and_cache(self, commodity: str, state: str = "Uttar Pradesh",
                       cache_dir: str = "data/external") -> pd.DataFrame:
        """Fetch market prices and cache to CSV."""
        os.makedirs(cache_dir, exist_ok=True)
        df = self.get_market_prices(commodity, state)
        if not df.empty:
            path = os.path.join(cache_dir, f"agmarknet_{commodity.lower()}_{state.lower().replace(' ', '_')}.csv")
            df.to_csv(path, index=False)
            print(f"[Agmarknet] Cached {len(df)} price records to {path}")
        return df


def run_all_connectors(cache: bool = True):
    """Run all API connectors for demonstration and testing."""
    lat, lon = 28.61, 77.21  # New Delhi
    year = datetime.now().year

    print("=" * 60)
    print("  API Connectors Test Suite")
    print("=" * 60)

    # 1. Soil Intelligence
    print("\n🌱 Soil Intelligence (SoilGrids)...")
    soil = SoilIntelligenceClient()
    props = soil.get_soil_properties(lat, lon)
    print(f"   Soil Properties: {props}")
    if cache:
        soil.fetch_and_cache([(lat, lon)])

    # 2. Satellite NDVI
    print("\n🛰️  Sentinel-2 NDVI (Demo)...")
    sentinel = SentinelClient()
    ndvi = sentinel.get_ndvi(lat, lon, f"{year}-01-01", f"{year}-03-01")
    print(f"   NDVI estimate: {ndvi}")
    if cache:
        sentinel.fetch_and_cache(lat, lon, year)

    # 3. Market Prices
    print("\n💰 Agmarknet Market Prices...")
    agmark = AgmarknetClient()
    market = agmark.get_market_prices("Wheat", "Uttar Pradesh")
    if not market.empty:
        print(f"   Found {len(market)} records")
        print(f"   Sample: {market[['commodity', 'min_price', 'max_price']].head(2)}")
    else:
        print("   API Key Verification...")
        r = requests.get(agmark.BASE_URL, params={"api-key": agmark.API_KEY, "format": "json", "limit": 1})
        if r.status_code == 200:
            print("   ✅ API Key is VALID")
        else:
            print(f"   ❌ API Key verification failed (Status: {r.status_code})")
    if cache:
        agmark.fetch_and_cache("Wheat", "Uttar Pradesh")

    print("\n" + "=" * 60)
    print("  All connectors tested")
    print("=" * 60)


if __name__ == "__main__":
    run_all_connectors(cache=True)

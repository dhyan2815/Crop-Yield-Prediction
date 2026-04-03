"""
API Connectors for Professional Agronomic Data.
Includes Soil Intelligence and Satellite (Proxy) data.
"""

import requests
import pandas as pd
from typing import Dict, Any

class SoilIntelligenceClient:
    """Client for SoilGrids (REST API) to fetch soil health metrics."""
    
    BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    def get_soil_properties(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch NPK, pH, and Organic Carbon for a location.
        Note: SoilGrids provides various depth levels; we take the 0-5cm layer.
        """
        # Mapping for relevant properties
        properties = ["nitrogen", "phh2o", "soc"] # nitrogen, pH, Soil Organic Carbon
        
        params = {
            "lon": lon,
            "lat": lat,
            "property": properties,
            "depth": "0-5cm",
            "value": "mean"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Simple extractor for the response structure
            results = {}
            layers = data.get("properties", {}).get("layers", [])
            for layer in layers:
                name = layer.get("name")
                # Get mean value from the 0-5cm depth
                depths = layer.get("depths", [])
                if depths:
                    val = depths[0].get("values", {}).get("mean")
                    # SoilGrids often uses scaling factors (e.g. nitrogen is g/kg * 10)
                    results[name] = val
            
            return results
        except Exception as e:
            print(f"Error fetching soil data: {e}")
            return {}

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
            "limit": 10
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

if __name__ == "__main__":
    print("Testing Soil Intelligence Connector...")
    soil = SoilIntelligenceClient()
    # Test for a point in India (Sample)
    props = soil.get_soil_properties(28.61, 77.21)
    print(f"Soil Properties for New Delhi: {props}")

    print("\nVerifying Agmarknet API Key...")
    agmark = AgmarknetClient()
    # Broaden test: fetch any records without filters first to verify key
    params = {"api-key": AgmarknetClient.API_KEY, "format": "json", "limit": 1}
    try:
        r = requests.get(AgmarknetClient.BASE_URL, params=params)
        if r.status_code == 200:
            print("✅ API Key is VALID (Status 200)")
            data = r.json()
            print(f"Sample Record Commodity: {data.get('records', [{}])[0].get('commodity', 'N/A')}")
        else:
            print(f"❌ API Key Verification Failed. Status: {r.status_code}")
            print(f"Response: {r.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

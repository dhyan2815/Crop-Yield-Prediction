"""
Unit tests for API connectors module.
"""

import pytest
import pandas as pd
from unittest.mock import patch, Mock
from scripts.api_connectors import (
    SentinelClient,
    SoilIntelligenceClient,
    AgmarknetClient,
)


@pytest.fixture
def mock_requests():
    with patch("scripts.api_connectors.requests") as mock_req:
        yield mock_req


def test_sentinel_approximate_ndvi():
    client = SentinelClient()
    # Test monsoon month (July) -> peak ndvi near max threshold
    ndvi_jul = client._approximate_ndvi(28.0, 77.0, "2025-07-15")
    assert 0.60 <= ndvi_jul <= 0.75
    # Test winter month (January) -> low ndvi
    ndvi_jan = client._approximate_ndvi(28.0, 77.0, "2025-01-15")
    assert 0.20 <= ndvi_jan <= 0.40


@patch("scripts.api_connectors.requests.get")
def test_sentinel_get_ndvi_with_mock(mock_get):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {}
    mock_get.return_value = mock_resp

    client = SentinelClient(client_id="test", client_secret="test")
    client.authenticate = Mock(return_value="fake_token")
    ndvi = client.get_ndvi(28.61, 77.21, "2025-01-01", "2025-01-31")
    assert isinstance(ndvi, float)
    assert -1.0 <= ndvi <= 1.0


@patch("scripts.api_connectors.requests.get")
def test_soil_properties_success(mock_get):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "properties": {
            "layers": [
                {
                    "name": "nitrogen",
                    "depths": [{"values": {"mean": 150}}],
                },
                {
                    "name": "phh2o",
                    "depths": [{"values": {"mean": 6.8}}],
                },
                {
                    "name": "soc",
                    "depths": [{"values": {"mean": 22.5}}],
                },
            ]
        }
    }
    mock_get.return_value = mock_resp

    client = SoilIntelligenceClient()
    props = client.get_soil_properties(28.61, 77.21)
    assert props["nitrogen"] == 150
    assert props["phh2o"] == 6.8
    assert props["soc"] == 22.5


@patch("scripts.api_connectors.requests.get")
def test_agmarknet_success(mock_get):
    mock_resp = Mock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "records": [
            {"commodity": "Wheat", "state": "Uttar Pradesh", "min_price": 2000, "max_price": 2200},
            {"commodity": "Wheat", "state": "Uttar Pradesh", "min_price": 2100, "max_price": 2300},
        ]
    }
    mock_get.return_value = mock_resp

    client = AgmarknetClient(api_key="fake_key")
    df = client.get_market_prices("Wheat", "Uttar Pradesh")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_agmarknet_no_api_key(monkeypatch):
    monkeypatch.setattr(AgmarknetClient, "API_KEY", None)
    client = AgmarknetClient(api_key=None)
    df = client.get_market_prices("Wheat")
    assert df.empty


def test_fetch_and_cache_directories(tmp_path):
    """Test that fetch_and_cache methods create expected file paths."""
    client = SentinelClient()
    cache_dir = tmp_path / "cache"
    df = pd.DataFrame({"date": ["2025-01-01"], "ndvi": [0.5]})
    with patch.object(client, "get_ndvi_timeseries", return_value=df):
        result = client.fetch_and_cache(28.61, 77.21, 2025, cache_dir=str(cache_dir))
    expected_path = cache_dir / "sentinel_ndvi_28.61_77.21_2025.csv"
    assert expected_path.exists()
    assert result.iloc[0]["ndvi"] == 0.5

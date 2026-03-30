# scripts/data_cleaner.py
"""Data cleaning module for crop yield prediction."""

import pandas as pd
from .config import RAW_DATA_PATH, RAINFALL_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_crop_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw crop yield data from CSV."""
    df = pd.read_csv(filepath)
    # Clean column names
    df.columns = df.columns.str.strip()
    return df


def load_rainfall_data(filepath: str = RAINFALL_DATA_PATH) -> pd.DataFrame:
    """Load rainfall data from filtered CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def filter_india_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data for India only."""
    df = df[df['Area'].str.strip().str.lower() == 'india'].copy()
    return df


def clean_crop_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean crop (Item) names: strip whitespace and remove quotes."""
    if 'Item' in df.columns:
        df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)
    return df


def calculate_average_rainfall(df: pd.DataFrame, rainfall_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average annual rainfall per state and merge with crop data.

    For this dataset, we'll use a simplified approach since the rainfall
    data is at state level. The average rainfall per year is used.
    """
    # Calculate yearly average rainfall across all states
    if 'Year' in rainfall_df.columns and 'ANNUAL' in rainfall_df.columns:
        yearly_rainfall = rainfall_df.groupby('Year')['ANNUAL'].mean().reset_index()
        yearly_rainfall.columns = ['Year', 'average_rain_fall_mm_per_year']
        # Merge with crop data
        df = df.merge(yearly_rainfall, on='Year', how='left')
    return df


def add_temperature_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add average temperature column.

    Note: In a full implementation, this would come from a weather API
    or temperature dataset. For now, we create a placeholder or use
    existing temp data if available.
    """
    # If avg_temp already exists in the data, keep it
    # Otherwise, create a synthetic temperature based on rainfall (correlation)
    if 'avg_temp' not in df.columns:
        # Use a simple correlation: higher rainfall often means lower avg temp in India
        df['avg_temp'] = 26 - (df['average_rain_fall_mm_per_year'] / 200)
    return df


def convert_yield_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yield from hg/ha to kg/ha."""
    if 'hg/ha_yield' in df.columns:
        df['kg_per_ha_yield'] = df['hg/ha_yield'] / 10
    return df


def clean_data(input_path: str = RAW_DATA_PATH,
               rainfall_path: str = RAINFALL_DATA_PATH,
               output_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Full data cleaning pipeline.

    Returns cleaned DataFrame and saves to output_path.
    """
    # Load data
    crop_df = load_raw_crop_data(input_path)
    rainfall_df = load_rainfall_data(rainfall_path)

    # Filter for India
    crop_df = filter_india_data(crop_df)

    # Clean crop names
    crop_df = clean_crop_names(crop_df)

    # Calculate average rainfall and merge
    crop_df = calculate_average_rainfall(crop_df, rainfall_df)

    # Add temperature column
    crop_df = add_temperature_column(crop_df)

    # Convert yield units
    crop_df = convert_yield_units(crop_df)

    # Drop rows with missing critical values
    crop_df = crop_df.dropna(subset=[
        'average_rain_fall_mm_per_year',
        'avg_temp',
        'pesticides_tonnes'
    ])

    # Save to processed path
    crop_df.to_csv(output_path, index=False)

    return crop_df


if __name__ == "__main__":
    print("Running data cleaning pipeline...")
    df = clean_data()
    print(f"Cleaned data saved. Shape: {df.shape}")

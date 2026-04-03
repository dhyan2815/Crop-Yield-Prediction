"""
DEPRECATED: Backward compatibility wrapper.
Use feature_engineer_v2.engineer_features_v2() for all new code.
"""

import warnings
import pandas as pd
from .config import PROCESSED_DATA_PATH, FEATURES_DATA_PATH

warnings.warn(
    "feature_engineer is deprecated; use feature_engineer_v2.engineer_features_v2 instead",
    DeprecationWarning,
    stacklevel=2
)


def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy wrapper - forwards to v2 pipeline."""
    from .feature_engineer_v2 import calculate_interaction_features as v2_fn
    return v2_fn(df.copy())


def add_year_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy wrapper - forwards to v2 pipeline."""
    from .feature_engineer_v2 import add_year_based_features as v2_fn
    return v2_fn(df.copy())


def engineer_features(input_path: str = PROCESSED_DATA_PATH,
                      output_path: str = FEATURES_DATA_PATH) -> pd.DataFrame:
    """
    Legacy wrapper - forwards to v2 pipeline.
    Maintains compatibility with old code that expects this interface.
    """
    from .feature_engineer_v2 import engineer_features_v2

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    result = engineer_features_v2(df)

    if output_path:
        result.to_csv(output_path, index=False)

    return result


def get_feature_names() -> list:
    """Return list of all engineered feature names."""
    from .feature_engineer_v2 import FEATURE_COLUMNS
    return FEATURE_COLUMNS

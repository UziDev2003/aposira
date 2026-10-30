"""
Feature Engineering Module
==========================
Generates features for demand forecasting including:
- Calendar features
- Pakistan-specific holiday features
- Lag features
- Rolling statistics
- Price and promotion features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

# Import Pakistan calendar
from backend.core.pakistan_calendar import generate_pakistan_features


def create_features(df: pd.DataFrame, 
                   target_col: str = "sales_qty",
                   date_col: str = "date",
                   lags: List[int] = None,
                   windows: List[int] = None) -> pd.DataFrame:
    """
    Create all features for demand forecasting.
    
    Args:
        df: DataFrame with sales data
        target_col: Name of the target column
        date_col: Name of the date column
        lags: List of lag periods (default: [7, 14, 30])
        windows: List of rolling window sizes (default: [7, 14, 30])
    
    Returns:
        DataFrame with all features added
    """
    if lags is None:
        lags = [7, 14, 30]
    if windows is None:
        windows = [7, 14, 30]
    
    df = df.copy()
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col]).reset_index(drop=True)
    
    # 1. Calendar features
    df = _add_calendar_features(df, date_col)
    
    # 2. Pakistan-specific features
    df = _add_pakistan_features(df, date_col)
    
    # 3. Lag features
    df = _add_lag_features(df, target_col, lags)
    
    # 4. Rolling features
    df = _add_rolling_features(df, target_col, windows)
    
    # 5. Exponential weighted moving features
    df = _add_ewm_features(df, target_col, windows)
    
    # 6. Price features
    df = _add_price_features(df)
    
    # 7. Promotion features
    df = _add_promotion_features(df)
    
    return df


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add standard calendar features."""
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)  # Friday is special in Pakistan
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    
    # Cyclical encoding for day of week
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def _add_pakistan_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add Pakistan-specific calendar features."""
    # Use the Pakistan calendar module
    pakistan_features = generate_pakistan_features(df[date_col])
    
    # Merge features
    for col in pakistan_features.columns:
        df[col] = pakistan_features[col].values
    
    return df


def _add_lag_features(df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    """Add lag features for the target variable."""
    for lag in lags:
        df[f"sales_lag_{lag}"] = df[target_col].shift(lag)
    
    # Difference features
    df["sales_diff_1"] = df[target_col].shift(1).diff(1)
    df["sales_diff_7"] = df[target_col].shift(1).diff(7)
    
    return df


def _add_rolling_features(df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
    """Add rolling window statistics."""
    for window in windows:
        # Rolling mean
        df[f"rolling_mean_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).mean()
        
        # Rolling std
        df[f"rolling_std_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).std()
        
        # Rolling min/max
        df[f"rolling_min_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).min()
        df[f"rolling_max_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).max()
        
        # Rolling median
        df[f"rolling_median_{window}"] = df[target_col].shift(1).rolling(window=window, min_periods=1).median()
    
    return df


def _add_ewm_features(df: pd.DataFrame, target_col: str, spans: List[int]) -> pd.DataFrame:
    """Add exponential weighted moving features."""
    for span in spans:
        df[f"ewm_{span}"] = df[target_col].shift(1).ewm(span=span, adjust=False).mean()
    
    return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-related features."""
    if "unit_price" in df.columns:
        # Price change
        df["price_change"] = df["unit_price"].diff()
        df["price_change_pct"] = df["unit_price"].pct_change()
        
        # Rolling price statistics
        df["price_rolling_mean_7"] = df["unit_price"].rolling(window=7, min_periods=1).mean()
        df["price_rolling_mean_30"] = df["unit_price"].rolling(window=30, min_periods=1).mean()
        
        # Relative price vs rolling average
        df["relative_price"] = df["unit_price"] / df["price_rolling_mean_7"]
    
    return df


def _add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add promotion-related features."""
    if "promotion_flag" in df.columns:
        # Days since last promotion
        df["days_since_promo"] = df["promotion_flag"].replace(0, np.nan).ffill().fillna(0)
        df["days_since_promo"] = df.groupby(df.index).cumcount() - df["days_since_promo"].cumsum()
        
        # Promotion in last N days
        for window in [7, 14, 30]:
            df[f"promo_count_{window}"] = df["promotion_flag"].rolling(
                window=window, min_periods=1
            ).sum()
        
        # Is currently on promotion
        df["is_promo"] = df["promotion_flag"]
    
    return df


def create_store_sku_features(df: pd.DataFrame,
                              target_col: str = "sales_qty",
                              date_col: str = "date",
                              group_col: str = "sku") -> pd.DataFrame:
    """
    Create features for each SKU-store combination separately.
    This ensures lag features are calculated within each group.
    """
    # Group by SKU and store, then apply feature engineering
    feature_dfs = []
    
    for (sku, store), group in df.groupby(["sku", "store_id"]):
        group = group.sort_values(date_col).reset_index(drop=True)
        group = create_features(group, target_col, date_col)
        feature_dfs.append(group)
    
    return pd.concat(feature_dfs, ignore_index=True)


def get_feature_names() -> List[str]:
    """Return list of all feature names that can be generated."""
    return [
        # Calendar features
        "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
        "day_of_year", "is_weekend", "is_friday", "is_month_start", "is_month_end",
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        
        # Pakistan features
        "is_ramadan", "is_eid_ul_fitr", "is_eid_ul_adha", "is_pre_eid_rush",
        "days_to_next_eid", "days_from_last_eid", "is_wedding_season",
        "is_monsoon", "is_summer", "is_winter", "is_salary_period",
        
        # Lag features
        "sales_lag_7", "sales_lag_14", "sales_lag_30",
        "sales_diff_1", "sales_diff_7",
        
        # Rolling features
        "rolling_mean_7", "rolling_std_7", "rolling_min_7", "rolling_max_7", "rolling_median_7",
        "rolling_mean_14", "rolling_std_14", "rolling_min_14", "rolling_max_14", "rolling_median_14",
        "rolling_mean_30", "rolling_std_30", "rolling_min_30", "rolling_max_30", "rolling_median_30",
        
        # EWM features
        "ewm_7", "ewm_14", "ewm_30",
        
        # Price features
        "price_change", "price_change_pct", "price_rolling_mean_7", 
        "price_rolling_mean_30", "relative_price",
        
        # Promotion features
        "days_since_promo", "promo_count_7", "promo_count_14", "promo_count_30", "is_promo"
    ]


def prepare_data_for_modeling(df: pd.DataFrame,
                              target_col: str = "sales_qty",
                              date_col: str = "date",
                              feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare data for machine learning modeling.
    
    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature column names
    """
    df = df.copy()
    
    # Ensure date is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create features if not already done
    if "day_of_week" not in df.columns:
        df = create_features(df, target_col, date_col)
    
    # Get feature columns
    if feature_cols is None:
        feature_cols = get_feature_names()
        # Only keep columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Remove rows with NaN in features (from lag/rolling calculations)
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y, feature_cols
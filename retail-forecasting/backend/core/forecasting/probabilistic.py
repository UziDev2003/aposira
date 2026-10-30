"""
Probabilistic Forecasting Module
=================================
LightGBM quantile regression for probabilistic forecasts.
Provides P10/P50/P90 predictions for inventory planning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import lightgbm as lgb
from datetime import timedelta

from backend.core.feature_engineering import create_features, get_feature_names


class ProbabilisticForecaster:
    """Quantile regression for probabilistic demand forecasting."""
    
    def __init__(self, quantiles: List[float] = None):
        """
        Initialize the probabilistic forecaster.
        
        Args:
            quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
        """
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.models: Dict[float, Any] = {}
        self.feature_names: List[str] = []
        
    def train(self, 
              df: pd.DataFrame,
              target_col: str = "sales_qty",
              date_col: str = "date",
              val_size: int = 30) -> Dict[str, Dict[str, float]]:
        """
        Train quantile regression models.
        
        Args:
            df: Training data
            target_col: Target column
            date_col: Date column
            val_size: Validation size
        
        Returns:
            Dictionary of metrics for each quantile
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Generate features
        df = create_features(df, target_col, date_col)
        
        # Get feature columns
        self.feature_names = get_feature_names()
        self.feature_names = [f for f in self.feature_names if f in df.columns]
        
        # Remove rows with NaN
        df = df.dropna(subset=self.feature_names + [target_col])
        
        # Split data
        train_df = df.iloc[:-val_size]
        val_df = df.iloc[-val_size:]
        
        X_train = train_df[self.feature_names]
        y_train = train_df[target_col]
        X_val = val_df[self.feature_names]
        y_val = val_df[target_col]
        
        metrics = {}
        
        # Train model for each quantile
        for q in self.quantiles:
            print(f"Training quantile {q}...")
            
            # LightGBM quantile regression
            params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 7,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbosity": -1,
                "n_jobs": -1,
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            self.models[q] = model
            
            # Calculate metrics
            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)
            
            metrics[f"q{int(q*100)}_mae"] = np.mean(np.abs(y_val - y_pred))
            metrics[f"q{int(q*100)}_mape"] = self._mape(y_val, y_pred)
        
        return metrics
    
    def predict(self, 
                df: pd.DataFrame,
                horizon: int = 7,
                date_col: str = "date") -> pd.DataFrame:
        """
        Generate probabilistic forecasts.
        
        Args:
            df: Historical data
            horizon: Days to forecast
            date_col: Date column
        
        Returns:
            DataFrame with P10, P50, P90 forecasts
        """
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Generate features
        df = create_features(df, "sales_qty", date_col)
        
        # Get last date
        last_date = df[date_col].max()
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq="D"
        )
        
        # Create future dataframe
        future_df = pd.DataFrame({date_col: future_dates})
        
        # Add features (use last known values for lag/rolling features)
        for col in self.feature_names:
            if col in df.columns:
                future_df[col] = df[col].iloc[-1]
            else:
                future_df[col] = 0
        
        # Add Pakistan-specific features
        from backend.core.pakistan_calendar import generate_pakistan_features
        pak_features = generate_pakistan_features(future_df[date_col])
        for col in pak_features.columns:
            future_df[col] = pak_features[col].values
        
        # Add calendar features
        future_df["day_of_week"] = future_df[date_col].dt.dayofweek
        future_df["month"] = future_df[date_col].dt.month
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in future_df.columns:
                future_df[col] = 0
        
        # Make predictions for each quantile
        X_future = future_df[self.feature_names]
        
        predictions = {}
        for q, model in self.models.items():
            pred = model.predict(X_future)
            predictions[f"q{int(q*100)}"] = np.maximum(pred, 0)
        
        # Create result dataframe
        result = pd.DataFrame({
            "date": future_dates,
            "forecast_p10": predictions.get(0.1, np.zeros(horizon)),
            "forecast_p50": predictions.get(0.5, np.zeros(horizon)),
            "forecast_p90": predictions.get(0.9, np.zeros(horizon)),
        })
        
        # Rename for clarity
        result = result.rename(columns={
            "forecast_p10": "lower_bound",
            "forecast_p50": "forecast_median",
            "forecast_p90": "upper_bound"
        })
        
        return result
    
    def get_safety_stock_estimate(self, 
                                   df: pd.DataFrame,
                                   horizon: int = 7,
                                   date_col: str = "date") -> float:
        """
        Get safety stock estimate based on forecast uncertainty.
        
        Args:
            df: Historical data
            horizon: Lead time horizon
            date_col: Date column
        
        Returns:
            Safety stock quantity
        """
        forecast = self.predict(df, horizon, date_col)
        
        # Safety stock = P90 - P50 (over lead time)
        safety_stock = (forecast["upper_bound"].sum() - 
                       forecast["forecast_median"].sum())
        
        return max(safety_stock, 0)
    
    @staticmethod
    def _mape(y_true, y_pred):
        """Calculate MAPE."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_probabilistic(df: pd.DataFrame,
                        target_col: str = "sales_qty",
                        date_col: str = "date",
                        val_size: int = 30) -> Tuple[ProbabilisticForecaster, Dict]:
    """
    Convenience function to train probabilistic forecaster.
    """
    forecaster = ProbabilisticForecaster()
    metrics = forecaster.train(df, target_col, date_col, val_size)
    return forecaster, metrics


def get_forecast_with_uncertainty(df: pd.DataFrame,
                                   horizon: int = 7,
                                   target_col: str = "sales_qty",
                                   date_col: str = "date") -> pd.DataFrame:
    """
    Convenience function to get probabilistic forecasts.
    """
    forecaster = ProbabilisticForecaster()
    forecaster.train(df, target_col, date_col, val_size=30)
    return forecaster.predict(df, horizon, date_col)
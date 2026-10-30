"""
Prophet Forecasting Engine
===========================
Facebook Prophet as baseline model for demand forecasting.
Custom seasonality for Pakistani holidays.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
except ImportError:
    # Fallback if prophet is not available
    Prophet = None

from backend.core.pakistan_calendar import ISLAMIC_HOLIDAYS, FIXED_HOLIDAYS


class ProphetEngine:
    """Prophet-based demand forecasting engine."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Prophet engine.
        
        Args:
            params: Prophet parameters
        """
        if Prophet is None:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        
        self.default_params = {
            "yearly_seasonality": "auto",
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10,
            "holidays_prior_scale": 10,
        }
        
        self.params = params or self.default_params
        self.model = None
        self.holidays_df = None
        self.metrics: Dict[str, float] = {}
        
    def _create_holidays_dataframe(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create holidays dataframe for Prophet."""
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        holidays = []
        
        # Add fixed holidays
        for year in range(start_year, end_year + 1):
            for name, (month, day) in FIXED_HOLIDAYS.items():
                holidays.append({
                    "holiday": name.lower().replace(" ", "_"),
                    "ds": f"{year}-{month:02d}-{day:02d}",
                    "lower_window": 0,
                    "upper_window": 0
                })
        
        # Add Islamic holidays
        for year in range(start_year, end_year + 1):
            if year in ISLAMIC_HOLIDAYS:
                for name, date in ISLAMIC_HOLIDAYS[year].items():
                    if date:
                        holidays.append({
                            "holiday": name.lower().replace("-", "_").replace(" ", "_"),
                            "ds": date.isoformat(),
                            "lower_window": 0,
                            "upper_window": 2 if "Eid" in name else 0  # 3-day holiday for Eid
                        })
        
        return pd.DataFrame(holidays)
    
    def train(self, 
              df: pd.DataFrame,
              target_col: str = "sales_qty",
              date_col: str = "date",
              val_size: int = 30) -> Dict[str, float]:
        """
        Train the Prophet model.
        
        Args:
            df: Training data
            target_col: Target column name
            date_col: Date column name
            val_size: Number of days for validation
        
        Returns:
            Dictionary of metrics
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]
        
        # Create holidays
        start_date = prophet_df["ds"].min().strftime("%Y-%m-%d")
        end_date = prophet_df["ds"].max().strftime("%Y-%m-%d")
        self.holidays_df = self._create_holidays_dataframe(start_date, end_date)
        
        # Split into train and validation
        train_df = prophet_df.iloc[:-val_size]
        val_df = prophet_df.iloc[-val_size:]
        
        # Create and train model
        self.model = Prophet(**self.params, holidays=self.holidays_df)
        
        self.model.fit(train_df)
        
        # Predict on validation
        future = self.model.make_future_dataframe(periods=val_size)
        forecast = self.model.predict(future)
        
        # Get validation predictions
        val_pred = forecast.iloc[-val_size:]["yhat"].values
        val_actual = val_df["y"].values
        
        # Calculate metrics
        self.metrics = {
            "val_mae": self._mae(val_actual, val_pred),
            "val_rmse": self._rmse(val_actual, val_pred),
            "val_mape": self._mape(val_actual, val_pred),
        }
        
        # Refit on the complete supplied history after measuring the holdout.
        self.model = Prophet(**self.params, holidays=self.holidays_df)
        self.model.fit(prophet_df)
        return self.metrics
    
    def predict(self, 
                df: pd.DataFrame,
                horizon: int = 7,
                date_col: str = "date") -> pd.DataFrame:
        """
        Generate forecasts for the next N days.
        
        Args:
            df: Historical data
            horizon: Number of days to forecast
            date_col: Date column name
        
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create future dataframe
        future = pd.DataFrame({"ds": pd.date_range(df[date_col].max() + timedelta(days=1), periods=horizon)})
        
        # Predict
        forecast = self.model.predict(future)
        
        # Get future predictions
        future_forecast = forecast.iloc[-horizon:]
        
        result = pd.DataFrame({
            "date": future_forecast["ds"].values,
            "forecast": np.maximum(future_forecast["yhat"].values, 0),
            "lower_bound": np.maximum(future_forecast["yhat_lower"].values, 0),
            "upper_bound": future_forecast["yhat_upper"].values
        })
        
        return result
    
    def get_components(self) -> pd.DataFrame:
        """Get trend, seasonality, and holiday components."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return self.model.plot_components(self.model.predict(
            self.model.make_future_dataframe(periods=0)
        ))
    
    def cross_validate(self, 
                       df: pd.DataFrame,
                       horizon: str = "30 days",
                       period: str = "30 days",
                       initial: str = "180 days") -> pd.DataFrame:
        """
        Perform time-series cross-validation.
        
        Args:
            df: Data
            horizon: Forecast horizon
            period: Period between cutoffs
            initial: Initial training period
        
        Returns:
            DataFrame with cross-validation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Prepare data
        prophet_df = df[["date", "sales_qty"]].copy()
        prophet_df.columns = ["ds", "y"]
        
        # Cross-validation
        cv_results = self.model.crossvalidate(
            prophet_df,
            horizon=horizon,
            period=period,
            initial=initial
        )
        
        return cv_results
    
    @staticmethod
    def _mae(y_true, y_pred):
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def _rmse(y_true, y_pred):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def _mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def forecast_with_prophet(df: pd.DataFrame,
                          horizon: int = 7,
                          target_col: str = "sales_qty",
                          date_col: str = "date",
                          val_size: int = 30) -> Tuple[ProphetEngine, pd.DataFrame]:
    """
    Convenience function to generate Prophet forecasts.
    
    Returns:
        Tuple of (trained model, forecasts)
    """
    engine = ProphetEngine()
    engine.train(df, target_col, date_col, val_size)
    forecasts = engine.predict(df, horizon, date_col)
    return engine, forecasts
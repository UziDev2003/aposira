"""
Ensemble Forecasting Module
===========================
Hybrid ensemble combining LightGBM and Prophet models.
Weighted average with automatic weight optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

# Import forecasting engines
from aposira.forecasting.backend.core.forecasting.lightgbm_engine import LightGBMEngine
from aposira.forecasting.backend.core.forecasting.prophet_engine import ProphetEngine


class EnsembleForecaster:
    """Hybrid ensemble of LightGBM and Prophet forecasters."""
    
    def __init__(self, 
                 lightgbm_weight: float = 0.7,
                 prophet_weight: float = 0.3,
                 auto_optimize: bool = True):
        """
        Initialize the ensemble forecaster.
        
        Args:
            lightgbm_weight: Weight for LightGBM predictions
            prophet_weight: Weight for Prophet predictions
            auto_optimize: Whether to automatically optimize weights
        """
        self.lightgbm_weight = lightgbm_weight
        self.prophet_weight = prophet_weight
        self.auto_optimize = auto_optimize
        
        self.lightgbm_engine = LightGBMEngine()
        self.prophet_engine = ProphetEngine()
        
        self.lightgbm_metrics = {}
        self.prophet_metrics = {}
        self.ensemble_metrics = {}
        
    def train(self, 
              df: pd.DataFrame,
              target_col: str = "sales_qty",
              date_col: str = "date",
              val_size: int = 30) -> Dict[str, float]:
        """
        Train both models and optimize ensemble weights.
        
        Args:
            df: Training data
            target_col: Target column
            date_col: Date column
            val_size: Validation size
        
        Returns:
            Dictionary of ensemble metrics
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Train LightGBM
        print("Training LightGBM...")
        self.lightgbm_metrics = self.lightgbm_engine.train(
            df, target_col, date_col, val_size
        )
        
        # Train Prophet
        print("Training Prophet...")
        try:
            self.prophet_metrics = self.prophet_engine.train(
                df, target_col, date_col, val_size
            )
        except Exception as e:
            print(f"Prophet training failed: {e}")
            self.prophet_metrics = {"val_mae": float('inf'), "val_mape": float('inf')}
        
        # Optimize weights if enabled
        if self.auto_optimize:
            self._optimize_weights(df, target_col, date_col, val_size)
        
        # Calculate ensemble metrics
        self.ensemble_metrics = self._calculate_ensemble_metrics(
            df, target_col, date_col, val_size
        )
        
        return self.ensemble_metrics
    
    def _optimize_weights(self,
                          df: pd.DataFrame,
                          target_col: str,
                          date_col: str,
                          val_size: int):
        """Optimize ensemble weights based on validation performance."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Get validation data
        val_df = df.iloc[-val_size:]
        
        # Get predictions from both models
        lgb_pred = self.lightgbm_engine.predict(df, val_size, date_col)["forecast"].values
        prophet_pred = self.prophet_engine.predict(df, val_size, date_col)["forecast"].values
        actual = val_df[target_col].values
        
        # Grid search for optimal weights
        best_mae = float('inf')
        best_weights = (self.lightgbm_weight, self.prophet_weight)
        
        for lgb_w in np.arange(0, 1.05, 0.1):
            prophet_w = 1 - lgb_w
            
            # Ensemble prediction
            ensemble_pred = lgb_w * lgb_pred + prophet_w * prophet_pred
            
            # Calculate MAE
            mae = np.mean(np.abs(actual - ensemble_pred))
            
            if mae < best_mae:
                best_mae = mae
                best_weights = (lgb_w, prophet_w)
        
        self.lightgbm_weight, self.prophet_weight = best_weights
        print(f"Optimized weights - LightGBM: {self.lightgbm_weight:.2f}, Prophet: {self.prophet_weight:.2f}")
    
    def _calculate_ensemble_metrics(self,
                                     df: pd.DataFrame,
                                     target_col: str,
                                     date_col: str,
                                     val_size: int) -> Dict[str, float]:
        """Calculate ensemble metrics on validation set."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        val_df = df.iloc[-val_size:]
        
        # Get predictions
        lgb_pred = self.lightgbm_engine.predict(df, val_size, date_col)["forecast"].values
        prophet_pred = self.prophet_engine.predict(df, val_size, date_col)["forecast"].values
        
        # Ensemble prediction
        ensemble_pred = (self.lightgbm_weight * lgb_pred + 
                        self.prophet_weight * prophet_pred)
        
        actual = val_df[target_col].values
        
        return {
            "ensemble_mae": np.mean(np.abs(actual - ensemble_pred)),
            "ensemble_rmse": np.sqrt(np.mean((actual - ensemble_pred) ** 2)),
            "ensemble_mape": self._mape(actual, ensemble_pred),
            "lightgbm_weight": self.lightgbm_weight,
            "prophet_weight": self.prophet_weight
        }
    
    def predict(self, 
                df: pd.DataFrame,
                horizon: int = 7,
                date_col: str = "date") -> pd.DataFrame:
        """
        Generate ensemble forecasts.
        
        Args:
            df: Historical data
            horizon: Days to forecast
            date_col: Date column
        
        Returns:
            DataFrame with ensemble forecasts
        """
        # Get predictions from both models
        lgb_forecast = self.lightgbm_engine.predict(df, horizon, date_col)
        prophet_forecast = self.prophet_engine.predict(df, horizon, date_col)
        
        # Ensemble prediction
        ensemble_forecast = (
            self.lightgbm_weight * lgb_forecast["forecast"].values +
            self.prophet_weight * prophet_forecast["forecast"].values
        )
        
        # Ensemble confidence interval (weighted)
        ensemble_lower = (
            self.lightgbm_weight * lgb_forecast["lower_bound"].values +
            self.prophet_weight * prophet_forecast["lower_bound"].values
        )
        
        ensemble_upper = (
            self.lightgbm_weight * lgb_forecast["upper_bound"].values +
            self.prophet_weight * prophet_forecast["upper_bound"].values
        )
        
        result = pd.DataFrame({
            "date": lgb_forecast["date"].values,
            "forecast": np.maximum(ensemble_forecast, 0),
            "lower_bound": np.maximum(ensemble_lower, 0),
            "upper_bound": ensemble_upper,
            "lightgbm_forecast": lgb_forecast["forecast"].values,
            "prophet_forecast": prophet_forecast["forecast"].values
        })
        
        return result
    
    def predict_with_uncertainty(self,
                                  df: pd.DataFrame,
                                  horizon: int = 7,
                                  date_col: str = "date") -> pd.DataFrame:
        """
        Generate forecasts with uncertainty estimates.
        
        Args:
            df: Historical data
            horizon: Days to forecast
            date_col: Date column
        
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        # Get base predictions
        base_predictions = self.predict(df, horizon, date_col)
        
        # Add uncertainty based on model disagreement
        lgb_forecast = self.lightgbm_engine.predict(df, horizon, date_col)
        prophet_forecast = self.prophet_engine.predict(df, horizon, date_col)
        
        # Model disagreement as uncertainty measure
        disagreement = np.abs(
            lgb_forecast["forecast"].values - prophet_forecast["forecast"].values
        )
        
        # Add disagreement to uncertainty
        base_predictions["lower_bound"] = np.maximum(
            base_predictions["forecast"] - disagreement, 0
        )
        base_predictions["upper_bound"] = base_predictions["forecast"] + disagreement
        
        return base_predictions
    
    @staticmethod
    def _mape(y_true, y_pred):
        """Calculate MAPE."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_ensemble(df: pd.DataFrame,
                   target_col: str = "sales_qty",
                   date_col: str = "date",
                   val_size: int = 30,
                   auto_optimize: bool = True) -> Tuple[EnsembleForecaster, Dict[str, float]]:
    """
    Convenience function to train an ensemble model.
    
    Returns:
        Tuple of (trained ensemble, metrics)
    """
    ensemble = EnsembleForecaster(auto_optimize=auto_optimize)
    metrics = ensemble.train(df, target_col, date_col, val_size)
    return ensemble, metrics


def forecast_ensemble(df: pd.DataFrame,
                      horizon: int = 7,
                      target_col: str = "sales_qty",
                      date_col: str = "date") -> pd.DataFrame:
    """
    Convenience function to generate ensemble forecasts.
    """
    ensemble = EnsembleForecaster()
    ensemble.train(df, target_col, date_col, val_size=30)
    return ensemble.predict(df, horizon, date_col)
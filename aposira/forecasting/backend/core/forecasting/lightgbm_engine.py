"""
LightGBM Forecasting Engine
===========================
Primary forecasting engine using LightGBM Regressor.
Supports multi-horizon forecasting with time-series cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path
import json
from datetime import datetime, timedelta

# Import feature engineering
from aposira.forecasting.backend.core.feature_engineering import create_features, get_feature_names


class LightGBMEngine:
    """LightGBM-based demand forecasting engine."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the LightGBM engine.
        
        Args:
            params: LightGBM parameters (uses defaults if not provided)
        """
        # Default parameters
        self.default_params = {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        
        self.params = params or self.default_params
        self.model = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.feature_importance: pd.DataFrame = None
        
    def train(self, 
              df: pd.DataFrame,
              target_col: str = "sales_qty",
              date_col: str = "date",
              val_size: int = 30) -> Dict[str, float]:
        """
        Train the LightGBM model.
        
        Args:
            df: Training data with features
            target_col: Target column name
            date_col: Date column name
            val_size: Number of days to use for validation
        
        Returns:
            Dictionary of training metrics
        """
        # Create features
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Generate features
        df = create_features(df, target_col, date_col)
        
        # Get feature columns
        self.feature_names = get_feature_names()
        self.feature_names = [f for f in self.feature_names if f in df.columns]
        
        # Remove rows with NaN in features
        df = df.dropna(subset=self.feature_names + [target_col])
        
        # Split into train and validation
        train_df = df.iloc[:-val_size]
        val_df = df.iloc[-val_size:]
        
        X_train = train_df[self.feature_names]
        y_train = train_df[target_col]
        X_val = val_df[self.feature_names]
        y_val = val_df[target_col]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Calculate metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        self.metrics = {
            "train_mae": self._mae(y_train, y_pred_train),
            "train_rmse": self._rmse(y_train, y_pred_train),
            "train_mape": self._mape(y_train, y_pred_train),
            "val_mae": self._mae(y_val, y_pred_val),
            "val_rmse": self._rmse(y_val, y_pred_val),
            "val_mape": self._mape(y_val, y_pred_val),
        }
        
        self.interval_radius = float(np.quantile(np.abs(np.asarray(y_val) - y_pred_val), .95))
        best_iteration = self.model.best_iteration or 100
        # The holdout metrics above remain unchanged; deploy a fit on all supplied history.
        self.model = lgb.train(self.params, lgb.Dataset(df[self.feature_names], label=df[target_col]),
                               num_boost_round=best_iteration)
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importance(importance_type="gain")
        }).sort_values("importance", ascending=False)
        
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
        df = df.sort_values(date_col).reset_index(drop=True)
        
        future_dates = pd.date_range(df[date_col].max() + timedelta(days=1), periods=horizon)
        history = df.copy()
        predictions = []
        for day in future_dates:
            row = history.iloc[-1].copy()
            row[date_col] = day
            row["sales_qty"] = np.nan
            # No future promotion schedule was supplied.
            row["promotion_flag"] = 0
            history = pd.concat([history, row.to_frame().T], ignore_index=True)
            history[date_col] = pd.to_datetime(history[date_col])
            for col in ("sales_qty", "unit_price", "promotion_flag"):
                if col in history:
                    history[col] = pd.to_numeric(history[col])
            features = create_features(history, "sales_qty", date_col)
            prediction = max(0., float(self.model.predict(features[self.feature_names].tail(1))[0]))
            predictions.append(prediction)
            history.loc[history.index[-1], "sales_qty"] = prediction
        predictions = np.asarray(predictions)
        radius = getattr(self, "interval_radius", 0.)
        result = pd.DataFrame({"date": future_dates, "forecast": predictions,
                               "lower_bound": np.maximum(predictions-radius, 0), "upper_bound": predictions+radius})

        return result
    
    def predict_with_uncertainty(self,
                                  df: pd.DataFrame,
                                  horizon: int = 7,
                                  date_col: str = "date",
                                  n_samples: int = 100) -> pd.DataFrame:
        """
        Generate forecasts with uncertainty estimates using Monte Carlo dropout.
        
        Args:
            df: Historical data
            horizon: Number of days to forecast
            date_col: Date column name
            n_samples: Number of Monte Carlo samples
        
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        # Get base predictions
        base_predictions = self.predict(df, horizon, date_col)
        
        # For simplicity, use percentage-based uncertainty
        # In production, use MC dropout or quantile regression
        uncertainty_factor = 0.15  # 15% uncertainty
        
        base_predictions["lower_bound"] = base_predictions["forecast"] * (1 - uncertainty_factor)
        base_predictions["upper_bound"] = base_predictions["forecast"] * (1 + uncertainty_factor)
        
        return base_predictions
    
    def cross_validate(self, 
                       df: pd.DataFrame,
                       n_splits: int = 5,
                       target_col: str = "sales_qty",
                       date_col: str = "date") -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation.
        
        Args:
            df: Data with features
            n_splits: Number of CV splits
            target_col: Target column
            date_col: Date column
        
        Returns:
            Dictionary of metrics for each fold
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Generate features
        df = create_features(df, target_col, date_col)
        
        # Get feature columns
        feature_cols = get_feature_names()
        feature_cols = [f for f in feature_cols if f in df.columns]
        
        # Drop NaN rows
        df = df.dropna(subset=feature_cols + [target_col])
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_metrics = {
            "mae": [],
            "rmse": [],
            "mape": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=300,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            # Predict
            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)
            
            # Calculate metrics
            cv_metrics["mae"].append(self._mae(y_val, y_pred))
            cv_metrics["rmse"].append(self._rmse(y_val, y_pred))
            cv_metrics["mape"].append(self._mape(y_val, y_pred))
        
        return cv_metrics
    
    def save(self, path: Path):
        """Save model to disk."""
        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "metrics": self.metrics
        }
        joblib.dump(model_data, path)
    
    def load(self, path: Path):
        """Load model from disk."""
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.params = model_data["params"]
        self.feature_names = model_data["feature_names"]
        self.metrics = model_data["metrics"]
    
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


def train_model(df: pd.DataFrame,
                target_col: str = "sales_qty",
                date_col: str = "date",
                val_size: int = 30,
                params: Dict[str, Any] = None) -> Tuple[LightGBMEngine, Dict[str, float]]:
    """
    Convenience function to train a LightGBM model.
    
    Returns:
        Tuple of (trained model, metrics)
    """
    engine = LightGBMEngine(params)
    metrics = engine.train(df, target_col, date_col, val_size)
    return engine, metrics


def forecast_demand(df: pd.DataFrame,
                    horizon: int = 7,
                    target_col: str = "sales_qty",
                    date_col: str = "date",
                    model_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Convenience function to generate forecasts.
    
    Args:
        df: Historical sales data
        horizon: Days to forecast
        target_col: Target column
        date_col: Date column
        model_path: Optional path to saved model
    
    Returns:
        DataFrame with forecasts
    """
    engine = LightGBMEngine()
    
    if model_path and model_path.exists():
        engine.load(model_path)
    else:
        # Train on all data
        engine.train(df, target_col, date_col, val_size=30)
    
    return engine.predict(df, horizon, date_col)
"""
Metrics Module
==============
Evaluation metrics for demand forecasting and inventory optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error."""
    n = len(y_true)
    if n <= seasonality:
        return np.nan
    
    # Calculate naive forecast error
    naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    naive_mae = np.mean(naive_errors)
    
    if naive_mae == 0:
        return np.nan
    
    return np.mean(np.abs(y_true - y_pred)) / naive_mae


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def coverage_probability(y_true: np.ndarray, 
                         lower_bound: np.ndarray, 
                         upper_bound: np.ndarray) -> float:
    """Coverage probability for prediction intervals."""
    covered = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
    return covered / len(y_true) * 100


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    """Quantile loss for probabilistic forecasts."""
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def calculate_forecast_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                lower_bound: Optional[np.ndarray] = None,
                                upper_bound: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate all forecast metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        lower_bound: Lower prediction bound (optional)
        upper_bound: Upper prediction bound (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    
    # Add coverage if bounds are provided
    if lower_bound is not None and upper_bound is not None:
        metrics["coverage"] = coverage_probability(y_true, lower_bound, upper_bound)
    
    return metrics


def calculate_inventory_metrics(actual_demand: np.ndarray,
                                 predicted_demand: np.ndarray,
                                 current_stock: np.ndarray,
                                 order_quantity: np.ndarray,
                                 holding_cost_rate: float = 0.2,
                                 order_cost: float = 500,
                                 unit_cost: float = 100) -> Dict[str, float]:
    """
    Calculate inventory-specific metrics.
    
    Args:
        actual_demand: Actual demand values
        predicted_demand: Predicted demand values
        current_stock: Current inventory levels
        order_quantity: Order quantities placed
        holding_cost_rate: Annual holding cost as percentage of unit cost
        order_cost: Cost per order (PKR)
        unit_cost: Unit cost (PKR)
    
    Returns:
        Dictionary of inventory metrics
    """
    # Calculate service level (fill rate)
    total_demand = np.sum(actual_demand)
    if total_demand > 0:
        fulfilled = np.sum(np.minimum(current_stock, actual_demand))
        fill_rate = fulfilled / total_demand * 100
    else:
        fill_rate = 100.0
    
    # Calculate stockout events
    stockouts = np.sum((current_stock < actual_demand).astype(int))
    
    # Calculate holding cost (average inventory * holding rate)
    avg_inventory = np.mean(current_stock)
    holding_cost = avg_inventory * unit_cost * holding_cost_rate / 365
    
    # Calculate order cost
    num_orders = np.sum(order_quantity > 0)
    total_order_cost = num_orders * order_cost
    
    # Calculate total inventory cost
    total_cost = holding_cost + total_order_cost
    
    # Forecast error metrics
    forecast_error = np.mean(np.abs(actual_demand - predicted_demand))
    forecast_bias = np.mean(actual_demand - predicted_demand)
    
    return {
        "fill_rate": fill_rate,
        "stockout_events": stockouts,
        "avg_inventory": avg_inventory,
        "holding_cost": holding_cost,
        "order_cost": total_order_cost,
        "total_inventory_cost": total_cost,
        "forecast_error": forecast_error,
        "forecast_bias": forecast_bias,
        "num_orders": num_orders
    }


def calculate_sku_level_metrics(df: pd.DataFrame,
                                 actual_col: str = "actual",
                                 predicted_col: str = "predicted",
                                 group_col: str = "sku") -> pd.DataFrame:
    """
    Calculate metrics for each SKU.
    
    Args:
        df: DataFrame with actual and predicted values
        actual_col: Column name for actual values
        predicted_col: Column name for predicted values
        group_col: Column to group by (e.g., SKU)
    
    Returns:
        DataFrame with metrics per SKU
    """
    metrics_list = []
    
    for sku, group in df.groupby(group_col):
        y_true = group[actual_col].values
        y_pred = group[predicted_col].values
        
        sku_metrics = {
            group_col: sku,
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "smape": smape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "total_demand": np.sum(y_true),
            "avg_demand": np.mean(y_true)
        }
        
        metrics_list.append(sku_metrics)
    
    return pd.DataFrame(metrics_list)


def weighted_aggregate_metrics(metrics_df: pd.DataFrame, 
                                weight_col: str = "total_demand") -> Dict[str, float]:
    """
    Aggregate metrics weighted by sales volume.
    
    Args:
        metrics_df: DataFrame with SKU-level metrics
        weight_col: Column to use for weighting
    
    Returns:
        Dictionary of weighted aggregate metrics
    """
    weights = metrics_df[weight_col].values
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        return {}
    
    weighted_metrics = {}
    
    for col in metrics_df.columns:
        if col in [weight_col, "sku"]:
            continue
        
        values = metrics_df[col].values
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            weighted_metrics[f"weighted_{col}"] = np.average(
                values[valid_mask], 
                weights=weights[valid_mask]
            )
    
    return weighted_metrics
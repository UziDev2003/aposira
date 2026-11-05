"""
Safety Stock Module
===================
Traditional and quantile-based safety stock calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List


# Z-scores for normal distribution
Z_SCORES = {
    0.80: 0.84,
    0.85: 1.04,
    0.90: 1.28,
    0.95: 1.65,
    0.97: 1.88,
    0.99: 2.33,
    0.999: 3.08
}


def calculate_safety_stock_traditional(demand_std: float,
                                        lead_time_days: int,
                                        service_level: float = 0.95) -> float:
    """
    Calculate safety stock using the traditional formula.
    
    SS = Z * σ_demand * sqrt(lead_time)
    
    Args:
        demand_std: Standard deviation of daily demand
        lead_time_days: Lead time in days
        service_level: Target service level (0-1)
    
    Returns:
        Safety stock quantity
    """
    z = Z_SCORES.get(service_level, 1.65)
    
    # Safety stock during lead time
    ss = z * demand_std * np.sqrt(lead_time_days)
    
    return round(ss, 2)


def calculate_safety_stock_quantile(forecast_p50: np.ndarray,
                                     forecast_p90: np.ndarray) -> float:
    """
    Calculate safety stock using quantile-based method.
    
    SS = P90_forecast - P50_forecast (over lead time)
    
    This is an innovation that uses probabilistic forecasts directly.
    
    Args:
        forecast_p50: Median (P50) forecasts for each day
        forecast_p90: P90 forecasts for each day
    
    Returns:
        Safety stock quantity
    """
    # Sum of upper bound minus median over the period
    ss = np.sum(forecast_p90 - forecast_p50)
    
    return round(max(ss, 0), 2)


def calculate_safety_stock_from_history(daily_demand: np.ndarray,
                                         lead_time_days: int,
                                         service_level: float = 0.95) -> Dict:
    """
    Calculate safety stock from historical demand data.
    
    Args:
        daily_demand: Array of daily demand values
        lead_time_days: Lead time in days
        service_level: Target service level
    
    Returns:
        Dictionary with safety stock and related metrics
    """
    # Calculate statistics
    mean_demand = np.mean(daily_demand)
    demand_std = np.std(daily_demand)
    
    # Calculate safety stock
    ss = calculate_safety_stock_traditional(demand_std, lead_time_days, service_level)
    
    # Calculate reorder point
    avg_daily_demand = mean_demand
    rop = avg_daily_demand * lead_time_days + ss
    
    return {
        "safety_stock": ss,
        "reorder_point": round(rop, 2),
        "avg_daily_demand": round(mean_demand, 2),
        "demand_std": round(demand_std, 2),
        "lead_time_days": lead_time_days,
        "service_level": service_level,
        "z_score": Z_SCORES.get(service_level, 1.65)
    }


def calculate_safety_stock_with_forecast(forecast_df: pd.DataFrame,
                                          lead_time_days: int) -> Dict:
    """
    Calculate safety stock using forecast data.
    
    Args:
        forecast_df: DataFrame with forecast_p10, forecast_p50, forecast_p90 columns
        lead_time_days: Lead time in days
    
    Returns:
        Dictionary with safety stock calculations
    """
    # Get forecasts for lead time period
    if len(forecast_df) < lead_time_days:
        lead_time_days = len(forecast_df)
    
    p50 = forecast_df["forecast_median"].values[:lead_time_days]
    p90 = forecast_df["upper_bound"].values[:lead_time_days]
    p10 = forecast_df["lower_bound"].values[:lead_time_days]
    
    # Quantile-based safety stock
    ss_quantile = calculate_safety_stock_quantile(p50, p90)
    
    # Traditional approach using forecast variability
    demand_std = np.std(p50 - p90) / 2  # Approximate std from interval
    ss_traditional = calculate_safety_stock_traditional(demand_std, lead_time_days, 0.95)
    
    # Average demand during lead time
    avg_demand_lead_time = np.sum(p50)
    
    # Reorder point
    rop = avg_demand_lead_time + ss_quantile
    
    return {
        "safety_stock_quantile": ss_quantile,
        "safety_stock_traditional": ss_traditional,
        "reorder_point": round(rop, 2),
        "avg_demand_lead_time": round(avg_demand_lead_time, 2),
        "lead_time_days": lead_time_days
    }


def calculate_service_level_from_safety_stock(safety_stock: float,
                                               demand_std: float,
                                               lead_time_days: int) -> float:
    """
    Calculate the service level achieved by a given safety stock.
    
    Args:
        safety_stock: Safety stock quantity
        demand_std: Standard deviation of daily demand
        lead_time_days: Lead time in days
    
    Returns:
        Achieved service level (0-1)
    """
    if demand_std <= 0 or lead_time_days <= 0:
        return 1.0
    
    # Calculate z-score from safety stock
    z = safety_stock / (demand_std * np.sqrt(lead_time_days))
    
    # Find closest service level
    achieved_service_level = 0.80  # Default
    for level, z_score in sorted(Z_SCORES.items()):
        if z >= z_score:
            achieved_service_level = level
    
    return achieved_service_level


def optimize_safety_stock(daily_demand: np.ndarray,
                          lead_time_days: int,
                          unit_cost: float,
                          stockout_cost: float,
                          holding_cost_rate: float = 0.2) -> Dict:
    """
    Optimize safety stock by balancing holding cost vs stockout cost.
    
    Args:
        daily_demand: Historical daily demand
        lead_time_days: Lead time in days
        unit_cost: Unit cost (PKR)
        stockout_cost: Cost per unit of stockout (PKR)
        holding_cost_rate: Annual holding cost as decimal
    
    Returns:
        Dictionary with optimal safety stock and costs
    """
    mean_demand = np.mean(daily_demand)
    demand_std = np.std(daily_demand)
    
    # Daily holding cost
    daily_holding_cost = unit_cost * holding_cost_rate / 365
    
    # Test different service levels
    results = []
    
    for service_level in [0.80, 0.85, 0.90, 0.95, 0.99]:
        ss = calculate_safety_stock_traditional(demand_std, lead_time_days, service_level)
        
        # Calculate expected stockouts per year
        z = Z_SCORES.get(service_level, 1.65)
        stockout_probability = 1 - service_level
        
        # Expected stockouts per year (simplified)
        expected_stockouts = stockout_probability * 365
        
        # Total cost
        holding_cost = ss * daily_holding_cost * 365
        stockout_expected_cost = expected_stockouts * stockout_cost
        
        total_cost = holding_cost + stockout_expected_cost
        
        results.append({
            "service_level": service_level,
            "safety_stock": ss,
            "holding_cost": round(holding_cost, 2),
            "stockout_expected_cost": round(stockout_expected_cost, 2),
            "total_cost": round(total_cost, 2)
        })
    
    # Find optimal
    optimal = min(results, key=lambda x: x["total_cost"])
    
    return {
        "optimal_safety_stock": optimal["safety_stock"],
        "optimal_service_level": optimal["service_level"],
        "optimal_total_cost": optimal["total_cost"],
        "alternatives": results
    }


def calculate_safety_stock_for_sku(sales_df: pd.DataFrame,
                                    lead_time_days: int,
                                    service_level: float = 0.95) -> Dict:
    """
    Calculate safety stock for a single SKU from sales data.
    
    Args:
        sales_df: DataFrame with daily sales data
        lead_time_days: Lead time in days
        service_level: Target service level
    
    Returns:
        Dictionary with safety stock calculations
    """
    daily_demand = sales_df["sales_qty"].values
    
    return calculate_safety_stock_from_history(
        daily_demand, lead_time_days, service_level
    )
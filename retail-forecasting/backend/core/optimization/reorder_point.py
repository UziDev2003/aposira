"""
Reorder Point (ROP) Module
===========================
Calculate reorder points and generate low-stock alerts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def calculate_rop(avg_daily_demand: float,
                  lead_time_days: int,
                  safety_stock: float = 0) -> float:
    """
    Calculate Reorder Point (ROP).
    
    ROP = avg_daily_demand * lead_time + safety_stock
    
    Args:
        avg_daily_demand: Average daily demand
        lead_time_days: Lead time in days
        safety_stock: Safety stock quantity
    
    Returns:
        Reorder point quantity
    """
    rop = avg_daily_demand * lead_time_days + safety_stock
    return round(rop, 2)


def calculate_rop_with_variability(daily_demand: np.ndarray,
                                    lead_time_days: int,
                                    service_level: float = 0.95) -> Dict:
    """
    Calculate ROP considering demand variability.
    
    Args:
        daily_demand: Array of daily demand values
        lead_time_days: Lead time in days
        service_level: Target service level
    
    Returns:
        Dictionary with ROP and components
    """
    mean_demand = np.mean(daily_demand)
    demand_std = np.std(daily_demand)
    
    # Calculate safety stock
    from backend.core.optimization.safety_stock import calculate_safety_stock_traditional
    safety_stock = calculate_safety_stock_traditional(
        demand_std, lead_time_days, service_level
    )
    
    # Calculate ROP
    rop = calculate_rop(mean_demand, lead_time_days, safety_stock)
    
    return {
        "reorder_point": rop,
        "safety_stock": safety_stock,
        "avg_daily_demand": round(mean_demand, 2),
        "demand_std": round(demand_std, 2),
        "lead_time_days": lead_time_days,
        "service_level": service_level,
        "demand_during_lead_time": round(mean_demand * lead_time_days, 2)
    }


def check_reorder_needed(current_stock: float,
                         rop: float,
                         incoming_orders: float = 0) -> Dict:
    """
    Check if reorder is needed.
    
    Args:
        current_stock: Current inventory level
        rop: Reorder point
        incoming_orders: Quantity of orders already placed but not received
    
    Returns:
        Dictionary with reorder status
    """
    available_stock = current_stock + incoming_orders
    
    return {
        "reorder_needed": available_stock < rop,
        "current_stock": current_stock,
        "incoming_orders": incoming_orders,
        "available_stock": available_stock,
        "reorder_point": rop,
        "stock_deficit": max(0, rop - available_stock),
        "urgency": "high" if available_stock < rop * 0.5 else ("medium" if available_stock < rop else "low")
    }


def generate_reorder_recommendations(inventory_df: pd.DataFrame,
                                      sales_dfs: Dict[str, pd.DataFrame],
                                      lead_times: Dict[str, int],
                                      service_level: float = 0.95) -> pd.DataFrame:
    """
    Generate reorder recommendations for all SKUs.
    
    Args:
        inventory_df: DataFrame with current stock levels (sku, store_id, current_stock)
        sales_dfs: Dict of {sku_store: sales DataFrame}
        lead_times: Dict of {sku: lead_time_days}
        service_level: Target service level
    
    Returns:
        DataFrame with reorder recommendations
    """
    recommendations = []
    
    for _, row in inventory_df.iterrows():
        sku = row["sku"]
        store_id = row["store_id"]
        current_stock = row.get("current_stock", 0)
        
        # Get sales data for this SKU-store combination
        key = f"{sku}_{store_id}"
        if key not in sales_dfs:
            continue
        
        sales_df = sales_dfs[key]
        if sales_df.empty:
            continue
        
        # Get lead time
        lead_time = lead_times.get(sku, 7)
        
        # Calculate ROP
        daily_demand = sales_df["sales_qty"].values
        rop_result = calculate_rop_with_variability(daily_demand, lead_time, service_level)
        
        # Check if reorder needed
        reorder_check = check_reorder_needed(current_stock, rop_result["reorder_point"])
        
        recommendations.append({
            "sku": sku,
            "store_id": store_id,
            "current_stock": current_stock,
            "reorder_point": rop_result["reorder_point"],
            "safety_stock": rop_result["safety_stock"],
            "reorder_needed": reorder_check["reorder_needed"],
            "urgency": reorder_check["urgency"],
            "stock_deficit": reorder_check["stock_deficit"],
            "lead_time_days": lead_time
        })
    
    return pd.DataFrame(recommendations)


def calculate_days_of_supply(current_stock: float,
                              avg_daily_demand: float) -> float:
    """
    Calculate days of supply remaining.
    
    Args:
        current_stock: Current inventory level
        avg_daily_demand: Average daily demand
    
    Returns:
        Days of supply
    """
    if avg_daily_demand <= 0:
        return float('inf')
    
    return round(current_stock / avg_daily_demand, 1)


def get_low_stock_alerts(inventory_df: pd.DataFrame,
                         sales_dfs: Dict[str, pd.DataFrame],
                         threshold_days: int = 7) -> pd.DataFrame:
    """
    Get low stock alerts based on days of supply.
    
    Args:
        inventory_df: DataFrame with current stock levels
        sales_dfs: Dict of sales DataFrames
        threshold_days: Alert threshold in days
    
    Returns:
        DataFrame with low stock alerts
    """
    alerts = []
    
    for _, row in inventory_df.iterrows():
        sku = row["sku"]
        store_id = row["store_id"]
        current_stock = row.get("current_stock", 0)
        
        key = f"{sku}_{store_id}"
        if key not in sales_dfs:
            continue
        
        sales_df = sales_dfs[key]
        if sales_df.empty:
            continue
        
        avg_daily_demand = sales_df["sales_qty"].mean()
        days_supply = calculate_days_of_supply(current_stock, avg_daily_demand)
        
        if days_supply <= threshold_days:
            alerts.append({
                "sku": sku,
                "store_id": store_id,
                "current_stock": current_stock,
                "avg_daily_demand": round(avg_daily_demand, 2),
                "days_of_supply": days_supply,
                "alert_level": "critical" if days_supply <= 3 else ("high" if days_supply <= 5 else "medium")
            })
    
    return pd.DataFrame(alerts).sort_values("days_of_supply")
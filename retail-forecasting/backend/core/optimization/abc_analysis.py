"""
ABC Analysis Module
===================
Pareto-based ABC classification of SKUs by revenue contribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def calculate_abc_classification(sales_df: pd.DataFrame,
                                  products_df: pd.DataFrame,
                                  method: str = "revenue") -> pd.DataFrame:
    """
    Perform ABC analysis on SKUs.
    
    Classifies SKUs into:
    - A: Top 20% SKUs = ~80% of revenue (high value, low quantity)
    - B: Next 30% SKUs = ~15% of revenue
    - C: Bottom 50% SKUs = ~5% of revenue
    
    Args:
        sales_df: DataFrame with sales data (sku, sales_qty, unit_price)
        products_df: DataFrame with product info
        method: Classification method ("revenue", "quantity", or "profit")
    
    Returns:
        DataFrame with ABC classification
    """
    # Calculate total revenue per SKU
    sales_df = sales_df.copy()
    sales_df["revenue"] = sales_df["sales_qty"] * sales_df["unit_price"]
    
    # Aggregate by SKU
    sku_metrics = sales_df.groupby("sku").agg({
        "sales_qty": "sum",
        "revenue": "sum"
    }).reset_index()
    
    # Merge with product info for cost
    if "unit_cost" in products_df.columns:
        products_df = products_df[["sku_id", "unit_cost"]].copy()
        products_df.columns = ["sku", "unit_cost"]
        sku_metrics = sku_metrics.merge(products_df, on="sku", how="left")
        sku_metrics["unit_cost"] = sku_metrics["unit_cost"].fillna(0)
        sku_metrics["profit"] = sku_metrics["revenue"] - (sku_metrics["sales_qty"] * sku_metrics["unit_cost"])
    
    # Sort by selected metric
    if method == "revenue":
        sort_col = "revenue"
    elif method == "quantity":
        sort_col = "sales_qty"
    elif method == "profit" and "profit" in sku_metrics.columns:
        sort_col = "profit"
    else:
        sort_col = "revenue"
    
    sku_metrics = sku_metrics.sort_values(sort_col, ascending=False).reset_index(drop=True)
    
    # Calculate cumulative percentage
    total = sku_metrics[sort_col].sum()
    sku_metrics["cumulative"] = sku_metrics[sort_col].cumsum()
    sku_metrics["cumulative_pct"] = sku_metrics["cumulative"] / total * 100 if total > 0 else 0.0
    
    # Assign ABC class
    def assign_class(cum_pct):
        if cum_pct <= 80:
            return "A"
        elif cum_pct <= 95:
            return "B"
        else:
            return "C"
    
    sku_metrics["abc_class"] = sku_metrics["cumulative_pct"].shift(1, fill_value=0).apply(lambda pct: "A" if pct < 80 else "B" if pct < 95 else "C")
    
    # Add additional metrics
    sku_metrics["sku_count"] = len(sku_metrics)
    sku_metrics["class_a_count"] = len(sku_metrics[sku_metrics["abc_class"] == "A"])
    sku_metrics["class_b_count"] = len(sku_metrics[sku_metrics["abc_class"] == "B"])
    sku_metrics["class_c_count"] = len(sku_metrics[sku_metrics["abc_class"] == "C"])
    
    return sku_metrics


def get_abc_summary(abc_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for ABC classification.
    
    Args:
        abc_df: DataFrame with ABC classification
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    for cls in ["A", "B", "C"]:
        class_data = abc_df[abc_df["abc_class"] == cls]
        
        summary[f"class_{cls.lower()}"] = {
            "sku_count": len(class_data),
            "total_revenue": round(class_data["revenue"].sum(), 2),
            "total_quantity": int(class_data["sales_qty"].sum()),
            "revenue_pct": round(class_data["revenue"].sum() / abc_df["revenue"].sum() * 100, 2),
            "avg_revenue_per_sku": round(class_data["revenue"].mean(), 2) if len(class_data) > 0 else 0
        }
    
    return summary


def apply_differentiated_service_levels(abc_df: pd.DataFrame,
                                         service_levels: Dict[str, float] = None) -> pd.DataFrame:
    """
    Apply different service level targets by ABC class.
    
    Args:
        abc_df: DataFrame with ABC classification
        service_levels: Dict of {class: service_level}
    
    Returns:
        DataFrame with service level targets
    """
    if service_levels is None:
        service_levels = {"A": 0.99, "B": 0.95, "C": 0.85}
    
    abc_df = abc_df.copy()
    abc_df["target_service_level"] = abc_df["abc_class"].map(service_levels)
    
    return abc_df


def get_inventory_prioritization(abc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get inventory prioritization based on ABC classification.
    
    Args:
        abc_df: DataFrame with ABC classification
    
    Returns:
        DataFrame with prioritization scores
    """
    priority_map = {"A": 1, "B": 2, "C": 3}
    review_freq_map = {"A": "Weekly", "B": "Bi-weekly", "C": "Monthly"}
    
    abc_df = abc_df.copy()
    abc_df["priority"] = abc_df["abc_class"].map(priority_map)
    abc_df["review_frequency"] = abc_df["abc_class"].map(review_freq_map)
    
    # Sort by priority
    abc_df = abc_df.sort_values("priority")
    
    return abc_df


def calculate_abc_with_stores(sales_df: pd.DataFrame,
                               products_df: pd.DataFrame,
                               store_id: str = None) -> pd.DataFrame:
    """
    Perform ABC analysis per store.
    
    Args:
        sales_df: Sales data
        products_df: Product data
        store_id: Optional store filter
    
    Returns:
        DataFrame with ABC classification per store
    """
    if store_id:
        sales_df = sales_df[sales_df["store_id"] == store_id]
    
    # Group by store and SKU
    store_sku = sales_df.groupby(["store_id", "sku"]).agg({
        "sales_qty": "sum",
        "unit_price": "mean"
    }).reset_index()
    
    store_sku["revenue"] = store_sku["sales_qty"] * store_sku["unit_price"]
    
    results = []
    
    for store in store_sku["store_id"].unique():
        store_data = store_sku[store_sku["store_id"] == store].copy()
        store_data = store_data.sort_values("revenue", ascending=False).reset_index(drop=True)
        
        total = store_data["revenue"].sum()
        store_data["cumulative"] = store_data["revenue"].cumsum()
        store_data["cumulative_pct"] = store_data["cumulative"] / total * 100
        
        def assign_class(cum_pct):
            if cum_pct <= 80:
                return "A"
            elif cum_pct <= 95:
                return "B"
            else:
                return "C"
        
        store_data["abc_class"] = store_data["cumulative_pct"].apply(assign_class)
        results.append(store_data)
    
    return pd.concat(results, ignore_index=True)


def visualize_abc_pareto(abc_df: pd.DataFrame) -> Dict:
    """
    Generate data for ABC Pareto chart.
    
    Args:
        abc_df: DataFrame with ABC classification
    
    Returns:
        Dictionary with chart data
    """
    # Sort by revenue
    abc_df = abc_df.sort_values("revenue", ascending=False).reset_index(drop=True)
    
    # Calculate cumulative
    total = abc_df["revenue"].sum()
    abc_df["cumulative_pct"] = abc_df["revenue"].cumsum() / total * 100
    
    # Get top SKUs for visualization
    top_skus = abc_df.head(20)
    
    return {
        "skus": top_skus["sku"].tolist(),
        "revenues": top_skus["revenue"].tolist(),
        "cumulative_pct": top_skus["cumulative_pct"].tolist(),
        "abc_classes": top_skus["abc_class"].tolist()
    }
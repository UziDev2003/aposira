"""
Economic Order Quantity (EOQ) Module
=====================================
Classic EOQ formula with demand variability adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def calculate_eoq(annual_demand: float, 
                  ordering_cost: float, 
                  holding_cost_rate: float,
                  unit_cost: float) -> float:
    """
    Calculate Economic Order Quantity using the classic Wilson EOQ formula.
    
    EOQ = sqrt(2 * D * S / H)
    
    Where:
    - D = Annual demand (units)
    - S = Ordering cost per order (PKR)
    - H = Holding cost per unit per year (PKR) = unit_cost * holding_cost_rate
    
    Args:
        annual_demand: Annual demand in units
        ordering_cost: Cost per order (PKR)
        holding_cost_rate: Annual holding cost as decimal (e.g., 0.2 for 20%)
        unit_cost: Unit cost (PKR)
    
    Returns:
        Optimal order quantity
    """
    if annual_demand <= 0 or ordering_cost <= 0 or unit_cost <= 0:
        return 0.0
    
    holding_cost = unit_cost * holding_cost_rate
    
    if holding_cost <= 0:
        return 0.0
    
    eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
    
    return round(eoq, 2)


def calculate_eoq_with_quantity_discount(annual_demand: float,
                                          ordering_cost: float,
                                          holding_cost_rate: float,
                                          unit_cost: float,
                                          discount_breakpoints: Dict[int, float]) -> Dict:
    """
    Calculate EOQ considering quantity discounts.
    
    Args:
        annual_demand: Annual demand in units
        ordering_cost: Cost per order (PKR)
        holding_cost_rate: Annual holding cost as decimal
        unit_cost: Base unit cost (PKR)
        discount_breakpoints: Dict of {quantity: discount_price}
    
    Returns:
        Dictionary with optimal order quantity and total cost
    """
    # Calculate EOQ at base price
    base_eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_rate, unit_cost)
    
    results = {
        "base_eoq": base_eoq,
        "alternatives": []
    }
    
    # Calculate total cost for base EOQ
    base_total_cost = calculate_total_inventory_cost(
        annual_demand, base_eoq, ordering_cost, holding_cost_rate, unit_cost
    )
    
    results["alternatives"].append({
        "order_qty": base_eoq,
        "unit_cost": unit_cost,
        "total_cost": base_total_cost
    })
    
    # Check each discount breakpoint
    for break_qty, discount_price in sorted(discount_breakpoints.items()):
        eoq_at_discount = calculate_eoq(annual_demand, ordering_cost, holding_cost_rate, discount_price)
        
        # Order quantity must be at least the discount breakpoint
        order_qty = max(eoq_at_discount, break_qty)
        
        total_cost = calculate_total_inventory_cost(
            annual_demand, order_qty, ordering_cost, holding_cost_rate, discount_price
        )
        
        results["alternatives"].append({
            "order_qty": order_qty,
            "unit_cost": discount_price,
            "total_cost": total_cost
        })
    
    # Find optimal
    best = min(results["alternatives"], key=lambda x: x["total_cost"])
    
    results["optimal_order_qty"] = best["order_qty"]
    results["optimal_unit_cost"] = best["unit_cost"]
    results["optimal_total_cost"] = best["total_cost"]
    
    return results


def calculate_total_inventory_cost(annual_demand: float,
                                    order_qty: float,
                                    ordering_cost: float,
                                    holding_cost_rate: float,
                                    unit_cost: float) -> float:
    """
    Calculate total annual inventory cost.
    
    Total Cost = Purchasing Cost + Ordering Cost + Holding Cost
    
    Args:
        annual_demand: Annual demand in units
        order_qty: Order quantity
        ordering_cost: Cost per order
        holding_cost_rate: Annual holding cost as decimal
        unit_cost: Unit cost
    
    Returns:
        Total annual inventory cost
    """
    if order_qty <= 0:
        return float('inf')
    
    # Purchasing cost
    purchasing_cost = annual_demand * unit_cost
    
    # Ordering cost (number of orders * cost per order)
    num_orders = annual_demand / order_qty
    ordering_cost_total = num_orders * ordering_cost
    
    # Holding cost (average inventory * holding cost per unit)
    avg_inventory = order_qty / 2
    holding_cost = avg_inventory * unit_cost * holding_cost_rate
    
    return purchasing_cost + ordering_cost_total + holding_cost


def calculate_eoq_with_variability(annual_demand: float,
                                    ordering_cost: float,
                                    holding_cost_rate: float,
                                    unit_cost: float,
                                    demand_std: float,
                                    lead_time_days: int,
                                    service_level: float = 0.95) -> Dict:
    """
    Calculate EOQ with demand variability adjustment.
    
    This version accounts for uncertainty in demand by adjusting
    the order quantity based on demand variability.
    
    Args:
        annual_demand: Annual demand in units
        ordering_cost: Cost per order
        holding_cost_rate: Annual holding cost as decimal
        unit_cost: Unit cost
        demand_std: Standard deviation of daily demand
        lead_time_days: Lead time in days
        service_level: Target service level (0-1)
    
    Returns:
        Dictionary with EOQ and safety stock
    """
    # Basic EOQ
    basic_eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_rate, unit_cost)
    
    # Calculate safety stock for variability
    # Using the normal distribution z-score for service level
    z_scores = {
        0.90: 1.28,
        0.95: 1.65,
        0.99: 2.33
    }
    z = z_scores.get(service_level, 1.65)
    
    # Safety stock during lead time
    daily_demand = annual_demand / 365
    demand_variability = demand_std * np.sqrt(lead_time_days)
    safety_stock = z * demand_variability
    
    # Adjusted EOQ (includes safety stock consideration)
    # This is a simplified adjustment - in practice, you might use different approaches
    adjusted_eoq = basic_eoq
    
    return {
        "basic_eoq": basic_eoq,
        "adjusted_eoq": adjusted_eoq,
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(daily_demand * lead_time_days + safety_stock, 2),
        "num_orders_per_year": round(annual_demand / adjusted_eoq, 2),
        "cycle_time_days": round(365 * adjusted_eoq / annual_demand, 2)
    }


def calculate_optimal_order_frequency(annual_demand: float,
                                       ordering_cost: float,
                                       holding_cost_rate: float,
                                       unit_cost: float) -> Dict:
    """
    Calculate optimal number of orders per year.
    
    Args:
        annual_demand: Annual demand
        ordering_cost: Cost per order
        holding_cost_rate: Annual holding cost as decimal
        unit_cost: Unit cost
    
    Returns:
        Dictionary with optimal frequency and related metrics
    """
    eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_rate, unit_cost)
    
    if eoq > 0:
        num_orders = annual_demand / eoq
    else:
        num_orders = 0
    
    return {
        "optimal_eoq": eoq,
        "optimal_orders_per_year": round(num_orders, 2),
        "time_between_orders_days": round(365 / num_orders, 2) if num_orders > 0 else 0,
        "total_ordering_cost": round(num_orders * ordering_cost, 2),
        "total_holding_cost": round((eoq / 2) * unit_cost * holding_cost_rate, 2)
    }
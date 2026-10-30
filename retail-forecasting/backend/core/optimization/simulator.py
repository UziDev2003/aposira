"""
Inventory Policy Simulator
==========================
Core innovation: Backtest inventory policies against historical data.
Compares baseline (simple moving average) vs proposed (LightGBM + quantile-based) policies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class InventorySimulator:
    """
    Simulates inventory policies over historical data to compare performance.
    """
    
    def __init__(self, 
                 holding_cost_rate: float = 0.2,
                 order_cost: float = 500,
                 stockout_cost: float = 100):
        """
        Initialize the simulator.
        
        Args:
            holding_cost_rate: Annual holding cost as decimal
            order_cost: Cost per order (PKR)
            stockout_cost: Cost per unit of stockout (PKR)
        """
        self.holding_cost_rate = holding_cost_rate
        self.order_cost = order_cost
        self.stockout_cost = stockout_cost
        
    def _simulate(self, sales_df, initial_stock, forecasts, safety_stocks):
        sales_df = sales_df.sort_values("date").reset_index(drop=True)
        if sales_df.empty or initial_stock < 0:
            raise ValueError("Provide sales history and nonnegative initial stock")
        stock = float(initial_stock)
        levels, orders, pending, stockouts = [], [], [], []
        demand_values = sales_df.sales_qty.to_numpy(dtype=float)
        for day, demand in enumerate(demand_values):
            arrivals = [order for order in pending if order["arrival_day"] == day]
            stock += sum(order["quantity"] for order in arrivals)
            pending = [order for order in pending if order["arrival_day"] > day]
            fulfilled = min(stock, demand)
            stock -= fulfilled
            if fulfilled < demand:
                stockouts.append({"day": day, "shortage": demand - fulfilled})
            forecast, safety = max(0, forecasts[day]), max(0, safety_stocks[day])
            position = stock + sum(order["quantity"] for order in pending)
            if position <= forecast * 5 + safety:
                quantity = max(forecast * 30, safety * 2)
                if quantity > 0:
                    order = {"day": day, "arrival_day": day + 5, "quantity": quantity}
                    orders.append(order); pending.append(order)
            levels.append(stock)
        return self._calculate_metrics(levels, orders, stockouts, demand_values, sales_df.unit_price.mean())

    def simulate_baseline_policy(self, sales_df, initial_stock, safety_stock_fixed=None):
        sales_df = sales_df.sort_values("date").reset_index(drop=True)
        means = sales_df.sales_qty.shift(1).rolling(30, min_periods=1).mean().fillna(0).to_numpy()
        safety = means * .5 if safety_stock_fixed is None else np.full(len(sales_df), safety_stock_fixed)
        return self._simulate(sales_df, initial_stock, means, safety)

    def simulate_proposed_policy(self, sales_df, initial_stock, forecasts, safety_stocks):
        if len(forecasts) != len(sales_df) or len(safety_stocks) != len(sales_df):
            raise ValueError("Forecast and safety arrays must match daily history")
        return self._simulate(sales_df, initial_stock, forecasts, safety_stocks)

    def _calculate_metrics(self,
                           stock_levels: List[float],
                           orders: List[Dict],
                           stockouts: List[Dict],
                           demand: np.ndarray,
                           avg_price: float) -> Dict:
        """Calculate simulation metrics."""
        total_demand = np.sum(demand)
        fulfilled_demand = total_demand - sum(s["shortage"] for s in stockouts)
        
        # Fill rate
        fill_rate = (fulfilled_demand / total_demand * 100) if total_demand > 0 else 100
        
        # Stockout events
        stockout_events = len(stockouts)
        
        # Average inventory
        avg_inventory = np.mean(stock_levels)
        
        # Holding cost (daily average * holding rate / 365)
        holding_cost = avg_inventory * avg_price * self.holding_cost_rate / 365 * len(stock_levels)
        
        # Order cost
        order_cost_total = len(orders) * self.order_cost
        
        # Stockout cost
        stockout_cost_total = sum(s["shortage"] for s in stockouts) * self.stockout_cost
        
        # Total cost
        total_cost = holding_cost + order_cost_total + stockout_cost_total
        
        return {
            "fill_rate": round(fill_rate, 2),
            "stockout_events": stockout_events,
            "avg_inventory": round(avg_inventory, 2),
            "holding_cost": round(holding_cost, 2),
            "order_cost": round(order_cost_total, 2),
            "stockout_cost": round(stockout_cost_total, 2),
            "total_cost": round(total_cost, 2),
            "num_orders": len(orders),
            "total_demand": int(total_demand),
            "fulfilled_demand": int(fulfilled_demand),
            "stock_levels": stock_levels
        }
    
    def compare_policies(self,
                         sales_df: pd.DataFrame,
                         initial_stock: float = 1000,
                         forecasts: np.ndarray = None,
                         safety_stocks: np.ndarray = None) -> Dict:
        """
        Compare baseline vs proposed policy.
        
        Args:
            sales_df: Historical sales data
            initial_stock: Starting inventory
            forecasts: Forecasts for proposed policy
            safety_stocks: Safety stocks for proposed policy
        
        Returns:
            Dictionary with comparison results
        """
        sales_df = sales_df.sort_values("date").reset_index(drop=True)
        # Run baseline simulation
        baseline = self.simulate_baseline_policy(sales_df, initial_stock)
        
        # Run proposed simulation
        if forecasts is None:
            # Generate simple forecasts if not provided
            sales_df = sales_df.copy()
            sales_df["rolling_avg_30"] = sales_df["sales_qty"].shift(1).rolling(30, min_periods=1).mean().fillna(0)
            forecasts = sales_df["rolling_avg_30"].values
        
        if safety_stocks is None:
            # Generate safety stocks if not provided
            safety_stocks = sales_df["sales_qty"].shift(1).rolling(30, min_periods=2).std().fillna(0).to_numpy() * 1.645 * np.sqrt(5)
        
        proposed = self.simulate_proposed_policy(sales_df, initial_stock, forecasts, safety_stocks)
        
        # Calculate improvement
        improvement = {
            "fill_rate_improvement": proposed["fill_rate"] - baseline["fill_rate"],
            "stockout_reduction": baseline["stockout_events"] - proposed["stockout_events"],
            "inventory_reduction": baseline["avg_inventory"] - proposed["avg_inventory"],
            "cost_savings": baseline["total_cost"] - proposed["total_cost"],
            "order_reduction": baseline["num_orders"] - proposed["num_orders"]
        }
        
        return {
            "baseline": baseline,
            "proposed": proposed,
            "improvement": improvement,
            "recommendation": "proposed" if improvement["cost_savings"] > 0 else "baseline"
        }


def run_simulation(sales_df: pd.DataFrame,
                   initial_stock: float = 1000,
                   forecasts: np.ndarray = None,
                   safety_stocks: np.ndarray = None) -> Dict:
    """
    Convenience function to run inventory simulation.
    """
    simulator = InventorySimulator()
    return simulator.compare_policies(sales_df, initial_stock, forecasts, safety_stocks)


def sensitivity_analysis(sales_df: pd.DataFrame,
                         service_levels: List[float] = None) -> pd.DataFrame:
    """
    Run sensitivity analysis on service level targets.
    
    Args:
        sales_df: Historical sales data
        service_levels: List of service levels to test
    
    Returns:
        DataFrame with results for each service level
    """
    if service_levels is None:
        service_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
    
    results = []
    
    for sl in service_levels:
        # Calculate safety stock for this service level
        demand_std = sales_df["sales_qty"].std()
        z_scores = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z = z_scores.get(sl, 1.65)
        
        lead_time = 5
        ss = z * demand_std * np.sqrt(lead_time)
        
        safety_stocks = np.full(len(sales_df), ss)
        
        # Run simulation
        sim = InventorySimulator()
        result = sim.compare_policies(sales_df, 1000, None, safety_stocks)
        
        results.append({
            "service_level": sl,
            "safety_stock": round(ss, 2),
            "baseline_fill_rate": result["baseline"]["fill_rate"],
            "proposed_fill_rate": result["proposed"]["fill_rate"],
            "baseline_cost": result["baseline"]["total_cost"],
            "proposed_cost": result["proposed"]["total_cost"],
            "cost_savings": result["improvement"]["cost_savings"]
        })
    
    return pd.DataFrame(results)
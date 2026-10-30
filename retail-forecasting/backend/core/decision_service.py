"""Shared APOSIRA decisions for the dashboard and API; no fabricated stock."""
from statistics import NormalDist
import numpy as np
import pandas as pd


def daily_history(sales):
    if sales.empty:
        raise ValueError('No sales history for this SKU and store')
    if sales[['sku', 'store_id']].drop_duplicates().shape[0] != 1:
        raise ValueError('Select exactly one SKU and store')
    frame = sales.copy()
    frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
    daily = frame.groupby('date').agg(sales_qty=('sales_qty', 'sum'), unit_price=('unit_price', 'mean'), promotion_flag=('promotion_flag', 'max'))
    # Missing dates are unknown observations, not assumed zero sales.
    if len(daily) != (daily.index.max() - daily.index.min()).days + 1:
        raise ValueError('Daily history has gaps; supply explicit zero-sales rows for days with no sales')
    daily['sku'] = str(frame['sku'].iloc[0])
    daily['store_id'] = str(frame['store_id'].iloc[0])
    return daily.reset_index()


def forecast_sales(sales, horizon=7, model_type='seasonal_naive'):
    if horizon not in (7, 14, 30):
        raise ValueError('Forecast horizon must be 7, 14 or 30 days')
    daily = daily_history(sales)
    if daily.date.max() < daily.date.min() + pd.DateOffset(months=6):
        raise ValueError('At least six months of daily history is required')
    train, actual = daily.iloc[:-30], daily.sales_qty.iloc[-30:].to_numpy(dtype=float)

    def predict(history, days):
        if model_type == 'seasonal_naive':
            return np.resize(history.sales_qty.tail(7).to_numpy(dtype=float), days)
        if model_type == 'lightgbm':
            from backend.core.forecasting.lightgbm_engine import LightGBMEngine
            engine = LightGBMEngine()
        elif model_type == 'prophet':
            from backend.core.forecasting.prophet_engine import ProphetEngine
            engine = ProphetEngine()
        elif model_type == 'ensemble':
            # Fixed weights avoid optimizing on the same holdout used to report accuracy.
            from backend.core.forecasting.lightgbm_engine import LightGBMEngine
            from backend.core.forecasting.prophet_engine import ProphetEngine
            engines = [LightGBMEngine(), ProphetEngine()]
            for engine in engines:
                engine.train(history, val_size=30)
            return np.mean([e.predict(history, days)['forecast'].to_numpy() for e in engines], axis=0)
        else:
            raise ValueError('Unsupported model type')
        engine.train(history, val_size=30)
        return engine.predict(history, days)['forecast'].to_numpy()

    heldout = np.maximum(predict(train, 30), 0)
    residuals = actual - heldout
    nonzero = actual != 0
    metrics = {'mae': float(np.mean(np.abs(residuals))), 'rmse': float(np.sqrt(np.mean(residuals ** 2))),
               'mape': float(np.mean(np.abs(residuals[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None}
    prediction = np.maximum(predict(daily, horizon), 0)
    # Finite-sample residual quantile. Coverage is nominal, not a measured guarantee.
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * .95)))
    radius = float(np.sort(np.abs(residuals))[rank - 1])
    result = pd.DataFrame({'date': pd.date_range(daily.date.max() + pd.Timedelta(days=1), periods=horizon),
                           'forecast': prediction, 'lower_bound': np.maximum(prediction - radius, 0),
                           'upper_bound': prediction + radius})
    return result, {'model': model_type, 'metrics': metrics, 'validation_days': 30,
                    'interval_level': .95, 'interval_method': 'held-out absolute residual quantile (nominal coverage)',
                    'retraining_recommended': metrics['mape'] is None or metrics['mape'] > 15}


def inventory_plan(sales, current_stock, unit_cost, lead_time=5, order_cost=500, holding_rate=.2, service_level=.95):
    values = (current_stock, unit_cost, lead_time, order_cost, holding_rate, service_level)
    if not all(np.isfinite(v) for v in values) or min(current_stock, lead_time, order_cost) < 0 or unit_cost <= 0 or holding_rate <= 0 or not 0 < service_level < 1:
        raise ValueError('Enter valid stock, positive costs, nonnegative lead time and a service level between 0 and 1')
    daily = daily_history(sales)
    demand = daily.sales_qty.to_numpy(dtype=float)
    average = float(demand.mean())
    safety = float(NormalDist().inv_cdf(service_level) * demand.std() * np.sqrt(lead_time))
    safety = max(0., safety)
    eoq = float(np.sqrt(2 * average * 365 * order_cost / (unit_cost * holding_rate)))
    rop = average * lead_time + safety
    days = current_stock / average if average > 0 else None
    return {'current_stock': current_stock, 'avg_daily_demand': average, 'eoq': eoq, 'safety_stock': safety,
            'reorder_point': rop, 'days_of_supply': days, 'reorder_needed': current_stock < rop,
            'recommended_quantity': float(np.ceil(max(eoq, rop - current_stock))) if current_stock < rop else 0.,
            'status': 'Critical' if current_stock == 0 and average > 0 else 'High' if current_stock < rop or (days is not None and days < 7) else 'Normal'}

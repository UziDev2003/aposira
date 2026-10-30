"""
Retail Forecasting - Configuration Module
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Create directories
for d in [DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Forecasting defaults
DEFAULT_FORECAST_HORIZONS = [7, 14, 30]
DEFAULT_CONFIDENCE_LEVEL = 0.95
LIGHTGBM_PARAMS = {
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

# Inventory defaults
DEFAULT_SERVICE_LEVEL = 0.95  # 95% fill rate target
DEFAULT_HOLDING_COST_RATE = 0.20  # 20% of unit cost per year
DEFAULT_ORDER_COST = 500  # PKR per order
DEFAULT_LEAD_TIME_DAYS = 5

# Pakistani currency
CURRENCY_SYMBOL = "PKR"
CURRENCY_NAME = "Pakistani Rupee"

# API
API_HOST = "127.0.0.1"
API_PORT = 8000
API_PREFIX = "/api/v1"

# Dashboard
DASHBOARD_PORT = 8501

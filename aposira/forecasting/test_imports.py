# Test script to verify all modules import correctly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

print("Testing imports...")

# Test database
from aposira.forecasting.backend.database import Database
print('✓ Database module loaded')

# Test Pakistan calendar
from aposira.forecasting.backend.core.pakistan_calendar import generate_pakistan_features
print('✓ Pakistan calendar loaded')

# Test data validator  
from aposira.forecasting.backend.core.data_validator import DataValidator
print('✓ Data validator loaded')

# Test feature engineering
from aposira.forecasting.backend.core.feature_engineering import create_features
print('✓ Feature engineering loaded')

# Test optimization modules
from aposira.forecasting.backend.core.optimization.eoq import calculate_eoq
from aposira.forecasting.backend.core.optimization.safety_stock import calculate_safety_stock_from_history
from aposira.forecasting.backend.core.optimization.reorder_point import calculate_rop
from aposira.forecasting.backend.core.optimization.abc_analysis import calculate_abc_classification
from aposira.forecasting.backend.core.optimization.simulator import InventorySimulator
print('✓ All optimization modules loaded')

# Test metrics
from aposira.forecasting.backend.utils.metrics import calculate_forecast_metrics
print('✓ Metrics module loaded')

print('')
print('All backend modules loaded successfully!')

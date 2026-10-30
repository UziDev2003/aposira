# Test script to verify all modules import correctly
import sys
sys.path.insert(0, '.')

print("Testing imports...")

# Test database
from backend.database import Database
print('✓ Database module loaded')

# Test Pakistan calendar
from backend.core.pakistan_calendar import generate_pakistan_features
print('✓ Pakistan calendar loaded')

# Test data validator  
from backend.core.data_validator import DataValidator
print('✓ Data validator loaded')

# Test feature engineering
from backend.core.feature_engineering import create_features
print('✓ Feature engineering loaded')

# Test optimization modules
from backend.core.optimization.eoq import calculate_eoq
from backend.core.optimization.safety_stock import calculate_safety_stock_from_history
from backend.core.optimization.reorder_point import calculate_rop
from backend.core.optimization.abc_analysis import calculate_abc_classification
from backend.core.optimization.simulator import InventorySimulator
print('✓ All optimization modules loaded')

# Test metrics
from backend.utils.metrics import calculate_forecast_metrics
print('✓ Metrics module loaded')

print('')
print('All backend modules loaded successfully!')

"""
Data Validator Module
=====================
Validates CSV/JSON uploads against expected schema for Retail Forecasting.
Provides data quality reports with missing values, outliers, and anomalies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json


# Required columns for sales data
REQUIRED_SALES_COLUMNS = ["date", "sku", "store_id", "sales_qty", "unit_price", "promotion_flag"]

# Optional columns
OPTIONAL_SALES_COLUMNS = ["cost", "category", "subcategory", "supplier_lead_time"]

# Column data types
SALES_COLUMN_TYPES = {
    "date": "datetime",
    "sku": "string",
    "store_id": "string",
    "sales_qty": "numeric",
    "unit_price": "numeric",
    "promotion_flag": "numeric",
    "cost": "numeric",
    "category": "string",
    "subcategory": "string",
    "supplier_lead_time": "numeric"
}


class DataValidator:
    """Validates uploaded data files for Retail Forecasting."""
    
    def __init__(self):
        self.validation_results: Dict[str, Any] = {}
    
    def validate_sales_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate sales data DataFrame.
        
        Returns a dictionary with:
        - is_valid: bool
        - errors: list of error messages
        - warnings: list of warning messages
        - quality_report: detailed quality metrics
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_report": {}
        }
        
        df = self.normalize_columns(df)
        # Check required columns
        missing_cols = set(REQUIRED_SALES_COLUMNS) - set(df.columns)
        if missing_cols:
            results["is_valid"] = False
            results["errors"].append(f"Missing required columns: {missing_cols}")
            return results
        
        # Check for empty DataFrame
        if df.empty:
            results["is_valid"] = False
            results["errors"].append("DataFrame is empty")
            return results
        
        # Validate date column
        date_errors = self._validate_date_column(df)
        if date_errors:
            results["errors"].extend(date_errors)
            results["is_valid"] = False
        
        # Validate numeric columns
        numeric_errors = self._validate_numeric_columns(df)
        if numeric_errors:
            results["errors"].extend(numeric_errors)
            results["is_valid"] = False
        
        for col in ("sku", "store_id"):
            if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
                results["errors"].append(f"{col} must contain non-empty identifiers")
        if not date_errors:
            for key, group in df.groupby(["store_id", "sku"]):
                if group["date"].max() < group["date"].min() + pd.DateOffset(months=6):
                    results["errors"].append(f"{key}: at least six months of history required")
        if results["errors"]:
            results["is_valid"] = False
            return results

        # Generate quality report
        results["quality_report"] = self._generate_quality_report(df)
        
        # Check for warnings
        results["warnings"] = self._generate_warnings(df, results["quality_report"])
        
        return results
    
    def _validate_date_column(self, df: pd.DataFrame) -> List[str]:
        """Validate date column."""
        errors = []
        
        if "date" not in df.columns:
            return ["Date column not found"]
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        if df["date"].isna().any():
            return ["Date contains missing or invalid values"]

        # Check for future dates
        future_dates = df[df["date"] > pd.Timestamp.now()]
        if not future_dates.empty:
            errors.append(f"Found {len(future_dates)} future dates")
        
        # Check for very old dates
        old_dates = df[df["date"] < pd.Timestamp("2020-01-01")]
        if not old_dates.empty:
            errors.append(f"Found {len(old_dates)} dates before 2020")
        
        return errors
    
    def _validate_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Validate numeric columns."""
        errors = []
        
        for col in ["sales_qty", "unit_price", "promotion_flag"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if not np.isfinite(df[col]).all():
                errors.append(f"{col} must contain finite numbers without missing values")
            elif (df[col] < 0).any():
                errors.append(f"{col} cannot be negative")
        if not df["promotion_flag"].isin([0, 1]).all():
            errors.append("promotion_flag must be 0 or 1")
        return errors

    @staticmethod
    def normalize_columns(df):
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "price" in df.columns and "unit_price" not in df.columns:
            df = df.rename(columns={"price": "unit_price"})
        return df

    def _generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed data quality report."""
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "date_range": {
                "start": str(df["date"].min()) if "date" in df.columns else None,
                "end": str(df["date"].max()) if "date" in df.columns else None
            },
            "missing_values": {},
            "outliers": {},
            "unique_counts": {}
        }
        
        # Missing values percentage
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            report["missing_values"][col] = {
                "count": int(df[col].isnull().sum()),
                "percentage": round(missing_pct, 2)
            }
        
        # Outlier detection using IQR
        if "sales_qty" in df.columns:
            q1 = df["sales_qty"].quantile(0.25)
            q3 = df["sales_qty"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df["sales_qty"] < lower_bound) | (df["sales_qty"] > upper_bound)]
            report["outliers"]["sales_qty"] = {
                "count": len(outliers),
                "percentage": round((len(outliers) / len(df)) * 100, 2),
                "bounds": {"lower": lower_bound, "upper": upper_bound}
            }
        
        # Unique counts
        for col in ["sku", "store_id", "category"]:
            if col in df.columns:
                report["unique_counts"][col] = int(df[col].nunique())
        
        return report
    
    def _generate_warnings(self, df: pd.DataFrame, quality_report: Dict) -> List[str]:
        """Generate warnings based on quality report."""
        warnings = []
        
        # Check missing values
        for col, info in quality_report["missing_values"].items():
            if info["percentage"] > 5:
                warnings.append(f"Column '{col}' has {info['percentage']}% missing values")
        
        # Check outliers
        if "sales_qty" in quality_report["outliers"]:
            outlier_pct = quality_report["outliers"]["sales_qty"]["percentage"]
            if outlier_pct > 10:
                warnings.append(f"Sales quantity has {outlier_pct}% outliers - may indicate data issues or promotions")
        
        # Check date range
        if quality_report["date_range"]["start"] and quality_report["date_range"]["end"]:
            start = pd.to_datetime(quality_report["date_range"]["start"])
            end = pd.to_datetime(quality_report["date_range"]["end"])
            days = (end - start).days
            if days < 30:
                warnings.append(f"Date range is only {days} days - recommend at least 30 days for forecasting")
        
        return warnings
    
    def validate_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a CSV file."""
        try:
            df = pd.read_csv(file_path)
            return self.validate_sales_data(df)
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Failed to read CSV: {str(e)}"],
                "warnings": [],
                "quality_report": {}
            }
    
    def validate_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a JSON file."""
        try:
            df = pd.read_json(file_path)
            return self.validate_sales_data(df)
        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Failed to read JSON: {str(e)}"],
                "warnings": [],
                "quality_report": {}
            }
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        df = self.normalize_columns(df)
        for col in ("sales_qty", "unit_price", "promotion_flag"):
            df[col] = pd.to_numeric(df[col])
        # Convert date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Fill missing sales_qty with 0
        if "sales_qty" in df.columns:
            df["sales_qty"] = df["sales_qty"].fillna(0)
        
        # Fill missing prices with median
        if "unit_price" in df.columns:
            df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
        
        # Fill missing promotion flag with 0
        if "promotion_flag" in df.columns:
            df["promotion_flag"] = df["promotion_flag"].fillna(0)
        
        # Remove negative sales
        if "sales_qty" in df.columns:
            df = df[df["sales_qty"] >= 0]
        
        return df


def validate_file(file_path: Path) -> Dict[str, Any]:
    """Convenience function to validate a file."""
    validator = DataValidator()
    
    if file_path.suffix.lower() == ".csv":
        return validator.validate_csv_file(file_path)
    elif file_path.suffix.lower() == ".json":
        return validator.validate_json_file(file_path)
    else:
        return {
            "is_valid": False,
            "errors": [f"Unsupported file type: {file_path.suffix}"],
            "warnings": [],
            "quality_report": {}
        }
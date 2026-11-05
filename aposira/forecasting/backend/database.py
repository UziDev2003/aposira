"""
Database Module
===============
SQLite database connection and models for Retail Forecasting.
For MVP, we use SQLite with CSV file storage for simplicity.
PostgreSQL can be swapped in later via configuration.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

# Import config
from aposira.forecasting.backend.config import DATA_DIR, SAMPLE_DATA_DIR


class Database:
    """SQLite database manager for Retail Forecasting."""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = DATA_DIR / "retail_optima.db"
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection and create tables."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Users table (for future auth)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Stores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stores (
                store_id TEXT PRIMARY KEY,
                store_name TEXT NOT NULL,
                city TEXT,
                area TEXT,
                store_type TEXT,
                size_sqft INTEGER,
                opening_date TEXT
            )
        """)
        
        # Products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                sku_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT,
                subcategory TEXT,
                unit_cost REAL,
                unit_price REAL,
                supplier TEXT,
                lead_time_days INTEGER,
                min_order_qty INTEGER
            )
        """)
        
        # Sales records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                sku TEXT NOT NULL,
                store_id TEXT NOT NULL,
                sales_qty INTEGER NOT NULL,
                unit_price REAL,
                promotion_flag INTEGER DEFAULT 0,
                FOREIGN KEY (sku) REFERENCES products(sku_id),
                FOREIGN KEY (store_id) REFERENCES stores(store_id)
            )
        """)
        
        # Forecasts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                store_id TEXT NOT NULL,
                forecast_date TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                predicted_qty REAL NOT NULL,
                lower_bound REAL,
                upper_bound REAL,
                model_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sku) REFERENCES products(sku_id),
                FOREIGN KEY (store_id) REFERENCES stores(store_id)
            )
        """)
        
        # Inventory recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                store_id TEXT NOT NULL,
                eoq REAL,
                safety_stock REAL,
                reorder_point REAL,
                service_level REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sku) REFERENCES products(sku_id),
                FOREIGN KEY (store_id) REFERENCES stores(store_id)
            )
        """)
        
        # Model metrics table (JSON logging instead of MLflow)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                sku TEXT,
                store_id TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                store_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT,
                severity TEXT DEFAULT 'warning',
                is_resolved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sku) REFERENCES products(sku_id),
                FOREIGN KEY (store_id) REFERENCES stores(store_id)
            )
        """)
        
        self.conn.commit()
    
    def load_sample_data(self):
        """Load sample data from CSV files into the database."""
        # Load stores
        stores_df = pd.read_csv(SAMPLE_DATA_DIR / "sample_stores.csv")
        stores_df.to_sql("stores", self.conn, if_exists="replace", index=False)
        
        # Load products
        products_df = pd.read_csv(SAMPLE_DATA_DIR / "sample_products.csv")
        products_df.to_sql("products", self.conn, if_exists="replace", index=False)
        
        # Load sales
        sales_df = pd.read_csv(SAMPLE_DATA_DIR / "sample_sales.csv")
        sales_df.to_sql("sales", self.conn, if_exists="replace", index=False)
        
        print(f"Loaded {len(stores_df)} stores, {len(products_df)} products, {len(sales_df)} sales records")
    
    def get_sales_data(self, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       sku: Optional[str] = None,
                       store_id: Optional[str] = None) -> pd.DataFrame:
        """Get sales data with optional filters."""
        query = "SELECT * FROM sales WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if sku:
            query += " AND sku = ?"
            params.append(sku)
        if store_id:
            query += " AND store_id = ?"
            params.append(store_id)
        
        query += " ORDER BY date"
        
        return pd.read_sql(query, self.conn, params=params)
    
    def get_products(self) -> pd.DataFrame:
        """Get all products."""
        return pd.read_sql("SELECT * FROM products", self.conn)
    
    def get_stores(self) -> pd.DataFrame:
        """Get all stores."""
        return pd.read_sql("SELECT * FROM stores", self.conn)
    
    def save_forecast(self, forecast_df: pd.DataFrame):
        """Save forecast results to database."""
        forecast_df = forecast_df.rename(columns={"date": "forecast_date", "forecast": "predicted_qty"}).copy()
        if "horizon_days" not in forecast_df:
            forecast_df["horizon_days"] = len(forecast_df)
        columns = ["sku", "store_id", "forecast_date", "horizon_days", "predicted_qty", "lower_bound", "upper_bound", "model_type"]
        forecast_df[columns].to_sql("forecasts", self.conn, if_exists="append", index=False)
    
    def save_inventory_recommendation(self, recommendation: Dict[str, Any]):
        """Save inventory recommendation."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO inventory_recommendations 
            (sku, store_id, eoq, safety_stock, reorder_point, service_level, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            recommendation.get("sku"),
            recommendation.get("store_id"),
            recommendation.get("eoq"),
            recommendation.get("safety_stock"),
            recommendation.get("reorder_point"),
            recommendation.get("service_level"),
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def log_metric(self, model_type: str, metric_name: str, metric_value: float,
                   sku: Optional[str] = None, store_id: Optional[str] = None):
        """Log model metric (simple JSON-based logging)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_metrics (model_type, sku, store_id, metric_name, metric_value)
            VALUES (?, ?, ?, ?, ?)
        """, (model_type, sku, store_id, metric_name, metric_value))
        self.conn.commit()
    
    def get_metrics(self, model_type: Optional[str] = None) -> pd.DataFrame:
        """Get logged metrics."""
        query = "SELECT * FROM model_metrics"
        if model_type:
            query += " WHERE model_type = ?"
        query += " ORDER BY created_at DESC"
        return pd.read_sql(query, self.conn, params=[model_type] if model_type else [])
    
    def create_alert(self, sku: str, store_id: str, alert_type: str, 
                     message: str, severity: str = "warning"):
        """Create a new alert."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (sku, store_id, alert_type, message, severity)
            VALUES (?, ?, ?, ?, ?)
        """, (sku, store_id, alert_type, message, severity))
        self.conn.commit()
    
    def get_alerts(self, unresolved_only: bool = True) -> pd.DataFrame:
        """Get alerts."""
        query = "SELECT * FROM alerts"
        if unresolved_only:
            query += " WHERE is_resolved = 0"
        query += " ORDER BY created_at DESC"
        return pd.read_sql(query, self.conn)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Global database instance
db: Optional[Database] = None


def get_database() -> Database:
    """Get or create database instance."""
    global db
    if db is None:
        db = Database()
    return db
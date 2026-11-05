"""
Retail Forecasting - FastAPI Backend
===================================
Main entry point for the Retail Forecasting API.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from aposira.forecasting.backend.core.decision_service import forecast_sales, inventory_plan
from aposira.forecasting.backend.core.reports import export_table
from fastapi.responses import Response
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import config
from aposira.forecasting.backend.config import API_HOST, API_PORT, API_PREFIX, DATA_DIR, SAMPLE_DATA_DIR

# Import database
from aposira.forecasting.backend.database import Database, get_database

# Import modules
from aposira.forecasting.backend.core.data_validator import DataValidator
from aposira.forecasting.backend.core.optimization.eoq import calculate_eoq
from aposira.forecasting.backend.core.optimization.safety_stock import calculate_safety_stock_from_history
from aposira.forecasting.backend.core.optimization.reorder_point import calculate_rop_with_variability, check_reorder_needed
from aposira.forecasting.backend.core.optimization.abc_analysis import calculate_abc_classification, get_abc_summary
from aposira.forecasting.backend.core.optimization.simulator import InventorySimulator

# Create FastAPI app
app = FastAPI(
    title="APOSIRA Forecasting API",
    description="Predictive Demand Forecasting & Intelligent Inventory Optimization Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = get_database()


# ─── Data Models ─────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    sku: str = Field(min_length=1)
    store_id: str = Field(min_length=1)
    horizon: Literal[7, 14, 30] = 7
    model_type: Literal["seasonal_naive", "lightgbm", "prophet", "ensemble"] = "seasonal_naive"


class InventoryRecommendationRequest(BaseModel):
    sku: str
    store_id: str
    current_stock: float = Field(ge=0)
    service_level: float = Field(default=0.95, gt=0, lt=1)


class SimulationRequest(BaseModel):
    sku: str
    store_id: str
    initial_stock: float = Field(default=1000, ge=0)


# ─── Health Check ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Retail Forecasting API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ─── Data Endpoints ─────────────────────────────────────────────────────────

@app.post(f"{API_PREFIX}/upload/sales")
async def upload_sales_data(file: UploadFile = File(...)):
    """Upload and validate sales data."""
    try:
        # Read file
        contents = await file.read()
        
        if (file.filename or '').lower().endswith('.csv'):
            import io
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif (file.filename or '').lower().endswith('.json'):
            import io
            df = pd.read_json(io.StringIO(contents.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate data
        validator = DataValidator()
        validation_result = validator.validate_sales_data(df)
        
        if not validation_result["is_valid"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Validation failed", "details": validation_result["errors"]}
            )
        
        # Clean data
        df = validator.clean_data(df)
        
        # Save to database
        df[["date", "sku", "store_id", "sales_qty", "unit_price", "promotion_flag"]].to_sql("sales", db.conn, if_exists="append", index=False)
        
        return {
            "message": "Data uploaded successfully",
            "rows": len(df),
            "quality_report": validation_result["quality_report"]
        }
    
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/data/load-sample")
async def load_sample_data():
    """Load sample data into the database."""
    try:
        db.load_sample_data()
        return {"message": "Sample data loaded successfully"}
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/data/sales")
async def get_sales_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sku: Optional[str] = None,
    store_id: Optional[str] = None
):
    """Get sales data with filters."""
    try:
        df = db.get_sales_data(start_date, end_date, sku, store_id)
        return {"data": df.to_dict(orient="records"), "count": len(df)}
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/data/products")
async def get_products():
    """Get all products."""
    try:
        df = db.get_products()
        return {"data": df.to_dict(orient="records"), "count": len(df)}
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/data/stores")
async def get_stores():
    """Get all stores."""
    try:
        df = db.get_stores()
        return {"data": df.to_dict(orient="records"), "count": len(df)}
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Forecast Endpoints ─────────────────────────────────────────────────────

@app.post(f"{API_PREFIX}/forecast/generate")
async def generate_forecast(request: ForecastRequest):
    """Generate demand forecasts."""
    try:
        # Get sales data
        df = db.get_sales_data(sku=request.sku, store_id=request.store_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified SKU/store")
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        forecast, details = forecast_sales(df, request.horizon, request.model_type)

        # Save forecast to database
        forecast["sku"] = request.sku or df["sku"].iloc[0]
        forecast["store_id"] = request.store_id or df["store_id"].iloc[0]
        forecast["model_type"] = request.model_type
        db.save_forecast(forecast)
        
        return {
            "message": "Forecast generated successfully",
            "forecast": forecast.to_dict(orient="records"),
            "metrics": details["metrics"],
            "details": details
        }
    
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/forecast/results")
async def get_forecast_results(sku: str, store_id: str):
    """Get forecast results for a SKU-store combination."""
    try:
        query = """
            SELECT * FROM forecasts 
            WHERE sku = ? AND store_id = ?
            ORDER BY forecast_date DESC
            LIMIT 30
        """
        df = pd.read_sql(query, db.conn, params=(sku, store_id))
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No forecasts found")
        
        return {"data": df.to_dict(orient="records"), "count": len(df)}
    
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Inventory Endpoints ───────────────────────────────────────────────────

@app.post(f"{API_PREFIX}/inventory/recommendations")
async def get_inventory_recommendations(request: InventoryRecommendationRequest):
    """Get inventory recommendations for a SKU."""
    try:
        # Get sales data
        df = db.get_sales_data(sku=request.sku, store_id=request.store_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Get product info
        products = db.get_products()
        matching = products[products["sku_id"] == request.sku]
        if matching.empty:
            raise HTTPException(404, "Product costs are missing; add the product first")
        product = matching.iloc[0]
        plan = inventory_plan(df, request.current_stock, float(product["unit_cost"]),
                              lead_time=float(product.get("lead_time_days", 5)), service_level=request.service_level)
        recommendation = {"sku": request.sku, "store_id": request.store_id,
                          "service_level": request.service_level, **plan}
        if plan["status"] != "Normal":
            db.create_alert(request.sku, request.store_id, "low_stock",
                            f"Available stock {request.current_stock}; reorder point {plan['reorder_point']:.2f}",
                            plan["status"].lower())

        # Save to database
        db.save_inventory_recommendation(recommendation)
        
        return recommendation
    
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/inventory/alerts")
async def get_inventory_alerts():
    """Get low stock alerts."""
    try:
        alerts = db.get_alerts(unresolved_only=True)
        return {"data": alerts.to_dict(orient="records"), "count": len(alerts)}
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{API_PREFIX}/inventory/simulate")
async def simulate_inventory_policy(request: SimulationRequest):
    """Run inventory policy simulation."""
    try:
        # Get sales data
        df = db.get_sales_data(sku=request.sku, store_id=request.store_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Run simulation
        simulator = InventorySimulator()
        result = simulator.compare_policies(df, request.initial_stock)
        
        return result
    
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{API_PREFIX}/inventory/abc-analysis")
async def get_abc_analysis():
    """Get ABC analysis for all SKUs."""
    try:
        sales_df = db.get_sales_data()
        products_df = db.get_products()
        
        abc_df = calculate_abc_classification(sales_df, products_df)
        summary = get_abc_summary(abc_df)
        
        return {
            "summary": summary,
            "details": abc_df.head(50).to_dict(orient="records")
        }
    
    except HTTPException:
        raise
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Reports Endpoints ─────────────────────────────────────────────────────

@app.get(f"{API_PREFIX}/reports/export")
async def export_report(report_type: Literal["sales", "forecast", "inventory"] = "forecast",
                        format_type: Literal["CSV", "Excel", "PDF"] = "CSV",
                        sku: str = "", store_id: str = "",
                        start_date: Optional[str] = None, end_date: Optional[str] = None):
    tables = {"sales": ("sales", "date"), "forecast": ("forecasts", "forecast_date"),
              "inventory": ("inventory_recommendations", "last_updated")}
    table, date_column = tables[report_type]
    query = f"SELECT * FROM {table} WHERE sku=? AND store_id=?"
    params = [sku, store_id]
    if start_date:
        query += f" AND {date_column} >= ?"; params.append(start_date)
    if end_date:
        query += f" AND date({date_column}) <= date(?)"; params.append(end_date)
    frame = pd.read_sql(query, db.conn, params=params)
    if frame.empty:
        raise HTTPException(404, "No report records for these filters")
    try:
        data, mime, ext = export_table(frame, format_type)
    except ImportError as e:
        raise HTTPException(503, str(e))
    return Response(data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="aposira-{report_type}.{ext}"'})


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
# Retail Forecasting

Predictive Demand Forecasting & Intelligent Inventory Optimization tailored for Pakistani SME Retailers.

This project features a powerful AI/ML forecasting engine and a beautiful, interactive Streamlit dashboard to help retailers optimize their stock levels, reduce stockouts, and boost overall revenue.

## Features
- **Dashboard Overview**: Key metrics including total revenue, active SKUs, and sales trends.
- **Demand Forecasting**: Machine learning based sales predictions (LightGBM/Prophet) with interactive confidence intervals.
- **Inventory Management**: Reorder recommendations, ABC analysis, and urgency tracking.
- **Policy Simulator**: Compare current inventory policies with proposed optimizations to estimate cost savings.

## Setup & Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/usman2003/retail-forecasting.git
   cd retail-forecasting
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Models**:
   The AI/ML models are too large to be hosted on GitHub. Please download them from the following Google Drive link:
   [Download Models from Google Drive](https://drive.google.com/drive/folders/1fOfo2BMJwX-qTgus97g094N5c1qaLfZH?usp=sharing)
   
   Once downloaded, place the model files inside the `models/` directory in the root of your project folder.

## Running the Application

### 1. Dashboard (Streamlit)
To launch the interactive frontend dashboard, run the following command from the root directory:
```bash
python -m streamlit run dashboard/app.py
```
This will open the application in your default web browser (typically at `http://localhost:8501`).

### 2. Backend API (FastAPI)
*(If you wish to run the backend separately for API integrations)*
```bash
uvicorn backend.main:app --reload
```
The API documentation will be available at `http://localhost:8000/docs`.

## Data
Sample data is included in the `data/` directory to help you get started testing the application immediately. You can also upload your own historical sales and inventory data via the dashboard.

## APOSIRA integration

This module now runs inside the parent workspace's `aposira/app.py`. See the parent README for the connected POS, stock, orders, loyalty, delivery and Store Vision workflows. The dashboard now validates real uploads and produces downloadable reports instead of placeholder success messages. The parent `requirements.txt` and `requirements-ml.txt` define the unified environment; this repository's original pinned requirements are retained for its historical standalone setup.

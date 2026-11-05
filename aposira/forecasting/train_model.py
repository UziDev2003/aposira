import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

warnings.filterwarnings("ignore")

DATA_PATH="data"
MODEL_PATH="models"
os.makedirs(MODEL_PATH,exist_ok=True)

sales=pd.read_csv(f"{DATA_PATH}/sample_sales.csv")
products=pd.read_csv(f"{DATA_PATH}/sample_products.csv")
stores=pd.read_csv(f"{DATA_PATH}/sample_stores.csv")

sales["date"]=pd.to_datetime(sales["date"])

required=["date","sku","store_id","sales_qty","unit_price","promotion_flag"]
missing=[c for c in required if c not in sales.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

sales=sales.drop_duplicates()
sales=sales[sales.sales_qty>=0]
sales=sales[sales.unit_price>0]

df=(sales.groupby(["date","sku","store_id"],as_index=False)
          .agg({"sales_qty":"sum",
                "unit_price":"mean",
                "promotion_flag":"max"}))

# Fix: Merge with 'products' DataFrame using 'sku' from df and 'sku_id' from products
df=df.merge(products,left_on="sku",right_on="sku_id",how="left")
df=df.merge(stores,on="store_id",how="left")

# Calendar features
df["dayofweek"]=df.date.dt.dayofweek
df["month"]=df.date.dt.month
df["quarter"]=df.date.dt.quarter
df["weekofyear"]=df.date.dt.isocalendar().week.astype(int)
df["dayofyear"]=df.date.dt.dayofyear
df["is_weekend"]=(df.dayofweek>=5).astype(int)

# TODO: Pakistan holiday features
df["is_ramadan"]=0
df["is_eid"]=0
df["is_independence_day"]=((df.date.dt.month==8)&(df.date.dt.day==14)).astype(int)

# Handle 'opening_date' from stores: convert to datetime and extract year
if 'opening_date' in df.columns:
    df['opening_date'] = pd.to_datetime(df['opening_date'])
    df['store_open_year'] = df['opening_date'].dt.year

# Encode categoricals
# Ensure 'sku_id' is also converted to category if it will be used as a feature,
# or drop 'sku' and use 'sku_id' if only one is needed.
# For now, let's include both sku and store_id from the original aggregation.
for c in ["sku","store_id","sku_id", "category", "subcategory", "brand", "city", "area", "store_type", "supplier"]:
    if c in df.columns:
        df[c]=df[c].astype("category")

df=df.sort_values(["sku","store_id","date"])

for lag in [1,7,14,28]:
    df[f"lag_{lag}"]=df.groupby(["sku","store_id"])["sales_qty"].shift(lag)

for w in [7,14,28]:
    grp=df.groupby(["sku","store_id"])["sales_qty"]
    df[f"roll_mean_{w}"]=grp.transform(lambda x:x.shift(1).rolling(w,1).mean())
    df[f"roll_std_{w}"]=grp.transform(lambda x:x.shift(1).rolling(w,1).std())
    df[f"ewm_{w}"]=grp.transform(lambda x:x.shift(1).ewm(span=w).mean())

df=df.dropna()

cut=df.date.quantile(.8)
train=df[df.date<=cut]
test=df[df.date>cut]

# Update ignored columns to reflect the new sku_id column, and other merged columns
# Also add 'opening_date' to ignore as we extracted 'store_open_year' from it
ignore=["date","sales_qty", "sku_id", "product_name", "store_name", "opening_date"]
features=[c for c in df.columns if c not in ignore]

X_train=train[features]
y_train=train.sales_qty
X_test=test[features]
y_test=test.sales_qty

cat=[c for c in X_train.columns if str(X_train[c].dtype)=="category"]

model=LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=.8,
    colsample_bytree=.8,
    random_state=42
)

model.fit(
    X_train,y_train,
    categorical_feature=cat,
    eval_set=[(X_test,y_test)],
    callbacks=[early_stopping(100,verbose=False)]
)

pred=model.predict(X_test)

metrics={
    "MAE":float(mean_absolute_error(y_test,pred)),
    "RMSE":float(np.sqrt(mean_squared_error(y_test,pred))),
    "MAPE":float(mean_absolute_percentage_error(y_test,pred)*100)
}

print(metrics)

with open(f"{MODEL_PATH}/lightgbm.pkl","wb") as f:
    pickle.dump(model,f)

with open(f"{MODEL_PATH}/training_metrics.json","w") as f:
    json.dump(metrics,f,indent=2)

fi=pd.DataFrame({
    "feature":features,
    "importance":model.feature_importances_
}).sort_values("importance",ascending=False)
fi.to_csv(f"{MODEL_PATH}/feature_importance.csv",index=False)

print("Training complete.")

# Optional PRD extensions:
# - Prophet baseline
# - LightGBM quantile models (0.1,0.5,0.9)
# - SHAP explainability
# - Inventory simulation
# - EOQ / Safety Stock / Reorder Point
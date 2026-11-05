# APOSIRA System Design

Version 1.0 | 4 November 2026 | Proposed product baseline

## Architecture baseline

The current system is a modular Python application. Streamlit renders a shared workspace and calls commerce services directly. Forecasting and vision now live inside the aposira package in the local working tree. This package move is staged and is not assumed to be on the remote main branch. The design below describes the inspected working tree rather than an aspirational microservice deployment.

| Component | Responsibility and boundary |
| --- | --- |
| aposira/app.py and ui.py | Navigation, presentation, session state and organization/branch selection. The UI is not an authorization boundary. |
| aposira/commerce.py | SQLite transactions, catalogue, customers, ledger, reservations, orders, loyalty and audit. |
| aposira/forecasting/backend/core | Validation, single-series forecasting, evaluation, inventory calculations, policy simulation and report generation. |
| aposira/forecasting/dashboard/app.py | Embedded forecasting UI; consumes completed POS sales or explicitly uploaded/sample history. |
| aposira/vision/streamlit_app.py and src | Video/RTSP input, detector, tracker, zones, behavior analytics and event persistence. |
| aposira/vision_events.py | Scoped timeline, filtering, CSV export and acknowledgement UI. Acknowledgement display mismatch is a known defect. |
| aposira/forecasting/backend/main.py | Separate legacy FastAPI service and database. Does not enforce production tenant access. |

## Two connected pipelines

Business flow: cart validation → order transaction → stock movement and loyalty on completion → sales history → validated daily series → forecast and inventory review. Non-POS orders reserve inventory before completion. Current inventory_plan uses historical daily demand; it does not automatically consume forecast output to execute purchase orders.

Physical-store flow: video/RTSP → frame processing → YOLO detections → ByteTrack tracks → zone and behavior measurements → scoped events and evidence references → operator review. Shelf/product conversion is a future integration, not a current link from anonymous tracks to purchases.

## Commerce and event data contracts

| Entity and keys | Fields and invariants |
| --- | --- |
| products — org plus sku | name; price and cost in integer paisa. Product catalogue is organization-scoped, not duplicated per branch. |
| customers — org plus customer | Customer ID and name. Contact consent, account identity and marketing attributes are planned. |
| movements — id | org, branch, sku, signed quantity, reason, reference, created_at. Unique org/branch/sku/reference; update and delete blocked by triggers. |
| orders — id | org, branch, source, customer, JSON items, total, payment, status, rider, proof, created_at. Item name and price snapshot preserve the sold value. |
| loyalty — id | org, customer, points, globally unique order reference. Completion awards once; reversal/redemption ledger is planned. |
| audit — id | org, branch, action, reference, detail and created_at. Order actions only; no authenticated actor field yet. |
| events — id | organization_id, store_id, epoch timestamp, camera_id, alert_type, severity_score, JSON payload and acknowledged_at. Indexed by scope/time. |

Logical links exist between orders, products, customers and movements, but the commerce schema does not declare foreign keys. Do not describe these logical relationships as database-enforced referential integrity. BEGIN IMMEDIATE serializes writer transactions; production concurrency must be measured before adopting this storage design at scale.

## Derived state and time semantics

On-hand is the sum of signed movements. Reserved quantity comes from order items whose status is neither completed nor cancelled, including failed delivery. Available stock is on-hand minus reserved. Completion inserts negative sale movements; cancellation releases reservation without a sale movement.

Commerce sales currently use order created_at, including delayed fulfillment, and promotion_flag is zero. A sale created on one day and completed later is therefore attributed to creation day. Production financial reporting needs a documented completed_at/business-day policy and migration. Orders have no separate payment-settlement timestamp.

## Storage locations

Commerce: data/aposira.db. Vision: aposira/vision/data/events.sqlite3 with uploads and runtime zone paths scoped by workspace helpers. Forecast API: aposira/forecasting/data/retail_optima.db; standalone inventory ledger: aposira/forecasting/data/stock_ledger.db. Generated files, snapshots, clips and trained weights need a separate retention and backup policy.

## Interfaces and input schema

The combined UI calls Python services; it is not a client of a complete commerce REST API. The following are existing legacy forecasting routes under /api/v1 unless stated otherwise. Keep them private until authenticated authorization and operational controls are implemented.

| Method and path | Contract and limit |
| --- | --- |
| GET /health | Service health response; does not prove database recovery or model readiness. |
| POST /upload/sales | Multipart file upload; validated CSV/JSON records. Required normalized fields: date, sku, store_id, sales_qty, unit_price, promotion_flag. |
| POST /forecast/generate | sku, store_id, horizon 7/14/30, model_type seasonal_naive/lightgbm/prophet/ensemble. Returns forecasts and metrics. |
| GET /forecast/results | sku and store_id select stored results. No authenticated tenant contract. |
| POST /inventory/recommendations | sku, store_id, nonnegative current_stock and service_level between zero and one. |
| POST /inventory/simulate | sku, store_id and nonnegative initial_stock. Historical policy comparison only. |
| GET /reports/export | Report type sales/forecast/inventory and export format; verify current OpenAPI for optional filters. |
| GET /data/load-sample | Mutates sample data despite GET semantics. Do not expose publicly; target design should use an authorized administrative command. |

The API also exposes data/sales, data/products, data/stores, inventory/alerts and inventory/abc-analysis. Use generated OpenAPI from the actual running revision as the authoritative request schema. Proposed commerce, identity, connector and webhook APIs must be versioned separately; these documents do not invent implemented endpoints.

## Failure and migration design

Import and prediction failures must return clear validation errors; missing dependencies must not yield fabricated forecasts. Long-running jobs need cancellation, bounded resource usage and observable status before a pilot. Retryable external operations require idempotency and durable correlation IDs.

For a production migration, preserve original IDs and item snapshots, convert monetary fields without floating-point rounding, reconcile totals per branch, validate reservations and loyalty, and compare event counts. Run a dry restore and migration first. Migration from old top-level module data paths to the reorganized paths needs explicit handling; moving code alone does not migrate runtime data.

## Decisions and constraints

Keep SQLite and Streamlit for R0. Resolve identity, storage, model/tracker substitutions and deployment envelope before R1. A private GitHub repository does not authenticate the publicly exposed Streamlit application. RunPod proxy origin configuration is an application connectivity control, not user access control.

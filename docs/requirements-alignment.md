# APOSIRA requirements alignment

The workspace combines the three supplied sources in one local application. Product descriptions and requirements in the reports are treated as specifications, not as authorization to publish, contact others, or make external account changes. The source documents remain unchanged.

## Sources and precedence

- **Final Year Project G1.docx**: forecasting and inventory SRS, especially sections 2.4 and 2.5. This describes Retail Optima AI, 7/14/30-day forecasts, six-month history, accuracy metrics, EOQ, safety stock, ABC analysis and exports.
- **Final Project Report G2.docx**: store vision SRS, especially sections 2.4 and 2.5. This describes detection/tracking, zones, dwell, congestion, suspicious activity, evidence and a retained timeline.
- **Initial Report (FYP - II).docx**: the larger APOSIRA product model, shared organization/branch context, inventory ledger and business/physical-store pipelines.
- **https://aposira.vercel.app/**, inspected 18 October 2026: APOSIRA branding, Pakistan-focused positioning, six product areas, and explicit under-development status.

The initial report supplies the umbrella product structure. G1 and G2 supply the requirements for the existing specialist modules. Claimed accuracy, latency, scale and availability are acceptance targets; they are not established measurements of this code.

## Implemented in this update

| Requirement | Implementation and limits |
|---|---|
| One connected APOSIRA application | Shared navigation and organization/branch context; embedded forecasting and camera pages, rather than separate dashboards to launch manually. |
| Product catalogue | Organization-scoped SKU, name, integer-paisa selling price and cost. Variants, barcodes, expiry and suppliers are not complete. |
| Ledger-based inventory | Permanent stock movements; no UPDATE/DELETE path. Checkout writes sale movements atomically. Negative available stock is rejected. Unique references prevent replay. |
| POS → inventory → customer → analytics | Multi-item cart, cash/external payment recording, customer attachment, receipt download, loyalty accrual and sales history. External payment is manual confirmation, not a gateway charge. |
| Order and delivery workflow | Reservations, acceptance, picking, packing, ready, dispatch, completion, failure and cancellation transitions. Rider/proof/COD checks. No GPS or platform integration. |
| Customer loyalty | Scoped customer registration and one point per PKR 100 on completed orders; no redemption/refunds yet. |
| G1 REQ-SF1-1 | CSV/JSON validation; supports SKU/price aliases; rejects invalid numeric values, missing dates/IDs, non-binary flags and histories shorter than six months per SKU/store. |
| G1 REQ-SF1-2 | 7/14/30 days, one SKU/store per forecast. Existing LightGBM and Prophet engines retained, plus explicit seasonal baseline and fixed-weight ensemble. ARIMA and LSTM remain unimplemented. |
| G1 REQ-SF1-3 | Separate final 30-day holdout, MAE/RMSE/MAPE, nominal 95% empirical residual intervals. Statistical coverage needs evaluation on representative retail data. |
| G1 REQ-SF1-4 | MAPE >15% review signal and model training during generation. No scheduler or automatic retraining job. |
| G1 REQ-SF2-1/2/3 | EOQ, safety stock, reorder point, available stock and low-supply warnings using actual stored stock and explicit costs. No hardcoded inventory examples. |
| G1 REQ-SF2-4 | Revenue-based ABC; top item belongs to A even if it crosses the 80% threshold. Percentages do not force item counts. |
| G1 REQ-SF3-4 | Actual CSV, Excel and PDF downloads; sales and ABC date filtering. Forecast/inventory/simulation exports use the current saved analysis, not date-range recalculation. |
| Inventory policy simulator | Uses prior observations, receives each order once, and calculates shortages from unmet demand. Default proposed policy is statistical; no invented ML savings. |
| G2 existing perception/analytics | YOLO/ByteTrack, zone editor, heatmaps, dwell and suspicious activity pipeline remain accessible inside APOSIRA. Hardware/model-specific performance still needs validation. |
| G2 REQ-SF4-2 | SQLite event metadata retained for 30 days; camera/date/message filters; acknowledgements persist across restart. Snapshot retention is separate. |
| G2 report export | Filtered event CSV. Full vision PDF analytics exports and WebSocket push remain pending. |
| Branch separation | Commerce queries, event records, video upload folders and runtime camera zone folders scoped to organization/branch. This does not provide authentication or authorization. |
| Audit trail | Order creation and status changes recorded in the same transaction; stock movements retain reasons/references. User attribution and general configuration audit remain pending. |

## Outstanding platform work

Production identity and role enforcement, API authorization/rate limits, encrypted managed storage, backup/restore, PostgreSQL/TimescaleDB migration, supplier purchasing, stock transfers, tax/shift/refund workflows, receipt printing, POS offline synchronization, customer storefront, loyalty redemption, social commerce, payment providers, marketplace integrations, delivery routing/GPS and cash reconciliation require additional implementation and service configuration.

Shelf product counting, gaps/misplacement detection, zone-to-SKU mapping, interest-to-purchase conversion, external weather/trend signals and pest-risk estimates require appropriate input data, trained models, evaluation and integration. Existing suspicious shelf interaction detections are not equivalent to product-level shelf recognition.

The G2 unattended-object detector is not the same as a staff-absence-at-checkout detector. DeepSORT, validated 1–16 simultaneous camera streams, hardware benchmarks, 95% detection accuracy, <5% false positives and subsecond alert targets are not claimed as achieved. No deployment to the marketing site's Vercel project was made because its source/deployment configuration is not present in this workspace.

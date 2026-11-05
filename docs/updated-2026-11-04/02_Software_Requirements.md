# APOSIRA Software Requirements

Version 1.0 | 4 November 2026 | Proposed product baseline

## Specification purpose

This SRS defines the combined APOSIRA target and its current implementation gaps. Each AP identifier is stable across the documentation pack. Shall statements define desired behavior; the status column records inspected implementation, not approval or successful acceptance.

## Status and priority conventions

| Term | Meaning |
| --- | --- |
| Implemented | Code provides the described R0 behavior; acceptance and deployment limitations still apply. |
| Partial | Some behavior exists, but the original or target requirement is not fully met or a known defect remains. |
| Planned | No complete implementation was established in the inspected working tree. |
| P0 | Integrity or access gate; must pass before a controlled pilot. |
| P1 | Core functionality or pilot readiness. |
| P2 | Later expansion after core gates. |

R0, R1, R2 and R3 are defined in the PRD. Planned security work may be P0 even though it is not in the demonstration. All source IDs are namespaced G1 or G2 because both reports reuse REQ-SF identifiers.

## System boundaries

R0 is a Streamlit application with direct Python service calls and SQLite files. The legacy forecasting FastAPI service is separate and unauthenticated. There is no production commerce API contract, identity service or external payment processor in the inspected baseline.

## Conventions

Money is displayed in PKR and persisted in integer paisa for commerce. Forecast quantities are numeric units and must not be presented as currency. Event timestamps and SQLite creation timestamps are UTC; business timezone handling needs a production policy. Anonymous tracking IDs are temporary analytical identifiers and are not proof of identity.

Source keys: G1 = Final Year Project G1.docx, Chapter 2; G2 = Final Project Report G2.docx, Chapter 2; I = Initial Report (FYP - II).docx, named module headings. Source performance figures are targets, not verified results.

## Functional requirements 1 to 4

## AP 01 Scope context

P0 | Partial | Source: I Organization and Store Management

The platform shall scope operations by organization and, where applicable, branch.

Acceptance: Products and customers are shared within an organization; orders, movements and events are branch-scoped. Switching branch clears prior cart and camera session state. Authorized membership enforcement is required for R1.

## AP 02 Identity and access

P0 | Planned | Source: I Authentication Roles and Security; G1 2.5.3; G2 2.5.3

The pilot shall authenticate users and enforce role and branch permissions on every protected operation.

Acceptance: Deny anonymous and unauthorized access server-side; test session expiry, logout, recovery, administrator MFA and cross-tenant object IDs. Provider choice is pending.

## AP 03 Product catalogue

P1 | Partial | Source: I Product and Catalogue Management

The platform shall maintain unique organization SKU, name, selling price and unit cost.

Acceptance: Reject blank identifiers and negative prices; store monetary values as integer paisa. Variants, barcodes, taxes, units, batches, expiry and channel prices require later acceptance cases.

## AP 04 Stock ledger

P0 | Implemented | Source: I Inventory and Warehouse Management

Stock changes shall append signed movements with quantity, reason and unique reference.

Acceptance: Reject zero/noninteger quantity, unknown SKU and reductions below reservations. Repeating the same reference and payload is a no-op; a different payload fails. SQL UPDATE/DELETE of movements fails.

## Functional requirements 5 to 8

## AP 05 Reservations

P0 | Implemented | Source: I Inventory and Warehouse Management; Omnichannel Order Management

Uncompleted, uncancelled orders shall reserve stock and reduce available-to-sell quantity.

Acceptance: On-hand equals signed movement sum; available equals on-hand minus active reservations. Two competing orders cannot oversell. Failed delivery retains reservation until completion or cancellation.

## AP 06 Checkout

P0 | Partial | Source: I Advanced Smart POS

Checkout shall validate each cart item and commit order, stock deduction, loyalty and audit atomically for a completed POS sale.

Acceptance: Empty carts, nonpositive quantities, insufficient stock and POS COD fail without partial writes. Retry returns the original order only when its original payload matches. Advanced POS features are AP-27.

## AP 07 Payment record

P0 | Partial | Source: I Advanced Smart POS Payment methods

R0 shall record cash, externally paid or COD arrangements without storing payment-card secrets.

Acceptance: Require operator confirmation in the UI; reject COD for immediate POS checkout. External paid means a recorded external payment, not gateway verification. Signed provider confirmation is required in R2.

## AP 08 Order fulfillment

P0 | Partial | Source: I Omnichannel Order Management; Delivery Management

Orders shall progress through the permitted state machine and maintain reservations until resolved.

Acceptance: Reject skipped states. Dispatch requires rider text; completion from dispatched requires proof; COD completion requires collection confirmation. GPS, rider accounts and cash settlement remain planned.

## Functional requirements 9 to 12

## AP 09 Customer loyalty

P1 | Partial | Source: I Customer and Loyalty Management

The platform shall register organization-scoped customers and accrue points on completed orders.

Acceptance: Unknown customer is rejected at checkout. Award floor(total paisa / 10000) points once per order. Redemption, tiers, consent, segmentation, refunds and rewards remain planned.

## AP 10 Audit and attribution

P0 | Partial | Source: I Administration Reliability and System Operations; G1 2.5.3

State-changing business actions shall be auditable with scope, time, reference and actor.

Acceptance: Existing order create/transition records are transactional. R1 must add authenticated actor, override justification, configuration/catalogue actions and access-controlled retention; stock reasons alone do not provide complete audit coverage.

## AP 11 Sales import

P1 | Implemented | Source: G1 REQ-SF1-1 and 2.5.5

Forecast intake shall accept CSV/JSON with date, sku, store_id, sales_qty, unit_price and promotion_flag, supporting SKU/price aliases.

Acceptance: Reject invalid dates, missing identifiers, negative numeric values and nonbinary flags. Validate six calendar months per group for forecasting. Rejected import must not overwrite usable history.

## AP 12 Daily history

P1 | Implemented | Source: G1 2.2.7 and 2.5.5

Forecasting shall operate on a single SKU/store with continuous daily observations.

Acceptance: Aggregate same-day sales; do not silently fill unknown dates with zero. Require explicit zero-sales rows. Reject mixed groups and history shorter than six months. Report source and selected group.

## Functional requirements 13 to 16

## AP 13 Forecast generation

P1 | Partial | Source: G1 REQ-SF1-2

The product shall generate nonnegative forecasts for 7, 14 or 30 days using the approved model baseline.

Acceptance: Current seasonal naive, LightGBM, Prophet and equal-weight ensemble produce dates after the last observation. ARIMA/LSTM specified in G1 are absent; revision requires approval. Missing model dependency produces a clear error.

## AP 14 Model evaluation

P1 | Partial | Source: G1 REQ-SF1-3

Every forecast shall show holdout MAE/RMSE/MAPE and clearly labeled nominal uncertainty intervals.

Acceptance: Keep final 30 days out of evaluation training. MAPE excludes zero actuals and is unavailable if all are zero. Current intervals use residual quantiles; measured 95% coverage is not established.

## AP 15 Retraining and review

P1 | Partial | Source: G1 REQ-SF1-4

The system shall flag poor model quality and provide a controlled retraining workflow.

Acceptance: Current generation trains on demand and warns for MAPE above 15% or unavailable. Scheduled change detection, durable model versions, automated retraining and rollback need implementation and failure/replay tests.

## AP 16 Inventory decisions

P1 | Partial | Source: G1 REQ-SF2-1 to REQ-SF2-3 and REQ-SF3-2

The system shall compute EOQ, safety stock, reorder point and days of supply from history, stock and explicit costs.

Acceptance: Reject invalid/nonfinite costs and lead times; flag stock below reorder point or under seven days supply. Zero demand returns no numeric days-of-supply. Promotion/overstock-risk alerts are not complete.

## Functional requirements 17 to 20

## AP 17 ABC classification

P1 | Partial | Source: G1 REQ-SF2-4 and 2.5.5

ABC classification shall rank SKU revenue using cumulative value thresholds of 80% and 95%.

Acceptance: Top SKU remains A even when crossing 80%; item percentages are measured, not forced. Quarterly scheduling is pending. Record this interpretation as a clarification of the original fixed item-share wording.

## AP 18 Scenario analysis

P2 | Partial | Source: G1 REQ-SF3-3

Decision support shall distinguish historical inventory-policy simulation from price or promotion scenario modeling.

Acceptance: Current simulator counts each arrival once and unmet demand as shortage, using prior data for default rules. Price elasticity, promotional counterfactuals and causal savings claims require separate implementation and validation.

## AP 19 Reporting

P1 | Partial | Source: G1 REQ-SF3-1 and REQ-SF3-4; I Unified Analytics

Reports shall identify scope, period, units, source and model where relevant and export real CSV, XLSX or PDF.

Acceptance: Verify signatures and readable output. Sales/ABC filters apply date ranges; saved forecast/plan/simulation exports use saved analysis. Full turnover charts and configurable branding remain pending.

## AP 20 Video perception

P1 | Partial | Source: G2 REQ-SF1-1 to REQ-SF1-4

Vision shall process authorized video/RTSP sources and pass timestamped boxes and anonymous track IDs to analytics.

Acceptance: Current YOLO/ByteTrack implementation must pass camera disconnect and model failure tests. DeepSORT, 1–16 stream capacity, 95% mAP and MOTA targets are unverified or unmet as specified.

## Functional requirements 21 to 24

## AP 21 Zones and movement

P1 | Partial | Source: G2 REQ-SF2-1 to REQ-SF2-4

Users shall configure camera polygons and review zone occupancy, dwell and heatmaps.

Acceptance: Persist camera zones within scope and reject invalid polygons. Test known trajectories at boundaries. Fixed 15-minute/hour/day aggregation and 90% dwell accuracy require separate evaluation.

## AP 22 Activity alerts

P1 | Partial | Source: G2 REQ-SF3-1 to REQ-SF3-4

Vision shall produce reviewable loitering, restricted-zone and shelf-interaction observations using configurable thresholds.

Acceptance: Evaluate threshold crossings and cooldown on labeled clips. Default source targets include 120-second loitering and 30-second shelf interaction. Dedicated unattended-staff checkout after 60 seconds is planned.

## AP 23 Event timeline

P1 | Partial | Source: G2 REQ-SF4-2

Events shall support scoped date/camera/text filters and durable acknowledgement with 30-day metadata retention.

Acceptance: Storage acknowledgement must survive reopening. UI must read acknowledged_at consistently; current UI reads acknowledged and needs correction. Query excludes expired events; pruning occurs on initialization/save, not a timer.

## AP 24 Evidence and delivery

P1 | Partial | Source: G2 REQ-SF4-1 REQ-SF4-3 REQ-SF4-4

Alerts shall expose available evidence and export authorized event records; real-time delivery shall be measured.

Acceptance: Filtered CSV exists. Snapshot/video retention, full vision PDF reports, push delivery under 500 ms and simultaneous-user access require implementation or validation. Streamlit WebSocket transport alone is not an event delivery SLA.

## Functional requirements 25 to 28

## AP 25 Supplier purchasing

P2 | Planned | Source: I Supplier and Purchase Management

The platform shall manage suppliers, approvals, purchase orders, receiving, costs, returns and balances.

Acceptance: Partial receipts create only the received stock movement; duplicate receipts do not duplicate inventory. Recommended orders require authorized human approval and reconciliation to supplier invoice.

## AP 26 Warehouses and transfers

P2 | Planned | Source: I Inventory and Warehouse Management

The platform shall support locations, stock counts, transfer states, batches, expiry and valuation.

Acceptance: A transfer records paired source/destination movements with in-transit status and exception handling. Count corrections preserve history; damaged/expired stock cannot silently become sellable.

## AP 27 Advanced POS

P2 | Planned | Source: I Advanced Smart POS

POS shall support approved discounts, shifts, receipts, returns, hardware and offline reconciliation as separate capabilities.

Acceptance: Manager override is attributed; returns cannot exceed original sold quantity; offline replay is idempotent. Barcode/scale/printer compatibility and tax calculations require explicit device and jurisdiction acceptance.

## AP 28 External orders and delivery

P2 | Planned | Source: I Omnichannel Order Management; Delivery Management

Official connectors shall import orders, synchronize availability and reconcile delivery/payment states.

Acceptance: Authenticate webhook origin, deduplicate provider event IDs, handle out-of-order updates and record settlement differences. Merchant API access is a prerequisite; do not infer it from a provider name.

## Functional requirements 29 to 32

## AP 29 Storefront and social commerce

P2 | Planned | Source: I Web Mobile App and Social Commerce

Customer channels shall offer catalogue, checkout, order status and permitted communications.

Acceptance: Public customers cannot see another account or branch inventory beyond policy. Price, stock and consent are validated server-side. Social publishing and campaign attribution require approved APIs and opt-in rules.

## AP 30 Shelf and interest intelligence

P2 | Planned | Source: I Shelf and Customer Interest Intelligence

The platform shall relate anonymous zone/product interactions to shelf observations and aggregate sales.

Acceptance: Define zone-to-SKU mappings and labeled product-facing data; measure gaps/misplacement precision. Aggregate interest-to-purchase with time/window rules; never equate anonymous tracks with identified customers.

## AP 31 Context signals and risk

P2 | Planned | Source: I Forecasting and Decision Intelligence

Approved weather, holiday, trend and sensor inputs shall be versioned, time-aligned and evaluated before influencing recommendations.

Acceptance: Use only information available at prediction time. Missing feeds degrade gracefully. Pest outputs are inspection-risk recommendations, never confirmation. Staffing, delivery and cash-flow forecasts need separate targets.

## AP 32 Platform administration

P1 | Planned | Source: I Organization and Store Management; Administration Reliability and System Operations

Administrators shall manage branch configuration, devices, feature flags, credentials, retention and service health.

Acceptance: Scope settings to authorized tenants; redact credentials; log changes. Subscription plans, business registration, operating hours, warehouses and device registration are not supplied by free-text selectors.

## Nonfunctional requirements

| ID | Target and verification |
| --- | --- |
| NFR-01 Integrity | P0. All transaction failures leave no partial checkout effects; replay and concurrent stock tests pass. Current transaction mechanisms exist. |
| NFR-02 Access | P0. Authenticated tenant/role enforcement and secure credential handling before R1; currently planned. Source JWT expiry is 15 minutes and API rate limit is 100 requests/min/user in G1; confirm provider design. |
| NFR-03 Forecast performance | G1 targets: dashboard p95 under 3 s, API p95 under 500 ms, 10,000 SKU forecasts under 5 min; 100 concurrent chain users and 1,000 platform users. Unbenchmarked. |
| NFR-04 Vision performance | G2 targets: at least 25 FPS, 95% mAP, MOTA at least 85%, restricted-zone alert under 1 s, all alert latency under 2 s, delivery under 500 ms. Define hardware and metric thresholds before validation. |
| NFR-05 Reliability | G1 99.5% and G2 99.9% monthly availability; G2 MTTR under 4 h. No achieved SLA. Measure services separately and define exclusions before pilot. |
| NFR-06 Recovery | G1 daily business-data backup; G2 hourly event/snapshot backup. Proposed R1 restore objective 4 h; approve RPO/RTO and prove an isolated restore. No automated backup confirmed. |
| NFR-07 Privacy | No facial recognition requirement. Restrict footage and snapshots; encryption, consent/notice, minimization and retention controls are production requirements, not compliance claims. |
| NFR-08 Usability and support | English and PKR default; clear errors, keyboard-accessible workflows and non-color status labels. G1 onboarding under 10 min, proposed accessibility target WCAG 2.2 AA; evaluation pending. |

Numeric source targets are retained for traceability. Benchmark results must disclose dataset, hardware, network, model, resolution, concurrency, duration and percentile. A successful smoke test does not establish any of these targets.

## Requirement reconciliations

| Conflict | Proposed baseline and remaining decision |
| --- | --- |
| Models and tracker | Current LightGBM/Prophet and ByteTrack differ from G1 ARIMA/LSTM and G2 DeepSORT. Preserve partial status until the owner approves substitutions or implementation meets the originals. |
| Data minimum | Use six calendar months of daily observations for forecasts. G1 later refers to 30-day aggregate data for another use case; that does not lower the forecast minimum. |
| Intervals and accuracy | Call the implemented band a nominal prediction interval, not a measured confidence guarantee. Define mAP IoU convention and dwell error tolerance before using source accuracy claims. |
| ABC shares | Use cumulative value boundaries; item shares are descriptive outcomes. Quarterly automatic recalculation is still required by G1 but absent. |
| Alert deadlines | Security acknowledgement target is 5 min in G2; critical inventory confirmation/escalation is 24 h in G1. These are distinct workflows, both need enforcement. |
| Retention and storage | 30-day vision metadata does not mean all footage is retained or deleted. G1 proposes 7-year transactions and 90-day override audit; approve policy before implementation. G2 500 GB/camera is an estimate, not a guarantee. |
| Architecture and UI | Streamlit/SQLite is R0; PostgreSQL/TimescaleDB, Docker orchestration and production APIs are targets. A shared APOSIRA theme replaces standalone dark/light styling for the combined UI. |
| Capacity | G2 says 16 cameras/node and 100-node cluster but also 1,000 cameras in its quality table. Select a tested deployment envelope; do not multiply claims into a promise. |

## Approval and maintenance

The product owner must resolve model/tracker substitutions, retention, identity provider and release scope. The technical lead must define metric protocols and infrastructure. QA must link every accepted AP/NFR item to evidence and record failed or deferred cases. Changes to this baseline require a version increment and updated traceability; no approval is implied by document creation.

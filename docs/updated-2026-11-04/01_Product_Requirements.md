# APOSIRA Product Requirements

Version 1.0 | 4 November 2026 | Proposed product baseline

## Purpose and product direction

APOSIRA combines retail operations, demand forecasting and store vision in one product. The first release should demonstrate a reliable path from a recorded sale to stock, loyalty and analytics, alongside a camera event timeline. Production rollout must follow identity, authorization, recovery and measurement work; the Streamlit prototype is not a production retail service.

This PRD is for the project owner, academic reviewers, product designers and developers. It defines what to build and why. The companion SRS defines testable requirements; the design, use cases, verification plan and operations guide explain how the release will be built and assessed.

## Problem and intended users

Retailers working across separate POS, spreadsheets and CCTV tools cannot easily reconcile sales, available inventory, customer history and physical store events. APOSIRA is intended to provide a shared organization and branch context with explainable recommendations that staff can review before acting.

| User group | Primary outcome |
| --- | --- |
| Owner and store manager | Review sales, inventory health and exceptions across authorized branches. |
| Cashier and inventory operator | Complete accurate sales, receive stock and preserve an auditable movement history. |
| Fulfillment staff and rider | Progress reserved orders and confirm delivery or collection. |
| Security officer and operations lead | Review camera observations, acknowledge events and assess store traffic. |
| Data administrator and system administrator | Validate history, evaluate models, configure access and recover service. |

## Baseline and scope of evidence

The inspected source is the local aposira-github working tree, including staged package moves under aposira/forecasting and aposira/vision. Its last committed ancestor is 354fafd. The local changes are not assumed to exist on GitHub or any current Pod. On 4 November 2026 the 12 root tests passed with one dependency deprecation warning; this does not validate production scale or all vision workflows.

Source keys: G1 = Final Year Project G1.docx, Chapter 2; G2 = Final Project Report G2.docx, Chapter 2; I = Initial Report (FYP - II).docx, named module headings. Source performance figures are targets, not verified results.

## Release scope and value

| Release | Scope and exit condition |
| --- | --- |
| R0 Demonstration | Commerce ledger, manual POS/order workflow, loyalty accrual, forecasting, vision and reports. Reproducible demonstration using non-sensitive data. |
| R1 Controlled pilot | Authentication and server-side branch permissions, attributed audit, tested backups, telemetry and representative model evaluation. Limited approved store pilot only. |
| R2 Retail operations | Supplier purchasing, transfers, returns, taxes, shift reconciliation and approved payment integration. Financial reconciliation verified. |
| R3 Connected intelligence | Customer storefront, official delivery/social connectors, shelf recognition and external signals, each gated by access and evaluation. |

These releases are sequencing proposals, not dated delivery commitments. The platform must remain usable when optional models or integrations are unavailable, with explicit status and actionable errors.

## What exists today

- Organization and branch selectors; organization-scoped products/customers; branch-scoped orders, stock movements and events. These selectors do not authenticate users.

- Integer-paisa prices, append-only stock ledger, reservations, multi-item checkout, manual payment confirmation, delivery state transitions and one loyalty point per PKR 100 of completed order value.

- CSV/JSON sales intake, seasonal naive, LightGBM, Prophet and fixed-weight ensemble; 7/14/30-day horizons, holdout metrics, inventory rules, simulation and exports.

- YOLO and ByteTrack pipeline, zones and behavior analytics, searchable 30-day event metadata and durable acknowledgement storage. Accuracy and stream capacity remain unverified.

## Not promised for R0

No production login, merchant payment processing, offline POS, receipt hardware, automatic purchasing, customer storefront, marketplace synchronization, loyalty redemption or trained product-facing recognition. A manually entered online order is not a connected e-commerce checkout. An unattended object detector is not an unattended checkout counter detector.

## Product measures and release gates

| Measure | Acceptance approach |
| --- | --- |
| Transaction integrity | Every accepted completed order has exactly one stock deduction per item and at most one loyalty award. Retries must not duplicate effects. |
| Scope isolation | Automated checks reject cross-organization and cross-branch access once identity is implemented; changing a UI selector is insufficient. |
| Forecast quality | Report MAE, RMSE, MAPE and interval coverage per SKU/store against a seasonal baseline. MAPE above 15% or unavailable requires review, not an automatic purchase. |
| Vision quality | Use labeled footage and declared hardware to measure precision, recall, mAP, tracking and alert delay. Record false-alert rates and lighting/occlusion conditions. |
| Usability | Proposed R1 gate: representative users complete sale, stock receipt, forecast and event-review tasks without critical assistance; onboarding target from G1 is under 10 minutes. |
| Reliability | Load, restore and outage tests must pass before a pilot. G1 99.5% and G2 99.9% availability remain separate unverified targets. |

## Dependencies and prioritization

P0 work prevents integrity, access or recovery failures. P1 work completes the documented demonstration and pilot workflows. P2 work expands the platform after P0/P1 exit criteria are met. Payment terminals, merchant APIs, usable sales history, labeled video, camera connectivity and storage capacity are external dependencies, not assumed entitlements.

## Decisions requiring owner approval

- Accept LightGBM/Prophet as the revised model baseline or fund ARIMA/LSTM to satisfy G1 exactly. Keep the source discrepancy visible until approved.

- Accept ByteTrack as the tracker or implement DeepSORT and reevaluate tracking targets. Confirm whether cross-camera identity continuity is required.

- Choose the identity provider, production database, hosting model, retention periods, support hours and pilot hardware. Confirm which roles may see footage and customer records.

## Change control

Every scope change must record affected requirement IDs, reason, acceptance tests, data/security impact and release assignment. Product owner approves scope; technical lead approves design; QA records evidence. Reviewers and approval dates remain unassigned until the team completes review. Original reports remain source records; this pack does not declare academic or legal compliance.

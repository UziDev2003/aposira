# APOSIRA Verification and Traceability

Version 1.0 | 4 November 2026 | Proposed product baseline

## Verification strategy

This plan links the requirements to observable evidence. Unit checks validate calculations and rules; integration tests validate durable state and interfaces; UI tests validate user workflows; hardware/load tests validate deployment-specific targets. Pass counts do not substitute for coverage or stakeholder acceptance.

## Evidence currently available

On 4 November 2026, the inspected local working tree passed 12 root tests using the workspace virtual environment. One FastAPI/Starlette test-client deprecation warning was reported. This run used temporary test data and did not establish live RunPod health, 16-camera throughput, model accuracy or production authorization. Earlier 39-test results refer to a previous iteration and must not be presented as a new full-suite result.

```bash
python -m pytest tests -q
python -m pytest aposira/vision/tests -q
```

Run the second command separately and record dependencies, skipped tests and failures. Some vision tests require model weights or hardware. A missing optional dependency is not evidence of a passing detector.

## Test fixtures

- Commerce: two organizations, two branches, repeated SKUs across tenants, one registered customer, known stock and deterministic idempotency references. Isolate temporary databases.

- Forecasting: at least six calendar months of daily single-SKU data, explicit zero-sales dates, missing dates, invalid columns/values, promotions and separate validation observations.

- Vision: consented labeled clips with known people counts, occlusions, zone entries/exits, dwell durations, restricted-zone events and negative examples. Preserve frame rate and timestamps.

- Operational: a disposable restore environment, deliberately disconnected camera, unavailable model, occupied port, invalid proxy host, corrupted import and simulated duplicate provider events.

## Evidence record fields

Test ID, AP/NFR IDs, revision including local-change status, operator/date, hardware/runtime, fixture checksum, expected result, actual result, pass/fail/blocked, logs/screenshots and linked defect. Measure coverage explicitly before citing G1 80% code coverage.

## Functional acceptance catalogue

| Test | Requirements and expected result |
| --- | --- |
| T01 Scope isolation | AP-01/02. Verify scope clearing and cross-scope service queries. After identity implementation, tampered scope/object IDs return denial. |
| T02 Catalogue and ledger | AP-03/04. Validate fields, repeat identical movement, reject changed replay, and reject UPDATE/DELETE. |
| T03 Atomic checkout | AP-05/06/07. Competing purchases cannot oversell; injected failure rolls back order, movements, loyalty and audit. |
| T04 Fulfillment | AP-08. Exercise every permitted transition plus invalid skips; verify rider/proof/COD guards and failed/cancelled reservations. |
| T05 Loyalty and audit | AP-09/10. Exact point floor, single award, audit in same transaction; R1 must prove actor attribution. |
| T06 Import and daily history | AP-11/12. Aliases normalize; bad columns, invalid dates/values, short series and gaps fail. |
| T07 Forecast and evaluation | AP-13/14/15. Each model/horizon returns correct dates and nonnegative bounds, separate holdout and truthful metrics; scheduler tests remain planned. |
| T08 Inventory and simulation | AP-16/17/18. Hand-calculated EOQ/ROP, zero demand, ABC threshold crossings, one receipt per purchase and shortage conservation. |
| T09 Exports | AP-19/24. Parse generated CSV/XLSX/PDF, verify scope and period, inspect spreadsheet formula handling and vision export content. |
| T10 Vision and zones | AP-20/21/22. Run labeled footage, check track schema, boundary crossings, dwell, threshold and cooldown behavior. |
| T11 Events and recovery | AP-23. Save/ack/reopen; verify acknowledged_at in UI and filtered count. Expired metadata excluded; file cleanup evaluated separately. |
| T12 Expanded platform | AP-25 to AP-32. Add PO/transfer/refund/connector/replay/consent/shelf/feed/admin cases as each capability is designed. No automatic pass for planned work. |

## Known checks to add before pilot

K01: timeline display and unacknowledged totals must use acknowledged_at rather than the absent acknowledged payload field. K02: delayed fulfillment sales need completion-time semantics. K03: post-reorganization runtime data paths need an explicit migration and backup test. K04: forecast cache must include selected model/horizon or clearly retain original labels. These are review findings or required regressions, not changes made by this documentation task.

## Original requirement traceability

| Original source identifiers | Updated IDs and disposition |
| --- | --- |
| G1 SF1 1 / 2 / 3 / 4 | AP-11/12; AP-13; AP-14; AP-15. Import/daily checks implemented; model substitution, coverage and automation remain partial. |
| G1 SF2 1 / 2 / 3 / 4 | AP-16; AP-16; AP-16; AP-17. EOQ/safety/ROP exist; source automation and ABC wording need resolution. |
| G1 SF3 1 / 2 / 3 / 4 | AP-19; AP-16; AP-18; AP-19. Turnover, promotion risk, causal scenarios and full branding/filter scope are incomplete. |
| G2 SF1 1 / 2 / 3 / 4 | AP-20. Perception code exists; DeepSORT/capacity/accuracy targets not established; track outputs feed analytics. |
| G2 SF2 1 / 2 / 3 / 4 | AP-21. Polygons and movement analytics exist; fixed time aggregation, dwell accuracy and occupancy validation pending. |
| G2 SF3 1 / 2 / 3 / 4 | AP-22. Threshold analytics exist; false-positive/latency validation and checkout staff-absence detector remain outstanding. |
| G2 SF4 1 / 2 / 3 / 4 | AP-24; AP-23; AP-24; AP-24. Metadata store exists; UI acknowledgement defect, evidence retention, PDF/push/multiuser gaps remain. |
| G1 and G2 sections 2.5 and 2.6 | NFR-01 to NFR-08 plus SRS reconciliations. Source numeric claims are unverified targets. |

## Umbrella report coverage

| Initial report module families | Updated IDs |
| --- | --- |
| Organization and security | AP-01, AP-02, AP-10, AP-32 |
| Catalogue suppliers inventory POS | AP-03 to AP-07, AP-16/17, AP-25/26/27 |
| Customer orders and delivery | AP-08, AP-09, AP-28 |
| Web mobile and social commerce | AP-29 |
| Store vision and shelf interest | AP-20 to AP-24, AP-30 |
| Forecasting and decision intelligence | AP-11 to AP-18, AP-31 |
| Unified analytics and operations | AP-19, AP-32, NFR-01 to NFR-08 |

## Performance and model evaluation protocol

Forecast benchmark: freeze the dataset and SKU count, six-month minimum, model/horizon and CPU/RAM. Report total wall time, queue time, failure rate and p95 request/page timings separately. Do not compare one seasonal baseline request with the 10,000-SKU G1 target. Evaluate rolling time splits across stores and demand patterns; keep test data separate from model selection.

Vision benchmark: specify camera count, input resolution/FPS, codec, model weights, confidence/IoU settings and GPU. Define mAP IoU convention, precision/recall, MOTA and dwell tolerance before running. Measure per-camera processing FPS and capture-to-alert delay, then event-to-dashboard delivery delay. Label false alerts over a declared observation window and include negative footage. Source hardware performance estimates are not measured results.

## Reliability and security protocol

Measure monthly availability separately for commerce/forecasting and vision if retaining G1/G2 targets. Run restore, camera-loss, process-restart and storage-exhaustion drills. Test that logs do not contain tokens or raw card data. For R1 test role restrictions, tenant isolation, expired sessions, credential revocation and exported evidence access. A Streamlit health endpoint alone does not test these controls.

## Release decisions

| Gate | Required evidence |
| --- | --- |
| R0 demonstration | T01 scope behavior, T02–T09 applicable paths, honest empty/error states, no secrets in delivered files and known defects disclosed. |
| R1 controlled pilot | All P0 requirements; K01–K04 disposition; successful restore; approved permissions/retention; representative forecasting and vision evaluation. |
| R2 transactional expansion | Reconciled supplier/transfer/return/payment records and duplicate/out-of-order integration tests. |
| R3 intelligence expansion | Labeled shelf results, authorized connector access, feed provenance and time-leakage checks. |

## Defect management and approval

Treat unauthorized access, inventory corruption and credential exposure as release blockers. Record other defects with reproducible steps, severity, affected AP IDs, owner and retest evidence. Product owner accepts deferred scope; QA verifies fixes; technical lead reviews operational readiness. Approval names/dates and the final pilot decision must be filled by the team.

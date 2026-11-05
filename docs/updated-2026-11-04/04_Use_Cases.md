# APOSIRA Use Cases and Workflows

Version 1.0 | 4 November 2026 | Proposed product baseline

## Actors and access model

These workflows describe R0 behavior and the role controls required for a pilot. Current roles are intended responsibilities only; the application does not enforce them. All examples use demonstration records and avoid real payment details or personal footage.

| Actor | Proposed permitted tasks |
| --- | --- |
| Owner and manager | Review authorized branches, approve overrides and investigate stock/order exceptions. |
| Cashier | Create sales and attach registered customers in assigned branch; no cross-branch administration. |
| Inventory operator | Maintain catalogue as permitted, receive and adjust stock with reasons; no silent deletion of movement history. |
| Delivery operator | Update assigned fulfillment state and record proof/COD; cannot change arbitrary order prices. |
| Security officer | Review assigned camera events and acknowledge observations; no unrestricted customer data access. |
| Data administrator | Import approved sales history, evaluate models and export scoped analytics. |
| System administrator | Manage identity, devices, backup and service configuration with attributed audit. |

## UC01 Stock receipt and correction

Precondition: select a nonempty organization/branch and register SKU A with a valid name, selling price and cost. Operator records +20 units with reason opening_balance and unique reference OPEN-A-01. System commits one movement and displays on-hand 20, reserved 0, available 20. Replaying the same movement has no effect; changing its quantity under the same reference is rejected. Correct errors with a new compensating movement, never by editing history. Trace: AP-01, AP-03, AP-04.

## UC02 Complete a POS sale

Precondition: SKU A has stock and any attached customer is registered in the organization. Add two units to cart, choose cash or externally paid, confirm payment arrangement and submit once. System atomically creates a completed order, deducts two units, adds loyalty if applicable, writes audit and offers a receipt. A double submit must reuse the request reference. Insufficient stock or invalid cart must leave no partial effects. Trace: AP-05 to AP-07, AP-09, AP-10.

## UC03 Fulfill an order

Precondition: a manual online or telephone order uses available inventory. Creation produces received status and reservation, without a sale movement. Staff progress through accepted, picking, packed and ready. Ready can complete a collection or dispatch a delivery. Dispatch requires an assigned rider. Delivery completion requires proof; COD also requires collection confirmation. Trace: AP-05, AP-08.

| Current state | Allowed next states |
| --- | --- |
| received | accepted, cancelled |
| accepted | picking, cancelled |
| picking | packed, cancelled |
| packed | ready, cancelled |
| ready | dispatched, completed, cancelled |
| dispatched | completed, failed |
| failed | dispatched, cancelled |
| completed or cancelled | None |

Cancellation releases reservations; completed orders cannot be cancelled in the current state machine. Failed delivery retains reservation and may be redispatched or cancelled. Returns, refunds and settlement adjustments require separate ledger workflows; do not force them through this state machine.

## UC04 Customer and points

Register a customer ID and name within the organization. Complete an order worth PKR 250 with that customer attached. Expected award is two points, once. A duplicate checkout must not add points again. Other organizations cannot attach that customer through the service. Redemption, membership tiers and point reversal after a return are planned. Trace: AP-09.

## UC05 Change workspace

Select another branch or organization. System clears previous cart, loaded camera state and analysis session values while preserving the navigation/scope widgets. Products and customers are organization-scoped; changing branch within the same organization does not create a new catalogue. Authorized scope membership is required in R1. Trace: AP-01, AP-02.

## UC06 Import history and forecast

Select POS sales, Uploaded history or Sample demonstration. For uploaded history, provide valid normalized columns and select the matching branch/store. Demonstration data must remain visibly labeled. Choose a SKU with six calendar months of continuous daily data and a 7-, 14- or 30-day horizon. Generate a forecast; inspect its chart, bounds, model and held-out metrics. Unknown dates are not zero sales. If MAPE is unavailable or above 15%, review before placing an order. Trace: AP-11 to AP-15.

Error paths: missing column, invalid date, wrong branch, mixed series, insufficient history, missing model dependency or training failure. Preserve usable prior data and show the reason. A new horizon/model selection must not mislabel a cached forecast from a different selection; include this regression case in acceptance testing.

## UC07 Review replenishment and reports

Confirm current available stock, cost, lead time, service level, ordering cost and holding rate. Review EOQ, safety stock, reorder point and days of supply. Use ABC for revenue contribution and policy simulation for historical comparisons. These outputs do not submit purchase orders. Export the selected report and verify units, period and group. Date ranges filter sales/ABC; saved analysis reports represent their original generated window. Trace: AP-16 to AP-19.

## UC08 Process and review video

Use an authorized upload or reachable RTSP source. Select model/device and camera, configure polygons and thresholds, then process. Review tracks, zones and event observations. If a source or model fails, display the failure and stop reporting the stale result as live. Cloud webcam access refers to the server device, not automatically to a laptop camera. Trace: AP-20 to AP-22.

## UC09 Acknowledge an event

Select dates and camera/text filters, review the event message and available evidence, then acknowledge. Storage writes acknowledged_at and should preserve it across restart. The current UI reads a different acknowledged field; the display/count issue must be fixed before this use case passes. Do not infer wrongdoing from an automated alert alone. Event metadata older than 30 days is excluded; evidence file retention is separate. Trace: AP-23, AP-24.

## Acceptance record

For each use case record actor, revision, environment, fixtures, steps, expected/actual result and evidence. Mark pass only after executing the complete path; a loaded page alone is not acceptance. Future workflows AP-25 to AP-32 require their own detailed scenarios before implementation.

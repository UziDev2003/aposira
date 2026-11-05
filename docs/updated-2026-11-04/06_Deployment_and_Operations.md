# APOSIRA Deployment and Operations

Version 1.0 | 4 November 2026 | Proposed product baseline

## Purpose and deployment boundary

This guide defines repeatable local and RunPod demonstration setup and the work needed before a controlled pilot. It contains no credentials or fixed Pod address. Use the current Pod connection details and HTTP proxy link. A private repository protects source access; it does not add login to the running application.

## Source and version control

Repository: https://github.com/UziDev2003/aposira. Local checkout: aposira-github within the development workspace. The inspected local tree includes staged reorganization under aposira/forecasting and aposira/vision; the last committed ancestor is 354fafd. Before deployment, review and commit the intended changes through the team workflow, record the revision and verify the clone contains the expected package paths. Do not reset or discard another contributor’s staged changes.

## Local setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-ml.txt
.venv/bin/python run_aposira.py
```

Use Python 3.12 as the project working baseline and compatible installed dependencies. The core app uses Streamlit 1.66 or later within major version 1. Vision and advanced models require the ML requirements and model weights. The local launcher uses port 8501 unless overridden; its macOS OpenMP handling is process-local.

## Private clone

```bash
gh auth login --hostname github.com --git-protocol https --web
cd /workspace
gh repo clone UziDev2003/aposira
cd aposira
```

These commands require Git and GitHub CLI on the Pod. Complete the device sign-in on a trusted browser as an account with repository access. Alternatively use HTTPS Git with a repository-scoped read token only at the password prompt. Never put a token in the command URL, source file or report. Revoke any token already exposed in chat/logs. Existing SSH keys authorize Pod access, not automatically GitHub access.

## RunPod launch and verification

```bash
cd /workspace/aposira
nohup bash run_runpod.sh > /workspace/aposira-runpod.log 2>&1 < /dev/null &
tail -f /workspace/aposira-runpod.log
```

The root setup script creates .venv-runpod, installs dependencies and system video libraries, reports CUDA availability, stops Jupyter to free port 8888 and launches Streamlit. It stops Jupyter processes but does not delete notebooks. If another service still holds the port, it exits rather than killing an unrelated process. Use the root script, not the standalone vision script.

The launcher derives the public hostname from RUNPOD_POD_ID, or accepts APOSIRA_PUBLIC_HOST. It sets browser.serverAddress and browser.serverPort 443 while keeping CORS and XSRF enabled. Open the current HTTPS proxy link in RunPod Connect. The old Jupyter token parameter is not an APOSIRA login credential. Do not rely on Streamlit’s printed HTTP URL with port 443.

```bash
APOSIRA_INSTALL=0 bash run_runpod.sh
```

Use this only after a successful installation and after stopping the existing APOSIRA process. It skips dependency installation. For a custom hostname, set APOSIRA_PUBLIC_HOST to the hostname without a token or URL path. Control+C stops tail output, not a background application.

## Postlaunch checks

- Open Overview, Products, POS, Fulfillment, Customers, Forecasting, Vision and Audit. Verify usable layout, accurate empty states and no application exceptions.

- Complete an isolated demonstration sale; verify movement, receipt, loyalty and audit. Generate a sample forecast and check source labels, model/horizon and CSV download.

- Run a consented test video, then verify event filters and acknowledgement after reload. Record GPU/model and measured performance rather than assuming hardware suitability.

- Review logs for WebSocket rejection, database errors and missing modules. Confirm configured data paths and available storage; never expose the unauthenticated legacy API as a production endpoint.

## Troubleshooting and safe recovery

| Symptom | Check and action |
| --- | --- |
| GitHub 403 during clone | Check account and repository access or use GitHub CLI device login. Cloning needs read access; a write-access error does not justify a broad write token. |
| SSH asks for password | Use the current Pod SSH command and an existing private key matching an authorized public key. A SHA256 fingerprint is not a private key. |
| Disallowed Origin or Host | Verify current proxy hostname, RUNPOD_POD_ID/APOSIRA_PUBLIC_HOST and browser address/port flags. Restart the intended process after correction. Keep protection enabled. |
| Port 8888 occupied | Identify the actual process or supervisor. Stop the known old application/Jupyter service before restarting; do not blindly kill all Python processes. |
| Forecast rejects data | Check SKU/store, six-month span, date gaps, explicit zero-sales rows and normalized fields. Do not lower validation to make an invalid dataset pass. |
| No video or CUDA | Check file permissions, model availability, torch CUDA, RTSP reachability and codec. A cloud Pod cannot directly reach private LAN camera IPs without approved connectivity. |
| Events look unacknowledged | Known UI mismatch: store returns acknowledged_at; UI reads acknowledged. Resolve and retest T11 before relying on acknowledgement totals. |

## Backup and restore procedure

Before a revision or path migration, stop writes or use SQLite’s online backup API; do not assume copying a live database is consistent. Back up commerce, forecasting/API ledger and vision event databases plus required uploads, zones, configs, snapshots and model artifacts. Exclude tokens and protect backups with access controls. Record revision, paths, file checksums and backup time.

Restore into an isolated directory or disposable Pod, install the recorded dependencies, reconcile branch stock/order/loyalty totals and event acknowledgements, and run acceptance checks before switching users. Source backup goals are daily business records and hourly vision evidence; scheduling and restore automation are still planned. Code rollback alone cannot undo data/schema changes safely.

## Operational ownership and retention

Assign an operator for service/log review, a data owner for retention and a release owner for recovery. Event metadata is filtered at 30 days and pruned on event-store initialization/save. Evidence files are not automatically covered. Approve source-proposed seven-year transactions, 90-day audit and footage periods before enabling automated deletion. Validate backup restoration and storage lifecycle before a pilot.

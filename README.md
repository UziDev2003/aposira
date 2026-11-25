# APOSIRA

One local application combining retail operations, the retail forecasting project, and Smart Store Vision AI. The application follows the APOSIRA product identity at https://aposira.vercel.app and the three supplied project reports.

## Start

From this workspace:

```bash
.venv/bin/python run_aposira.py
```

Open http://127.0.0.1:8501. The supplied `.streamlit/config.toml` binds the development server to localhost.

For a fresh installation, use Python 3.12 and a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-ml.txt
.venv/bin/python run_aposira.py
```

The core application and seasonal baseline run without the optional ML dependencies. Store Vision requires the optional dependencies and YOLO weights; a selected model may need its first download. The launcher discovers the OpenMP runtime shipped with scikit-learn on macOS and sets a process-local library path; it does not change system libraries. If that runtime is absent, LightGBM requires an OpenMP installation. If a model is unavailable, the UI reports the error rather than inventing model output.

## Connected workflows

1. Select an organization and branch.
2. Add a product in **Products & Stock**, then record an opening balance or purchase with a unique movement reference.
3. Optionally register a customer in **Customers & Loyalty**.
4. Add items in **POS & Orders** and confirm cash or an externally processed payment. A completed sale decreases stock, earns loyalty points, appears in the audit trail, and becomes a forecasting input. The receipt is downloadable JSON.
5. An online or telephone order reserves stock. Progress it through acceptance, picking, packing, ready, dispatch, and completion. Dispatch requires a rider; delivery completion requires a proof reference and COD collection confirmation where applicable. Cancelling releases the reservation. “Online” is currently a manually entered order source, not a public storefront or marketplace integration.
6. In **Forecasting & Inventory**, select POS sales, upload historical sales, or explicitly choose demonstration data. Forecasts require six months of daily history per SKU/store. Missing days are not silently converted to zero sales; include explicit zero-sales records where appropriate. Uploaded history stays in the session and is not automatically posted to stock or mixed with POS records.
7. In **Store Vision**, uploaded videos start analysis automatically and display annotated frames, progress, metrics, and processing warnings while running. Disable automatic analysis before uploading to configure zones first. Upload preparation is reused across reruns. Processing history and errors are stored in SQLite by organization and branch alongside the existing searchable 30-day event timeline. Use the start button to retry a failed run or apply changed settings. Video inference speed depends on the selected device and model.

Store Vision includes **Possible Concealment** review alerts when a recognized shelf item moves toward a tracked person's bag or torso and becomes obscured for a sustained interval. It also flags possible unpaid exits when a new carried item is observed from shelf to exit without an observed checkout visit. Align shelf, checkout, and entrance/exit zones with each camera. These rules depend on visible object detections and stable person tracks. Their evidence is saved with the alert for human review.

Forecasting provides 7/14/30-day horizons, held-out MAE/RMSE/MAPE, nominal 95% residual intervals, and an accuracy review warning above 15% MAPE. Models available through the common service are seasonal naive, LightGBM, Prophet and a fixed-weight LightGBM/Prophet ensemble. Forecast accuracy and interval coverage are not guaranteed. Inventory decisions calculate EOQ, safety stock, reorder points and seven-day supply alerts from stock and cost inputs. The corrected simulator counts each delivery once and uses only prior demand for default policy signals.

## Structure and storage

For a complete system map, subsystem breakdown, and token optimization guide, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

- `aposira/app.py`: combined Streamlit entry point and shared navigation.
- `aposira/commerce.py`: transactional product, customer, order, reservation, stock movement, loyalty and audit storage.
- `aposira/ui.py`: shared APOSIRA design system and UI components.
- `aposira/forecasting/`: forecasting, optimization, exports and the standalone API.
- `aposira/vision/`: computer vision pipeline, zone editor, and persistent event storage.
- `data/aposira.db`: local commerce state. Organizations and branches are query scopes, not access controls.
- `aposira/vision/data/events.sqlite3`: scoped event metadata, retained for 30 days.
- `aposira/vision/data/uploads/`: video uploads grouped by a hash of organization/branch.
- `aposira/vision/data/snapshots/`: evidence snapshots. Automatic snapshot deletion is not implemented; configure an appropriate storage/retention process before using real footage.

This repository includes both forecasting and vision pipelines directly under the `aposira` package, sharing a single cohesive theme and UI design.

## Verification

```bash
.venv/bin/python -m pytest -v
```

The root suite exercises checkout idempotency, stock reservations, fulfillment transitions, branch isolation, immutable stock movements, forecast validation, simulator conservation, exports, retained alerts, and dashboard navigation/checkout. It uses temporary databases rather than production records.

## Implementation boundaries

This is a combined local prototype, not a finished implementation of every requirement in the initial report. Authentication/RBAC, PostgreSQL/TimescaleDB, production hosting, payment gateways, tax/shift reconciliation, refunds, loyalty redemption, public e-commerce, social/marketplace connectors, routing/GPS, weather/trend feeds, automated retraining, and a trained shelf-product/interest-to-purchase model remain future work. The legacy forecasting API is separate from the commerce store and has no production authentication; it is not a tenant-safe public service.

See `docs/requirements-alignment.md` for source traceability, implemented behavior, and remaining report targets. The Vercel marketing website and original DOCX files were not modified or deployed.

## RunPod deployment

This GitHub repository is private. Authenticate on the Pod with a GitHub account that has access. If GitHub CLI is installed:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
cd /workspace
gh repo clone UziDev2003/aposira
cd aposira
bash run_runpod.sh
```

Without GitHub CLI, run `git clone https://github.com/UziDev2003/aposira.git` in `/workspace`. At the prompts use your GitHub username and a fine-grained personal access token restricted to this repository with Contents read permission as the password. Never place the token in the URL or a committed file.

The root setup script installs dependencies, stops Jupyter, and starts the combined app on port 8888. Open the Pod's HTTP port 8888 link. This app has no login; use demonstration data on the public proxy. Run the root script, not the older script inside aposira/vision.

To keep it running after disconnecting:

```bash
cd /workspace/aposira
nohup bash run_runpod.sh > /workspace/aposira-runpod.log 2>&1 < /dev/null &
tail -f /workspace/aposira-runpod.log
```

Use `Ctrl+C` to stop following the log; the background application keeps running. After a successful first installation, `APOSIRA_INSTALL=0 bash run_runpod.sh` reuses the environment. Stop an already running APOSIRA process before launching another instance.

The RunPod launcher derives the public proxy hostname from `RUNPOD_POD_ID` and keeps CORS/XSRF enabled. For a custom hostname, set `APOSIRA_PUBLIC_HOST=your-hostname` before launching. Use the HTTPS proxy URL from RunPod, even if Streamlit logs an HTTP URL with port 443.

## Simulate a live RTSP camera

Run `video_to_rtsp.py` on the same Pod as APOSIRA with a video file stored on that Pod:

```bash
python3 video_to_rtsp.py "/workspace/demo.mp4"
```

The standalone script uses FFmpeg and downloads a checksum-verified MediaMTX release when needed. It loops the file at normal playback speed and publishes `rtsp://127.0.0.1:8554/store`. In Store Vision, choose Live Stream, select RTSP, paste that URL, and press Start. Keep the script running while testing. Ctrl+C stops both processes. Server and publisher errors appear in the terminal and in the printed log directory.

FFmpeg is included by `run_runpod.sh`. To install it separately on RunPod, use `apt-get update && apt-get install -y ffmpeg`. The script also supports Windows and macOS when FFmpeg is on PATH. Use `--port 8555` for another port, `--fps 10` to reduce encoding work, or `--duration 60` for a one-minute test.

The default URL works for readers on the same machine. For a reader on another host, use `--bind 0.0.0.0` and a reachable TCP address/port. On RunPod, expose TCP port 8554 and use its external TCP IP and mapped port. APOSIRA's HTTPS proxy link serves the web UI. MediaMTX's RTSP stream uses its TCP listener.

References: [MediaMTX FFmpeg publishing](https://mediamtx.org/docs/publish/ffmpeg), [MediaMTX configuration](https://mediamtx.org/docs/references/configuration-file).

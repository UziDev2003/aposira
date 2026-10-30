#!/usr/bin/env bash
# Run inside your RunPod after uploading the complete APOSIRA workspace.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PORT=8888

echo '=================================================='
echo '       APOSIRA - RunPod Automated Setup'
echo '=================================================='

if [[ "$(uname -s)" != "Linux" ]]; then
    echo 'Run this script in the RunPod terminal, not on your Mac.' >&2
    exit 1
fi
for required in aposira/app.py requirements.txt requirements-ml.txt run_aposira.py; do
    [[ -f "$required" ]] || { echo "Missing $required: upload the complete APOSIRA workspace." >&2; exit 1; }
done

if [[ "${APOSIRA_INSTALL:-1}" == "1" ]]; then
    echo '[1/4] Installing system dependencies...'
    if [[ "$EUID" == '0' ]]; then
        apt-get update -qq
        apt-get install -y python3-venv ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 libgomp1 procps
    else
        echo 'Not root: system dependencies must already be installed.'
    fi

    echo '[2/4] Installing APOSIRA and ML dependencies...'
    python3 -m venv --system-site-packages .venv-runpod
    .venv-runpod/bin/python -m pip install --upgrade pip
    .venv-runpod/bin/python -m pip install -r requirements.txt -r requirements-ml.txt
else
    echo '[1/4] Skipping system installation.'
    echo '[2/4] Reusing the existing RunPod environment.'
fi

PYTHON_BIN="$ROOT_DIR/.venv-runpod/bin/python"
[[ -x "$PYTHON_BIN" ]] || { echo 'Run once with APOSIRA_INSTALL=1.' >&2; exit 1; }
"$PYTHON_BIN" -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mode")'

echo '[3/4] Stopping Jupyter to release port 8888...'
# Stops Jupyter servers, without deleting notebooks or uninstalling packages.
# Bracketed patterns prevent pkill from matching its own command line.
pkill -TERM -f '(^|/)[j]upyter-(lab|notebook|server)([[:space:]]|$)|(^|/)[j]upyter[[:space:]]+(lab|notebook|server)([[:space:]]|$)|-m[[:space:]]+[j]upyter(lab|_server|[[:space:]]+(lab|notebook|server))([[:space:]]|$)' || true

# Allow graceful shutdown; never kill an unrelated service occupying this port.
"$PYTHON_BIN" - <<'PY'
import socket
import time
for attempt in range(20):
    try:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', 8888))
        break
    except OSError:
        if attempt == 19:
            raise SystemExit('Port 8888 is still occupied. Jupyter may be restarting under a supervisor; stop its managed service in the Pod terminal and rerun.')
        time.sleep(.5)
PY

echo '[4/4] Starting the combined APOSIRA application...'
echo 'Open RunPod Console -> Connect -> HTTP Port 8888.'
echo 'The link may still be labelled Jupyter Lab; it now serves APOSIRA.'
echo 'This prototype has no login: use demonstration data on the public proxy.'

# Command-line settings override the local-only project config. Keep security
# protections enabled and avoid rewriting the global Streamlit configuration.
exec "$PYTHON_BIN" run_aposira.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true \
    --server.enableCORS true \
    --server.enableXsrfProtection true \
    --browser.gatherUsageStats false

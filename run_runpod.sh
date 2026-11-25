#!/usr/bin/env bash
# Run inside your RunPod after uploading the complete APOSIRA workspace.
set -euo pipefail
trap 'echo "APOSIRA setup failed at line $LINENO. Review the error above, then rerun bash run_runpod.sh." >&2' ERR

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PORT=8888
PUBLIC_HOST="${APOSIRA_PUBLIC_HOST:-${RUNPOD_POD_ID:+${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net}}"
if [[ -z "$PUBLIC_HOST" ]]; then
    echo 'Set APOSIRA_PUBLIC_HOST to your RunPod proxy hostname.' >&2
    exit 1
fi

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

# Hold the lock through startup so concurrent launchers cannot modify the same venv.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$ROOT_DIR/.runpod-setup.lock"
    flock -n 9 || { echo 'APOSIRA is already installing or running from this directory.' >&2; exit 1; }
fi

# Container-local storage avoids unpacking thousands of wheel files on /workspace.
ENV_KEY="$(printf '%s' "$ROOT_DIR" | sha256sum | cut -c1-16)"
ENV_ROOT="${APOSIRA_ENV_ROOT:-/tmp/aposira-envs}"
VENV_DIR="$ENV_ROOT/$ENV_KEY"
PYTHON_BIN="$VENV_DIR/bin/python"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

install_packages() {
    "$PYTHON_BIN" -m pip install --no-input --no-compile "$@" &
    local installer_pid=$!
    (
        while kill -0 "$installer_pid" 2>/dev/null; do
            sleep 20
            if kill -0 "$installer_pid" 2>/dev/null; then
                echo '[setup] pip is still running; large packages can take several minutes to install.'
            fi
        done
    ) &
    local progress_pid=$!
    local result=0
    wait "$installer_pid" || result=$?
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
    return "$result"
}

REQUIREMENTS_KEY="$(cat requirements.txt requirements-ml.txt; python3 --version; printf 'runpod-env-v2')"
REQUIREMENTS_KEY="$(printf '%s' "$REQUIREMENTS_KEY" | sha256sum | cut -d' ' -f1)"
READY_FILE="$VENV_DIR/.aposira-ready"

if [[ "${APOSIRA_INSTALL:-1}" == "1" ]] && { [[ ! -f "$READY_FILE" ]] || [[ "$(cat "$READY_FILE")" != "$REQUIREMENTS_KEY" ]]; }; then
    echo '[1/4] Checking system dependencies...'
    SYSTEM_PACKAGES=(python3-venv ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 libgomp1 procps)
    if [[ "$EUID" == '0' ]] && ! dpkg-query -W -f='${Status}\n' "${SYSTEM_PACKAGES[@]}" 2>/dev/null | awk '
        $0 != "install ok installed" { missing=1 }
        END { exit missing }
    '; then
        apt-get update -qq
        apt-get install -y "${SYSTEM_PACKAGES[@]}"
    fi

    echo "[2/4] Installing dependencies into container-local storage: $VENV_DIR"
    mkdir -p "$ENV_ROOT"
    python3 -m venv --system-site-packages "$VENV_DIR"
    # Shadow the container's dateutil without attempting to uninstall it.
    install_packages --ignore-installed --no-deps 'python-dateutil>=2.8,<3'
    # Preserve the template's matching torch/torchvision build and CUDA runtime.
    python3 - "$VENV_DIR/torch-constraints.txt" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
constraints = []
for package in ('torch', 'torchvision', 'torchaudio'):
    try:
        constraints.append(f'{package}=={version(package)}')
    except PackageNotFoundError:
        pass
Path(sys.argv[1]).write_text('\n'.join(constraints) + '\n')
print('Reusing container packages:', ', '.join(constraints) or 'no preinstalled PyTorch')
PY
    install_packages -c "$VENV_DIR/torch-constraints.txt" -r requirements.txt -r requirements-ml.txt
else
    echo '[1/4] Reusing system dependencies.'
    echo '[2/4] Reusing installed Python dependencies.'
fi

[[ -x "$PYTHON_BIN" ]] || { echo 'Run once with APOSIRA_INSTALL=1.' >&2; exit 1; }
"$PYTHON_BIN" -c 'import streamlit, pandas, numpy, openpyxl, cv2, ultralytics, lightgbm, prophet; print("APOSIRA dependency imports passed.")'
"$PYTHON_BIN" -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mode")'

if [[ "${APOSIRA_INSTALL:-1}" == "1" ]]; then
    printf '%s' "$REQUIREMENTS_KEY" > "$READY_FILE"
fi

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
    --browser.serverAddress "$PUBLIC_HOST" \
    --browser.serverPort 443 \
    --browser.gatherUsageStats false

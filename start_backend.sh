#!/usr/bin/env bash
# Kimodo-G1 BACKEND — models + generation API on port 8000.
#
#   ./start_backend.sh              # foreground, Ctrl-C to stop
#   PORT=9000 ./start_backend.sh
#
# Supervised: if the process dies it is restarted automatically. That matters
# because a CUDA fault poisons the context for the whole process and cannot be
# repaired in place -- the app exits with code 75 and the fresh process re-probes
# the GPU, falling back to CPU if the driver is still unhappy. So a crash costs a
# reload, not your afternoon.
#
# Logs: backend.log

set -uo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
LOG=backend.log

PY="${PY:-${PYTHON:-}}"
if [ -z "$PY" ]; then
    if [ -x "$(pwd)/venv/bin/python" ]; then
        PY="$(pwd)/venv/bin/python"
    elif [ -x "/home/sai/g1_venv/bin/python" ]; then
        PY="/home/sai/g1_venv/bin/python"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PY="$VIRTUAL_ENV/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        PY="$CONDA_PREFIX/bin/python"
    elif [ -x "/home/sai/miniforge3/envs/kimodo/bin/python" ]; then
        PY="/home/sai/miniforge3/envs/kimodo/bin/python"
    else
        PY=$(which python3 2>/dev/null || echo python3)
    fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)/kimodo_repo"
# Force CPU mode globally across all backend processes and sub-workers
export CUDA_VISIBLE_DEVICES=""

if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: python interpreter not found at $PY" >&2
    exit 1
fi
if [ ! -f "$HOME/.cache/kimodo/llm2vec_merged/model.safetensors.index.json" ]; then
    echo "WARNING: merged encoder missing ($HOME/.cache/kimodo/llm2vec_merged)." >&2
    echo "         Presentation & Sentence tabs work immediately." >&2
    echo "         To enable live diffusion generation, run: $PY scripts/download_models.py && $PY scripts/merge_llm2vec_lora.py" >&2
fi

# Single instance only. Two supervisors would each kill the other's child on
# startup and loop forever, so take an exclusive lock and re-exec under it.
LOCK=/tmp/kimodo_backend_$PORT.lock
if [ "${_KIMODO_LOCKED:-}" != "1" ]; then
    export _KIMODO_LOCKED=1
    exec flock -n "$LOCK" "$0" "$@" || {
        echo "ERROR: another backend supervisor already holds $LOCK" >&2
        echo "       stop it first, or run with a different PORT." >&2
        exit 1
    }
fi

# Free the port if a previous run is still holding it, so restarts never fail
# with "address already in use".
existing=$(pgrep -f "webapp.py --port $PORT" || true)
if [ -n "$existing" ]; then
    echo "stopping previous backend (pid $existing)"
    # shellcheck disable=SC2086
    kill $existing 2>/dev/null || true
    sleep 3
fi

echo "=============================================="
echo " Kimodo-G1 backend   http://$HOST:$PORT"
echo " logs: $(pwd)/$LOG"
echo " Ctrl-C to stop"
echo "=============================================="

trap 'echo; echo "backend stopped."; exit 0' INT TERM

WEBAPP="src/webapp.py"
[ ! -f "$WEBAPP" ] && WEBAPP="webapp.py"

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[$(date +%H:%M:%S)] starting backend (attempt $attempt)" | tee -a "$LOG"
    "$PY" -u "$WEBAPP" --port "$PORT" --host "$HOST" 2>&1 | tee -a "$LOG"
    code=${PIPESTATUS[0]}
    if [ "$code" -eq 0 ]; then
        echo "backend exited cleanly."
        exit 0
    fi
    echo "[$(date +%H:%M:%S)] backend exited with code $code — restarting in 5s" | tee -a "$LOG"
    sleep 5
done

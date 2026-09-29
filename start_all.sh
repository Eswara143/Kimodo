#!/usr/bin/env bash
# Start the whole Kimodo-G1 web stack with one command.
#
#   ./start_all.sh          # backend + frontend, browser opens, Ctrl-C stops both
#
# Backend logs to backend.log, frontend to frontend.log. Both are supervised the
# same way as when started individually; this only saves opening two terminals.

set -uo pipefail
cd "$(dirname "$0")"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
# Force CPU mode globally across all services
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)/kimodo_repo"

cleanup() {
    echo
    echo "stopping..."
    [ -n "${BE_PID:-}" ] && kill "$BE_PID" 2>/dev/null
    [ -n "${FE_PID:-}" ] && kill "$FE_PID" 2>/dev/null
    pkill -f "webapp.py --port $BACKEND_PORT" 2>/dev/null
    pkill -f "http.server $FRONTEND_PORT" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

echo "starting backend on :$BACKEND_PORT ..."
PORT="$BACKEND_PORT" ./start_backend.sh > backend.log 2>&1 &
BE_PID=$!

# The models take ~40 s to load. Wait for readiness rather than guessing, so the
# browser does not open onto a dead page.
printf "waiting for backend..."
for _ in $(seq 1 30); do
    if curl -s --max-time 2 "http://127.0.0.1:$BACKEND_PORT/api/status" 2>/dev/null | grep -q '"device"'; then
        echo " ready"
        break
    fi
    if ! kill -0 "$BE_PID" 2>/dev/null; then
        echo
        echo "backend died during startup — see backend.log" >&2
        tail -20 backend.log >&2
        exit 1
    fi
    printf "."
    sleep 2
done

echo "starting frontend on :$FRONTEND_PORT ..."
PORT="$FRONTEND_PORT" API="http://127.0.0.1:$BACKEND_PORT" ./start_frontend.sh > frontend.log 2>&1 &
FE_PID=$!

sleep 3
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo
echo "=========================================================="
echo "  Kimodo-G1 Web Stack Started!"
echo "  💻 Local PC URL:      http://127.0.0.1:$BACKEND_PORT"
echo "  📱 Mobile Wi-Fi URL:  http://$LOCAL_IP:$BACKEND_PORT"
echo "  (Also accessible at http://$LOCAL_IP:$FRONTEND_PORT)"
echo "  Logs: backend.log | frontend.log"
echo "  Press Ctrl-C to stop"
echo "=========================================================="

wait

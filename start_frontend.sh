#!/usr/bin/env bash
# Kimodo-G1 FRONTEND — serves the web UI on port 8080 and opens your browser.
#
#   ./start_frontend.sh                       # backend assumed on :8000
#   PORT=8081 API=http://127.0.0.1:9000 ./start_frontend.sh
#
# Static files only; all model work happens in the backend. Safe to restart at
# any time, and it does not need the backend to be up first -- the page polls
# and shows "backend unreachable" until it appears.
#
# Logs: frontend.log

set -uo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
API="${API:-http://127.0.0.1:8000}"
SERVE_DIR=.frontend
LOG=frontend.log

# The UI reads window.KIMODO_API from config.js. Generating it dynamically
mkdir -p "$SERVE_DIR"
WEBUI="src/webui.html"
[ ! -f "$WEBUI" ] && WEBUI="webui.html"
cp -f "$WEBUI" "$SERVE_DIR/index.html"
printf 'window.KIMODO_API = "http://" + (location.hostname || "127.0.0.1") + ":8000";\n' > "$SERVE_DIR/config.js"

existing=$(pgrep -f "http.server $PORT" || true)
if [ -n "$existing" ]; then
    echo "stopping previous frontend (pid $existing)"
    # shellcheck disable=SC2086
    kill $existing 2>/dev/null || true
    sleep 1
fi

LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "=============================================="
echo " Kimodo-G1 UI (Local)  -> http://127.0.0.1:$PORT"
echo " Kimodo-G1 UI (Mobile) -> http://$LOCAL_IP:$PORT"
echo " backend API (Mobile)  -> http://$LOCAL_IP:8000"
echo " Ctrl-C to stop"
echo "=============================================="

if curl -s --max-time 3 "http://127.0.0.1:8000/api/status" >/dev/null 2>&1; then
    echo "backend: reachable"
else
    echo "backend: NOT reachable yet — start it in another terminal:  ./start_backend.sh"
    echo "         (the page will connect on its own once it comes up)"
fi

( sleep 2; xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true ) &

trap 'echo; echo "frontend stopped."; exit 0' INT TERM

cd "$SERVE_DIR"
exec python3 -u -m http.server "$PORT" --bind 0.0.0.0 2>&1 | tee -a "../$LOG"

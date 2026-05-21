#!/usr/bin/env bash
# StudioLite one-click launcher (Linux/macOS).
# Starts FastAPI (uvicorn) on :8000 and Next.js dev on :3000, waits for both
# to come up, opens the default browser, then tails interleaved logs in this
# terminal. Ctrl+C cleanly stops both servers.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/.mp/launcher"
mkdir -p "$LOG_DIR"
API_OUT="$LOG_DIR/api.out.log"
API_ERR="$LOG_DIR/api.err.log"
WEB_OUT="$LOG_DIR/web.out.log"
WEB_ERR="$LOG_DIR/web.err.log"
: > "$API_OUT"; : > "$API_ERR"; : > "$WEB_OUT"; : > "$WEB_ERR"

VENV_PY="$ROOT/venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "venv missing at $VENV_PY" >&2
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
    read -r -p "Press Enter to close..." _
    exit 1
fi

WEB_DIR="$ROOT/web"
if [[ ! -d "$WEB_DIR/node_modules" ]]; then
    echo "web/node_modules missing - running 'npm install' first..."
    (cd "$WEB_DIR" && npm install) || { echo "npm install failed" >&2; exit 1; }
fi

echo
echo "  StudioLite launcher"
echo "  ==================="
echo "  Root : $ROOT"
echo "  Logs : $LOG_DIR"
echo

port_in_use() {
    # bash /dev/tcp probe - no external tools needed
    (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1 && {
        exec 3<&- 3>&-
        return 0
    }
    return 1
}

for port in 8000 3000; do
    if port_in_use "$port"; then
        echo "Port $port is already in use. Close the existing server (or kill old uvicorn/node) and retry." >&2
        read -r -p "Press Enter to close..." _
        exit 1
    fi
done

echo "Starting API server  (uvicorn) on http://localhost:8000 ..."
"$VENV_PY" -u -m uvicorn api_server:app --host 127.0.0.1 --port 8000 \
    --ws-ping-interval 30 --ws-ping-timeout 90 \
    >"$API_OUT" 2>"$API_ERR" &
API_PID=$!

echo "Starting Next.js dev server on http://localhost:3000 ..."
(
    cd "$WEB_DIR"
    npm run dev
) >"$WEB_OUT" 2>"$WEB_ERR" &
WEB_PID=$!

TAIL_PID=""
cleanup() {
    local rc=$?
    echo
    echo "Stopping services..."
    [[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null || true
    for pid in "$API_PID" "$WEB_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            # kill child tree first (uvicorn workers, next dev subprocesses)
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 0.3
    for pid in "$API_PID" "$WEB_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    echo "Done."
    exit "$rc"
}
trap cleanup EXIT INT TERM

DEADLINE=$(( $(date +%s) + 180 ))
api_up=0
web_up=0
while (( $(date +%s) < DEADLINE )) && { (( api_up == 0 )) || (( web_up == 0 )); }; do
    if (( api_up == 0 )) && port_in_use 8000; then api_up=1; fi
    if (( web_up == 0 )) && port_in_use 3000; then web_up=1; fi
    api_tag="[..]"; (( api_up == 1 )) && api_tag="[OK]"
    web_tag="[..]"; (( web_up == 1 )) && web_tag="[OK]"
    printf "\r  api %s   web %s   " "$api_tag" "$web_tag"
    sleep 0.6
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo
        echo "API process exited unexpectedly."
        for f in "$API_ERR" "$API_OUT"; do
            if [[ -s "$f" ]]; then
                echo "--- $(basename "$f") (last 30 lines) ---"
                tail -n 30 "$f" | sed 's/^/  /'
            fi
        done
        exit 1
    fi
    if ! kill -0 "$WEB_PID" 2>/dev/null; then
        echo
        echo "Web process exited unexpectedly."
        for f in "$WEB_ERR" "$WEB_OUT"; do
            if [[ -s "$f" ]]; then
                echo "--- $(basename "$f") (last 30 lines) ---"
                tail -n 30 "$f" | sed 's/^/  /'
            fi
        done
        exit 1
    fi
done
echo

if (( api_up == 0 )); then echo "API did not respond on :8000 within 3 min. See $LOG_DIR"; exit 1; fi
if (( web_up == 0 )); then echo "Next.js did not respond on :3000 within 3 min. See $LOG_DIR"; exit 1; fi

echo
echo "  Both servers are up."
echo "  Web : http://localhost:3000"
echo "  API : http://localhost:8000/docs"
echo

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:3000" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
    open "http://localhost:3000" >/dev/null 2>&1 &
fi

echo "Press Ctrl+C to stop both servers."
echo "================================================="

# Tail all four log files with a tag prefix per service.
{
    tail -n 0 -F "$API_OUT" "$API_ERR" 2>/dev/null | sed -u 's/^/[api] /' &
    tail -n 0 -F "$WEB_OUT" "$WEB_ERR" 2>/dev/null | sed -u 's/^/[web] /' &
    wait
} &
TAIL_PID=$!

# Block until either server exits; trap handles teardown.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
    sleep 1
done
echo
echo "A server exited. Stopping the other."

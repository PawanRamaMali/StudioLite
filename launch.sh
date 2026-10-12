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
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA GPU detected. Run:" >&2
        echo "  python3 -m venv venv && source venv/bin/activate" >&2
        echo "  pip install -r requirements-cuda.txt && pip install -r requirements.txt" >&2
    else
        echo "No NVIDIA GPU detected - installing CPU-only torch wheels:" >&2
        echo "  python3 -m venv venv && source venv/bin/activate" >&2
        echo "  pip install -r requirements-cpu.txt && pip install -r requirements.txt" >&2
    fi
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

port_holder_pid() {
    # Return the pid of the process listening on $1, or empty.
    local pid=""
    if command -v lsof >/dev/null 2>&1; then
        pid="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -n 1)"
    elif command -v ss >/dev/null 2>&1; then
        pid="$(ss -ltnp 2>/dev/null | awk -v p=":$1\$" '$4 ~ p { print $0 }' \
               | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -n 1)"
    fi
    printf "%s" "$pid"
}

# Classify a port holder as "ours" (a stale StudioLite we can safely reclaim)
# or a stranger (another project's server we leave alone and route around).
# $1 = pid, $2 = 'api' or 'web'. Returns 0 when ours.
is_studiolite_holder() {
    local pid="$1" kind="$2"
    [[ -z "$pid" ]] && return 1
    local cmd=""
    if [[ -r "/proc/$pid/cmdline" ]]; then
        cmd="$(tr '\0' ' ' < /proc/$pid/cmdline)"
    elif command -v ps >/dev/null 2>&1; then
        cmd="$(ps -o command= -p "$pid" 2>/dev/null)"
    fi
    [[ -z "$cmd" ]] && return 1
    if [[ "$kind" == "api" ]]; then
        [[ "$cmd" == *"uvicorn api_server:app"* || "$cmd" == *"api_server.py"* ]]
        return
    fi
    if [[ "$kind" == "web" ]]; then
        [[ "$cmd" == *"next-server"* || ( "$cmd" == *"next"* && "$cmd" == *"$WEB_DIR"* ) ]]
        return
    fi
    return 1
}

find_free_port() {
    # $1 = default port, $2 = kind. Scans $1..$1+19 for a free port. If the
    # default is held by a stale StudioLite, reclaim it; otherwise warn and
    # move on. Prints the chosen port to stdout, or empty on failure.
    local start="$1" kind="$2"
    local port
    for (( port = start; port < start + 20; port++ )); do
        if ! port_in_use "$port"; then
            printf "%s" "$port"
            return 0
        fi
        if (( port == start )); then
            local pid="$(port_holder_pid "$port")"
            if is_studiolite_holder "$pid" "$kind"; then
                echo "  Port $port held by a stale StudioLite (pid $pid); reclaiming." >&2
                kill -TERM "$pid" 2>/dev/null || true
                sleep 0.4
                kill -KILL "$pid" 2>/dev/null || true
                sleep 0.2
                if ! port_in_use "$port"; then
                    printf "%s" "$port"
                    return 0
                fi
            elif [[ -n "$pid" ]]; then
                echo "  Port $port held by pid $pid; using an alternate port." >&2
            else
                echo "  Port $port appears taken; using an alternate port." >&2
            fi
        fi
    done
    return 1
}

API_PORT="$(find_free_port 8000 api)" || { echo "No free API port found in 8000-8019." >&2; read -r -p "Press Enter to close..." _; exit 1; }
WEB_PORT="$(find_free_port 3000 web)" || { echo "No free web port found in 3000-3019." >&2; read -r -p "Press Enter to close..." _; exit 1; }

API_URL="http://localhost:$API_PORT"
WEB_URL="http://localhost:$WEB_PORT"

# Mint the API token up front (api_server reuses .auth) so the web server
# can hand the same token to the UI; see web/app/layout.tsx.
AUTH_FILE="$ROOT/.auth"
if [[ -z "$(tr -d '[:space:]' 2>/dev/null < "$AUTH_FILE")" ]]; then
    (umask 077 && "$VENV_PY" -c "import secrets; print(secrets.token_urlsafe(32), end='')" > "$AUTH_FILE")
fi
STUDIOLITE_API_TOKEN="$(tr -d '[:space:]' < "$AUTH_FILE")"
export STUDIOLITE_API_TOKEN

# API's CORS allowlist defaults to :3000; when the launcher moves the web
# port, teach the API about the new origin before it boots or every browser
# fetch gets rejected at the CORS preflight.
export STUDIOLITE_ALLOWED_ORIGINS="http://localhost:$WEB_PORT,http://127.0.0.1:$WEB_PORT"

echo "Starting API server  (uvicorn) on $API_URL ..."
"$VENV_PY" -u -m uvicorn api_server:app --host 127.0.0.1 --port "$API_PORT" \
    --ws-ping-interval 30 --ws-ping-timeout 90 \
    >"$API_OUT" 2>"$API_ERR" &
API_PID=$!

# Both env vars reach the Next.js child: STUDIOLITE_API_URL feeds the dev
# rewrite target in next.config.ts, NEXT_PUBLIC_API_URL is the direct base
# the browser uses from web/lib/api.ts.
export STUDIOLITE_API_URL="$API_URL"
export NEXT_PUBLIC_API_URL="$API_URL"

echo "Starting Next.js dev server on $WEB_URL ..."
(
    cd "$WEB_DIR"
    # Loopback only: the page carries the API token (web/app/layout.tsx).
    npm run dev -- --hostname 127.0.0.1 --port "$WEB_PORT"
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
    if (( api_up == 0 )) && port_in_use "$API_PORT"; then api_up=1; fi
    if (( web_up == 0 )) && port_in_use "$WEB_PORT"; then web_up=1; fi
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

if (( api_up == 0 )); then echo "API did not respond on :$API_PORT within 3 min. See $LOG_DIR"; exit 1; fi
if (( web_up == 0 )); then echo "Next.js did not respond on :$WEB_PORT within 3 min. See $LOG_DIR"; exit 1; fi

echo
echo "  Both servers are up."
echo "  Web : $WEB_URL"
echo "  API : $API_URL/docs"
echo

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$WEB_URL" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
    open "$WEB_URL" >/dev/null 2>&1 &
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

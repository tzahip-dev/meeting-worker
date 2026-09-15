#!/bin/bash
# Container entrypoint.
#
# The image bakes vast_worker.py + runpod_handler.py, but a fresh copy is pulled
# from the VPS receiver (/static/, unauthenticated read-only) when it is
# reachable. That way worker-code fixes take effect on the next cold start
# without a 10-minute image rebuild. A downloaded file is only accepted if it
# parses as Python, so a truncated transfer can never break the worker.
set -u

VPS_URL="${VPS_URL:-}"
export CURL_CA_BUNDLE=""
export SSL_CERT_FILE=""

if [ -n "$VPS_URL" ]; then
    for f in vast_worker.py runpod_handler.py; do
        tmp="/${f}.new"
        if curl -sfk --max-time 25 "${VPS_URL}/static/${f}" -o "$tmp" \
           && [ -s "$tmp" ] \
           && python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$tmp" 2>/dev/null; then
            mv "$tmp" "/${f}"
            echo "[start] refreshed ${f} from VPS"
        else
            rm -f "$tmp"
            echo "[start] keeping baked ${f}"
        fi
    done
else
    echo "[start] VPS_URL unset — using baked worker code"
fi

exec python3 -u /runpod_handler.py

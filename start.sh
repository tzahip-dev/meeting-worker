#!/bin/bash
# Container entrypoint.
#
# The image bakes vast_worker.py + runpod_handler.py, but a fresh copy is pulled
# from the VPS receiver (/static/, unauthenticated read-only) when it is
# reachable. That way worker-code fixes take effect on the next cold start
# without a 10-minute image rebuild. A downloaded file is only accepted if it
# parses as Python, so a truncated transfer can never break the worker.
#
# IMPORTANT — do NOT export empty SSL_CERT_FILE / CURL_CA_BUNDLE here.
# An empty SSL_CERT_FILE makes rustls/reqwest fail to build its TLS connector,
# which surfaces as `Reqwest error: builder error` from hf_xet and kills every
# HuggingFace download. `curl -k` already handles the self-signed VPS cert, and
# the worker's own HTTP calls pass verify=False, so no global env is needed.
set -u

VPS_URL="${VPS_URL:-}"

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

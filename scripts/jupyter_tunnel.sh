#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${1:-runpod-jlens}"
LOCAL_PORT="${LOCAL_JUPYTER_PORT:-8888}"
REMOTE_PORT="${JUPYTER_PORT:-8889}"

echo "Forwarding http://127.0.0.1:${LOCAL_PORT} to ${SSH_HOST}:127.0.0.1:${REMOTE_PORT}"
echo "Keep this terminal open. Stop the tunnel with Ctrl-C."

exec ssh \
  -o ExitOnForwardFailure=yes \
  -N \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "$SSH_HOST"

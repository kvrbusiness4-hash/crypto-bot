#!/usr/bin/env bash
set -euo pipefail

echo "Starting loop runner…"
while true; do
  python -u bot_paid.py
  code=$?
  echo "Bot exited with code ${code}. Restarting in 5s…"
  sleep 5
done

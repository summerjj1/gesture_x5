#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec /usr/bin/python3 -m gesture_x5.apps.web \
  --device /dev/video0 \
  --source-width 4000 \
  --source-height 1200 \
  --source-fps 30 \
  --port 8088 \
  "$@"

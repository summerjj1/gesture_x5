#!/usr/bin/env bash
set -euo pipefail

# A normal SSH session has no DISPLAY, while the RDK's local XFCE desktop runs
# on :0. Bind OpenCV's window to that desktop automatically when available.
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
  export XAUTHORITY="${XAUTHORITY:-/home/sunrise/.Xauthority}"
fi

cd "$(dirname "$0")"
exec /usr/bin/python3 -m gesture_x5.apps.camera "$@"

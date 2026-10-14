#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
  export XAUTHORITY="${XAUTHORITY:-/home/sunrise/.Xauthority}"
fi

cd "$(dirname "$0")"
if [[ ! -f ./libhand_roi_bridge.so ]]; then
  ./build_hand_roi_bridge.sh
fi
exec /usr/bin/python3 ./gesture_opencv_bpu.py "$@"

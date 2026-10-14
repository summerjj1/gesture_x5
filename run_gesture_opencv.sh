#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
  export XAUTHORITY="${XAUTHORITY:-/home/sunrise/.Xauthority}"
fi

cd "$(dirname "$0")"
if [[ ! -f ./build/libhand_roi_bridge.so || ./native/hand_roi_bridge.c -nt ./build/libhand_roi_bridge.so ]]; then
  ./build_hand_roi_bridge.sh
fi
exec /usr/bin/python3 -m gesture_x5.apps.gesture "$@"

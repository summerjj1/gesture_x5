#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
gcc -O3 -shared -fPIC \
  ./hand_roi_bridge.c \
  -o ./libhand_roi_bridge.so \
  -ldnn
echo "Built: $(pwd)/libhand_roi_bridge.so"

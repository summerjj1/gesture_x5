#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${project_root}/build"

gcc -O3 -shared -fPIC \
  "${project_root}/native/hand_roi_bridge.c" \
  -o "${project_root}/build/libhand_roi_bridge.so" \
  -ldnn -lm

echo "Built: ${project_root}/build/libhand_roi_bridge.so"

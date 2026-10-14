#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

/usr/bin/python3 -m compileall -q gesture_x5 \
  gesture_opencv_bpu.py camera_opencv.py camera_web.py
/usr/bin/python3 -m unittest discover -s tests -v
bash -n ./*.sh scripts/*.sh

echo "All static checks and unit tests passed."

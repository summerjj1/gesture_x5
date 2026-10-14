#!/usr/bin/env bash
set -eo pipefail

# The TROS setup script probes optional variables that may be unset, so enable
# nounset only after the environment has been loaded.
source /opt/tros/humble/setup.bash
set -u
cd "$(dirname "$0")"
exec ros2 launch ./launch/camera_web.launch.py

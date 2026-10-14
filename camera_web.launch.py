#!/usr/bin/env python3
"""Compatibility wrapper for the organized ROS 2 launch file."""

import runpy
from pathlib import Path


_LAUNCH_FILE = Path(__file__).resolve().parent / "launch" / "camera_web.launch.py"
generate_launch_description = runpy.run_path(str(_LAUNCH_FILE))[
    "generate_launch_description"
]

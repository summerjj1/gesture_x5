#!/usr/bin/env python3
"""Launch the detected DECXIN camera and the TROS web viewer together."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    shm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_shm"),
                "launch",
                "hobot_shm.launch.py",
            )
        )
    )

    camera = Node(
        package="hobot_usb_cam",
        executable="hobot_usb_cam",
        name="gesture_x5_usb_camera",
        output="screen",
        parameters=[
            {"video_device": "/dev/video0"},
            {"image_width": 4000},
            {"image_height": 1200},
            {"framerate": 30},
            {"pixel_format": "mjpeg"},
            {"io_method": "mmap"},
            {"zero_copy": False},
        ],
        arguments=["--ros-args", "--log-level", "warn"],
    )

    web = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("websocket"),
                "launch",
                "websocket.launch.py",
            )
        ),
        launch_arguments={
            "websocket_image_topic": "/image",
            "websocket_image_type": "mjpeg",
            "websocket_only_show_image": "True",
            "websocket_output_fps": "15",
        }.items(),
    )

    return LaunchDescription([shm_launch, camera, web])

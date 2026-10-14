#!/usr/bin/env bash
set -euo pipefail

WITH_ROS_GESTURE=0
if [[ "${1:-}" == "--with-ros-gesture" ]]; then
  WITH_ROS_GESTURE=1
elif [[ $# -gt 0 ]]; then
  echo "用法: $0 [--with-ros-gesture]" >&2
  exit 2
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "警告：本项目的 X5 BPU 模型和 libdnn 桥接层需要在 RDK X5/aarch64 上运行。" >&2
fi

sudo apt update
sudo apt install -y \
  gcc \
  python3-numpy \
  python3-opencv \
  tros-humble-palm-detection-mediapipe \
  tros-humble-hand-landmarks-mediapipe

if [[ $WITH_ROS_GESTURE -eq 1 ]]; then
  sudo apt install -y tros-humble-hand-gesture-detection
fi

cd "$(dirname "$0")"

/usr/bin/python3 -c \
  "import cv2, numpy; from hobot_dnn import pyeasy_dnn; print('Python/OpenCV/BPU dependencies OK')"

test -f /opt/tros/humble/lib/palm_detection_mediapipe/config/palm_det_192_192.bin
test -f /opt/tros/humble/lib/hand_landmarks_mediapipe/config/hand_224_224.bin

./build_hand_roi_bridge.sh
echo "Gesture_X5 dependencies installed successfully."

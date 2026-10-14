#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--with-official-gesture" ) ]]; then
  echo "用法: $0 [--with-official-gesture]" >&2
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

if [[ "${1:-}" == "--with-official-gesture" ]]; then
  sudo apt install -y tros-humble-hand-gesture-detection
fi

cd "${project_root}"

/usr/bin/python3 -c \
  "import cv2, numpy; from hobot_dnn import pyeasy_dnn; print('Python/OpenCV/BPU dependencies OK')"

test -f /opt/tros/humble/lib/palm_detection_mediapipe/config/palm_det_192_192.bin
test -f /opt/tros/humble/lib/hand_landmarks_mediapipe/config/hand_224_224.bin
if [[ "${1:-}" == "--with-official-gesture" ]]; then
  test -f /opt/tros/humble/lib/hand_gesture_detection/config/gestureDet_8x21.hbm
fi

./scripts/build_native.sh
echo "Gesture_X5 dependencies installed successfully."

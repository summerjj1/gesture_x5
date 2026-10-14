#!/usr/bin/env python3
"""Display the RDK X5 USB camera directly with OpenCV."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2


def device_index(device: str) -> int | str:
    if device.startswith("/dev/video") and device[10:].isdigit():
        return int(device[10:])
    if device.isdigit():
        return int(device)
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct OpenCV camera viewer")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="start recording immediately; optional output .avi path",
    )
    parser.add_argument(
        "--record-dir",
        default="recordings",
        help="directory for auto-named recordings and screenshots",
    )
    return parser.parse_args()


def timestamped_path(directory: Path, prefix: str, suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{prefix}_{stamp}{suffix}"


def main() -> None:
    args = parse_args()
    if not os.environ.get("DISPLAY"):
        raise SystemExit(
            "当前没有 DISPLAY。若要显示到板载桌面，请设置 DISPLAY=:0 和 "
            "XAUTHORITY=/home/sunrise/.Xauthority。"
        )
    cap = cv2.VideoCapture(device_index(args.device), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise SystemExit(
            f"无法打开 {args.device}；请先停止占用相机的 ROS 或其他 OpenCV 程序"
        )

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    print(
        "按键：1=完整画面，2=左半画面，3=右半画面，"
        "R=开始/停止录像，S=截图，Q/ESC=退出"
    )

    window = "Gesture_X5 OpenCV Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    record_dir = Path(args.record_dir).expanduser()
    if not record_dir.is_absolute():
        record_dir = Path.cwd() / record_dir
    record_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    record_path = None
    pending_record = args.record
    view = "full"
    frames = 0
    measured_fps = 0.0
    started = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取图像失败")
                break

            if pending_record is not None:
                if pending_record:
                    record_path = Path(pending_record).expanduser()
                    if not record_path.is_absolute():
                        record_path = Path.cwd() / record_path
                    if record_path.suffix.lower() != ".avi":
                        record_path = record_path.with_suffix(".avi")
                else:
                    record_path = timestamped_path(record_dir, "camera", ".avi")
                record_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(record_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    float(args.fps),
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise RuntimeError(f"无法创建录像文件：{record_path}")
                print(f"开始录像：{record_path}")
                pending_record = None

            if writer is not None:
                writer.write(frame)

            frames += 1
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= 1.0:
                measured_fps = frames / elapsed
                frames = 0
                started = now

            source_width = frame.shape[1]
            if view == "left":
                shown = frame[:, : source_width // 2]
            elif view == "right":
                shown = frame[:, source_width // 2 :]
            else:
                shown = frame

            if args.display_width > 0 and shown.shape[1] > args.display_width:
                display_height = round(
                    shown.shape[0] * args.display_width / shown.shape[1]
                )
                shown = cv2.resize(
                    shown,
                    (args.display_width, display_height),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.putText(
                shown,
                f"{view.upper()}  {measured_fps:.1f} FPS",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if writer is not None:
                cv2.circle(shown, (shown.shape[1] - 32, 28), 10, (0, 0, 255), -1)
                cv2.putText(
                    shown,
                    "REC",
                    (shown.shape[1] - 100, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("1"):
                view = "full"
            elif key == ord("2"):
                view = "left"
            elif key == ord("3"):
                view = "right"
            elif key in (ord("r"), ord("R")):
                if writer is None:
                    pending_record = ""
                else:
                    writer.release()
                    writer = None
                    print(f"录像已保存：{record_path}")
                    record_path = None
            elif key in (ord("s"), ord("S")):
                screenshot = timestamped_path(record_dir, "frame", ".jpg")
                if cv2.imwrite(str(screenshot), frame):
                    print(f"截图已保存：{screenshot}")
                else:
                    print(f"截图保存失败：{screenshot}")
    finally:
        if writer is not None:
            writer.release()
            print(f"录像已保存：{record_path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

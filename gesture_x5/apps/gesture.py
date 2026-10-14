#!/usr/bin/env python3
"""OpenCV camera + MediaPipe palm/21-landmark models on the RDK X5 BPU.

This program does not use ROS.  It uses the D-Robotics converted MediaPipe
models already installed on the board. Palm detection uses pyeasy_dnn and the
21-landmark resizer model uses a tiny ctypes bridge around hbDNNRoiInfer.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn

from gesture_x5.core.control import GestureRecordController
from gesture_x5.core.video import (
    encoding_for_format,
    resolve_video_path,
    video_format_names,
)


PALM_MODEL = "/opt/tros/humble/lib/palm_detection_mediapipe/config/palm_det_192_192.bin"
HAND_MODEL = "/opt/tros/humble/lib/hand_landmarks_mediapipe/config/hand_224_224.bin"
GESTURE_MODEL = (
    "/opt/tros/humble/lib/hand_gesture_detection/config/gestureDet_8x21.hbm"
)

PALM_SIZE = 192
HAND_SIZE = 224
HAND_ROI_MAX = 640
GESTURE_SEQUENCE_LENGTH = 8
GESTURE_OUTPUT_SIZE = 23
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_LIBRARY = PROJECT_ROOT / "build" / "libhand_roi_bridge.so"

GESTURE_NAMES = {
    2: "Thumb Up",
    3: "Victory",
    4: "Mute",
    5: "Palm",
    11: "Okay",
    12: "Thumb Left",
    13: "Thumb Right",
    14: "Awesome",
}
GESTURE_MERGE_GROUPS = (
    (0, 1, 2, 3, 6, 12, 13, 14, 15, 16),
    (),
    (4,),
    (5,),
    (7,),
    (8, 9, 10),
    (),
    (),
    (),
    (),
    (),
    (17,),
    (18,),
    (19,),
    (20,),
    (11,),
    (21,),
    (22,),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class Palm:
    box: tuple[int, int, int, int]
    score: float
    keypoints: np.ndarray


@dataclass
class Hand:
    landmarks: np.ndarray
    score: float
    handedness: float
    roi: tuple[int, int, int, int]
    gesture: str
    gesture_score: float
    track_id: int


@dataclass
class GestureTrack:
    track_id: int
    center: np.ndarray
    roi: tuple[int, int, int, int]
    landmarks: deque[np.ndarray]
    boxes: deque[tuple[int, int, int, int]]
    last_seen: float


class _LegacyGestureRecordController:
    def __init__(
        self,
        palm_seconds: float,
        missing_seconds: float,
        thumb_seconds: float,
        gesture_grace_seconds: float,
    ) -> None:
        self.palm_seconds = palm_seconds
        self.missing_seconds = missing_seconds
        self.thumb_seconds = thumb_seconds
        self.gesture_grace_seconds = gesture_grace_seconds
        self.palm_since: float | None = None
        self.palm_last_seen: float | None = None
        self.missing_since: float | None = None
        self.thumb_since: float | None = None
        self.thumb_last_seen: float | None = None
        self.alarm = False
        self.alarm_reported = False
        self.status = "AUTO: waiting for 2 Palms"

    @staticmethod
    def elapsed(since: float | None, now: float) -> float:
        return 0.0 if since is None else max(0.0, now - since)

    def update(
        self,
        hands: list[Hand] | None,
        recording: bool,
        now: float,
    ) -> tuple[bool, bool]:
        """Return (start, stop); hands=None pauses gesture control."""
        if hands is None:
            self.palm_since = None
            self.palm_last_seen = None
            self.missing_since = None
            self.thumb_since = None
            self.thumb_last_seen = None
            self.alarm = False
            self.alarm_reported = False
            self.status = "AUTO: paused (control eye is not active)"
            return False, False

        palm_count = sum(hand.gesture == "Palm" for hand in hands)
        thumb_count = sum(hand.gesture == "Thumb Up" for hand in hands)

        if not recording:
            self.missing_since = None
            self.thumb_since = None
            self.thumb_last_seen = None
            self.alarm = False
            self.alarm_reported = False
            if palm_count >= 2:
                if self.palm_since is None:
                    self.palm_since = now
                self.palm_last_seen = now
                held = self.elapsed(self.palm_since, now)
                self.status = (
                    f"AUTO START: 2 Palms {min(held, self.palm_seconds):.1f}/"
                    f"{self.palm_seconds:.1f}s"
                )
                if held >= self.palm_seconds:
                    self.palm_since = None
                    self.palm_last_seen = None
                    self.status = "AUTO START: recording"
                    return True, False
            else:
                gap = self.elapsed(self.palm_last_seen, now)
                if (
                    self.palm_since is not None
                    and self.palm_last_seen is not None
                    and gap <= self.gesture_grace_seconds
                ):
                    held = self.elapsed(self.palm_since, now)
                    self.status = (
                        f"AUTO START: Palm tracking "
                        f"{min(held, self.palm_seconds):.1f}/{self.palm_seconds:.1f}s "
                        f"(grace {gap:.1f}s)"
                    )
                else:
                    self.palm_since = None
                    self.palm_last_seen = None
                    self.status = f"AUTO: waiting for 2 Palms ({palm_count}/2)"
            return False, False

        self.palm_since = None
        self.palm_last_seen = None
        if len(hands) == 0:
            if self.missing_since is None:
                self.missing_since = now
            missing = self.elapsed(self.missing_since, now)
            if missing >= self.missing_seconds:
                self.alarm = True
                self.status = f"ALARM: both hands missing {missing:.1f}s"
                if not self.alarm_reported:
                    print("\a报警：左右手同时消失超过设定时间", flush=True)
                    self.alarm_reported = True
            else:
                self.status = (
                    f"AUTO REC: hands missing {missing:.1f}/"
                    f"{self.missing_seconds:.1f}s"
                )
        else:
            self.missing_since = None
            self.alarm = False
            self.alarm_reported = False
            self.status = f"AUTO REC: hands visible ({len(hands)})"

        if thumb_count >= 2:
            if self.thumb_since is None:
                self.thumb_since = now
            self.thumb_last_seen = now
            held = self.elapsed(self.thumb_since, now)
            self.status = (
                f"AUTO STOP: 2 Thumb Up {min(held, self.thumb_seconds):.1f}/"
                f"{self.thumb_seconds:.1f}s"
            )
            if held >= self.thumb_seconds:
                self.thumb_since = None
                self.thumb_last_seen = None
                self.missing_since = None
                self.alarm = False
                self.alarm_reported = False
                self.status = "AUTO STOP: recording saved"
                return False, True
        else:
            gap = self.elapsed(self.thumb_last_seen, now)
            if (
                self.thumb_since is not None
                and self.thumb_last_seen is not None
                and gap <= self.gesture_grace_seconds
            ):
                held = self.elapsed(self.thumb_since, now)
                self.status = (
                    f"AUTO STOP: Thumb Up tracking {min(held, self.thumb_seconds):.1f}/"
                    f"{self.thumb_seconds:.1f}s (grace {gap:.1f}s)"
                )
            else:
                self.thumb_since = None
                self.thumb_last_seen = None
        return False, False

    def draw(self, image: np.ndarray) -> None:
        if self.alarm:
            banner_width = min(image.shape[1] - 1, 1900)
            cv2.rectangle(image, (0, 56), (banner_width, 150), (0, 0, 220), -1)
            cv2.putText(
                image,
                "ALARM: BOTH HANDS LOST",
                (24, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            return
        cv2.putText(
            image,
            self.status,
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


class HandRoiModel:
    def __init__(self, model_path: str) -> None:
        if not BRIDGE_LIBRARY.exists():
            raise RuntimeError(
                f"缺少 {BRIDGE_LIBRARY.name}，请先运行 ./build_hand_roi_bridge.sh"
            )
        self.lib = ctypes.CDLL(str(BRIDGE_LIBRARY))
        self.lib.hand_roi_create.argtypes = [ctypes.c_char_p]
        self.lib.hand_roi_create.restype = ctypes.c_void_p
        self.lib.hand_roi_ready.argtypes = [ctypes.c_void_p]
        self.lib.hand_roi_ready.restype = ctypes.c_int
        self.lib.hand_roi_last_error.argtypes = [ctypes.c_void_p]
        self.lib.hand_roi_last_error.restype = ctypes.c_char_p
        self.lib.hand_roi_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.hand_roi_infer.restype = ctypes.c_int
        self.lib.hand_roi_destroy.argtypes = [ctypes.c_void_p]
        self.lib.hand_roi_destroy.restype = None
        self.context = self.lib.hand_roi_create(model_path.encode())
        if not self.context or not self.lib.hand_roi_ready(self.context):
            error = self.last_error()
            self.close()
            raise RuntimeError(f"加载 21 点 BPU 模型失败：{error}")

    def last_error(self) -> str:
        if not self.context:
            return "null context"
        value = self.lib.hand_roi_last_error(self.context)
        return value.decode(errors="replace") if value else "unknown error"

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = image.shape[:2]
        scale = min(1.0, HAND_ROI_MAX / max(width, height))
        target_width = max(18, int(width * scale) // 2 * 2)
        target_height = max(18, int(height * scale) // 2 * 2)
        if target_width != width or target_height != height:
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
        nv12 = bgr_to_nv12(image)
        landmarks = np.empty(63, dtype=np.float32)
        score = ctypes.c_float()
        handedness = ctypes.c_float()
        ret = self.lib.hand_roi_infer(
            self.context,
            nv12.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            target_width,
            target_height,
            landmarks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(score),
            ctypes.byref(handedness),
        )
        if ret != 0:
            raise RuntimeError(f"21 点 ROI 推理失败：{self.last_error()}")
        return landmarks.reshape(21, 3), float(score.value), float(handedness.value)

    def close(self) -> None:
        if getattr(self, "context", None):
            self.lib.hand_roi_destroy(self.context)
            self.context = None

    def __del__(self) -> None:
        self.close()


class OfficialGestureModel:
    def __init__(self, model_path: str, threshold: float) -> None:
        if not BRIDGE_LIBRARY.exists():
            raise RuntimeError(
                f"缺少 {BRIDGE_LIBRARY.name}，请先运行 ./build_hand_roi_bridge.sh"
            )
        self.threshold = threshold
        self.lib = ctypes.CDLL(str(BRIDGE_LIBRARY))
        self.lib.gesture_model_create.argtypes = [ctypes.c_char_p]
        self.lib.gesture_model_create.restype = ctypes.c_void_p
        self.lib.gesture_model_ready.argtypes = [ctypes.c_void_p]
        self.lib.gesture_model_ready.restype = ctypes.c_int
        self.lib.gesture_model_last_error.argtypes = [ctypes.c_void_p]
        self.lib.gesture_model_last_error.restype = ctypes.c_char_p
        self.lib.gesture_model_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.gesture_model_infer.restype = ctypes.c_int
        self.lib.gesture_model_destroy.argtypes = [ctypes.c_void_p]
        self.lib.gesture_model_destroy.restype = None
        self.context = self.lib.gesture_model_create(model_path.encode())
        if not self.context or not self.lib.gesture_model_ready(self.context):
            error = self.last_error()
            self.close()
            raise RuntimeError(f"加载官方 gestureDet BPU 模型失败：{error}")

    def last_error(self) -> str:
        if not self.context:
            return "null context"
        value = self.lib.gesture_model_last_error(self.context)
        return value.decode(errors="replace") if value else "unknown error"

    def infer(self, features: np.ndarray) -> tuple[str, float]:
        if features.shape != (GESTURE_SEQUENCE_LENGTH, 21, 3):
            raise ValueError(f"错误的 gestureDet 输入尺寸：{features.shape}")
        features = np.ascontiguousarray(features, dtype=np.float32)
        logits = np.empty(GESTURE_OUTPUT_SIZE, dtype=np.float32)
        ret = self.lib.gesture_model_infer(
            self.context,
            features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if ret != 0:
            raise RuntimeError(f"gestureDet 推理失败：{self.last_error()}")
        probabilities = np.exp(logits - float(np.max(logits)))
        probabilities /= max(float(np.sum(probabilities)), 1e-12)
        group_scores = np.asarray(
            [
                float(np.sum(probabilities[list(group)])) if group else 0.0
                for group in GESTURE_MERGE_GROUPS
            ],
            dtype=np.float32,
        )
        gesture_id = int(np.argmax(group_scores))
        score = float(group_scores[gesture_id])
        if gesture_id not in GESTURE_NAMES or score < self.threshold:
            return "Unknown", score
        return GESTURE_NAMES[gesture_id], score

    def close(self) -> None:
        if getattr(self, "context", None):
            self.lib.gesture_model_destroy(self.context)
            self.context = None

    def __del__(self) -> None:
        self.close()


class OfficialGestureClassifier:
    def __init__(self, model_path: str, threshold: float) -> None:
        self.model = OfficialGestureModel(model_path, threshold)
        self.tracks: dict[str, dict[int, GestureTrack]] = {}
        self.next_track_id = 1
        self.track_timeout = 1.0
        self.match_distance = 1.5

    @staticmethod
    def hand_center(hand: Hand) -> np.ndarray:
        return hand.landmarks[9, :2].copy()

    @staticmethod
    def hand_size(roi: tuple[int, int, int, int]) -> float:
        return float(max(roi[2] - roi[0], roi[3] - roi[1], 1))

    def assign_tracks(
        self, hands: list[Hand], stream_id: str, now: float
    ) -> list[tuple[Hand, GestureTrack]]:
        tracks = self.tracks.setdefault(stream_id, {})
        for track_id in list(tracks):
            if now - tracks[track_id].last_seen > self.track_timeout:
                del tracks[track_id]

        candidates = []
        for hand_index, hand in enumerate(hands):
            center = self.hand_center(hand)
            for track_id, track in tracks.items():
                scale = max(self.hand_size(hand.roi), self.hand_size(track.roi))
                relative_distance = float(np.linalg.norm(center - track.center)) / scale
                if relative_distance <= self.match_distance:
                    candidates.append((relative_distance, hand_index, track_id))

        assignments: dict[int, GestureTrack] = {}
        used_tracks: set[int] = set()
        for _, hand_index, track_id in sorted(candidates):
            if hand_index in assignments or track_id in used_tracks:
                continue
            assignments[hand_index] = tracks[track_id]
            used_tracks.add(track_id)

        for hand_index, hand in enumerate(hands):
            if hand_index in assignments:
                continue
            track = GestureTrack(
                track_id=self.next_track_id,
                center=self.hand_center(hand),
                roi=hand.roi,
                landmarks=deque(maxlen=GESTURE_SEQUENCE_LENGTH),
                boxes=deque(maxlen=GESTURE_SEQUENCE_LENGTH),
                last_seen=now,
            )
            tracks[track.track_id] = track
            assignments[hand_index] = track
            self.next_track_id += 1
        return [(hand, assignments[index]) for index, hand in enumerate(hands)]

    @staticmethod
    def make_features(track: GestureTrack) -> np.ndarray:
        sequence = np.stack(tuple(track.landmarks), axis=0)
        # gestureDet uses a legacy 21-point layout. Indices 2..20 match
        # MediaPipe, but its first two slots are wrist anchors rather than
        # MediaPipe's wrist-center/thumb-CMC pair. Use the wrist center for
        # both anchors before applying the official normalization.
        wrist = sequence[:, 0, :2].copy()
        sequence[:, 0, :2] = wrist
        sequence[:, 1, :2] = wrist
        center = sequence[0, 9, :2].copy()
        max_border = max(
            max(box[2] - box[0], box[3] - box[1]) for box in track.boxes
        )
        features = np.zeros((GESTURE_SEQUENCE_LENGTH, 21, 3), dtype=np.float32)
        features[:, :, :2] = (sequence[:, :, :2] - center) / max(max_border, 1)
        features[:, :, 2] = 1.0
        return features

    def process(
        self,
        hands: list[Hand],
        stream_id: str,
        bootstrap: bool = False,
    ) -> None:
        now = time.monotonic()
        for hand, track in self.assign_tracks(hands, stream_id, now):
            track.center = self.hand_center(hand)
            track.roi = hand.roi
            track.last_seen = now
            track.landmarks.append(hand.landmarks.copy())
            track.boxes.append(hand.roi)
            if bootstrap and len(track.landmarks) == 1:
                while len(track.landmarks) < GESTURE_SEQUENCE_LENGTH:
                    track.landmarks.append(hand.landmarks.copy())
                    track.boxes.append(hand.roi)
            hand.track_id = track.track_id
            if len(track.landmarks) < GESTURE_SEQUENCE_LENGTH:
                hand.gesture = f"Warmup {len(track.landmarks)}/{GESTURE_SEQUENCE_LENGTH}"
                hand.gesture_score = 0.0
                continue
            hand.gesture, hand.gesture_score = self.model.infer(
                self.make_features(track)
            )

    def close(self) -> None:
        self.model.close()


def device_index(device: str) -> int | str:
    if device.startswith("/dev/video") and device[10:].isdigit():
        return int(device[10:])
    if device.isdigit():
        return int(device)
    return device


def timestamped_path(directory: Path, prefix: str, suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{prefix}_{stamp}{suffix}"


def make_record_frame(
    regions: list[tuple[str, np.ndarray]], max_width: int
) -> np.ndarray:
    frame = (
        np.hstack([region for _, region in regions])
        if len(regions) == 2
        else regions[0][1].copy()
    )
    if max_width > 0 and frame.shape[1] > max_width:
        height = round(frame.shape[0] * max_width / frame.shape[1])
        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
    return frame


def bgr_to_nv12(image: np.ndarray) -> np.ndarray:
    """Convert an even-sized BGR image into a packed NV12 byte vector."""
    height, width = image.shape[:2]
    if width % 2 or height % 2:
        raise ValueError(f"NV12 requires even dimensions, got {width}x{height}")
    i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y_size = width * height
    uv_size = y_size // 4
    y_plane = i420[:y_size]
    u_plane = i420[y_size : y_size + uv_size]
    v_plane = i420[y_size + uv_size : y_size + 2 * uv_size]
    uv_plane = np.empty(uv_size * 2, dtype=np.uint8)
    uv_plane[0::2] = u_plane
    uv_plane[1::2] = v_plane
    return np.ascontiguousarray(np.concatenate((y_plane, uv_plane)))


def make_anchors() -> np.ndarray:
    """Generate the exact 2016 anchor centers used by the official model."""
    anchors: list[tuple[float, float]] = []
    for grid_size, repetitions in ((24, 2), (12, 6)):
        for y in range(grid_size):
            for x in range(grid_size):
                center = ((x + 0.5) / grid_size, (y + 0.5) / grid_size)
                anchors.extend([center] * repetitions)
    result = np.asarray(anchors, dtype=np.float32)
    if result.shape != (2016, 2):
        raise RuntimeError(f"Unexpected anchor shape: {result.shape}")
    return result


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def letterbox_top_left(image: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Resize with preserved aspect ratio and pad on bottom/right."""
    height, width = image.shape[:2]
    scale = size / max(width, height)
    resized_width = max(2, min(size, int(width * scale)))
    resized_height = max(2, min(size, int(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    return padded, float(max(width, height))


def normalize_indices(indices) -> list[int]:
    if indices is None:
        return []
    values = np.asarray(indices).reshape(-1)
    return [int(value) for value in values]


def clip_box(box: tuple[float, float, float, float], width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def move_box(
    box: tuple[int, int, int, int],
    direction: np.ndarray,
    width: int,
    height: int,
    scale: float = 2.6,
    offset: float = 0.5,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    angle = math.atan2(float(direction[1]), float(direction[0]))
    distance = offset * box_height
    center_x += distance * math.cos(angle)
    center_y += distance * math.sin(angle)
    side = max(box_width, box_height) * scale
    return clip_box(
        (
            center_x - side / 2.0,
            center_y - side / 2.0,
            center_x + side / 2.0,
            center_y + side / 2.0,
        ),
        width,
        height,
    )


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def finger_extended(points: np.ndarray, tip: int, pip: int, mcp: int, thumb: bool) -> bool:
    a = points[mcp, :2]
    b = points[pip, :2]
    c = points[tip, :2]
    ba = a - b
    bc = c - b
    if thumb:
        d = points[1, :2]
        da = a - d
        denominator = np.linalg.norm(da) * np.linalg.norm(bc)
        if denominator < 1e-6:
            return False
        cosine = float(np.clip(np.dot(da, bc) / denominator, -1.0, 1.0))
        return math.degrees(math.acos(cosine)) < 45.0
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-6:
        return False
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    return angle > 160.0 and distance(a, c) > distance(a, b)


def recognize_gesture(points: np.ndarray) -> str:
    joints = ((4, 3, 2), (8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17))
    fingers = tuple(
        finger_extended(points, tip, pip, mcp, index == 0)
        for index, (tip, pip, mcp) in enumerate(joints)
    )
    if fingers == (False, True, True, False, False):
        return "Victory"
    if fingers == (False, True, False, False, False):
        return "Mute"
    if fingers == (True, True, True, True, True):
        return "Palm"
    if fingers in ((False, False, True, True, True), (True, False, True, True, True)):
        hand_scale = max(distance(points[0], points[9]), 1.0)
        if distance(points[4], points[8]) < 0.35 * hand_scale:
            return "Okay"
    if fingers == (True, False, False, False, True):
        return "Awesome"
    thumb_vector = points[4, :2] - points[2, :2]
    hand_scale = max(distance(points[0], points[9]), 1.0)
    thumb_points_up = (
        thumb_vector[1] < -0.25 * hand_scale
        and abs(float(thumb_vector[1])) >= 0.7 * abs(float(thumb_vector[0]))
        and points[4, 1] < min(points[2, 1], points[3, 1])
    )
    if thumb_points_up and sum(fingers[1:]) <= 1:
        return "Thumb Up"
    if fingers == (True, False, False, False, False):
        vector = points[4, :2] - points[1, :2]
        theta = -math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        if 0.0 < theta < 45.0:
            return "Thumb Right"
        if 45.0 <= theta < 135.0:
            return "Thumb Up"
        if theta >= 135.0 or theta < -135.0:
            return "Thumb Left"
    return "Unknown"


class BpuHandPipeline:
    def __init__(
        self,
        palm_score: float,
        hand_score: float,
        max_hands: int,
        palm_tiles: int,
        gesture_backend: str,
        gesture_model: str,
        gesture_score: float,
    ) -> None:
        print("Loading BPU palm model...")
        self.palm_model = dnn.load(PALM_MODEL)[0]
        print("Loading BPU 21-landmark model...")
        self.hand_model = HandRoiModel(HAND_MODEL)
        self.gesture_backend = gesture_backend
        self.gesture_classifier = None
        if gesture_backend == "official":
            print("Loading official BPU gestureDet model...")
            self.gesture_classifier = OfficialGestureClassifier(
                gesture_model, gesture_score
            )
        self.anchors = make_anchors()
        self.palm_score = palm_score
        self.hand_score = hand_score
        self.max_hands = max_hands
        self.palm_tiles = max(1, palm_tiles)

    @staticmethod
    def output_array(output) -> np.ndarray:
        return np.asarray(output.buffer, dtype=np.float32).reshape(-1)

    def detect_palms_once(
        self,
        image: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[Palm]:
        model_image, coordinate_scale = letterbox_top_left(image, PALM_SIZE)
        outputs = self.palm_model.forward(bgr_to_nv12(model_image))
        boxes_raw = self.output_array(outputs[0]).reshape(2016, 18)
        scores = sigmoid(self.output_array(outputs[1]))
        candidates = np.flatnonzero(scores >= self.palm_score)
        if candidates.size == 0:
            return []

        candidate_boxes = boxes_raw[candidates]
        anchor_points = self.anchors[candidates]
        centers = (candidate_boxes[:, 0:2] / PALM_SIZE + anchor_points) * coordinate_scale
        sizes = candidate_boxes[:, 2:4] / PALM_SIZE * coordinate_scale
        top_left = centers - sizes / 2.0
        cv_boxes = np.column_stack((top_left, sizes)).tolist()
        selected = cv2.dnn.NMSBoxes(
            cv_boxes,
            scores[candidates].tolist(),
            self.palm_score,
            0.75,
            top_k=max(6, self.max_hands * 3),
        )

        image_height, image_width = image.shape[:2]
        palms: list[Palm] = []
        for relative_index in normalize_indices(selected):
            source_index = int(candidates[relative_index])
            raw = boxes_raw[source_index]
            anchor = self.anchors[source_index]
            center = (raw[0:2] / PALM_SIZE + anchor) * coordinate_scale
            size = raw[2:4] / PALM_SIZE * coordinate_scale
            box = clip_box(
                (
                    center[0] - size[0] / 2.0,
                    center[1] - size[1] / 2.0,
                    center[0] + size[0] / 2.0,
                    center[1] + size[1] / 2.0,
                ),
                image_width,
                image_height,
            )
            keypoints = raw[4:18].reshape(7, 2)
            keypoints = (keypoints / PALM_SIZE + anchor) * coordinate_scale
            if offset_x or offset_y:
                x1, y1, x2, y2 = box
                box = (
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                )
                keypoints = keypoints + np.asarray(
                    (offset_x, offset_y), dtype=np.float32
                )
            palms.append(Palm(box, float(scores[source_index]), keypoints))
        return palms

    def merge_palms(self, palms: list[Palm]) -> list[Palm]:
        """Remove duplicates produced by overlapping full-frame/tile inference."""
        if len(palms) <= 1:
            return palms
        kept: list[Palm] = []
        for palm in sorted(palms, key=lambda item: item.score, reverse=True):
            x1, y1, x2, y2 = palm.box
            center = np.asarray(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            side = max(x2 - x1, y2 - y1)
            duplicate = False
            for existing in kept:
                ex1, ey1, ex2, ey2 = existing.box
                ix1, iy1 = max(x1, ex1), max(y1, ey1)
                ix2, iy2 = min(x2, ex2), min(y2, ey2)
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (
                    (x2 - x1) * (y2 - y1)
                    + (ex2 - ex1) * (ey2 - ey1)
                    - intersection
                )
                iou = intersection / max(1, union)
                existing_center = np.asarray(
                    ((ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0)
                )
                existing_side = max(ex2 - ex1, ey2 - ey1)
                center_gap = float(np.linalg.norm(center - existing_center))
                if iou >= 0.50 and center_gap <= 0.35 * min(side, existing_side):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(palm)
                if len(kept) >= self.max_hands:
                    break
        return kept

    def detect_palms(self, image: np.ndarray) -> list[Palm]:
        # Fast path: use one full-frame BPU pass. A 2000x1200 stereo eye is
        # compressed to only 192x115 model pixels, so small/far hands may be
        # missed. Only then retry with overlapping, higher-resolution tiles.
        full_palms = self.merge_palms(self.detect_palms_once(image))
        height, width = image.shape[:2]
        if (
            self.palm_tiles <= 1
            or len(full_palms) >= self.max_hands
            or width <= height * 1.25
        ):
            return full_palms[: self.max_hands]

        overlap = 0.20
        denominator = self.palm_tiles - overlap * (self.palm_tiles - 1)
        tile_width = min(width, max(height, int(math.ceil(width / denominator))))
        starts = np.linspace(
            0,
            max(0, width - tile_width),
            self.palm_tiles,
            dtype=np.int32,
        )
        candidates = list(full_palms)
        for start in np.unique(starts):
            end = min(width, int(start) + tile_width)
            tile = image[:, int(start) : end]
            candidates.extend(self.detect_palms_once(tile, offset_x=int(start)))
        return self.merge_palms(candidates)

    def detect_hand(self, image: np.ndarray, palm: Palm) -> Hand | None:
        image_height, image_width = image.shape[:2]
        roi = move_box(
            palm.box,
            palm.keypoints[2] - palm.keypoints[0],
            image_width,
            image_height,
        )
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        points, score, handedness = self.hand_model.infer(crop)
        if score < self.hand_score:
            return None

        roi_width = x2 - x1
        roi_height = y2 - y1
        landmarks = points.copy()
        landmarks[:, 0] = points[:, 0] / HAND_SIZE * roi_width + x1
        landmarks[:, 1] = points[:, 1] / HAND_SIZE * roi_height + y1
        landmarks[:, 2] = points[:, 2] / HAND_SIZE * max(roi_width, roi_height)
        return Hand(
            landmarks=landmarks,
            score=score,
            handedness=handedness,
            roi=roi,
            gesture=(
                recognize_gesture(landmarks)
                if self.gesture_backend == "rules"
                else "Unknown"
            ),
            gesture_score=1.0 if self.gesture_backend == "rules" else 0.0,
            track_id=-1,
        )

    def process(
        self,
        image: np.ndarray,
        stream_id: str = "default",
        bootstrap_gesture: bool = False,
    ) -> tuple[list[Palm], list[Hand]]:
        palms = self.detect_palms(image)
        hands = []
        for palm in palms[: self.max_hands]:
            hand = self.detect_hand(image, palm)
            if hand is not None:
                hands.append(hand)
        if self.gesture_classifier is not None:
            self.gesture_classifier.process(hands, stream_id, bootstrap_gesture)
        return palms, hands

    def close(self) -> None:
        self.hand_model.close()
        if self.gesture_classifier is not None:
            self.gesture_classifier.close()


def draw_results(image: np.ndarray, palms: list[Palm], hands: list[Hand]) -> None:
    for palm in palms:
        x1, y1, x2, y2 = palm.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
        for point in palm.keypoints:
            cv2.circle(image, tuple(np.round(point).astype(int)), 3, (0, 200, 255), -1)

    for hand in hands:
        points = np.round(hand.landmarks[:, :2]).astype(int)
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, tuple(points[start]), tuple(points[end]), (70, 230, 70), 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            color = (0, 80, 255) if index in (4, 8, 12, 16, 20) else (255, 180, 0)
            cv2.circle(image, tuple(point), 4, color, -1, cv2.LINE_AA)
        x1 = int(np.min(points[:, 0]))
        y1 = int(np.min(points[:, 1]))
        x2 = int(np.max(points[:, 0]))
        y2 = int(np.max(points[:, 1]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (70, 230, 70), 2)
        if hand.gesture_score > 0:
            label = (
                f"{hand.gesture} G:{hand.gesture_score:.2f} "
                f"H:{hand.score:.2f}"
            )
        else:
            label = f"{hand.gesture} H:{hand.score:.2f}"
        cv2.putText(
            image,
            label,
            (max(0, x1), max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (70, 230, 70),
            2,
            cv2.LINE_AA,
        )


def draw_controller_status(
    image: np.ndarray, controller: GestureRecordController
) -> None:
    """Render state-machine status without coupling core logic to OpenCV."""
    if controller.alarm:
        banner_width = min(image.shape[1] - 1, 1900)
        cv2.rectangle(image, (0, 56), (banner_width, 150), (0, 0, 220), -1)
        cv2.putText(
            image,
            "ALARM: BOTH HANDS LOST",
            (24, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        return
    cv2.putText(
        image,
        controller.status,
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def select_eye(frame: np.ndarray, eye: str) -> np.ndarray:
    if eye == "left":
        return frame[:, : frame.shape[1] // 2]
    if eye == "right":
        return frame[:, frame.shape[1] // 2 :]
    return frame


def view_regions(frame: np.ndarray, eye: str) -> list[tuple[str, np.ndarray]]:
    if eye == "both":
        middle = frame.shape[1] // 2
        return [("left", frame[:, :middle]), ("right", frame[:, middle:])]
    return [(eye, select_eye(frame, eye))]


def region_uses_bpu(name: str, view: str, compute_eye: str) -> bool:
    """Limit BPU work only when showing the split stereo view."""
    if view != "both":
        return True
    return compute_eye == "both" or name == compute_eye


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV + X5 BPU MediaPipe hand landmarks")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=4000)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--eye",
        choices=("left", "right", "both", "full"),
        default="both",
        help="识别左眼、右眼、左右分别识别，或未拆分的完整画面",
    )
    parser.add_argument(
        "--compute-eye",
        choices=("left", "right", "both"),
        default="left",
        help="双路显示时运行 BPU 的镜头；默认只计算左侧",
    )
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--detect-every", type=int, default=2)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--palm-score", type=float, default=0.5)
    parser.add_argument("--hand-score", type=float, default=0.35)
    parser.add_argument(
        "--palm-tiles",
        type=int,
        default=2,
        help="整图未找到足够手掌时使用的水平重叠检测块数；1 表示关闭",
    )
    parser.add_argument(
        "--gesture-backend",
        choices=("official", "rules"),
        default="rules",
        help="rules 使用即时几何规则（默认、无 warmup）；official 使用可选的 gestureDet 时序模型",
    )
    parser.add_argument("--gesture-model", default=GESTURE_MODEL)
    parser.add_argument(
        "--gesture-score",
        type=float,
        default=0.5,
        help="官方 gestureDet 分类置信度阈值",
    )
    parser.add_argument(
        "--control-eye",
        choices=("left", "right"),
        default="left",
        help="双目模式中用于自动录像手势判定的镜头",
    )
    parser.add_argument("--palm-start-seconds", type=float, default=3.0)
    parser.add_argument("--missing-alarm-seconds", type=float, default=5.0)
    parser.add_argument("--thumb-stop-seconds", type=float, default=4.0)
    parser.add_argument(
        "--gesture-grace-seconds",
        type=float,
        default=1.0,
        help="Thumb Up 连续计时允许的短暂漏检秒数",
    )
    parser.add_argument(
        "--no-auto-record",
        action="store_true",
        help="关闭双手手势控制的自动录像状态机",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="启动后立即录制无识别标注的原始视频；扩展名可选 .avi/.mpeg/.mp4",
    )
    parser.add_argument(
        "--record-format",
        choices=video_format_names(),
        default="avi",
        help="自动录像格式；默认 avi，可选 mpeg 或 mp4",
    )
    parser.add_argument(
        "--record-dir",
        default="recordings",
        help="自动命名的录像和截图保存目录",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=1280,
        help="录像最大宽度；0 表示保存相机原始分辨率",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=0.0,
        help="录像文件帧率；0 表示使用开始录像时测得的显示帧率",
    )
    parser.add_argument("--image", help="process one image instead of the camera")
    parser.add_argument("--output", default="gesture_result.jpg")
    parser.add_argument(
        "--no-display",
        "--headless",
        dest="no_display",
        action="store_true",
        help="无窗口运行，并跳过画面复制、骨架绘制和显示缩放",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "palm_start_seconds",
        "missing_alarm_seconds",
        "thumb_stop_seconds",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} 必须大于 0")
    if args.record_width < 0 or args.record_fps < 0:
        raise SystemExit("--record-width 和 --record-fps 不能为负数")
    if args.palm_tiles < 1:
        raise SystemExit("--palm-tiles 必须大于等于 1")
    if args.gesture_grace_seconds < 0:
        raise SystemExit("--gesture-grace-seconds 不能为负数")
    if not 0.0 <= args.gesture_score <= 1.0:
        raise SystemExit("--gesture-score 必须在 0 到 1 之间")
    pipeline = BpuHandPipeline(
        args.palm_score,
        args.hand_score,
        args.max_hands,
        args.palm_tiles,
        args.gesture_backend,
        args.gesture_model,
        args.gesture_score,
    )

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"无法读取图片：{args.image}")
        started = time.perf_counter()
        result_parts = []
        palms_count = 0
        hands_count = 0
        for name, region in view_regions(frame, args.eye):
            if region_uses_bpu(name, args.eye, args.compute_eye):
                palms, hands = pipeline.process(
                    region, name, bootstrap_gesture=True
                )
            else:
                palms, hands = [], []
            part = region.copy()
            draw_results(part, palms, hands)
            if args.eye == "both":
                role = "BPU" if region_uses_bpu(name, args.eye, args.compute_eye) else "VIEW"
                cv2.putText(
                    part,
                    f"{name.upper()} [{role}]",
                    (20, max(30, part.shape[0] - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            result_parts.append(part)
            palms_count += len(palms)
            hands_count += len(hands)
        infer_ms = (time.perf_counter() - started) * 1000.0
        result = np.hstack(result_parts) if len(result_parts) == 2 else result_parts[0]
        cv2.putText(
            result,
            f"BPU pipeline {infer_ms:.1f} ms | palms {palms_count} hands {hands_count}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), result):
            raise SystemExit(f"无法保存结果：{output}")
        print(
            f"palms={palms_count} hands={hands_count} "
            f"time={infer_ms:.2f}ms output={output}"
        )
        pipeline.close()
        return

    if not args.no_display and not os.environ.get("DISPLAY"):
        raise SystemExit("没有 DISPLAY；请使用 run_gesture_opencv.sh 启动")

    cap = cv2.VideoCapture(device_index(args.device), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"无法打开 {args.device}，请确认没有其他程序占用相机")

    eye = args.eye
    results: dict[str, tuple[list[Palm], list[Hand]]] = {
        "left": ([], []),
        "right": ([], []),
        "full": ([], []),
    }
    frame_index = 0
    infer_ms = 0.0
    display_fps = 0.0
    fps_frames = 0
    fps_started = time.monotonic()
    window = "Gesture_X5 BPU Hand Landmarks"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    else:
        print("无可视化模式：保留识别、自动录像和报警，跳过所有显示绘制")
    record_dir = Path(args.record_dir).expanduser()
    if not record_dir.is_absolute():
        record_dir = Path.cwd() / record_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    record_path = None
    pending_record = args.record
    auto_controller = GestureRecordController(
        args.palm_start_seconds,
        args.missing_alarm_seconds,
        args.thumb_stop_seconds,
        args.gesture_grace_seconds,
    )
    print("按键：E=切换左/右/双路/完整画面，R=录像，S=截图，Q/ESC=退出")
    if not args.no_auto_record:
        print(
            f"自动录像：{args.control_eye}镜头检测到双手 Palm "
            f"{args.palm_start_seconds:g}秒开始；双手消失 "
            f"{args.missing_alarm_seconds:g}秒报警；双手 Thumb Up "
            f"{args.thumb_stop_seconds:g}秒结束"
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            regions = view_regions(frame, eye)
            computed_names = {
                name
                for name, _ in regions
                if region_uses_bpu(name, eye, args.compute_eye)
            }
            frame_index += 1
            if frame_index % max(1, args.detect_every) == 0:
                started = time.perf_counter()
                for name, region in regions:
                    if name in computed_names:
                        results[name] = pipeline.process(region, name)
                infer_ms = (time.perf_counter() - started) * 1000.0

            palm_counts = []
            hand_counts = []
            shown_parts = []
            for name, region in regions:
                palms, hands = results[name]
                palm_counts.append(len(palms))
                hand_counts.append(len(hands))
                if args.no_display:
                    continue
                part = region.copy()
                draw_results(part, palms, hands)
                if eye == "both":
                    role = "BPU" if name in computed_names else "VIEW"
                    cv2.putText(
                        part,
                        f"{name.upper()} [{role}]",
                        (20, max(30, part.shape[0] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                shown_parts.append(part)
            shown = None
            if shown_parts:
                shown = (
                    np.hstack(shown_parts)
                    if len(shown_parts) == 2
                    else shown_parts[0]
                )
            fps_frames += 1
            now = time.monotonic()
            elapsed = now - fps_started
            if elapsed >= 1.0:
                display_fps = fps_frames / elapsed
                fps_frames = 0
                fps_started = now
            count_text = ""
            if eye == "both":
                count_text = (
                    f"  L:P{palm_counts[0]}/H{hand_counts[0]}"
                    f" R:P{palm_counts[1]}/H{hand_counts[1]}"
                )
            else:
                count_text = f"  P:{palm_counts[0]} H:{hand_counts[0]}"

            if not args.no_auto_record:
                control_hands = (
                    results[args.control_eye][1]
                    if args.control_eye in computed_names
                    else None
                )
                auto_start, auto_stop = auto_controller.update(
                    control_hands,
                    writer is not None or pending_record is not None,
                    time.monotonic(),
                )
                if auto_start and writer is None and pending_record is None:
                    pending_record = ""
                    print("自动录像已触发：双手 Palm 保持时间达到阈值", flush=True)
                if auto_stop:
                    pending_record = None
                    if writer is not None:
                        writer.release()
                        writer = None
                        print(f"自动录像结束并保存：{record_path}", flush=True)
                        record_path = None
            if shown is not None:
                cv2.putText(
                    shown,
                    f"{eye.upper()}  {display_fps:.1f} FPS  BPU {infer_ms:.1f} ms{count_text}",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if not args.no_auto_record:
                    draw_controller_status(shown, auto_controller)

            record_frame = None
            if pending_record is not None or writer is not None:
                record_frame = make_record_frame(regions, args.record_width)

            if pending_record is not None:
                if pending_record:
                    record_path = Path(pending_record).expanduser()
                    if not record_path.is_absolute():
                        record_path = Path.cwd() / record_path
                    record_path, record_encoding = resolve_video_path(
                        record_path, args.record_format
                    )
                else:
                    record_encoding = encoding_for_format(args.record_format)
                    record_path = timestamped_path(
                        record_dir, "gesture", record_encoding.suffix
                    )
                record_path.parent.mkdir(parents=True, exist_ok=True)
                record_fps = args.record_fps
                if record_fps <= 0:
                    record_fps = min(
                        float(args.fps),
                        display_fps if display_fps >= 1.0 else 15.0,
                    )
                writer = cv2.VideoWriter(
                    str(record_path),
                    cv2.VideoWriter_fourcc(*record_encoding.fourcc),
                    record_fps,
                    (record_frame.shape[1], record_frame.shape[0]),
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise RuntimeError(f"无法创建录像文件：{record_path}")
                print(
                    f"开始录像：{record_path} | {record_frame.shape[1]}x"
                    f"{record_frame.shape[0]} @ {record_fps:.1f} FPS | "
                    f"{record_encoding.name}/{record_encoding.fourcc}"
                )
                pending_record = None

            if writer is not None:
                if record_frame is None:
                    record_frame = make_record_frame(regions, args.record_width)
                writer.write(record_frame)
                if shown is not None:
                    cv2.circle(
                        shown,
                        (shown.shape[1] - 32, 28),
                        10,
                        (0, 0, 255),
                        -1,
                    )
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

            if args.no_display:
                continue
            if args.display_width > 0 and shown.shape[1] > args.display_width:
                out_height = round(shown.shape[0] * args.display_width / shown.shape[1])
                shown = cv2.resize(
                    shown,
                    (args.display_width, out_height),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("e"), ord("E")):
                if writer is not None:
                    writer.release()
                    writer = None
                    print(f"切换画面，录像已保存：{record_path}")
                    record_path = None
                eye = {
                    "left": "right",
                    "right": "both",
                    "both": "full",
                    "full": "left",
                }[eye]
                results = {"left": ([], []), "right": ([], []), "full": ([], [])}
            elif key in (ord("r"), ord("R")):
                if writer is None:
                    pending_record = ""
                else:
                    writer.release()
                    writer = None
                    print(f"录像已保存：{record_path}")
                    record_path = None
            elif key in (ord("s"), ord("S")):
                screenshot = timestamped_path(record_dir, "gesture", ".jpg")
                if cv2.imwrite(str(screenshot), shown):
                    print(f"截图已保存：{screenshot}")
                else:
                    print(f"截图保存失败：{screenshot}")
    except KeyboardInterrupt:
        print("收到中断，正在保存录像并退出")
    finally:
        if writer is not None:
            writer.release()
            print(f"录像已保存：{record_path}")
        cap.release()
        pipeline.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

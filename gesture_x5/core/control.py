"""Recording state machine driven by classified hand gestures.

This module deliberately has no OpenCV or BPU dependency, so its timing
behaviour can be unit-tested on a development computer.
"""

from __future__ import annotations

from typing import Protocol


class GestureResult(Protocol):
    gesture: str


class GestureRecordController:
    """Convert Palm/Thumb Up observations into record start/stop events."""

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
        hands: list[GestureResult] | None,
        recording: bool,
        now: float,
    ) -> tuple[bool, bool]:
        """Return ``(start_recording, stop_recording)`` for this frame."""
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
                        "AUTO START: Palm tracking "
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

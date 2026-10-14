from types import SimpleNamespace
from unittest import TestCase

from gesture_x5.core.control import GestureRecordController


def hands(gesture: str, count: int = 2):
    return [SimpleNamespace(gesture=gesture) for _ in range(count)]


class GestureRecordControllerTest(TestCase):
    def test_palm_starts_after_three_seconds_with_short_gap(self):
        controller = GestureRecordController(3.0, 5.0, 5.0, 1.0)
        self.assertEqual(controller.update(hands("Palm"), False, 0.0), (False, False))
        self.assertEqual(controller.update(hands("Palm"), False, 1.5), (False, False))
        self.assertEqual(controller.update([], False, 2.2), (False, False))
        self.assertIn("grace", controller.status)
        self.assertEqual(controller.update(hands("Palm"), False, 3.1), (True, False))

    def test_palm_gap_over_grace_resets_timer(self):
        controller = GestureRecordController(3.0, 5.0, 5.0, 1.0)
        controller.update(hands("Palm"), False, 0.0)
        controller.update([], False, 1.1)
        self.assertIsNone(controller.palm_since)
        self.assertEqual(controller.update(hands("Palm"), False, 3.2), (False, False))

    def test_single_thumb_up_stops_recording_after_four_seconds(self):
        controller = GestureRecordController(3.0, 5.0, 4.0, 1.0)
        controller.update(hands("Thumb Up", 1), True, 10.0)
        controller.update([], True, 10.5)
        self.assertEqual(
            controller.update(hands("Thumb Up", 1), True, 14.1),
            (False, True),
        )

    def test_single_thumb_up_does_not_stop_before_four_seconds(self):
        controller = GestureRecordController(3.0, 5.0, 4.0, 1.0)
        controller.update(hands("Thumb Up", 1), True, 20.0)
        self.assertEqual(
            controller.update(hands("Thumb Up", 1), True, 23.9),
            (False, False),
        )

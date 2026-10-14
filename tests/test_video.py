from pathlib import Path
from unittest import TestCase

from gesture_x5.core.video import resolve_video_path


class VideoEncodingTest(TestCase):
    def test_adds_selected_mpeg_suffix(self):
        path, encoding = resolve_video_path(Path("recordings/gesture"), "mpeg")
        self.assertEqual(path.suffix, ".mpeg")
        self.assertEqual(encoding.fourcc, "MPG2")

    def test_explicit_mp4_suffix_selects_mp4_codec(self):
        path, encoding = resolve_video_path(Path("clip.mp4"), "avi")
        self.assertEqual(path, Path("clip.mp4"))
        self.assertEqual(encoding.fourcc, "mp4v")

    def test_mpg_alias_uses_mpeg_codec(self):
        path, encoding = resolve_video_path(Path("clip.mpg"), "avi")
        self.assertEqual(path.suffix, ".mpg")
        self.assertEqual(encoding.name, "mpeg")

"""Video container and OpenCV FourCC selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoEncoding:
    name: str
    suffix: str
    fourcc: str


VIDEO_ENCODINGS = {
    "avi": VideoEncoding("avi", ".avi", "MJPG"),
    "mpeg": VideoEncoding("mpeg", ".mpeg", "MPG2"),
    "mp4": VideoEncoding("mp4", ".mp4", "mp4v"),
}

SUFFIX_FORMATS = {
    ".avi": "avi",
    ".mpeg": "mpeg",
    ".mpg": "mpeg",
    ".mp4": "mp4",
}


def video_format_names() -> tuple[str, ...]:
    return tuple(VIDEO_ENCODINGS)


def encoding_for_format(name: str) -> VideoEncoding:
    try:
        return VIDEO_ENCODINGS[name]
    except KeyError as exc:
        supported = ", ".join(video_format_names())
        raise ValueError(f"不支持的视频格式 {name!r}；可选：{supported}") from exc


def resolve_video_path(
    path: Path,
    default_format: str,
) -> tuple[Path, VideoEncoding]:
    """Honor a recognized suffix or append the selected default suffix."""
    suffix_format = SUFFIX_FORMATS.get(path.suffix.lower())
    encoding = encoding_for_format(suffix_format or default_format)
    if not suffix_format:
        path = path.with_suffix(encoding.suffix)
    return path, encoding

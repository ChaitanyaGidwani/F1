"""Decode an uploaded video into an evenly-spaced set of frames.

The brief's input is "photos or video frames", and a judge dropping an mp4 on
the page should get the same trend analysis as a folder of stills. Frames are
sampled across the whole clip rather than taken from the front: a wet-to-dry
sequence has its entire story in the spread, and reading only the first N frames
would show a wet track and nothing else.
"""

from __future__ import annotations

import io
import math
from typing import List, Optional

from PIL import Image

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}


def is_video(filename: Optional[str], content_type: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("video/"):
        return True
    if filename and "." in filename:
        return f".{filename.rsplit('.', 1)[-1].lower()}" in VIDEO_SUFFIXES
    return False


def frames_from_video(raw: bytes, max_frames: int = 36, fallback_stride: int = 15) -> List[Image.Image]:
    """Return up to `max_frames` RGB frames spread across the clip.

    Raises ValueError if the bytes are not a decodable video, so the caller can
    turn that into a 400 rather than a 500.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise ValueError("video support requires the 'av' package") from exc

    try:
        container = av.open(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - av raises several unrelated types
        raise ValueError("could not decode this video") from exc

    frames: List[Image.Image] = []
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Prefer a stride computed from the real frame count so coverage is even.
        # Containers do not always report it, hence the fallback.
        total = stream.frames or 0
        stride = max(1, math.ceil(total / max_frames)) if total else fallback_stride

        for index, frame in enumerate(container.decode(video=0)):
            if index % stride:
                continue
            frames.append(frame.to_image().convert("RGB"))
            if len(frames) >= max_frames:
                break
    except (IndexError, StopIteration) as exc:
        raise ValueError("no video stream found") from exc
    finally:
        container.close()

    if not frames:
        raise ValueError("no frames could be decoded from this video")
    return frames

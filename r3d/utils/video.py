# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared video writing utilities.

Provides ``write_video`` for encoding RGB frames to MP4 using OpenCV.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def write_video(
    frames: list[np.ndarray] | Iterator[np.ndarray],
    output_path: str | Path,
    fps: float = 30.0,
    output_height: int | None = None,
) -> None:
    """Write frames to an MP4 video file using OpenCV.

    Frames are expected in RGB uint8 format (H, W, 3).  They are converted
    to BGR before encoding.  The encoder tries H.264 (``avc1``) first and
    falls back to ``mp4v`` if the codec is unavailable.

    Args:
        frames: RGB uint8 images, either a list or an iterator.
        output_path: Destination ``.mp4`` file path.
        fps: Output frame rate.
        output_height: If set, resize all frames to this height
            (preserving aspect ratio) before encoding. Dimensions
            are clamped to even numbers for codec compatibility.

    Raises:
        ValueError: If *frames* is empty.
        RuntimeError: If the VideoWriter cannot be opened.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Materialise an iterator so we can inspect the first frame for size.
    if not isinstance(frames, list):
        frames = list(frames)

    if not frames:
        raise ValueError("No frames provided")

    if output_height is not None:
        resized = []
        for frame in frames:
            fh, fw = frame.shape[:2]
            scale = output_height / fh
            new_w = int(fw * scale)
            resized.append(cv2.resize(frame, (new_w, output_height)))
        frames = resized

    h, w = frames[0].shape[:2]

    # Clamp to even dimensions for codec compatibility.
    if h % 2 != 0:
        h -= 1
        frames = [f[:h, :] for f in frames]
    if w % 2 != 0:
        w -= 1
        frames = [f[:, :w] for f in frames]

    # Try avc1 (H.264) first, fall back to mp4v.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        logger.warning("avc1 codec unavailable, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {output_path} with size ({w}, {h})"
        )

    try:
        for frame in frames:
            bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    logger.info(f"Wrote video ({len(frames)} frames) to {output_path}")

# Copyright (c) Meta Platforms, Inc. and affiliates.

"""V2 annotation visualization.

Renders full sequence video with segmentation mask overlays, 2D bounding
boxes with labels, and highlighted reference frames for pointing annotations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from projectaria_tools.core.stream_id import StreamId
from r3d.data_gen.generators.base import ObjectRef
from r3d.data_gen.generators.disambiguation import _extract_object_mask
from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.utils.aria_images import get_corrected_rgb
from r3d.utils.video import write_video
from r3d.utils.viz import (
    draw_bbox_from_mask as _draw_bbox_from_mask,
    overlay_mask as _overlay_mask,
)

logger: logging.Logger = logging.getLogger(__name__)

OUTPUT_SIZE = 512
TEXT_PANEL_HEIGHT = 120
FRAME_SKIP = 1
FPS = 5.0

COLORS_NORMAL = [
    np.array([255, 80, 80]),
    np.array([80, 120, 255]),
]
COLORS_REFERENCE = [
    np.array([255, 255, 0]),
    np.array([0, 255, 255]),
]
BBOX_COLORS_NORMAL = [
    (255, 80, 80),
    (80, 120, 255),
]
BBOX_COLORS_REFERENCE = [
    (255, 255, 0),
    (0, 255, 255),
]
BLEND_NORMAL = 0.35
BLEND_REFERENCE = 0.55


def _wrap_text(text: str, font: int, scale: float, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels using cv2 font metrics."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, 1)
        if tw > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)
    return lines


def _draw_text_panel(
    frame: np.ndarray,
    question: str,
    answer: str,
    is_reference_frame: bool = False,
) -> np.ndarray:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    line_height = 18
    margin = 8
    max_width = w - 2 * margin

    lines = _wrap_text(question, font, scale, max_width)
    gt_lines = _wrap_text(f"GT: {answer}", font, scale, max_width)
    total_lines = len(lines) + len(gt_lines) + (1 if is_reference_frame else 0)
    panel_height = max(TEXT_PANEL_HEIGHT, line_height * total_lines + 2 * margin)

    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    y = line_height
    for line in lines:
        cv2.putText(panel, line, (margin, y), font, scale, (255, 255, 255), 1)
        y += line_height

    y += 4
    for line in gt_lines:
        cv2.putText(panel, line, (margin, y), font, scale, (0, 255, 255), 1)
        y += line_height

    if is_reference_frame:
        cv2.putText(panel, "REFERENCE FRAME", (w - 200, y), font, 0.5, (255, 255, 0), 2)

    return np.vstack([frame, panel])


def _render_video_overlay(
    annotation: Annotation,
    refs: list[ObjectRef],
    gt_provider: Any,
    timestamps_ns: list[int],
    stream_id: StreamId,
    output_dir: Path,
    frame_cache: dict[int, np.ndarray] | None = None,
) -> Path | None:
    ql = annotation.query_layer
    answer_str = f"{ql.gt_answer} {ql.gt_answer_unit or ''}".strip()

    subsampled_ts = timestamps_ns[::FRAME_SKIP]
    instance_ids = [r.obj.instance_id for r in refs]
    obj_names = [o.canonical_name for o in annotation.identity_layer.referenced_objects]

    ref_ts_by_instance: dict[int, int] = {}
    pointing_masks: dict[int, Any] = {}
    for r in refs:
        if r.reference_timestamp_ns is not None:
            ref_ts_by_instance[r.obj.instance_id] = r.reference_timestamp_ns
        if r.pointing_mask is not None:
            pointing_masks[r.obj.instance_id] = r.pointing_mask

    frames = []
    for ts in subsampled_ts:
        if frame_cache is not None and ts in frame_cache:
            rgb = frame_cache[ts].copy()
        else:
            raw = get_corrected_rgb(gt_provider, ts, stream_id)
            if raw is None:
                continue
            rgb = cv2.resize(raw, (OUTPUT_SIZE, OUTPUT_SIZE))
            if frame_cache is not None:
                frame_cache[ts] = rgb
        any_ref = False

        for i, inst_id in enumerate(instance_ids):
            ci = min(i, len(COLORS_NORMAL) - 1)
            label = obj_names[i] if i < len(obj_names) else f"obj_{inst_id}"
            is_ref_for_obj = ref_ts_by_instance.get(inst_id) == ts

            if is_ref_for_obj and inst_id in pointing_masks:
                mask = pointing_masks[inst_id]
                color = COLORS_REFERENCE[ci]
                bbox_color = BBOX_COLORS_REFERENCE[ci]
                blend = BLEND_REFERENCE
                any_ref = True
            else:
                mask = _extract_object_mask(gt_provider, inst_id, ts, stream_id)
                if mask is None:
                    continue
                color = COLORS_NORMAL[ci]
                bbox_color = BBOX_COLORS_NORMAL[ci]
                blend = BLEND_NORMAL

            rgb = _overlay_mask(rgb, mask, color, blend)
            rgb = _draw_bbox_from_mask(rgb, mask, bbox_color, label)

        rgb = _draw_text_panel(rgb, ql.question_text, answer_str, any_ref)
        frames.append(rgb)

    if not frames:
        return None

    ann_id = annotation.annotation_id[:8]
    video_path = output_dir / f"{ql.question_type.value}_{ann_id}.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_video(frames, video_path, fps=FPS)
    logger.info(f"Wrote {len(frames)} frames to {video_path}")
    return video_path


def render_annotation_video(
    annotation: Annotation,
    refs: list[ObjectRef],
    gt_provider: Any,
    timestamps_ns: list[int],
    stream_id: StreamId,
    output_dir: Path,
    frame_cache: dict[int, np.ndarray] | None = None,
) -> Path | None:
    return _render_video_overlay(
        annotation,
        refs,
        gt_provider,
        timestamps_ns,
        stream_id,
        output_dir,
        frame_cache=frame_cache,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared RGB (+ optional SAM3 mask overlay) evaluation helpers.

Used by both the main eval pipeline (generate_responses, rgb strategy) and the
standalone rgb_overlay_evals.py script, so the overlay/prompt/query logic lives
in exactly one place.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.prompts import ANSWER_FORMAT_INSTRUCTIONS
from r3d.pipeline.eval.responses import Response
from r3d.pipeline.eval.vlm import _overlay_masks_on_images, VLMClient
from r3d.pipeline.frame_data import CameraIntrinsics
from r3d.pipeline.stores.base import FrameStore, SegmentationStore

def load_frames_for_window(
    sequence_id: str,
    start_ns: int,
    end_ns: int,
    frame_store: FrameStore,
    image_size: int,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], list[int], CameraIntrinsics]:
    """Load RGB frames (resized) in [start_ns, end_ns].

    When ``max_frames`` is set, evenly subsamples down to that many frames;
    when ``None`` (default), every frame in the window is used.
    """
    all_ts = frame_store.get_all_timestamps(sequence_id)
    window_ts = [ts for ts in all_ts if start_ns <= ts <= end_ns]
    if not window_ts:
        raise RuntimeError(f"No frames in [{start_ns}, {end_ns}] for {sequence_id}")
    if max_frames is not None and len(window_ts) > max_frames:
        idx = np.linspace(0, len(window_ts) - 1, max_frames, dtype=int)
        window_ts = [window_ts[i] for i in idx]
    frame_datas = [frame_store.load_frame(sequence_id, ts) for ts in window_ts]
    frames = [cv2.resize(fd.rgb, (image_size, image_size)) for fd in frame_datas]
    return frames, window_ts, frame_datas[0].intrinsics


def load_masks_for_annotation(
    annotation: Annotation,
    timestamps_ns: list[int],
    sequence_id: str,
    seg_store: SegmentationStore,
    image_size: int,
) -> tuple[list[list[np.ndarray | None]], list[str]]:
    """Load per-frame SAM3 masks for an annotation's referenced objects."""
    ref_objects = annotation.identity_layer.referenced_objects
    object_names = [obj.canonical_name for obj in ref_objects]
    per_frame_masks: list[list[np.ndarray | None]] = []
    for ts in timestamps_ns:
        frame_masks: list[np.ndarray | None] = []
        for ref_obj in ref_objects:
            seg = seg_store.get_segmentation(
                sequence_id, ts, query_name=ref_obj.canonical_name
            )
            if not seg.objects:
                frame_masks.append(None)
                continue
            obj_seg = next(iter(seg.objects.values()))
            frame_masks.append(
                cv2.resize(
                    obj_seg.mask.astype(np.uint8),
                    (image_size, image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
        per_frame_masks.append(frame_masks)
    return per_frame_masks, object_names


def build_rgb_prompt(annotation: Annotation) -> str:
    return (
        "Look at the images and answer this spatial question.\n\n"
        f"Question: {annotation.query_layer.question_text}\n\n"
        f"{ANSWER_FORMAT_INSTRUCTIONS}"
    )


def run_rgb_overlay_eval(
    annotation: Annotation,
    sequence_id: str,
    frame_store: FrameStore,
    vlm: VLMClient,
    model: str,
    image_size: int,
    seg_store: SegmentationStore | None = None,
    max_frames: int | None = None,
) -> Response:
    """Run one annotation with the RGB (+ optional mask overlay) strategy.

    When seg_store is given, the referenced objects' SAM3 masks are overlaid on
    the frames before they are sent to the model. ``max_frames`` caps the number
    of frames sent (``None`` = all frames in the window).
    """
    t_start = time.monotonic()
    ql = annotation.query_layer
    frames, window_ts, _ = load_frames_for_window(
        sequence_id,
        ql.query_timestamp_ns_start or 0,
        ql.query_timestamp_ns_end or 0,
        frame_store,
        image_size,
        max_frames=max_frames,
    )

    if seg_store is not None:
        masks, object_names = load_masks_for_annotation(
            annotation, window_ts, sequence_id, seg_store, image_size
        )
        frames = _overlay_masks_on_images(frames, masks, object_names)

    response_text = vlm.query(build_rgb_prompt(annotation), frames, model)
    return Response(
        annotation_id=annotation.annotation_id,
        model=model,
        strategy="rgb_overlay" if seg_store is not None else "rgb",
        response=response_text,
        tool_call_log=None,
        created_ns=int(time.time() * 1e9),
        latency_s=time.monotonic() - t_start,
    )

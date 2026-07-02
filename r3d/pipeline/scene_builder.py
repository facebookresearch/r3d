# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Scene builder: coverage computation and scene compilation.

Builds scene.db from segmentation data. Coverage computation functions
calculate angular span and viewpoint diversity for each object.
"""

from __future__ import annotations

import logging

import numpy as np
from r3d.pipeline.frame_data import FrameData
from r3d.pipeline.stores.base import (
    FrameStore,
    FrameVisibility,
    ObjectCoverage,
    SceneObject,
    SceneStore,
    SegmentationStore,
)

logger: logging.Logger = logging.getLogger(__name__)


def build_scene(
    sequence_id: str,
    seg_store: SegmentationStore,
    scene_store: SceneStore,
    frame_store: FrameStore | None = None,
) -> None:
    """Build scene.db for a single sequence from segmentations.

    Called once per sequence. Multiple sequences can be written to the same
    scene_store -- each sequence's objects and reconstructions are scoped
    by sequence_id.
    """
    scene_store.delete_sequence(sequence_id)
    _build_scene_objects(sequence_id, seg_store, scene_store)
    _build_frame_visibility(sequence_id, seg_store, scene_store)
    if frame_store is not None:
        _build_object_coverage(sequence_id, frame_store, scene_store)


def _build_scene_objects(
    sequence_id: str,
    seg_store: SegmentationStore,
    scene_store: SceneStore,
) -> None:
    """Populate scene_objects from segmentation data."""
    object_info: dict[int, tuple[str, int, int]] = {}

    for query_name in seg_store.get_all_query_names(sequence_id):
        for ts in seg_store.get_segmented_timestamps(sequence_id, query_name):
            frame_seg = seg_store.get_segmentation(sequence_id, ts, query_name)
            for obj_id, obj in frame_seg.objects.items():
                if obj_id not in object_info:
                    object_info[obj_id] = (obj.query_name, ts, ts)
                else:
                    qn, first, last = object_info[obj_id]
                    object_info[obj_id] = (qn, min(first, ts), max(last, ts))

    for obj_id, (query_name, first_ns, last_ns) in object_info.items():
        scene_store.write_scene_object(
            SceneObject(
                sequence_id=sequence_id,
                object_id=obj_id,
                query_name=query_name,
                first_seen_ns=first_ns,
                last_seen_ns=last_ns,
            )
        )

    logger.info(
        f"[{sequence_id}] Built {len(object_info)} scene objects from segmentations"
    )


def _build_frame_visibility(
    sequence_id: str,
    seg_store: SegmentationStore,
    scene_store: SceneStore,
) -> None:
    """Populate frame_visibility from segmentation data."""
    count = 0
    for query_name in seg_store.get_all_query_names(sequence_id):
        for ts in seg_store.get_segmented_timestamps(sequence_id, query_name):
            frame_seg = seg_store.get_segmentation(sequence_id, ts, query_name)
            for obj_id, obj in frame_seg.objects.items():
                scene_store.write_frame_visibility(
                    FrameVisibility(
                        sequence_id=sequence_id,
                        timestamp_ns=ts,
                        object_id=obj_id,
                        bbox_2d=obj.bbox_2d,
                        mask_rle=obj.mask_rle,
                        sam3_score=obj.score,
                    )
                )
                count += 1

    logger.info(f"[{sequence_id}] Built {count} frame_visibility entries")


def _compute_viewing_angles(
    camera_pos: np.ndarray,
    object_pos: np.ndarray,
) -> tuple[float, float]:
    """Compute azimuth and elevation of camera relative to object center."""
    d = camera_pos - object_pos
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        raise RuntimeError(
            f"Camera and object positions coincide: {camera_pos} == {object_pos}"
        )
    d_norm = d / dist
    azimuth = np.degrees(np.arctan2(d_norm[0], d_norm[2]))
    elevation = np.degrees(np.arcsin(np.clip(d_norm[1], -1.0, 1.0)))
    return float(azimuth), float(elevation)


def _build_visibility_index(
    scene_store: SceneStore,
    sequence_id: str,
    all_timestamps: list[int],
) -> dict[int, list[tuple[int, FrameVisibility]]]:
    """Build per-object visibility index in one pass over timestamps."""
    index: dict[int, list[tuple[int, FrameVisibility]]] = {}
    for ts in all_timestamps:
        for fv in scene_store.get_frame_visibility(sequence_id, ts):
            index.setdefault(fv.object_id, []).append((ts, fv))
    return index


def _collect_angles_and_visibility(
    vis_entries: list[tuple[int, FrameVisibility]],
    obj_pos: np.ndarray,
    frame_store: FrameStore,
    sequence_id: str,
    frame_cache: dict[int, FrameData],
) -> tuple[list[tuple[float, float]], list[float]]:
    angles: list[tuple[float, float]] = []
    vis_ratios: list[float] = []
    for ts, fv in vis_entries:
        if ts not in frame_cache:
            frame_cache[ts] = frame_store.load_frame(sequence_id, ts)
        frame = frame_cache[ts]
        T_scene_camera = frame.T_scene_device @ frame.T_device_camera
        cam_pos = T_scene_camera[:3, 3]
        az, el = _compute_viewing_angles(cam_pos, obj_pos)
        angles.append((az, el))

        bbox = fv.bbox_2d
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = frame.intrinsics.width * frame.intrinsics.height
        vis_ratios.append(bbox_area / img_area)
    return angles, vis_ratios


def _compute_angular_coverage(
    angles: list[tuple[float, float]],
) -> tuple[float, int]:
    directions = np.array(
        [
            [
                np.cos(np.radians(el)) * np.sin(np.radians(az)),
                np.sin(np.radians(el)),
                np.cos(np.radians(el)) * np.cos(np.radians(az)),
            ]
            for az, el in angles
        ]
    )
    dots = np.clip(directions @ directions.T, -1.0, 1.0)
    angular_span = float(np.degrees(np.arccos(dots.min())))

    viewpoint_bins: set[tuple[int, int]] = set()
    for az, el in angles:
        az_bin = int(round(az / 30)) % 12
        el_bin = round(el / 30)
        viewpoint_bins.add((az_bin, el_bin))
    return angular_span, len(viewpoint_bins)


def _best_recon_positions(
    scene_store: SceneStore,
    sequence_id: str,
    objects: list[SceneObject],
) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    for obj in objects:
        recons = scene_store.get_reconstructions(sequence_id, obj.object_id)
        if recons:
            best = max(recons, key=lambda r: r.psnr or 0.0)
            positions[obj.object_id] = best.position
    return positions


def _build_object_coverage(
    sequence_id: str,
    frame_store: FrameStore,
    scene_store: SceneStore,
) -> None:
    objects = scene_store.get_all_scene_objects(sequence_id)
    all_timestamps = frame_store.get_all_timestamps(sequence_id)
    recon_positions = _best_recon_positions(scene_store, sequence_id, objects)
    vis_index = _build_visibility_index(scene_store, sequence_id, all_timestamps)
    frame_cache: dict[int, FrameData] = {}

    count = 0
    for obj in objects:
        if obj.object_id not in recon_positions:
            logger.info(
                f"  obj {obj.object_id} ({obj.query_name}): "
                f"no reconstruction, skipping coverage"
            )
            continue
        vis_entries = vis_index.get(obj.object_id, [])
        if not vis_entries:
            continue
        angles, vis_ratios = _collect_angles_and_visibility(
            vis_entries,
            recon_positions[obj.object_id],
            frame_store,
            sequence_id,
            frame_cache,
        )
        angular_span, n_viewpoints = _compute_angular_coverage(angles)
        scene_store.write_object_coverage(
            ObjectCoverage(
                sequence_id=sequence_id,
                object_id=obj.object_id,
                num_views=len(vis_entries),
                angular_span_deg=angular_span,
                num_distinct_viewpoints=n_viewpoints,
                mean_visibility_ratio=float(np.mean(vis_ratios)),
            )
        )
        count += 1

    logger.info(f"[{sequence_id}] Built {count} object_coverage entries")

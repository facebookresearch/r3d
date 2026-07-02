# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Object movement analysis for ADT dataset.

This module provides functions to analyze object movement over time
in ADT sequences.
"""

from __future__ import annotations

import numpy as np
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.object_info import BBox3D, ObjectInfo


def get_all_objects(
    gt_provider: AriaDigitalTwinDataProvider,
    timestamps_ns: list[int],
) -> dict[int, ObjectInfo]:
    """Build an ObjectInfo dict for every object in the scene.

    Each ObjectInfo gets its bbox set to the value from the *last* sampled
    timestamp at which a 3D bbox is available (matching the previous
    behaviour of ``analyze_object_movement``).

    Args:
        gt_provider: Data provider for the sequence.
        timestamps_ns: List of valid timestamps.

    Returns:
        Dictionary mapping instance_id to ObjectInfo.
    """
    instance_ids = gt_provider.get_instance_ids()

    # Collect static metadata
    names: dict[int, str] = {}
    categories: dict[int, str] = {}
    for obj_id in instance_ids:
        if gt_provider.has_instance_id(obj_id):
            info = gt_provider.get_instance_info_by_id(obj_id)
            names[obj_id] = info.name
            categories[obj_id] = info.category

    # Sample timestamps for efficiency (check every 10th frame)
    sample_indices = list(range(0, len(timestamps_ns), 10))
    if len(sample_indices) > 0 and sample_indices[-1] != len(timestamps_ns) - 1:
        sample_indices.append(len(timestamps_ns) - 1)

    # Collect last bbox per object (overwriting keeps the last one)
    last_bbox: dict[int, BBox3D] = {}
    for idx in sample_indices:
        timestamp_ns = timestamps_ns[idx]
        bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
            timestamp_ns
        )
        if not bbox3d_with_dt.is_valid():
            continue

        bboxes3d = bbox3d_with_dt.data()
        for obj_id, bbox3d in bboxes3d.items():
            if obj_id in names:
                last_bbox[obj_id] = BBox3D.from_aabb(
                    bbox3d.aabb, bbox3d.transform_scene_object
                )

    # Build frozen ObjectInfo instances with bbox set at construction
    objects: dict[int, ObjectInfo] = {}
    for obj_id in names:
        objects[obj_id] = ObjectInfo(
            instance_id=obj_id,
            name=names[obj_id],
            category=categories[obj_id],
            bbox=last_bbox.get(obj_id),
        )

    return objects


def analyze_object_movement(
    gt_provider: AriaDigitalTwinDataProvider,
    timestamps_ns: list[int],
    movement_threshold: float = 0.1,  # 10cm threshold for "moved"
) -> dict[int, tuple[bool, float, np.ndarray | None, np.ndarray | None]]:
    """Analyze object positions over time to detect movement.

    Returns per-object temporal data (is_static, movement_distance,
    first_position, last_position) *without* modifying ObjectInfo, which
    now only holds static fields.

    Args:
        gt_provider: Data provider for the sequence.
        timestamps_ns: List of valid timestamps.
        movement_threshold: Minimum distance (meters) to consider an
            object as "moved".

    Returns:
        Dictionary mapping instance_id to a tuple of
        (is_static, movement_distance, first_position, last_position).
    """
    # Sample timestamps for efficiency (check every 10th frame)
    sample_indices = list(range(0, len(timestamps_ns), 10))
    if len(sample_indices) > 0 and sample_indices[-1] != len(timestamps_ns) - 1:
        sample_indices.append(len(timestamps_ns) - 1)

    first_positions: dict[int, np.ndarray] = {}
    last_positions: dict[int, np.ndarray] = {}

    for idx in sample_indices:
        timestamp_ns = timestamps_ns[idx]
        bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
            timestamp_ns
        )
        if not bbox3d_with_dt.is_valid():
            continue

        bboxes3d = bbox3d_with_dt.data()
        for obj_id, bbox3d in bboxes3d.items():
            center = BBox3D.from_aabb(
                bbox3d.aabb, bbox3d.transform_scene_object
            ).center.copy()
            if obj_id not in first_positions:
                first_positions[obj_id] = center
            last_positions[obj_id] = center

    result: dict[int, tuple[bool, float, np.ndarray | None, np.ndarray | None]] = {}
    for obj_id in set(first_positions) | set(last_positions):
        first = first_positions.get(obj_id)
        last = last_positions.get(obj_id)
        if first is not None and last is not None:
            dist = float(np.linalg.norm(last - first))
            result[obj_id] = (dist < movement_threshold, dist, first, last)
        else:
            result[obj_id] = (True, 0.0, first, last)

    return result

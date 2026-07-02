# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Scene-level helper functions shared across question generators.

Relocated from global_qs.common, global_qs.complex_reasoning_utils,
and global_qs.how_long so that v2 generators can depend on utils
rather than reaching back into global_qs.
"""

from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.movement import analyze_object_movement, get_all_objects
from r3d.data_gen.extractor.object_info import ObjectMovementEvent, VideoObjectInfo

# Import from filter.global using importlib since 'global' is a Python keyword
_global_module = importlib.import_module("r3d.data_gen.filter.global")
get_globally_unique_objects = _global_module.get_globally_unique_objects

# Distance threshold (meters) for considering an object "held" by the user.
HELD_OBJECT_DISTANCE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# From global_qs/common.py
# ---------------------------------------------------------------------------


def is_likely_held(
    gt_provider: AriaDigitalTwinDataProvider,
    obj_id: int,
    timestamp_ns: int,
) -> bool:
    """Check if an object is likely being held by the user at a timestamp.

    An object is considered "held" if its 3D bbox center is within
    HELD_OBJECT_DISTANCE_THRESHOLD of the Aria device position. This
    helps avoid nonsensical questions like "how far is this from me"
    when the user is holding the object.

    Args:
        gt_provider: Data provider for the sequence.
        obj_id: Object instance ID.
        timestamp_ns: Timestamp to check.

    Returns:
        True if the object is likely being held.
    """
    aria_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
    if not aria_pose_with_dt.is_valid():
        return False
    aria_pos = aria_pose_with_dt.data().transform_scene_device.to_matrix()[:3, 3]

    bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
        timestamp_ns
    )
    if not bbox3d_with_dt.is_valid():
        return False
    bboxes3d = bbox3d_with_dt.data()
    if obj_id not in bboxes3d:
        return False

    obj_pos = bboxes3d[obj_id].transform_scene_object.to_matrix()[:3, 3]
    return float(np.linalg.norm(obj_pos - aria_pos)) < HELD_OBJECT_DISTANCE_THRESHOLD


def find_frame_with_object(
    gt_provider: AriaDigitalTwinDataProvider,
    obj_id: int,
    timestamps_ns: list[int],
    stream_id: StreamId,
) -> int:
    """Find a frame where the object is visible.

    Args:
        gt_provider: AriaDigitalTwinDataProvider instance.
        obj_id: Object instance ID.
        timestamps_ns: List of valid timestamps.
        stream_id: Camera stream ID.

    Returns:
        Timestamp where object is visible.

    Raises:
        ValueError: If no frame contains the object.
    """
    for ts in timestamps_ns:
        bbox2d_with_dt = gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(
            ts, stream_id
        )
        if bbox2d_with_dt.is_valid():
            if obj_id in bbox2d_with_dt.data():
                return ts

    raise ValueError(f"Could not find frame with object {obj_id}")


def get_video_object_info(
    gt_provider: AriaDigitalTwinDataProvider,
    timestamps_ns: list[int],
    stream_id: StreamId,
) -> list[VideoObjectInfo]:
    """Build a fully-populated list of VideoObjectInfo for global questions.

    This consolidates the previous ``build_video_object_info_dict`` and
    ``prepare_video_objects`` into a single call.  For each object in the
    scene it:

    1. Creates an ``ObjectInfo`` with static fields (name, category, bbox).
    2. Runs movement analysis to populate temporal fields (``is_static``,
       ``movement_distance``, ``first_position``, ``last_position``).
    3. Wraps everything in a ``VideoObjectInfo``.

    Args:
        gt_provider: Data provider for the sequence.
        timestamps_ns: All valid timestamps in the sequence.
        stream_id: Camera stream ID.

    Returns:
        List of VideoObjectInfo ready for global question generation.
    """
    # 1. Build static ObjectInfo dict
    objects_dict = get_all_objects(gt_provider, timestamps_ns)

    # 2. Analyze movement (returns temporal data only)
    movement_data = analyze_object_movement(gt_provider, timestamps_ns)

    # 3. Build VideoObjectInfo list
    video_objects: dict[int, VideoObjectInfo] = {}
    for instance_id, obj in objects_dict.items():
        is_static, movement_distance, first_pos, last_pos = movement_data.get(
            instance_id, (True, 0.0, None, None)
        )

        # Build movement_event from the overall first->last displacement.
        # A non-static object gets an ObjectMovementEvent capturing
        # the net displacement detected during the sequence.
        movement_event: ObjectMovementEvent | None = None
        if not is_static and first_pos is not None and last_pos is not None:
            movement_event = ObjectMovementEvent(
                instance_id=instance_id,
                start_timestamp_ns=timestamps_ns[0],
                end_timestamp_ns=timestamps_ns[-1],
                start_position=first_pos,
                end_position=last_pos,
                movement_distance=movement_distance,
            )

        video_objects[instance_id] = VideoObjectInfo(
            object_info=obj,
            movement_event=movement_event,
            is_static=is_static,
            movement_distance=movement_distance,
            first_position=first_pos,
            last_position=last_pos,
        )

    # 4. Mark globally unique objects
    unique_objs = get_globally_unique_objects(video_objects)
    unique_ids = {obj.instance_id for obj in unique_objs}
    for instance_id in unique_ids:
        video_objects[instance_id] = replace(
            video_objects[instance_id], is_globally_unique=True
        )

    return list(video_objects.values())


# ---------------------------------------------------------------------------
# From global_qs/how_long.py
# ---------------------------------------------------------------------------


def get_longest_dimension_at_timestamp(
    gt_provider: AriaDigitalTwinDataProvider,
    obj: VideoObjectInfo,
    timestamp_ns: int,
) -> float | None:
    """Get the longest dimension of an object at a specific timestamp.

    Args:
        gt_provider: Data provider for the sequence.
        obj: The object to measure.
        timestamp_ns: Timestamp in nanoseconds.

    Returns:
        Longest dimension in meters, or None if not available.
    """
    bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
        timestamp_ns
    )
    assert bbox3d_with_dt.is_valid()

    bboxes3d = bbox3d_with_dt.data()
    assert obj.instance_id in bboxes3d

    bbox3d = bboxes3d[obj.instance_id]
    # ADT aabb format: [xmin, xmax, ymin, ymax, zmin, zmax]
    aabb = bbox3d.aabb
    width = float(aabb[1] - aabb[0])
    height = float(aabb[3] - aabb[2])
    depth = float(aabb[5] - aabb[4])
    return max(width, height, depth)

# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Functions for extracting scene objects from ADT dataset.

This module provides functions to get information about objects in ADT scenes,
including objects visible at specific timestamps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.object_info import BBox3D, ObjectInfo


def get_visible_objects_at_timestamp(
    gt_provider: AriaDigitalTwinDataProvider,
    timestamp_ns: int,
    stream_id: StreamId,
    objects: dict[int, ObjectInfo] | None = None,
) -> list[ObjectInfo] | list[dict[str, Any]]:
    """Get objects visible at a specific timestamp.

    Args:
        gt_provider: AriaDigitalTwinDataProvider instance.
        timestamp_ns: Timestamp in nanoseconds.
        stream_id: Camera stream ID.
        objects: Optional pre-computed object info dictionary. If provided,
            returns list of ObjectInfo. Otherwise returns list of dicts.

    Returns:
        List of visible objects (either ObjectInfo or dict depending on input).
    """
    bbox2d_with_dt = gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(
        timestamp_ns, stream_id
    )

    if not bbox2d_with_dt.is_valid():
        return []

    visible_ids = set(bbox2d_with_dt.data().keys())

    # If objects dict is provided, update bboxes and return ObjectInfo list
    if objects is not None:
        # Build updated copies with current bboxes for visible objects
        bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
            timestamp_ns
        )
        updated_bboxes: dict[int, BBox3D] = {}
        if bbox3d_with_dt.is_valid():
            bboxes3d = bbox3d_with_dt.data()
            for obj_id in visible_ids:
                if obj_id in bboxes3d and obj_id in objects:
                    updated_bboxes[obj_id] = BBox3D.from_aabb(
                        bboxes3d[obj_id].aabb, bboxes3d[obj_id].transform_scene_object
                    )

        result: list[ObjectInfo] = []
        for obj_id in visible_ids:
            if obj_id not in objects:
                continue
            obj = objects[obj_id]
            if obj_id in updated_bboxes:
                obj = replace(obj, bbox=updated_bboxes[obj_id])
            result.append(obj)
        return result

    # Otherwise return simple dict list
    visible_objects = []
    for obj_id in visible_ids:
        if gt_provider.has_instance_id(obj_id):
            info = gt_provider.get_instance_info_by_id(obj_id)
            visible_objects.append(
                {
                    "id": obj_id,
                    "name": info.name,
                    "category": info.category,
                }
            )

    return visible_objects

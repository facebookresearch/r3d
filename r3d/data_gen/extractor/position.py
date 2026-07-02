# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Functions for extracting position information from ADT dataset.

This module provides functions to get 2D and 3D position information
for objects and the Aria device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.object_info import ObjectInfo


def get_object_3d_position(
    gt_provider: AriaDigitalTwinDataProvider,
    obj_id: int,
    timestamp_ns: int,
) -> dict[str, float] | None:
    """Get 3D position of an object at a specific timestamp.

    Args:
        gt_provider: AriaDigitalTwinDataProvider instance.
        obj_id: Object instance ID.
        timestamp_ns: Timestamp in nanoseconds.

    Returns:
        Dictionary with x, y, z coordinates, or None if not available.
    """
    bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
        timestamp_ns
    )
    if not bbox3d_with_dt.is_valid():
        return None

    bboxes3d = bbox3d_with_dt.data()
    if obj_id not in bboxes3d:
        return None

    bbox3d = bboxes3d[obj_id]
    transform = bbox3d.transform_scene_object.to_matrix()
    position = transform[:3, 3]

    return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}


def get_object_3d_info(
    gt_provider: AriaDigitalTwinDataProvider,
    obj_id: int,
    timestamp_ns: int,
) -> dict[str, Any] | None:
    """Get full 3D info of an object at a specific timestamp including size.

    Args:
        gt_provider: AriaDigitalTwinDataProvider instance.
        obj_id: Object instance ID.
        timestamp_ns: Timestamp in nanoseconds.

    Returns:
        Dictionary with position and size information, or None if not available.
    """
    bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
        timestamp_ns
    )
    if not bbox3d_with_dt.is_valid():
        return None

    bboxes3d = bbox3d_with_dt.data()
    if obj_id not in bboxes3d:
        return None

    bbox3d = bboxes3d[obj_id]
    transform = bbox3d.transform_scene_object.to_matrix()
    position = transform[:3, 3]

    # Get AABB (axis-aligned bounding box) dimensions
    aabb = bbox3d.aabb
    # ADT aabb format: [xmin, xmax, ymin, ymax, zmin, zmax]
    dimensions = {
        "width": float(aabb[1] - aabb[0]),
        "height": float(aabb[3] - aabb[2]),
        "depth": float(aabb[5] - aabb[4]),
    }

    return {
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "dimensions": dimensions,
        "aabb": {
            "x_min": float(aabb[0]),
            "x_max": float(aabb[1]),
            "y_min": float(aabb[2]),
            "y_max": float(aabb[3]),
            "z_min": float(aabb[4]),
            "z_max": float(aabb[5]),
        },
    }


def get_object_2d_bbox(
    gt_provider: AriaDigitalTwinDataProvider,
    obj_id: int,
    timestamp_ns: int,
    stream_id: StreamId,
) -> dict[str, Any] | None:
    """Get 2D bounding box of an object at a specific timestamp.

    Args:
        gt_provider: AriaDigitalTwinDataProvider instance.
        obj_id: Object instance ID.
        timestamp_ns: Timestamp in nanoseconds.
        stream_id: Camera stream ID.

    Returns:
        Dictionary with 2D bounding box information, or None if not available.
    """
    bbox2d_with_dt = gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(
        timestamp_ns, stream_id
    )
    if not bbox2d_with_dt.is_valid():
        return None

    bboxes2d = bbox2d_with_dt.data()
    if obj_id not in bboxes2d:
        return None

    bbox2d = bboxes2d[obj_id]
    return {
        "x_min": float(bbox2d.box_range[0]),
        "x_max": float(bbox2d.box_range[1]),
        "y_min": float(bbox2d.box_range[2]),
        "y_max": float(bbox2d.box_range[3]),
        "visibility_ratio": float(bbox2d.visibility_ratio),
    }


def get_aria_position_at_timestamp(
    gt_provider: AriaDigitalTwinDataProvider,
    timestamp_ns: int,
) -> np.ndarray | None:
    """Get the Aria device position at a timestamp.

    Args:
        gt_provider: AriaDigitalTwinDataProvider instance.
        timestamp_ns: Timestamp in nanoseconds.

    Returns:
        3D position as numpy array, or None if not available.
    """
    aria_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
    if not aria_pose_with_dt.is_valid():
        return None

    aria_pose = aria_pose_with_dt.data()
    # Extract translation from transform
    transform_matrix = aria_pose.transform_scene_device.to_matrix()
    return transform_matrix[:3, 3]


def distance_between_objects(obj1: ObjectInfo, obj2: ObjectInfo) -> float | None:
    """Calculate shortest distance between two objects' bounding boxes.

    Args:
        obj1: First object.
        obj2: Second object.

    Returns:
        Distance in meters, or None if bboxes are not available.
    """
    if obj1.bbox is None or obj2.bbox is None:
        return None
    return obj1.bbox.shortest_distance_to_bbox(obj2.bbox)


def distance_from_aria(obj: ObjectInfo, aria_position: np.ndarray) -> float | None:
    """Calculate distance from Aria device to closest point on object bounding box.

    Args:
        obj: Object to measure distance to.
        aria_position: 3D position of the Aria device.

    Returns:
        Distance in meters, or None if bbox is not available.
    """
    if obj.bbox is None:
        return None
    closest_point = obj.bbox.closest_point_to(aria_position)
    return float(np.linalg.norm(closest_point - aria_position))


def get_relative_position_description(
    obj: ObjectInfo,
    reference_obj: ObjectInfo,
    gt_provider: AriaDigitalTwinDataProvider,
    timestamp_ns: int,
) -> str | None:
    """Get a natural language description of obj's position relative to reference_obj.

    Args:
        obj: The object to describe the position of.
        reference_obj: The reference object.
        gt_provider: AriaDigitalTwinDataProvider instance.
        timestamp_ns: Timestamp in nanoseconds.

    Returns:
        A string like "to the left of", "above", "behind", etc., or None if
        positions cannot be determined.
    """
    obj_pos = get_object_3d_position(gt_provider, obj.instance_id, timestamp_ns)
    ref_pos = get_object_3d_position(
        gt_provider, reference_obj.instance_id, timestamp_ns
    )

    if obj_pos is None or ref_pos is None:
        return None

    # Calculate relative position
    dx = obj_pos["x"] - ref_pos["x"]
    dy = obj_pos["y"] - ref_pos["y"]
    dz = obj_pos["z"] - ref_pos["z"]

    # Get distance between objects
    dist = distance_between_objects(obj, reference_obj)
    if dist is None or dist < 0.1:
        return None

    # Determine dominant direction
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

    # Threshold for "significant" movement in each direction
    threshold = 0.15  # 15cm

    descriptions = []

    # Horizontal (left/right) - x axis
    if abs_dx > threshold:
        if dx > 0:
            descriptions.append("to the right of")
        else:
            descriptions.append("to the left of")

    # Vertical (above/below) - Y-up (gravity along -Y in ADT scene coords)
    if abs_dy > threshold:
        if dy > 0:
            descriptions.append("above")
        else:
            descriptions.append("below")

    # Depth (in front/behind) - z axis
    if abs_dz > threshold:
        if dz > 0:
            descriptions.append("behind")
        else:
            descriptions.append("in front of")

    if not descriptions:
        # Objects are very close, use "next to"
        return "next to"

    # Combine descriptions for compound positions
    if len(descriptions) == 1:
        return descriptions[0]
    elif len(descriptions) == 2:
        # e.g., "to the left of and above"
        return f"{descriptions[0]} and {descriptions[1]}"
    else:
        # e.g., "to the left of, above, and behind"
        return f"{descriptions[0]}, {descriptions[1]}, and {descriptions[2]}"

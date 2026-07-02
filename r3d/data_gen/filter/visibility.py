# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Visibility-based filtering for ADT objects.

This module provides functions to filter objects based on their visibility
in the camera view, including bbox projection checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataProvider,
    utils as adt_utils,
)
from r3d.data_gen.extractor.calibration import get_corrected_camera_calibration
from r3d.data_gen.extractor.object_info import ObjectInfo
from r3d.data_gen.extractor.position import get_object_2d_bbox
from r3d.data_gen.filter.distance import compute_angular_size_degrees
from r3d.data_gen.filter.uniqueness import filter_unique_names


def can_project_bbox_to_image(
    obj: ObjectInfo,
    gt_provider: AriaDigitalTwinDataProvider,
    timestamp_ns: int,
    stream_id: StreamId,
    corrected_calib: Any,
) -> bool:
    """Check if an object's bounding box can be projected onto the corrected image.

    Args:
        obj: Object to check.
        gt_provider: Data provider for the sequence.
        timestamp_ns: Timestamp in nanoseconds.
        stream_id: Camera stream ID.
        corrected_calib: Corrected camera calibration.

    Returns:
        True if the bbox can be projected, False otherwise.
    """
    if corrected_calib is None:
        return False

    # Get aria pose at timestamp
    aria3dpose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
    if not aria3dpose_with_dt.is_valid():
        return False
    aria3dpose = aria3dpose_with_dt.data()

    # Get 3D bounding boxes
    bboxes3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
        timestamp_ns
    )
    if not bboxes3d_with_dt.is_valid():
        return False
    bboxes3d = bboxes3d_with_dt.data()

    # Check if this object's bbox can be projected
    if obj.instance_id not in bboxes3d:
        return False

    bbox3d = bboxes3d[obj.instance_id]

    # Compute transform chain
    transform_cam_device = corrected_calib.get_transform_device_camera().inverse()
    transform_cam_scene = (
        transform_cam_device.to_matrix()
        @ aria3dpose.transform_scene_device.inverse().to_matrix()
    )
    transform_cam_obj = transform_cam_scene @ bbox3d.transform_scene_object.to_matrix()

    try:
        projected_bbox = adt_utils.project_3d_bbox_to_image(
            bbox3d.aabb, transform_cam_obj, corrected_calib
        )
    except Exception:
        return False

    return projected_bbox is not None and len(projected_bbox) > 0


def filter_objects(
    objects: list[ObjectInfo],
    gt_provider: AriaDigitalTwinDataProvider,
    timestamp_ns: int,
    stream_id: StreamId,
    viewer_position: np.ndarray,
    unique: bool = False,
    min_fov_degrees: float = 0.0,
    max_distance: float = 0.0,
    require_2d_bbox: bool = True,
    min_visibility_ratio: float = 0.0,
) -> list[ObjectInfo]:
    """Filter objects based on specified criteria.

    Args:
        objects: List of objects to filter.
        gt_provider: Data provider for the sequence.
        timestamp_ns: Timestamp in nanoseconds.
        stream_id: Camera stream ID.
        viewer_position: The 3D position of the viewer.
        unique: If True, only include objects with unique natural names
            (after removing size modifiers like "large" or "small").
        min_fov_degrees: Filter out objects whose angular size
            (as seen from viewer_position) is less than this many degrees.
            Set to 0 to disable.
        max_distance: Filter out objects whose closest point is farther
            than this many meters from the viewer. Set to 0 to disable.
        require_2d_bbox: If True, require objects to have a visible 2D bbox
            that can be projected onto the image.
        min_visibility_ratio: Minimum visibility ratio (0-1) for 2D bbox.
            Set to 0 to disable.

    Returns:
        Filtered list of objects meeting all specified criteria.
    """
    filtered = list(objects)

    # Filter to objects with natural names and bboxes
    filtered = [
        obj for obj in filtered if obj.natural_name is not None and obj.bbox is not None
    ]

    # Filter for uniqueness
    if unique:
        filtered = filter_unique_names(filtered)

    # Filter by angular size
    if min_fov_degrees > 0:
        filtered = [
            obj
            for obj in filtered
            if (angular_size := compute_angular_size_degrees(obj, viewer_position))
            is not None
            and angular_size >= min_fov_degrees
        ]

    # Filter by distance from viewer
    if max_distance > 0:
        filtered = [
            obj
            for obj in filtered
            if obj.bbox is not None
            and np.linalg.norm(
                obj.bbox.closest_point_to(viewer_position) - viewer_position
            )
            <= max_distance
        ]

    # Filter by 2D bounding box visibility and projection
    if require_2d_bbox or min_visibility_ratio > 0:
        corrected_calib = get_corrected_camera_calibration(
            gt_provider, timestamp_ns, stream_id
        )

        filtered_with_2d = []
        for obj in filtered:
            # Check if bbox can be projected
            if require_2d_bbox:
                if not can_project_bbox_to_image(
                    obj, gt_provider, timestamp_ns, stream_id, corrected_calib
                ):
                    continue

            # Check visibility ratio
            if min_visibility_ratio > 0:
                bbox_2d = get_object_2d_bbox(
                    gt_provider, obj.instance_id, timestamp_ns, stream_id
                )
                if bbox_2d is None:
                    continue
                visibility = bbox_2d.get("visibility_ratio", 1.0)
                if visibility < min_visibility_ratio:
                    continue

            filtered_with_2d.append(obj)
        filtered = filtered_with_2d

    return filtered

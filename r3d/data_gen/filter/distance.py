# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Distance-based filtering for ADT objects.

This module provides functions to filter objects based on distance
from the viewer and angular size in the field of view.
"""

from __future__ import annotations

import numpy as np
from r3d.data_gen.extractor.object_info import ObjectInfo


def compute_angular_size_degrees(
    obj: ObjectInfo,
    viewer_position: np.ndarray,
) -> float | None:
    """Compute the angular size of an object as seen from the viewer.

    Args:
        obj: Object to compute angular size for.
        viewer_position: 3D position of the viewer.

    Returns:
        Angular size in degrees, or None if bbox is not available.
    """
    if obj.bbox is None:
        return None

    # Get the closest point on the bbox to the viewer
    closest_point = obj.bbox.closest_point_to(viewer_position)
    distance = np.linalg.norm(closest_point - viewer_position)

    if distance < 0.01:  # Very close, avoid division issues
        return 180.0

    # Use the longest dimension as the object size
    object_size = obj.bbox.longest_dimension

    # Angular size in radians: 2 * arctan(size / (2 * distance))
    angular_size_rad = 2 * np.arctan(object_size / (2 * distance))
    angular_size_deg = np.degrees(angular_size_rad)

    return float(angular_size_deg)

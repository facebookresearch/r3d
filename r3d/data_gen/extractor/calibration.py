# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Camera calibration utilities for ADT dataset.

This module provides functions to get corrected camera calibration
(undistorted and rotated upright) for the Aria device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from projectaria_tools.utils.calibration_utils import (
    rotate_upright_image_and_calibration,
    undistort_image_and_calibration,
)


def get_corrected_camera_calibration(
    gt_provider: AriaDigitalTwinDataProvider,
    timestamp_ns: int,
    stream_id: StreamId,
) -> Any | None:
    """Get the corrected camera calibration (undistorted and rotated upright).

    Args:
        gt_provider: Data provider for the sequence.
        timestamp_ns: Timestamp in nanoseconds.
        stream_id: Camera stream ID.

    Returns:
        Corrected camera calibration, or None if not available.
    """
    # Get the image to ensure valid timestamp
    image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, stream_id)
    if not image_with_dt.is_valid():
        return None

    image = image_with_dt.data().to_numpy_array()
    if len(image.shape) < 3:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)

    camera_calib = gt_provider.get_aria_camera_calibration(stream_id)
    if camera_calib is None:
        return None

    # Apply undistortion and rotation corrections
    _, corrected_calib = undistort_image_and_calibration(image, camera_calib)
    _, corrected_calib = rotate_upright_image_and_calibration(image, corrected_calib)

    return corrected_calib

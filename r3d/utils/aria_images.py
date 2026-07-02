# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared utilities for loading and correcting Aria RGB images, depth maps, and poses.

Consolidates the common "load image, undistort, rotate upright" pattern used
across the pipeline. Also provides depth loading and device-pose retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from projectaria_tools.utils.calibration_utils import (
    rotate_upright_image_and_calibration,
    undistort_image_and_calibration,
)
from r3d.utils.transforms import invert_rigid_transform

logger: logging.Logger = logging.getLogger(__name__)


def get_raw_stream_image(
    provider: Any,
    timestamp_ns: int,
    stream_id: Any,
) -> np.ndarray | None:
    """Load a raw image frame without undistortion or rotation."""
    image_with_dt = provider.get_aria_image_by_timestamp_ns(timestamp_ns, stream_id)
    if not image_with_dt.is_valid():
        return None
    image = image_with_dt.data().to_numpy_array()
    if image.ndim == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    return image


def get_camera_calibration_for_timestamp(
    provider: Any,
    timestamp_ns: int,
    stream_id: Any,
) -> Any | None:
    """Get the best available camera calibration for a timestamp."""
    camera_calib = provider.get_aria_camera_calibration(stream_id)
    if camera_calib is None:
        return None

    mps_provider = provider.mps_data_provider_ptr()
    raw_provider = provider.raw_data_provider_ptr()
    if (
        mps_provider is None
        or raw_provider is None
        or not hasattr(mps_provider, "has_online_calibrations")
        or not mps_provider.has_online_calibrations()
    ):
        return camera_calib

    stream_label = raw_provider.get_label_from_stream_id(stream_id)
    online_calib = mps_provider.get_online_calibration(timestamp_ns)
    online_camera_calib = online_calib.get_camera_calib(stream_label)
    if online_camera_calib is None:
        return camera_calib

    online_size = tuple(int(x) for x in online_camera_calib.get_image_size())
    static_size = tuple(int(x) for x in camera_calib.get_image_size())
    if online_size != static_size:
        logger.warning(
            "Ignoring online calibration for %s at %d due to image-size mismatch: %s vs %s",
            stream_label,
            timestamp_ns,
            online_size,
            static_size,
        )
        return camera_calib

    return online_camera_calib


def get_scene_camera_transform(
    provider: Any,
    timestamp_ns: int,
    stream_id: Any,
    camera_calib: Any,
) -> np.ndarray | None:
    """Get T_scene_camera for the corrected (undistorted+rotated) camera.

    Follows the Project Aria tutorial pattern:
      T_scene_device from closed-loop pose
      T_device_camera from the corrected calibration
      T_scene_camera = T_scene_device @ T_device_camera
    """
    pose_result = provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)
    if not pose_result.is_valid():
        return None
    T_scene_device = np.array(pose_result.data().transform_scene_device.to_matrix())
    T_device_camera = np.array(camera_calib.get_transform_device_camera().to_matrix())
    return T_scene_device @ T_device_camera


def get_rgb_corrected_timestamp_ns(
    provider: Any,
    capture_timestamp_ns: int,
    stream_id: Any,
) -> int:
    """Get the corrected RGB timestamp when MPS online calibration is available."""
    if str(stream_id) != "214-1":
        return capture_timestamp_ns
    mps_provider = provider.mps_data_provider_ptr()
    if (
        mps_provider is None
        or not hasattr(mps_provider, "has_online_calibrations")
        or not mps_provider.has_online_calibrations()
    ):
        return capture_timestamp_ns
    corrected_ts = mps_provider.get_rgb_corrected_timestamp_ns(capture_timestamp_ns)
    return int(corrected_ts) if corrected_ts is not None else capture_timestamp_ns


def get_corrected_rgb_image_and_calibration(
    provider: Any,
    capture_timestamp_ns: int,
    stream_id: Any,
) -> tuple[np.ndarray | None, Any | None, int]:
    """Load corrected upright RGB plus timestamp-specific calibration."""
    image = get_raw_stream_image(provider, capture_timestamp_ns, stream_id)
    if image is None:
        return None, None, capture_timestamp_ns

    camera_calib = get_camera_calibration_for_timestamp(
        provider, capture_timestamp_ns, stream_id
    )
    if camera_calib is None:
        return None, None, capture_timestamp_ns

    corrected_image, corrected_calib = undistort_image_and_calibration(
        image, camera_calib
    )
    corrected_image, corrected_calib = rotate_upright_image_and_calibration(
        corrected_image, corrected_calib
    )
    corrected_image = np.ascontiguousarray(corrected_image)
    if corrected_image.dtype != np.uint8:
        corrected_image = corrected_image.astype(np.uint8, copy=False)

    corrected_timestamp_ns = get_rgb_corrected_timestamp_ns(
        provider, capture_timestamp_ns, stream_id
    )
    return corrected_image, corrected_calib, corrected_timestamp_ns


def get_corrected_rgb(
    provider: Any,
    timestamp_ns: int,
    stream_id: Any,
) -> np.ndarray | None:
    """Load, undistort, and rotate an Aria RGB image to upright orientation.

    Args:
        provider: An ADT data provider (AriaDigitalTwinDataProvider or similar)
            that exposes ``get_aria_image_by_timestamp_ns`` and
            ``get_aria_camera_calibration``.
        timestamp_ns: Capture timestamp in nanoseconds.
        stream_id: Camera stream ID (e.g. ``StreamId("214-1")``).

    Returns:
        (H, W, 3) uint8 array in upright orientation, or None if the image
        is unavailable.
    """
    image, _calib = get_corrected_rgb_with_calibration(
        provider, timestamp_ns, stream_id
    )
    return image


def get_corrected_rgb_with_calibration(
    provider: Any,
    timestamp_ns: int,
    stream_id: Any,
) -> tuple[np.ndarray | None, Any | None]:
    """Load, undistort, and rotate an Aria RGB image. Returns image + calibration.

    Uses timestamp-specific online calibration (when available) via
    ``get_corrected_rgb_image_and_calibration``, discarding the corrected
    timestamp. Prefer ``get_corrected_rgb_image_and_calibration`` directly
    when you also need the corrected timestamp for pose lookups.

    Args:
        provider: An ADT data provider that exposes
            ``get_aria_image_by_timestamp_ns`` and
            ``get_aria_camera_calibration``.
        timestamp_ns: Capture timestamp in nanoseconds.
        stream_id: Camera stream ID.

    Returns:
        ``(image, calibration)`` where *image* is a (H, W, 3) uint8 array and
        *calibration* is the corrected ``CameraCalibration``, or
        ``(None, None)`` if the image is unavailable.
    """
    image, calib, _corrected_ts = get_corrected_rgb_image_and_calibration(
        provider, timestamp_ns, stream_id
    )
    return image, calib


def get_raw_depth(
    provider: Any,
    timestamp_ns: int,
    stream_id: Any,
    depth_scale: float = 1.0,
) -> tuple[np.ndarray, Any] | None:
    """Fetch raw metric depth and corrected calibration at a timestamp.

    Args:
        provider: ADT data provider.
        timestamp_ns: Timestamp in nanoseconds.
        stream_id: Camera stream ID.
        depth_scale: Multiplier to convert raw depth to meters.
            Use 1.0 if depth is already in meters, 0.001 if in mm.

    Returns:
        (depth_meters, corrected_calibration) or None if unavailable.
        depth_meters has shape (H, W).
    """
    depth_with_dt = provider.get_depth_image_by_timestamp_ns(timestamp_ns, stream_id)
    if not depth_with_dt.is_valid():
        return None

    depth_array = depth_with_dt.data().to_numpy_array().astype(np.float32)
    depth_array = depth_array * depth_scale

    camera_calib = provider.get_aria_camera_calibration(stream_id)
    if camera_calib is None:
        raise RuntimeError(f"No camera calibration for stream {stream_id}")

    depth_array, corrected_calib = undistort_image_and_calibration(
        depth_array, camera_calib
    )
    depth_array, corrected_calib = rotate_upright_image_and_calibration(
        depth_array, corrected_calib
    )

    return depth_array, corrected_calib


def get_corrected_rgb_pose(
    provider: Any,
    timestamp_ns: int,
    stream_id: Any,
) -> np.ndarray | None:
    """Get T_camera_world for the corrected (undistorted + rotated) RGB camera.

    Returns:
        (4, 4) T_camera_world matrix, or None if pose unavailable.
    """
    camera_calib = provider.get_aria_camera_calibration(stream_id)
    if camera_calib is None:
        return None
    # The calibration correction functions require an image argument but only
    # use its dimensions to compute the undistortion/rotation transform. A 1x1
    # dummy works because the resulting calibration is independent of image
    # content -- we only need the corrected intrinsics, not the corrected pixels.
    dummy = np.zeros((1, 1), dtype=np.uint8)
    _, corrected_calib = undistort_image_and_calibration(dummy, camera_calib)
    _, corrected_calib = rotate_upright_image_and_calibration(dummy, corrected_calib)

    T_scene_camera = get_scene_camera_transform(
        provider, timestamp_ns, stream_id, corrected_calib
    )
    if T_scene_camera is None:
        return None
    return invert_rigid_transform(T_scene_camera)

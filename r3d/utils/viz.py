# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared visualization primitives for R3D pipelines.

Drawing, projection, and overlay helpers for 3D bounding boxes and
segmentation masks.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from r3d.utils.transforms import invert_rigid_transform

logger: logging.Logger = logging.getLogger(__name__)

# 12 edges of a box connecting the 8 corners produced by aabb_to_world_corners.
BOX_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top face
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # vertical edges
]


def track_color(object_id: int) -> tuple[int, int, int]:
    """Return a deterministic RGB color for a given track/object ID.

    Uses a modular hash to keep the seed within numpy's uint32 range,
    then draws three independent uniform samples in [60, 255) so every
    channel is comfortably visible on both light and dark backgrounds.
    """
    seed = (object_id * 7 + 13) % (2**32)
    rng = np.random.RandomState(seed)
    return (
        int(rng.randint(60, 255)),
        int(rng.randint(60, 255)),
        int(rng.randint(60, 255)),
    )


def aabb_to_world_corners(aabb: list[float], transform: np.ndarray) -> np.ndarray:
    """Convert a 6-element AABB + 4x4 transform to 8 world-frame corners.

    Args:
        aabb: [x_min, x_max, y_min, y_max, z_min, z_max].
        transform: 4x4 rigid transform placing the local box in world frame.

    Returns:
        shape (8, 3) array of world-frame corner coordinates.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = aabb
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ]
    )
    T = np.asarray(transform, dtype=np.float64)
    h = np.hstack([corners, np.ones((8, 1))])
    return (T @ h.T).T[:, :3]


def project_obb_to_2d(
    obb_aabb: np.ndarray,
    obb_transform: np.ndarray,
    T_scene_device: np.ndarray,
    T_device_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    """Project an OBB to 2D image coordinates using pinhole projection.

    For use with pipeline FrameData (pre-undistorted, no rotation needed).
    Unlike project_3d_box_corners, this does NOT apply the 90-degree Aria
    rotation -- pipeline frames are already in upright orientation.

    Args:
        obb_aabb: shape (6,) [xmin, xmax, ymin, ymax, zmin, zmax] in local frame.
        obb_transform: shape (4,4) local-to-world rigid transform.
        T_scene_device: shape (4,4) device pose in world/scene frame.
        T_device_camera: shape (4,4) camera-in-device transform.
        fx, fy: focal lengths in pixels.
        cx, cy: principal point in pixels.

    Returns:
        shape (8, 2) projected pixel coordinates, or None if all corners
        are behind the camera.
    """
    corners = aabb_to_world_corners(obb_aabb.tolist(), obb_transform)
    T_camera_scene = invert_rigid_transform(T_scene_device @ T_device_camera)

    corners_h = np.hstack([corners, np.ones((8, 1))])
    corners_cam = (T_camera_scene @ corners_h.T).T[:, :3]

    # Skip if centroid is behind camera
    centroid_z = corners_cam[:, 2].mean()
    if centroid_z <= 0:
        return None

    # Skip if all corners are behind camera
    front_mask = corners_cam[:, 2] > 0
    if not np.any(front_mask):
        return None

    # Project in-front corners; mark behind-camera as NaN
    z_safe = np.where(front_mask, corners_cam[:, 2], np.nan)
    px = np.where(front_mask, corners_cam[:, 0] / z_safe * fx + cx, np.nan)
    py = np.where(front_mask, corners_cam[:, 1] / z_safe * fy + cy, np.nan)

    # Clamp behind-camera corners to nearest in-front projection bounds
    if not np.all(front_mask):
        px_valid = px[front_mask]
        py_valid = py[front_mask]
        px_clamp_min = px_valid.min()
        py_clamp_min = py_valid.min()
        px = np.where(front_mask, px, px_clamp_min)
        py = np.where(front_mask, py, py_clamp_min)

    return np.stack([px, py], axis=1)


def draw_wireframe(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dashed: bool = False,
    dash_len: int = 8,
) -> None:
    """Draw a 3D box wireframe on an image from projected 2D corners.

    Mutates *image* in place.

    Args:
        image: BGR or RGB image array.
        corners_2d: shape (8, 2) projected corner coordinates.
        color: RGB/BGR color tuple matching *image* channel order.
        thickness: Line thickness.
        dashed: If True, draw dashed lines.
        dash_len: Length of each dash segment (only used when *dashed* is True).
    """
    for i, j in BOX_EDGES:
        pt1 = (int(corners_2d[i, 0]), int(corners_2d[i, 1]))
        pt2 = (int(corners_2d[j, 0]), int(corners_2d[j, 1]))
        if dashed:
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = max(abs(dx), abs(dy))
            if length == 0:
                continue
            steps = max(1, length // (dash_len * 2))
            for s in range(steps):
                sx = pt1[0] + dx * (2 * s) // (2 * steps)
                sy = pt1[1] + dy * (2 * s) // (2 * steps)
                ex = pt1[0] + dx * (2 * s + 1) // (2 * steps)
                ey = pt1[1] + dy * (2 * s + 1) // (2 * steps)
                cv2.line(image, (sx, sy), (ex, ey), color, thickness)
        else:
            cv2.line(image, pt1, pt2, color, thickness)


def overlay_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray | tuple[int, int, int],
    blend: float = 0.4,
    inplace: bool = False,
) -> np.ndarray:
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    out = frame if inplace else frame.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (out[mask] * (1 - blend) + color_arr * blend).astype(np.uint8)
    return out


def draw_bbox_from_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
    inplace: bool = False,
) -> np.ndarray:
    """Draw a 2D bounding box derived from the segmentation mask, with a text label."""
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_u8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
    x_min, y_min, bw, bh = cv2.boundingRect(mask_u8)
    if bw == 0 or bh == 0:
        return frame
    x_max = x_min + bw - 1
    y_max = y_min + bh - 1

    out = frame if inplace else frame.copy()
    cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    label_y = max(y_min - 4, th + 4)
    cv2.rectangle(
        out, (x_min, label_y - th - 4), (x_min + tw + 6, label_y + 2), color, -1
    )
    cv2.putText(out, label, (x_min + 3, label_y - 2), font, scale, (0, 0, 0), 1)

    return out

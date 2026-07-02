# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Object information data structures for ADT dataset.

This module defines the core data structures for representing objects
and their bounding boxes in the Aria Digital Twin dataset, including
temporal (video-level) object info with movement tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from r3d.data_gen.utils.name_mapping import get_natural_name


@dataclass(frozen=True)
class BBox3D:
    """Represents an oriented 3D bounding box.

    Stores the bounding box in ADT convention: a local-frame AABB
    (aabb_local) plus an SE3 transform (transform_scene_object) that
    maps from the local OBB frame to scene coordinates.
    """

    aabb_local: np.ndarray  # (6,) [xmin, xmax, ymin, ymax, zmin, zmax] in OBB frame
    transform_scene_object: np.ndarray  # (4, 4) SE3 from OBB frame to scene

    @classmethod
    def from_aabb(cls, aabb: np.ndarray, transform_scene_object: Any) -> BBox3D:
        """Create BBox3D from ADT local AABB + SE3 transform.

        Args:
            aabb: numpy array [6,] with format [xmin, xmax, ymin, ymax, zmin, zmax]
            transform_scene_object: SE3 transform from object to scene coordinates
                (must have .to_matrix() method)
        """
        aabb_np = np.array([float(x) for x in aabb])
        T = transform_scene_object.to_matrix()
        return cls(aabb_local=aabb_np, transform_scene_object=T)

    @property
    def center(self) -> np.ndarray:
        """Return the center of the bounding box in scene coordinates."""
        a = self.aabb_local
        local_center = np.array(
            [(a[0] + a[1]) / 2, (a[2] + a[3]) / 2, (a[4] + a[5]) / 2]
        )
        center_h = np.append(local_center, 1.0)
        return (self.transform_scene_object @ center_h)[:3]

    @property
    def dimensions(self) -> np.ndarray:
        """Return the local-frame dimensions (width, height, depth)."""
        a = self.aabb_local
        return np.array([a[1] - a[0], a[3] - a[2], a[5] - a[4]])

    @property
    def half_extents(self) -> np.ndarray:
        """Return half of the dimensions (extents from center to each face)."""
        return self.dimensions / 2

    @property
    def height_off_ground(self) -> float:
        """Return the minimum Z coordinate in scene coordinates."""
        corners = self.get_corners()
        return float(corners[:, 2].min())

    @property
    def longest_dimension(self) -> float:
        """Return the longest dimension of the bounding box."""
        return float(np.max(self.dimensions))

    def get_corners(self) -> np.ndarray:
        """Return all 8 corners of the bounding box in scene coordinates.

        Returns:
            Array of shape (8, 3) containing the 8 corner points.
        """
        a = self.aabb_local
        local_min = np.array([a[0], a[2], a[4]])
        local_max = np.array([a[1], a[3], a[5]])

        corners_local = np.array(
            [
                [local_min[0], local_min[1], local_min[2]],
                [local_max[0], local_min[1], local_min[2]],
                [local_min[0], local_max[1], local_min[2]],
                [local_max[0], local_max[1], local_min[2]],
                [local_min[0], local_min[1], local_max[2]],
                [local_max[0], local_min[1], local_max[2]],
                [local_min[0], local_max[1], local_max[2]],
                [local_max[0], local_max[1], local_max[2]],
            ]
        )
        corners_h = np.hstack([corners_local, np.ones((8, 1))])
        return (self.transform_scene_object @ corners_h.T).T[:, :3]

    def closest_point_to(self, point: np.ndarray) -> np.ndarray:
        """Find the closest point on the OBB surface to a given point.

        Transforms the query point into the OBB local frame, clips to
        the local AABB, and transforms back to scene coordinates.

        Args:
            point: A 3D point in scene coordinates.

        Returns:
            The closest point on the OBB surface in scene coordinates.
        """
        T_object_scene = np.linalg.inv(self.transform_scene_object)
        point_h = np.append(point, 1.0)
        point_local = (T_object_scene @ point_h)[:3]

        a = self.aabb_local
        local_min = np.array([a[0], a[2], a[4]])
        local_max = np.array([a[1], a[3], a[5]])
        clipped = np.clip(point_local, local_min, local_max)

        return (self.transform_scene_object @ np.append(clipped, 1.0))[:3]

    def shortest_distance_to_bbox(self, other: BBox3D) -> float:
        """Calculate shortest distance between two oriented bounding boxes.

        Args:
            other: Another BBox3D instance.

        Returns:
            The shortest distance between the two bounding boxes.
        """
        closest_on_self = self.closest_point_to(other.center)
        closest_on_other = other.closest_point_to(self.center)

        # Check containment
        if np.allclose(closest_on_self, other.center):
            return 0.0
        if np.allclose(closest_on_other, self.center):
            return 0.0

        return float(np.linalg.norm(closest_on_self - closest_on_other))


@dataclass(frozen=True)
class ObjectInfo:
    """Information about an object in the ADT scene.

    Contains only static / image-based fields needed for local (per-frame)
    question generation.  Temporal fields (movement, visibility) live on
    VideoObjectInfo in generators.global_qs.common.
    """

    instance_id: int
    name: str
    category: str
    bbox: BBox3D | None = None

    @property
    def natural_name(self) -> str | None:
        """Get the natural language name for this object."""
        return get_natural_name(self.name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "instance_id": self.instance_id,
            "name": self.name,
            "category": self.category,
            "natural_name": self.natural_name,
        }
        if self.bbox is not None:
            result["bbox"] = {
                "center": self.bbox.center.tolist(),
                "dimensions": self.bbox.dimensions.tolist(),
            }
        return result


@dataclass(frozen=True)
class ObjectMovementEvent:
    """Represents a movement event for an object."""

    instance_id: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    start_position: np.ndarray
    end_position: np.ndarray
    movement_distance: float


@dataclass(frozen=True)
class VideoObjectInfo:
    """Object info with video-specific temporal tracking.

    Uses composition: wraps an ``ObjectInfo`` and adds temporal fields
    (movement events, global uniqueness, etc.).  Forwarding properties
    (``instance_id``, ``name``, ``category``, ``natural_name``, ``bbox``)
    delegate to the inner ``object_info`` so that code which accesses
    these fields on either type continues to work unchanged.
    """

    object_info: ObjectInfo

    movement_event: ObjectMovementEvent | None = None
    is_static: bool = True
    movement_distance: float = 0.0
    first_position: np.ndarray | None = None
    last_position: np.ndarray | None = None

    is_globally_unique: bool = False
    visible_timestamps: list[int] = field(default_factory=list)

    @property
    def instance_id(self) -> int:
        return self.object_info.instance_id

    @property
    def name(self) -> str:
        return self.object_info.name

    @property
    def category(self) -> str:
        return self.object_info.category

    @property
    def natural_name(self) -> str | None:
        return self.object_info.natural_name

    @property
    def bbox(self) -> Any:
        return self.object_info.bbox

    @property
    def was_moved(self) -> bool:
        return self.movement_event is not None

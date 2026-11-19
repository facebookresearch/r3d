# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Ground-truth oracle store for the ``gt_oracle`` tool backend.

``gt_oracle.db`` holds, for every ``(sequence_id, canonical object name)``, the
ADT ground-truth box the R3D-Bench GT answers were computed from, the object's
GT functional volume, and the GT Aria device pose at each query end timestamp.
Scene objects are matched by ``query_name``, so ``list_objects`` and the object
IDs stay those of the reconstructed scene the model normally sees.
"""

from __future__ import annotations

import functools
import itertools
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from r3d.data_gen.extractor.object_info import BBox3D
from r3d.pipeline import volume


def _min_area_yaw(points_xz: np.ndarray) -> float:
    """Yaw (about +Y) of the minimum-area rectangle enclosing 2D (x, z) points."""
    best_area, best_yaw = np.inf, 0.0
    for a, b in itertools.combinations(points_xz, 2):
        d = b - a
        if np.linalg.norm(d) < 1e-9:
            continue
        yaw = float(np.arctan2(d[1], d[0]))
        u = np.array([np.cos(yaw), np.sin(yaw)])
        v = np.array([-u[1], u[0]])
        area = float(np.ptp(points_xz @ u) * np.ptp(points_xz @ v))
        if area < best_area - 1e-12:
            best_area, best_yaw = area, yaw
    return best_yaw


def scene_convention_box(gt_box: BBox3D) -> BBox3D:
    """Gravity-aligned (Y-up) box enclosing a GT box, centered on its origin.

    This is the scene backend's box convention (scene.db OBBs are yaw-only,
    with a symmetric local AABB and ``position`` = box center). For an upright
    GT box (local Y vertical) it is the same box re-centered; for tilted ADT
    object frames its ``height`` is the true vertical extent.
    """
    corners = gt_box.get_corners()
    yaw = _min_area_yaw(corners[:, [0, 2]])
    ex = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.cross(ex, ey)
    rot = np.stack([ex, ey, ez], axis=1)
    local = corners @ rot
    lo, hi = local.min(axis=0), local.max(axis=0)
    half = (hi - lo) / 2
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = rot @ ((hi + lo) / 2)
    aabb = np.array([-half[0], half[0], -half[1], half[1], -half[2], half[2]])
    return BBox3D(aabb_local=aabb, transform_scene_object=transform)


@dataclass(frozen=True)
class OracleObject:
    """GT data for one ADT object instance.

    ``bbox`` is the raw ADT GT box (what the GT answers were computed from);
    tools read ``tool_box``, the same GT box in the scene tools' convention.
    """

    adt_instance_id: int
    adt_instance_name: str
    category: str
    bbox: BBox3D
    volume_m3: float | None

    @functools.cached_property
    def tool_box(self) -> BBox3D:
        return scene_convention_box(self.bbox)

    @property
    def position(self) -> np.ndarray:
        """GT box center, matching the scene backend's ``get_position``."""
        return self.tool_box.center.copy()


def _unblob(data: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float64).reshape(shape).copy()


class GTOracle:
    """In-memory, read-only view of ``gt_oracle.db`` (safe to share across threads)."""

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise RuntimeError(f"gt_oracle.db not found at {db_path}")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            self._objects: dict[tuple[str, str], OracleObject] = {
                (row[0], row[1]): OracleObject(
                    adt_instance_id=row[2],
                    adt_instance_name=row[3],
                    category=row[4],
                    bbox=BBox3D(
                        aabb_local=_unblob(row[5], (6,)),
                        transform_scene_object=_unblob(row[6], (4, 4)),
                    ),
                    volume_m3=row[7],
                )
                for row in conn.execute("SELECT * FROM oracle_objects")
            }
            self._poses: dict[tuple[str, int], np.ndarray] = {
                (row[0], row[1]): _unblob(row[2], (4, 4))
                for row in conn.execute("SELECT * FROM oracle_device_poses")
            }
        finally:
            conn.close()

    def get_object(self, sequence_id: str, query_name: str) -> OracleObject:
        key = (sequence_id, query_name.lower())
        if key not in self._objects:
            raise RuntimeError(
                f"gt_oracle has no object '{query_name}' in sequence {sequence_id}"
            )
        return self._objects[key]

    def get_device_position(self, sequence_id: str, timestamp_ns: int) -> np.ndarray:
        key = (sequence_id, timestamp_ns)
        if key not in self._poses:
            raise RuntimeError(
                f"gt_oracle has no device pose for {sequence_id} at {timestamp_ns}"
            )
        return self._poses[key][:3, 3].copy()


def format_oracle_volume(obj: OracleObject, object_id: int, volume_method: str) -> str:
    """Format a GT functional volume exactly like ``get_object_volume``."""
    if obj.volume_m3 is None:
        return volume.format_volume_error(volume_method, object_id)
    return volume.format_volume_result(obj.volume_m3, volume_method)

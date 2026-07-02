# Copyright (c) Meta Platforms, Inc. and affiliates.

"""SQLite WAL implementations of pipeline stores.

Each store wraps a separate .db file. WAL mode allows concurrent readers
with a single writer. All numpy arrays are stored as raw bytes via tobytes()
and reconstructed with np.frombuffer().
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from r3d.data_gen.utils.annotation_schema import (
    Annotation,
    BBox3D,
    DisambiguationContext,
    DisambiguationLayer,
    DisambiguationMethod,
    EvalMetric,
    EvalMode,
    IdentityLayer,
    QueryLayer,
    QuestionType,
    ReferencedObject,
    ReleaseType,
)
from r3d.pipeline.frame_data import CameraIntrinsics, DepthSource, FrameData
from r3d.pipeline.segmentation import FrameSegmentation, ObjectSegmentation
from r3d.pipeline.stores.base import (
    AnnotationStore,
    FrameStore,
    FrameVisibility,
    MeshStore,
    ObjectCoverage,
    ObjectMesh,
    ObjectPointsStore,
    ObjectReconstruction,
    SceneObject,
    SceneStore,
    SegmentationStore,
)
from r3d.utils.rle import decode_rle_to_mask


def _init_connection(db_path: Path, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=30.0, check_same_thread=False
        )
    else:
        conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _commit_with_retry(
    conn: sqlite3.Connection,
    max_retries: int = 10,
    base_delay: float = 0.05,
) -> None:
    for attempt in range(max_retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            raise


_UINT16_MAX_MM: int = 65535


def _depth_map_to_uint16_mm(depth_map: np.ndarray) -> np.ndarray:
    """Convert a float depth map (meters) to a uint16 millimeter array.

    Validates that the depth map is finite and within the representable
    uint16 millimeter range, failing loudly rather than silently overflowing
    or corrupting NaN/inf values.

    :param depth_map: Depth in meters, shape (H, W).
    :returns: Depth in millimeters as a uint16 array.
    :raises ValueError: if the depth map contains NaN/inf or values that fall
        outside the [0, 65.535] meter range representable as uint16 mm.
    """
    if not np.isfinite(depth_map).all():
        raise ValueError("depth_map contains NaN or inf values")
    depth_mm = depth_map * 1000.0
    if depth_mm.min() < 0 or depth_mm.max() > _UINT16_MAX_MM:
        raise ValueError(
            "depth_map values out of uint16 mm range "
            f"[0, {_UINT16_MAX_MM}], got [{depth_mm.min()}, {depth_mm.max()}]"
        )
    return np.rint(depth_mm).astype(np.uint16)


def _ndarray_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float64).tobytes()


def _blob_to_ndarray(blob: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64).reshape(shape).copy()


# ---------------------------------------------------------------------------
# FrameStore
# ---------------------------------------------------------------------------


class SQLiteFrameStore(FrameStore):
    """SQLite-backed frame store (frames.db)."""

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._db_dir = db_path.parent
        self._conn = _init_connection(db_path, read_only=read_only)
        if read_only:
            return
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                sequence_id TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                rgb_path TEXT NOT NULL,
                depth_path TEXT NOT NULL,
                fx REAL NOT NULL,
                fy REAL NOT NULL,
                cx REAL NOT NULL,
                cy REAL NOT NULL,
                img_width INTEGER NOT NULL,
                img_height INTEGER NOT NULL,
                T_scene_device BLOB NOT NULL,
                T_device_camera BLOB NOT NULL,
                gravity_world BLOB,
                depth_source TEXT NOT NULL,
                PRIMARY KEY (sequence_id, timestamp_ns)
            )
        """)
        self._conn.commit()

    def write_frame(
        self,
        sequence_id: str,
        timestamp_ns: int,
        rgb_path: str,
        depth_path: str,
        intrinsics: CameraIntrinsics,
        T_scene_device: np.ndarray,
        T_device_camera: np.ndarray,
        depth_source: DepthSource,
        gravity_world: np.ndarray | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO frames
            (sequence_id, timestamp_ns, rgb_path, depth_path, fx, fy, cx, cy,
             img_width, img_height, T_scene_device, T_device_camera, gravity_world,
             depth_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sequence_id,
                timestamp_ns,
                rgb_path,
                depth_path,
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                intrinsics.width,
                intrinsics.height,
                _ndarray_to_blob(T_scene_device),
                _ndarray_to_blob(T_device_camera),
                _ndarray_to_blob(gravity_world) if gravity_world is not None else None,
                depth_source.value,
            ),
        )
        _commit_with_retry(self._conn)

    def write_frame_data(self, sequence_id: str, frame: FrameData) -> None:
        """Persist an in-memory frame (RGB + depth arrays) to disk and metadata.

        Writes RGB as a BGR PNG and depth as a uint16 millimeter PNG under a
        ``frame_data/`` subdirectory (relative paths, mirroring ``load_frame``),
        then inserts the metadata row via :meth:`write_frame`.
        """
        prefix = f"{sequence_id}_{frame.timestamp_ns}"
        rgb_rel = f"frame_data/{prefix}_rgb.png"
        depth_rel = f"frame_data/{prefix}_depth.png"
        rgb_path = self._db_dir / rgb_rel
        depth_path = self._db_dir / depth_rel
        rgb_tmp = rgb_path.with_name(f".{rgb_path.name}.tmp.png")
        depth_tmp = depth_path.with_name(f".{depth_path.name}.tmp.png")

        depth_mm = _depth_map_to_uint16_mm(frame.depth_map)
        frame_dir = self._db_dir / "frame_data"
        frame_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = [rgb_tmp, depth_tmp]
        try:
            if not cv2.imwrite(
                str(rgb_tmp),
                cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            ):
                raise RuntimeError(f"Failed to write RGB image: {rgb_path}")
            if not cv2.imwrite(
                str(depth_tmp), depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 0]
            ):
                raise RuntimeError(f"Failed to write depth image: {depth_path}")
            rgb_tmp.replace(rgb_path)
            written[written.index(rgb_tmp)] = rgb_path
            depth_tmp.replace(depth_path)
            written[written.index(depth_tmp)] = depth_path
            self.write_frame(
                sequence_id=sequence_id,
                timestamp_ns=frame.timestamp_ns,
                rgb_path=rgb_rel,
                depth_path=depth_rel,
                intrinsics=frame.intrinsics,
                T_scene_device=frame.T_scene_device,
                T_device_camera=frame.T_device_camera,
                gravity_world=frame.gravity_world,
                depth_source=frame.depth_source,
            )
        except BaseException:
            self._conn.rollback()
            for path in written:
                path.unlink(missing_ok=True)
            raise

    def get_all_timestamps(self, sequence_id: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT timestamp_ns FROM frames WHERE sequence_id = ? ORDER BY timestamp_ns",
            (sequence_id,),
        ).fetchall()
        return [row["timestamp_ns"] for row in rows]

    def get_frame_count(self, sequence_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM frames WHERE sequence_id = ?",
            (sequence_id,),
        ).fetchone()
        assert row is not None
        return int(row["cnt"])

    def load_frame(self, sequence_id: str, timestamp_ns: int) -> FrameData:
        row = self._conn.execute(
            "SELECT * FROM frames WHERE sequence_id = ? AND timestamp_ns = ?",
            (sequence_id, timestamp_ns),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"No frame at timestamp {timestamp_ns} for sequence {sequence_id}"
            )

        intrinsics = CameraIntrinsics(
            fx=row["fx"],
            fy=row["fy"],
            cx=row["cx"],
            cy=row["cy"],
            width=row["img_width"],
            height=row["img_height"],
        )
        gravity = (
            _blob_to_ndarray(row["gravity_world"], (3,))
            if row["gravity_world"] is not None
            else None
        )

        rgb_path = str(self._db_dir / row["rgb_path"])
        depth_path = str(self._db_dir / row["depth_path"])

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise RuntimeError(f"Failed to load depth image: {depth_path}")
        depth = depth_mm.astype(np.float32) / 1000.0

        return FrameData(
            timestamp_ns=timestamp_ns,
            rgb=rgb,
            depth_map=depth,
            depth_source=DepthSource(row["depth_source"])
            if "depth_source" in row.keys()
            else DepthSource.GROUND_TRUTH,
            intrinsics=intrinsics,
            T_scene_device=_blob_to_ndarray(row["T_scene_device"], (4, 4)),
            T_device_camera=_blob_to_ndarray(row["T_device_camera"], (4, 4)),
            gravity_world=gravity,
        )

    def load_frame_pose(
        self, sequence_id: str, timestamp_ns: int
    ) -> tuple[CameraIntrinsics, np.ndarray, np.ndarray]:
        """Load only camera intrinsics and poses, skipping RGB/depth decode.

        Used by the multiview membership vote, which needs geometry only and
        would otherwise decode two PNGs per (object, frame) redundantly.
        """
        row = self._conn.execute(
            "SELECT fx, fy, cx, cy, img_width, img_height, T_scene_device, "
            "T_device_camera FROM frames WHERE sequence_id = ? AND timestamp_ns = ?",
            (sequence_id, timestamp_ns),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"No frame at timestamp {timestamp_ns} for sequence {sequence_id}"
            )
        intrinsics = CameraIntrinsics(
            fx=row["fx"],
            fy=row["fy"],
            cx=row["cx"],
            cy=row["cy"],
            width=row["img_width"],
            height=row["img_height"],
        )
        return (
            intrinsics,
            _blob_to_ndarray(row["T_scene_device"], (4, 4)),
            _blob_to_ndarray(row["T_device_camera"], (4, 4)),
        )

    def get_all_sequence_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT sequence_id FROM frames ORDER BY sequence_id"
        ).fetchall()
        return [row["sequence_id"] for row in rows]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# MeshStore
# ---------------------------------------------------------------------------


class SQLiteMeshStore(MeshStore):
    """SQLite-backed mesh store (mesh.db).

    Stores per-object single-view mesh reconstructions keyed by
    (sequence_id, object_name). mesh_path is stored relative to the
    db directory; get_mesh_abs_path reconstructs the absolute path so
    artifacts stay portable across machines.
    """

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._db_dir = db_path.parent
        self._conn = _init_connection(db_path, read_only=read_only)
        if read_only:
            return
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS meshes (
                sequence_id TEXT NOT NULL,
                object_name TEXT NOT NULL,
                annotation_id TEXT NOT NULL,
                adt_instance_name TEXT NOT NULL,
                mesh_path TEXT NOT NULL,
                source_timestamp_ns INTEGER NOT NULL,
                num_vertices INTEGER NOT NULL,
                num_faces INTEGER NOT NULL,
                metric_scale_x REAL NOT NULL,
                metric_scale_y REAL NOT NULL,
                metric_scale_z REAL NOT NULL,
                created_ns INTEGER NOT NULL,
                PRIMARY KEY (sequence_id, object_name)
            )
        """)
        self._conn.commit()

    def write_mesh(self, mesh: ObjectMesh) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO meshes
            (sequence_id, object_name, annotation_id, adt_instance_name,
             mesh_path, source_timestamp_ns, num_vertices, num_faces,
             metric_scale_x, metric_scale_y, metric_scale_z,
             created_ns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mesh.sequence_id,
                mesh.object_name,
                mesh.annotation_id,
                mesh.adt_instance_name,
                mesh.mesh_path,
                mesh.source_timestamp_ns,
                mesh.num_vertices,
                mesh.num_faces,
                mesh.metric_scale_x,
                mesh.metric_scale_y,
                mesh.metric_scale_z,
                mesh.created_ns,
            ),
        )
        _commit_with_retry(self._conn)

    def _row_to_mesh(self, row: sqlite3.Row) -> ObjectMesh:
        return ObjectMesh(
            sequence_id=row["sequence_id"],
            object_name=row["object_name"],
            annotation_id=row["annotation_id"],
            adt_instance_name=row["adt_instance_name"],
            mesh_path=row["mesh_path"],
            source_timestamp_ns=row["source_timestamp_ns"],
            num_vertices=row["num_vertices"],
            num_faces=row["num_faces"],
            metric_scale_x=row["metric_scale_x"],
            metric_scale_y=row["metric_scale_y"],
            metric_scale_z=row["metric_scale_z"],
            created_ns=row["created_ns"],
        )

    def get_mesh(self, sequence_id: str, object_name: str) -> ObjectMesh | None:
        row = self._conn.execute(
            "SELECT * FROM meshes WHERE sequence_id = ? AND object_name = ?",
            (sequence_id, object_name),
        ).fetchone()
        return self._row_to_mesh(row) if row is not None else None

    def get_mesh_abs_path(self, sequence_id: str, object_name: str) -> str | None:
        mesh = self.get_mesh(sequence_id, object_name)
        if mesh is None:
            return None
        return str(self._db_dir / mesh.mesh_path)

    def get_all_meshes(self) -> list[ObjectMesh]:
        rows = self._conn.execute(
            "SELECT * FROM meshes ORDER BY sequence_id, object_name"
        ).fetchall()
        return [self._row_to_mesh(row) for row in rows]

    def get_all_sequence_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT sequence_id FROM meshes ORDER BY sequence_id"
        ).fetchall()
        return [row["sequence_id"] for row in rows]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# SegmentationStore
# ---------------------------------------------------------------------------


class SQLiteSegmentationStore(SegmentationStore):
    """SQLite-backed segmentation store (segmentations.db)."""

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._conn = _init_connection(db_path, read_only=read_only)
        if read_only:
            return
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS query_status (
                sequence_id TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                query_name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                PRIMARY KEY (sequence_id, timestamp_ns, query_name)
            );
            CREATE TABLE IF NOT EXISTS segmentations (
                sequence_id TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                query_name TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                bbox_2d BLOB NOT NULL,
                mask_rle TEXT NOT NULL,
                score REAL NOT NULL,
                obj_ptr BLOB,
                min_depth_m REAL NOT NULL,
                PRIMARY KEY (sequence_id, timestamp_ns, query_name, object_id)
            );
        """)
        self._conn.commit()

    def register_query(
        self, sequence_id: str, query_name: str, timestamps: list[int]
    ) -> None:
        self._conn.executemany(
            """INSERT OR IGNORE INTO query_status
            (sequence_id, timestamp_ns, query_name, status)
            VALUES (?, ?, ?, 'pending')""",
            [(sequence_id, ts, query_name) for ts in timestamps],
        )
        self._conn.commit()

    def write_segmentation(
        self,
        sequence_id: str,
        timestamp_ns: int,
        obj_seg: ObjectSegmentation,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO segmentations
            (sequence_id, timestamp_ns, query_name, object_id,
             bbox_2d, mask_rle, score, obj_ptr, min_depth_m)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sequence_id,
                timestamp_ns,
                obj_seg.query_name,
                obj_seg.object_id,
                _ndarray_to_blob(obj_seg.bbox_2d),
                json.dumps(obj_seg.mask_rle),
                obj_seg.score,
                (
                    obj_seg.obj_ptr.astype(np.float32).tobytes()
                    if obj_seg.obj_ptr is not None
                    else None
                ),
                obj_seg.min_depth_m,
            ),
        )
        _commit_with_retry(self._conn)

    def mark_segmented(
        self, sequence_id: str, timestamp_ns: int, query_name: str
    ) -> None:
        self._conn.execute(
            """UPDATE query_status SET status = 'segmented'
            WHERE sequence_id = ? AND timestamp_ns = ? AND query_name = ?""",
            (sequence_id, timestamp_ns, query_name),
        )
        _commit_with_retry(self._conn)

    def get_pending_timestamps(self, sequence_id: str, query_name: str) -> list[int]:
        rows = self._conn.execute(
            """SELECT timestamp_ns FROM query_status
            WHERE sequence_id = ? AND query_name = ? AND status = 'pending'
            ORDER BY timestamp_ns""",
            (sequence_id, query_name),
        ).fetchall()
        return [row["timestamp_ns"] for row in rows]

    def get_segmented_timestamps(self, sequence_id: str, query_name: str) -> list[int]:
        rows = self._conn.execute(
            """SELECT timestamp_ns FROM query_status
            WHERE sequence_id = ? AND query_name = ? AND status = 'segmented'
            ORDER BY timestamp_ns""",
            (sequence_id, query_name),
        ).fetchall()
        return [row["timestamp_ns"] for row in rows]

    def get_segmentation(
        self,
        sequence_id: str,
        timestamp_ns: int,
        query_name: str | None = None,
    ) -> FrameSegmentation:
        if query_name is not None:
            rows = self._conn.execute(
                """SELECT * FROM segmentations
                WHERE sequence_id = ? AND timestamp_ns = ? AND query_name = ?""",
                (sequence_id, timestamp_ns, query_name),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM segmentations
                WHERE sequence_id = ? AND timestamp_ns = ?""",
                (sequence_id, timestamp_ns),
            ).fetchall()

        objects: dict[int, ObjectSegmentation] = {}
        for row in rows:
            rle = json.loads(row["mask_rle"])
            obj_ptr = (
                np.frombuffer(row["obj_ptr"], dtype=np.float32).copy()
                if row["obj_ptr"] is not None
                else None
            )
            bbox_2d = _blob_to_ndarray(row["bbox_2d"], (4,))
            mask = decode_rle_to_mask(rle)
            objects[row["object_id"]] = ObjectSegmentation(
                object_id=row["object_id"],
                query_name=row["query_name"],
                bbox_2d=bbox_2d,
                mask=mask,
                mask_rle=rle,
                score=row["score"],
                obj_ptr=obj_ptr,
                min_depth_m=row["min_depth_m"],
            )

        return FrameSegmentation(timestamp_ns=timestamp_ns, objects=objects)

    def get_all_query_names(self, sequence_id: str) -> list[str]:
        rows = self._conn.execute(
            """SELECT DISTINCT query_name FROM query_status
            WHERE sequence_id = ? ORDER BY query_name""",
            (sequence_id,),
        ).fetchall()
        return [row["query_name"] for row in rows]

    def get_query_names_for_object(self, sequence_id: str, object_id: int) -> list[str]:
        """Query names under which a given object_id has any segmentation."""
        rows = self._conn.execute(
            """SELECT DISTINCT query_name FROM segmentations
            WHERE sequence_id = ? AND object_id = ? ORDER BY query_name""",
            (sequence_id, object_id),
        ).fetchall()
        return [row["query_name"] for row in rows]

    def count_segmentations(self, sequence_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM segmentations WHERE sequence_id = ?",
            (sequence_id,),
        ).fetchone()
        return int(row[0])

    def get_all_segmented_timestamps(self, sequence_id: str) -> list[int]:
        rows = self._conn.execute(
            """SELECT DISTINCT timestamp_ns FROM segmentations
            WHERE sequence_id = ? ORDER BY timestamp_ns""",
            (sequence_id,),
        ).fetchall()
        return [row["timestamp_ns"] for row in rows]

    def get_all_sequence_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT sequence_id FROM query_status ORDER BY sequence_id"
        ).fetchall()
        return [row["sequence_id"] for row in rows]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# SceneStore
# ---------------------------------------------------------------------------


class SQLiteSceneStore(SceneStore):
    """SQLite-backed scene store (scene.db)."""

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._conn = _init_connection(db_path, read_only=read_only)
        if read_only:
            return
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS scene_objects (
                sequence_id TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                query_name TEXT NOT NULL,
                first_seen_ns INTEGER NOT NULL,
                last_seen_ns INTEGER NOT NULL,
                PRIMARY KEY (sequence_id, object_id)
            );
            CREATE TABLE IF NOT EXISTS object_reconstructions (
                reconstruction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                time_range_start_ns INTEGER NOT NULL,
                time_range_end_ns INTEGER NOT NULL,
                obb_aabb BLOB NOT NULL,
                obb_transform BLOB NOT NULL,
                position BLOB NOT NULL,
                initial_obb_aabb BLOB NOT NULL,
                initial_obb_transform BLOB NOT NULL,
                initial_position BLOB NOT NULL,
                num_gaussians INTEGER,
                psnr REAL,
                ssim REAL,
                lpips REAL,
                created_ns INTEGER NOT NULL
            );
        """)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS frame_visibility (
                sequence_id TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                object_id INTEGER NOT NULL,
                bbox_2d BLOB NOT NULL,
                mask_rle TEXT,
                sam3_score REAL,
                PRIMARY KEY (sequence_id, timestamp_ns, object_id)
            );
            CREATE TABLE IF NOT EXISTS object_coverage (
                sequence_id TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                num_views INTEGER NOT NULL,
                angular_span_deg REAL NOT NULL,
                num_distinct_viewpoints INTEGER NOT NULL,
                mean_visibility_ratio REAL NOT NULL,
                PRIMARY KEY (sequence_id, object_id)
            );
        """)
        self._conn.commit()

    def delete_sequence(self, sequence_id: str) -> None:
        self._conn.execute(
            "DELETE FROM frame_visibility WHERE sequence_id = ?", (sequence_id,)
        )
        self._conn.execute(
            "DELETE FROM object_reconstructions WHERE sequence_id = ?",
            (sequence_id,),
        )
        self._conn.execute(
            "DELETE FROM scene_objects WHERE sequence_id = ?", (sequence_id,)
        )
        self._conn.execute(
            "DELETE FROM object_coverage WHERE sequence_id = ?", (sequence_id,)
        )
        self._conn.commit()

    def write_scene_object(self, obj: SceneObject) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO scene_objects
            (sequence_id, object_id, query_name, first_seen_ns, last_seen_ns)
            VALUES (?, ?, ?, ?, ?)""",
            (
                obj.sequence_id,
                obj.object_id,
                obj.query_name,
                obj.first_seen_ns,
                obj.last_seen_ns,
            ),
        )
        self._conn.commit()

    def write_reconstruction(self, recon: ObjectReconstruction) -> None:
        self._conn.execute(
            """INSERT INTO object_reconstructions
            (sequence_id, object_id, time_range_start_ns,
             time_range_end_ns, obb_aabb, obb_transform, position,
             initial_obb_aabb, initial_obb_transform, initial_position,
             num_gaussians,
             psnr, ssim, lpips, created_ns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                recon.sequence_id,
                recon.object_id,
                recon.time_range_start_ns,
                recon.time_range_end_ns,
                _ndarray_to_blob(recon.obb_aabb),
                _ndarray_to_blob(recon.obb_transform),
                _ndarray_to_blob(recon.position),
                _ndarray_to_blob(recon.initial_obb_aabb),
                _ndarray_to_blob(recon.initial_obb_transform),
                _ndarray_to_blob(recon.initial_position),
                recon.num_gaussians,
                recon.psnr,
                recon.ssim,
                recon.lpips,
                recon.created_ns,
            ),
        )
        self._conn.commit()

    def write_frame_visibility(self, vis: FrameVisibility) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO frame_visibility
            (sequence_id, timestamp_ns, object_id,
             bbox_2d, mask_rle, sam3_score)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                vis.sequence_id,
                vis.timestamp_ns,
                vis.object_id,
                _ndarray_to_blob(vis.bbox_2d),
                json.dumps(vis.mask_rle) if vis.mask_rle is not None else None,
                vis.sam3_score,
            ),
        )
        self._conn.commit()

    def get_all_scene_objects(self, sequence_id: str) -> list[SceneObject]:
        rows = self._conn.execute(
            "SELECT * FROM scene_objects WHERE sequence_id = ? ORDER BY object_id",
            (sequence_id,),
        ).fetchall()
        return [
            SceneObject(
                sequence_id=row["sequence_id"],
                object_id=row["object_id"],
                query_name=row["query_name"],
                first_seen_ns=row["first_seen_ns"],
                last_seen_ns=row["last_seen_ns"],
            )
            for row in rows
        ]

    def get_reconstructions(
        self, sequence_id: str, object_id: int
    ) -> list[ObjectReconstruction]:
        rows = self._conn.execute(
            """SELECT * FROM object_reconstructions
            WHERE sequence_id = ? AND object_id = ? ORDER BY created_ns""",
            (sequence_id, object_id),
        ).fetchall()
        return [self._row_to_recon(row) for row in rows]

    def get_all_reconstructions(self) -> list[ObjectReconstruction]:
        rows = self._conn.execute(
            "SELECT * FROM object_reconstructions ORDER BY reconstruction_id"
        ).fetchall()
        return [self._row_to_recon(row) for row in rows]

    def get_frame_visibility(
        self, sequence_id: str, timestamp_ns: int
    ) -> list[FrameVisibility]:
        rows = self._conn.execute(
            """SELECT * FROM frame_visibility
            WHERE sequence_id = ? AND timestamp_ns = ?""",
            (sequence_id, timestamp_ns),
        ).fetchall()
        return [
            FrameVisibility(
                sequence_id=row["sequence_id"],
                timestamp_ns=row["timestamp_ns"],
                object_id=row["object_id"],
                bbox_2d=_blob_to_ndarray(row["bbox_2d"], (4,)),
                mask_rle=(
                    json.loads(row["mask_rle"]) if row["mask_rle"] is not None else None
                ),
                sam3_score=row["sam3_score"],
            )
            for row in rows
        ]

    def is_non_empty(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM scene_objects").fetchone()
        assert row is not None
        return int(row["cnt"]) > 0

    def get_all_sequence_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT sequence_id FROM scene_objects ORDER BY sequence_id"
        ).fetchall()
        return [row["sequence_id"] for row in rows]

    @staticmethod
    def _row_to_recon(row: sqlite3.Row) -> ObjectReconstruction:
        return ObjectReconstruction(
            reconstruction_id=row["reconstruction_id"],
            sequence_id=row["sequence_id"],
            object_id=row["object_id"],
            time_range_start_ns=row["time_range_start_ns"],
            time_range_end_ns=row["time_range_end_ns"],
            obb_aabb=_blob_to_ndarray(row["obb_aabb"], (6,)),
            obb_transform=_blob_to_ndarray(row["obb_transform"], (4, 4)),
            position=_blob_to_ndarray(row["position"], (3,)),
            initial_obb_aabb=_blob_to_ndarray(row["initial_obb_aabb"], (6,)),
            initial_obb_transform=_blob_to_ndarray(
                row["initial_obb_transform"], (4, 4)
            ),
            initial_position=_blob_to_ndarray(row["initial_position"], (3,)),
            num_gaussians=row["num_gaussians"],
            psnr=row["psnr"],
            ssim=row["ssim"],
            lpips=row["lpips"],
            created_ns=row["created_ns"],
        )

    def write_object_coverage(self, coverage: ObjectCoverage) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO object_coverage
            (sequence_id, object_id, num_views, angular_span_deg,
             num_distinct_viewpoints, mean_visibility_ratio)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                coverage.sequence_id,
                coverage.object_id,
                coverage.num_views,
                coverage.angular_span_deg,
                coverage.num_distinct_viewpoints,
                coverage.mean_visibility_ratio,
            ),
        )
        self._conn.commit()

    def get_object_coverage(
        self, sequence_id: str, object_id: int
    ) -> ObjectCoverage | None:
        row = self._conn.execute(
            "SELECT * FROM object_coverage WHERE sequence_id = ? AND object_id = ?",
            (sequence_id, object_id),
        ).fetchone()
        if row is None:
            return None
        return ObjectCoverage(
            sequence_id=row["sequence_id"],
            object_id=row["object_id"],
            num_views=row["num_views"],
            angular_span_deg=row["angular_span_deg"],
            num_distinct_viewpoints=row["num_distinct_viewpoints"],
            mean_visibility_ratio=row["mean_visibility_ratio"],
        )

    def get_all_object_coverages(self, sequence_id: str) -> list[ObjectCoverage]:
        rows = self._conn.execute(
            "SELECT * FROM object_coverage WHERE sequence_id = ? ORDER BY object_id",
            (sequence_id,),
        ).fetchall()
        return [
            ObjectCoverage(
                sequence_id=row["sequence_id"],
                object_id=row["object_id"],
                num_views=row["num_views"],
                angular_span_deg=row["angular_span_deg"],
                num_distinct_viewpoints=row["num_distinct_viewpoints"],
                mean_visibility_ratio=row["mean_visibility_ratio"],
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# AnnotationStore
# ---------------------------------------------------------------------------


class SQLiteAnnotationStore(AnnotationStore):
    """SQLite-backed annotation store (annotations.db)."""

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._conn = _init_connection(db_path, read_only=read_only)
        if read_only:
            return
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS annotations (
                annotation_id TEXT PRIMARY KEY,
                sequence_id TEXT NOT NULL,
                release_type TEXT NOT NULL,
                question_type TEXT NOT NULL,
                question_text TEXT NOT NULL,
                gt_answer TEXT NOT NULL,
                gt_answer_type TEXT NOT NULL,
                eval_mode TEXT NOT NULL,
                eval_metric TEXT NOT NULL,
                timestamp_ns_start INTEGER NOT NULL,
                timestamp_ns_end INTEGER NOT NULL,
                disambiguation_method TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS referenced_objects (
                annotation_id TEXT NOT NULL REFERENCES annotations(annotation_id),
                object_position INTEGER NOT NULL,
                object_id TEXT NOT NULL,
                adt_instance_id INTEGER NOT NULL,
                adt_instance_name TEXT NOT NULL,
                prototype_name TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                adt_mesh_path TEXT,
                is_dynamic INTEGER NOT NULL,
                reference_frame_idx INTEGER,
                reference_timestamp_ns INTEGER,
                PRIMARY KEY (annotation_id, object_position)
            );

            CREATE TABLE IF NOT EXISTS numeric_answer_details (
                annotation_id TEXT PRIMARY KEY REFERENCES annotations(annotation_id),
                gt_answer_unit TEXT NOT NULL,
                gt_computation_method TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS disambiguation_context (
                annotation_id TEXT PRIMARY KEY REFERENCES annotations(annotation_id),
                spatial_description TEXT,
                temporal_description TEXT
            );

            CREATE TABLE IF NOT EXISTS gt_object_bboxes (
                annotation_id TEXT NOT NULL,
                object_position INTEGER NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                obb_aabb BLOB NOT NULL,
                obb_transform BLOB NOT NULL,
                PRIMARY KEY (annotation_id, object_position, timestamp_ns)
            );

            CREATE INDEX IF NOT EXISTS idx_ann_sequence
                ON annotations(sequence_id);
            CREATE INDEX IF NOT EXISTS idx_ann_question_type
                ON annotations(question_type);
            CREATE INDEX IF NOT EXISTS idx_refobj_annotation
                ON referenced_objects(annotation_id);
            CREATE INDEX IF NOT EXISTS idx_refobj_canonical
                ON referenced_objects(canonical_name);
        """)
        self._conn.commit()

    # -- writes ---------------------------------------------------------------

    def _write_annotation_core(self, annotation: Annotation) -> None:
        il = annotation.identity_layer
        ql = annotation.query_layer
        dl = annotation.disambiguation_layer
        self._conn.execute(
            """INSERT OR REPLACE INTO annotations
            (annotation_id, sequence_id, release_type, question_type,
             question_text, gt_answer, gt_answer_type, eval_mode,
             eval_metric, timestamp_ns_start, timestamp_ns_end,
             disambiguation_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                annotation.annotation_id,
                il.sequence_id,
                il.release_type.value,
                ql.question_type.value,
                ql.question_text,
                ql.gt_answer,
                ql.gt_answer_type,
                ql.eval_mode.value,
                ql.eval_metric.value,
                ql.query_timestamp_ns_start,
                ql.query_timestamp_ns_end,
                dl.method.value,
            ),
        )

    def _write_referenced_objects(
        self,
        annotation_id: str,
        objects: list[ReferencedObject],
    ) -> None:
        for pos, obj in enumerate(objects):
            self._conn.execute(
                """INSERT OR REPLACE INTO referenced_objects
                (annotation_id, object_position, object_id,
                 adt_instance_id, adt_instance_name, prototype_name,
                 canonical_name, adt_mesh_path, is_dynamic,
                 reference_frame_idx, reference_timestamp_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    annotation_id,
                    pos,
                    obj.object_id,
                    obj.adt_instance_id,
                    obj.adt_instance_name,
                    obj.prototype_name,
                    obj.canonical_name,
                    obj.adt_mesh_path,
                    int(obj.is_dynamic),
                    obj.reference_frame_idx,
                    obj.reference_timestamp_ns,
                ),
            )

    def _write_numeric_details(
        self, annotation_id: str, query_layer: QueryLayer
    ) -> None:
        if (
            query_layer.gt_answer_unit is not None
            and query_layer.gt_computation_method is not None
        ):
            self._conn.execute(
                """INSERT OR REPLACE INTO numeric_answer_details
                (annotation_id, gt_answer_unit, gt_computation_method)
                VALUES (?, ?, ?)""",
                (
                    annotation_id,
                    query_layer.gt_answer_unit,
                    query_layer.gt_computation_method,
                ),
            )

    def _write_disambiguation(
        self,
        annotation_id: str,
        disambiguation_layer: DisambiguationLayer,
    ) -> None:
        if disambiguation_layer.method != DisambiguationMethod.GLOBAL:
            ctx = disambiguation_layer.disambiguation_context
            self._conn.execute(
                """INSERT OR REPLACE INTO disambiguation_context
                (annotation_id, spatial_description, temporal_description)
                VALUES (?, ?, ?)""",
                (
                    annotation_id,
                    ctx.spatial_description,
                    ctx.temporal_description,
                ),
            )

    def write_annotation(self, annotation: Annotation) -> None:
        aid = annotation.annotation_id
        with self._conn:
            self._write_annotation_core(annotation)
            self._write_referenced_objects(
                aid, annotation.identity_layer.referenced_objects
            )
            self._write_numeric_details(aid, annotation.query_layer)
            self._write_disambiguation(aid, annotation.disambiguation_layer)

    def get_all_annotations(self) -> list[Annotation]:
        rows = self._conn.execute(
            "SELECT * FROM annotations ORDER BY annotation_id"
        ).fetchall()
        return [self._row_to_annotation(row) for row in rows]

    def get_annotations_by_sequence(self, sequence_id: str) -> list[Annotation]:
        rows = self._conn.execute(
            "SELECT * FROM annotations WHERE sequence_id = ? ORDER BY annotation_id",
            (sequence_id,),
        ).fetchall()
        return [self._row_to_annotation(row) for row in rows]

    def get_all_sequence_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT sequence_id FROM annotations ORDER BY sequence_id"
        ).fetchall()
        return [row["sequence_id"] for row in rows]

    def get_object_sequence_pairs(self) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            """SELECT DISTINCT canonical_name, sequence_id
            FROM referenced_objects
            JOIN annotations USING(annotation_id)
            ORDER BY canonical_name, sequence_id"""
        ).fetchall()
        return [(row["canonical_name"], row["sequence_id"]) for row in rows]

    # -- helpers --------------------------------------------------------------

    def _load_referenced_objects(self, annotation_id: str) -> list[ReferencedObject]:
        obj_rows = self._conn.execute(
            """SELECT * FROM referenced_objects
            WHERE annotation_id = ? ORDER BY object_position""",
            (annotation_id,),
        ).fetchall()
        return [
            ReferencedObject(
                object_id=r["object_id"],
                adt_instance_id=r["adt_instance_id"],
                adt_instance_name=r["adt_instance_name"],
                prototype_name=r["prototype_name"],
                canonical_name=r["canonical_name"],
                adt_mesh_path=r["adt_mesh_path"],
                is_dynamic=bool(r["is_dynamic"]),
                reference_frame_idx=r["reference_frame_idx"],
                reference_timestamp_ns=r["reference_timestamp_ns"],
            )
            for r in obj_rows
        ]

    def _load_query_layer(self, row: sqlite3.Row, annotation_id: str) -> QueryLayer:
        nad_row = self._conn.execute(
            "SELECT * FROM numeric_answer_details WHERE annotation_id = ?",
            (annotation_id,),
        ).fetchone()
        return QueryLayer(
            question_type=QuestionType(row["question_type"]),
            question_text=row["question_text"],
            gt_answer=row["gt_answer"],
            gt_answer_type=row["gt_answer_type"],
            gt_answer_unit=nad_row["gt_answer_unit"] if nad_row else None,
            gt_computation_method=(
                nad_row["gt_computation_method"] if nad_row else None
            ),
            eval_mode=EvalMode(row["eval_mode"]),
            eval_metric=EvalMetric(row["eval_metric"]),
            query_timestamp_ns_start=row["timestamp_ns_start"],
            query_timestamp_ns_end=row["timestamp_ns_end"],
        )

    def _load_disambiguation_layer(
        self, row: sqlite3.Row, annotation_id: str
    ) -> DisambiguationLayer:
        dc_row = self._conn.execute(
            "SELECT * FROM disambiguation_context WHERE annotation_id = ?",
            (annotation_id,),
        ).fetchone()
        disambiguation_context = DisambiguationContext(
            spatial_description=(dc_row["spatial_description"] if dc_row else None),
            temporal_description=(dc_row["temporal_description"] if dc_row else None),
        )
        return DisambiguationLayer(
            method=DisambiguationMethod(row["disambiguation_method"]),
            disambiguation_context=disambiguation_context,
        )

    def _row_to_annotation(self, row: sqlite3.Row) -> Annotation:
        aid = row["annotation_id"]
        identity_layer = IdentityLayer(
            sequence_id=row["sequence_id"],
            release_type=ReleaseType(row["release_type"]),
            referenced_objects=self._load_referenced_objects(aid),
        )
        return Annotation(
            annotation_id=aid,
            identity_layer=identity_layer,
            disambiguation_layer=self._load_disambiguation_layer(row, aid),
            query_layer=self._load_query_layer(row, aid),
        )

    def write_gt_bbox(
        self,
        annotation_id: str,
        object_position: int,
        timestamp_ns: int,
        obb_aabb: np.ndarray,
        obb_transform: np.ndarray,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO gt_object_bboxes
            (annotation_id, object_position, timestamp_ns,
             obb_aabb, obb_transform)
            VALUES (?, ?, ?, ?, ?)""",
            (
                annotation_id,
                object_position,
                timestamp_ns,
                _ndarray_to_blob(obb_aabb),
                _ndarray_to_blob(obb_transform),
            ),
        )

    def flush_gt_bboxes(self) -> None:
        _commit_with_retry(self._conn)

    def get_gt_bboxes(
        self,
        annotation_id: str,
        object_position: int,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT * FROM gt_object_bboxes
            WHERE annotation_id = ? AND object_position = ?
            ORDER BY timestamp_ns""",
            (annotation_id, object_position),
        ).fetchall()
        if not rows:
            raise RuntimeError(
                f"No GT bboxes for annotation {annotation_id}, "
                f"object_position {object_position}. "
                f"Regenerate annotations with GT bbox population."
            )
        return [
            {
                "timestamp_ns": row["timestamp_ns"],
                "obb_aabb": _blob_to_ndarray(row["obb_aabb"], (6,)),
                "obb_transform": _blob_to_ndarray(row["obb_transform"], (4, 4)),
            }
            for row in rows
        ]

    def get_nearest_gt_bbox(
        self,
        annotation_id: str,
        object_position: int,
        timestamp_ns: int,
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            """SELECT obb_aabb, obb_transform FROM gt_object_bboxes
            WHERE annotation_id = ? AND object_position = ?
            ORDER BY ABS(timestamp_ns - ?) LIMIT 1""",
            (annotation_id, object_position, timestamp_ns),
        ).fetchone()
        if row is None:
            return None
        return {
            "obb_aabb": _blob_to_ndarray(row["obb_aabb"], (6,)),
            "obb_transform": _blob_to_ndarray(row["obb_transform"], (4, 4)),
        }

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# ObjectPointsStore
# ---------------------------------------------------------------------------


class SQLiteObjectPointsStore(ObjectPointsStore):
    """SQLite-backed store for per-object 3D point sets (object_points.db)."""

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self._conn = _init_connection(db_path, read_only=read_only)
        if read_only:
            return
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS object_points (
                sequence_id TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                points_blob BLOB NOT NULL,
                num_points INTEGER NOT NULL,
                PRIMARY KEY (sequence_id, object_id)
            )
        """)
        self._conn.commit()

    def write_points(
        self, sequence_id: str, object_id: int, points: np.ndarray
    ) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got shape {points.shape}")
        self._conn.execute(
            """INSERT OR REPLACE INTO object_points
            (sequence_id, object_id, points_blob, num_points)
            VALUES (?, ?, ?, ?)""",
            (
                sequence_id,
                object_id,
                points.astype(np.float32).tobytes(),
                len(points),
            ),
        )
        _commit_with_retry(self._conn)

    def get_points(self, sequence_id: str, object_id: int) -> np.ndarray | None:
        row = self._conn.execute(
            "SELECT points_blob, num_points FROM object_points WHERE sequence_id = ? AND object_id = ?",
            (sequence_id, object_id),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["points_blob"], dtype=np.float32).reshape(-1, 3).copy()

    def close(self) -> None:
        self._conn.close()

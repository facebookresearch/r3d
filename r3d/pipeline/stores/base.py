# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Abstract base classes for pipeline storage.

Each store maps to a separate .db file with a single writer process.
Implementations: SQLite (local dev).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.frame_data import CameraIntrinsics, DepthSource, FrameData
from r3d.pipeline.segmentation import FrameSegmentation, ObjectSegmentation


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SceneObject(BaseModel):
    """An object known in the scene."""

    model_config = ConfigDict(frozen=True)

    sequence_id: str
    object_id: int
    query_name: str
    first_seen_ns: int
    last_seen_ns: int


class ObjectReconstruction(BaseModel):
    """A 3D reconstruction of an object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reconstruction_id: int
    sequence_id: str
    object_id: int
    time_range_start_ns: int
    time_range_end_ns: int
    obb_aabb: np.ndarray
    obb_transform: np.ndarray
    position: np.ndarray
    initial_obb_aabb: np.ndarray
    initial_obb_transform: np.ndarray
    initial_position: np.ndarray
    num_gaussians: int | None = None
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    created_ns: int = 0


class FrameVisibility(BaseModel):
    """Per-frame per-object 2D visibility data from SAM3."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    sequence_id: str
    timestamp_ns: int
    object_id: int
    bbox_2d: np.ndarray
    mask_rle: dict[str, object] | None = None
    sam3_score: float | None = None


class ObjectCoverage(BaseModel):
    """Per-object camera coverage metrics."""

    model_config = ConfigDict(frozen=True)

    sequence_id: str
    object_id: int
    num_views: int
    angular_span_deg: float
    num_distinct_viewpoints: int
    mean_visibility_ratio: float


class ObjectMesh(BaseModel):
    """A single-view 3D mesh reconstruction of an object (e.g. from SAM3D).

    Keyed by (sequence_id, object_name) where object_name is the
    annotation's canonical_name, so it can be looked up from the eval
    scene by the object's query_name. mesh_path is stored relative to
    the mesh.db directory for portability (see SQLiteMeshStore).
    """

    model_config = ConfigDict(frozen=True)

    sequence_id: str
    object_name: str
    annotation_id: str
    adt_instance_name: str
    mesh_path: str
    source_timestamp_ns: int
    num_vertices: int
    num_faces: int
    metric_scale_x: float
    metric_scale_y: float
    metric_scale_z: float
    created_ns: int


class ObjectPoints(BaseModel):
    """A set of 3D points for an object (e.g. from depth back-projection)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    sequence_id: str
    object_id: int
    points: np.ndarray
    num_points: int


# ---------------------------------------------------------------------------
# Store ABCs
# ---------------------------------------------------------------------------


class FrameStore(ABC):
    """Stores frame metadata and paths to RGB/depth files.

    Writer: Frame capture process.
    Readers: SAM3, Track builder.
    """

    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
    def write_frame_data(self, sequence_id: str, frame: FrameData) -> None:
        """Persist an in-memory frame (RGB + depth arrays) and its metadata."""
        ...

    @abstractmethod
    def get_all_timestamps(self, sequence_id: str) -> list[int]: ...

    @abstractmethod
    def get_frame_count(self, sequence_id: str) -> int: ...

    @abstractmethod
    def load_frame(self, sequence_id: str, timestamp_ns: int) -> FrameData: ...

    @abstractmethod
    def get_all_sequence_ids(self) -> list[str]: ...


class SegmentationStore(ABC):
    """Stores per-(frame, query, object) segmentation results.

    Writer: SAM3 process.
    Readers: Orchestrator, Scene builder.
    """

    @abstractmethod
    def register_query(
        self, sequence_id: str, query_name: str, timestamps: list[int]
    ) -> None:
        """Register a query for the given timestamps (status='pending')."""
        ...

    @abstractmethod
    def write_segmentation(
        self,
        sequence_id: str,
        timestamp_ns: int,
        obj_seg: ObjectSegmentation,
    ) -> None: ...

    @abstractmethod
    def mark_segmented(
        self, sequence_id: str, timestamp_ns: int, query_name: str
    ) -> None: ...

    @abstractmethod
    def get_pending_timestamps(
        self, sequence_id: str, query_name: str
    ) -> list[int]: ...

    @abstractmethod
    def get_segmentation(
        self,
        sequence_id: str,
        timestamp_ns: int,
        query_name: str | None = None,
    ) -> FrameSegmentation: ...

    @abstractmethod
    def get_all_query_names(self, sequence_id: str) -> list[str]: ...

    @abstractmethod
    def get_segmented_timestamps(
        self, sequence_id: str, query_name: str
    ) -> list[int]: ...

    @abstractmethod
    def get_all_sequence_ids(self) -> list[str]: ...


class SceneStore(ABC):
    """Stores the compiled scene representation.

    Writer: Scene builder.
    Reader: LLM query process.
    """

    @abstractmethod
    def delete_sequence(self, sequence_id: str) -> None:
        """Delete all data for a sequence (idempotent rebuild)."""
        ...

    @abstractmethod
    def write_scene_object(self, obj: SceneObject) -> None: ...

    @abstractmethod
    def write_reconstruction(self, recon: ObjectReconstruction) -> None: ...

    @abstractmethod
    def write_frame_visibility(self, vis: FrameVisibility) -> None: ...

    @abstractmethod
    def get_all_scene_objects(self, sequence_id: str) -> list[SceneObject]: ...

    @abstractmethod
    def get_reconstructions(
        self, sequence_id: str, object_id: int
    ) -> list[ObjectReconstruction]: ...

    @abstractmethod
    def get_all_reconstructions(self) -> list[ObjectReconstruction]: ...

    @abstractmethod
    def get_frame_visibility(
        self, sequence_id: str, timestamp_ns: int
    ) -> list[FrameVisibility]: ...

    @abstractmethod
    def write_object_coverage(self, coverage: ObjectCoverage) -> None: ...

    @abstractmethod
    def get_object_coverage(
        self, sequence_id: str, object_id: int
    ) -> ObjectCoverage | None: ...

    @abstractmethod
    def get_all_object_coverages(self, sequence_id: str) -> list[ObjectCoverage]: ...

    @abstractmethod
    def is_non_empty(self) -> bool:
        """Return True if the scene store already has data."""
        ...

    @abstractmethod
    def get_all_sequence_ids(self) -> list[str]: ...


class AnnotationStore(ABC):
    """Stores R3D annotations (annotations.db).

    Writer: generate_qa.py
    Readers: eval.py, filter_annotations.py, run_segmentation.py
    """

    @abstractmethod
    def write_annotation(self, annotation: Annotation) -> None: ...

    @abstractmethod
    def get_all_annotations(self) -> list[Annotation]: ...

    @abstractmethod
    def get_annotations_by_sequence(self, sequence_id: str) -> list[Annotation]: ...

    @abstractmethod
    def get_all_sequence_ids(self) -> list[str]: ...

    @abstractmethod
    def get_object_sequence_pairs(self) -> list[tuple[str, str]]: ...

    @abstractmethod
    def write_gt_bbox(
        self,
        annotation_id: str,
        object_position: int,
        timestamp_ns: int,
        obb_aabb: np.ndarray,
        obb_transform: np.ndarray,
    ) -> None: ...

    @abstractmethod
    def get_gt_bboxes(
        self,
        annotation_id: str,
        object_position: int,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    def flush_gt_bboxes(self) -> None: ...

    @abstractmethod
    def get_nearest_gt_bbox(
        self,
        annotation_id: str,
        object_position: int,
        timestamp_ns: int,
    ) -> dict[str, Any] | None:
        """Get the GT bbox closest to the given timestamp.

        Returns a dict with keys 'obb_aabb' (ndarray shape (6,)) and
        'obb_transform' (ndarray shape (4,4)), or None if no GT bbox
        exists for the given annotation and object_position.
        """
        ...

    @abstractmethod
    def close(self) -> None: ...


class MeshStore(ABC):
    """Stores per-object single-view mesh reconstructions (mesh.db).

    Writer: run_sam3d.py
    Reader: generate_responses.py / tool_use.py (mesh volume tool).
    """

    @abstractmethod
    def write_mesh(self, mesh: ObjectMesh) -> None: ...

    @abstractmethod
    def get_mesh(self, sequence_id: str, object_name: str) -> ObjectMesh | None: ...

    @abstractmethod
    def get_mesh_abs_path(self, sequence_id: str, object_name: str) -> str | None: ...

    @abstractmethod
    def get_all_meshes(self) -> list[ObjectMesh]: ...

    @abstractmethod
    def get_all_sequence_ids(self) -> list[str]: ...

    @abstractmethod
    def close(self) -> None: ...


class ObjectPointsStore(ABC):
    """Stores per-object 3D point sets (e.g. from depth back-projection).

    Writer: depth back-projection / track builder.
    Reader: SceneState (volume estimation).
    """

    @abstractmethod
    def write_points(
        self, sequence_id: str, object_id: int, points: np.ndarray
    ) -> None: ...

    @abstractmethod
    def get_points(self, sequence_id: str, object_id: int) -> np.ndarray | None: ...

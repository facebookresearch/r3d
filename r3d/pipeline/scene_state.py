# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Read-only query interface for the compiled scene (scene.db).

SceneState is the consumer-facing API used by the LLM query process
and the eval pipeline. It reads from scene.db and provides typed
access to objects, reconstructions, and visibility.
"""

from __future__ import annotations

import logging

import numpy as np
from r3d.pipeline import volume
from r3d.pipeline.eval.config import VOLUME_METHODS
from r3d.pipeline.stores.base import (
    CubifyObjectResult,
    CubifyStore,
    FrameVisibility,
    MeshStore,
    ObjectReconstruction,
    SceneObject,
    SceneStore,
)

logger: logging.Logger = logging.getLogger(__name__)


class ObjectNotDetectedError(RuntimeError):
    """The CuTR store has no box for an object and R3D fallback is disabled."""


class SceneState:
    """Read-only view of the current 3D scene.

    Wraps a SceneStore and provides query methods for LLM tool calling
    and evaluation. Thread-safe for concurrent reads.
    """

    def __init__(
        self,
        store: SceneStore,
        sequence_id: str,
        mesh_store: MeshStore | None = None,
        volume_method: str = "mesh_rescaled",
        cubify_store: CubifyStore | None = None,
        allow_cutr_fallback: bool = True,
    ) -> None:
        if volume_method not in VOLUME_METHODS:
            raise ValueError(f"Unknown volume_method: {volume_method}")
        if not allow_cutr_fallback and cubify_store is None:
            raise ValueError("allow_cutr_fallback=False requires a cubify_store")
        self._cubify_store = cubify_store
        self._allow_cutr_fallback = allow_cutr_fallback
        self._store = store
        self._sequence_id = sequence_id
        self._mesh_store = mesh_store
        self._volume_method = volume_method
        self._object_map: dict[int, SceneObject] | None = None

    def _get_object_map(self) -> dict[int, SceneObject]:
        if self._object_map is None:
            self._object_map = {
                obj.object_id: obj
                for obj in self._store.get_all_scene_objects(self._sequence_id)
            }
        return self._object_map

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    @property
    def volume_method(self) -> str:
        return self._volume_method

    def get_object_ids(self) -> list[int]:
        return list(self._get_object_map().keys())

    def get_object(self, object_id: int) -> SceneObject | None:
        return self._get_object_map().get(object_id)

    def get_all_objects(self) -> list[SceneObject]:
        return self._store.get_all_scene_objects(self._sequence_id)

    def get_best_reconstruction(
        self,
        object_id: int,
        policy: str = "best_psnr",
    ) -> ObjectReconstruction | None:
        """Get the best reconstruction for an object based on selection policy.

        Policies:
            best_psnr: highest PSNR (default)
            latest: most recently created

        With a cubify_store (the CuTR baseline), the object's box and position
        come from CuTR. If CuTR has no box for the object, the scene box is
        returned, or ObjectNotDetectedError is raised when allow_cutr_fallback
        is False.
        """
        recon = self._get_scene_reconstruction(object_id, policy)
        if self._cubify_store is None:
            return recon
        cubify = self._cubify_store.get_object_result(self._sequence_id, object_id)
        if cubify is None:
            if not self._allow_cutr_fallback:
                raise ObjectNotDetectedError(
                    f"object {object_id} was not detected by the 3D box detector "
                    f"(CuTR); no geometry is available for it."
                )
            return recon
        return _merge_cubify_box(recon, cubify)

    def _get_scene_reconstruction(
        self, object_id: int, policy: str
    ) -> ObjectReconstruction | None:
        recons = self._store.get_reconstructions(self._sequence_id, object_id)
        if not recons:
            return None

        if policy == "best_psnr":
            with_psnr = [r for r in recons if r.psnr is not None]
            if with_psnr:

                def _psnr(r: ObjectReconstruction) -> float:
                    assert r.psnr is not None
                    return r.psnr

                return max(with_psnr, key=_psnr)
            return recons[-1]
        elif policy == "latest":
            return max(recons, key=lambda r: r.created_ns)
        else:
            raise ValueError(f"Unknown reconstruction policy: {policy}")

    def get_object_position(self, object_id: int) -> np.ndarray | None:
        """Get the latest 3D position for an object (from best reconstruction)."""
        recon = self.get_best_reconstruction(object_id)
        if recon is None:
            return None
        return recon.position

    def get_object_bbox_3d(
        self, object_id: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get (aabb_local, T_scene_obb) for an object's best reconstruction."""
        recon = self.get_best_reconstruction(object_id)
        if recon is None:
            return None
        return recon.obb_aabb, recon.obb_transform

    def get_initial_object_bbox_3d(
        self, object_id: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get initial (pre-training) (aabb_local, T_scene_obb) for an object."""
        recon = self.get_best_reconstruction(object_id)
        if recon is None:
            return None
        return recon.initial_obb_aabb, recon.initial_obb_transform

    def get_visible_objects_at(self, timestamp_ns: int) -> list[int]:
        """Get object IDs visible at a given timestamp."""
        vis = self._store.get_frame_visibility(self._sequence_id, timestamp_ns)
        return [v.object_id for v in vis]

    def get_object_query_name(self, object_id: int) -> str | None:
        obj = self.get_object(object_id)
        return obj.query_name if obj else None

    def get_all_reconstructions(self, object_id: int) -> list[ObjectReconstruction]:
        return self._store.get_reconstructions(self._sequence_id, object_id)

    def get_frame_visibility(self, timestamp_ns: int) -> list[FrameVisibility]:
        return self._store.get_frame_visibility(self._sequence_id, timestamp_ns)

    def resolve_by_name(self, query_name: str) -> SceneObject:
        target = query_name.lower().strip()
        for obj in self._store.get_all_scene_objects(self._sequence_id):
            if obj.query_name.lower() == target:
                return obj
        raise RuntimeError(
            f"No object with query_name '{query_name}' in scene. "
            f"Available: {[o.query_name for o in self._store.get_all_scene_objects(self._sequence_id)]}"
        )

    def get_tracked_object_ids_in_window(self, start_ns: int, end_ns: int) -> list[int]:
        seen: set[int] = set()
        for obj in self._store.get_all_scene_objects(self._sequence_id):
            if obj.last_seen_ns >= start_ns and obj.first_seen_ns <= end_ns:
                seen.add(obj.object_id)
        return sorted(seen)

    def get_object_volume(self, object_id: int) -> float | None:
        """Get estimated object volume in cubic meters.

        Volume is the SAM3D mesh cavity (functional) volume rescaled to match
        the object's fitted OBB, or the OBB box volume when volume_method is
        "bbox" (the no-SAM3D ablation).

        Returns:
            Volume in cubic meters, or None if the object has no
            reconstruction or its mesh is unavailable.
        """
        recon = self.get_best_reconstruction(object_id)
        if recon is None:
            return None
        if self._volume_method == "bbox":
            return volume.bbox_volume(recon.obb_aabb)

        if self._mesh_store is None:
            raise RuntimeError("MeshStore required for volume estimation")
        query_name = self.get_object_query_name(object_id)
        if query_name is None:
            return None
        mesh_path = self._mesh_store.get_mesh_abs_path(self._sequence_id, query_name)
        if mesh_path is None:
            return None
        return volume.mesh_rescaled_volume(mesh_path, recon.obb_aabb)


def _merge_cubify_box(
    recon: ObjectReconstruction | None, cubify: CubifyObjectResult
) -> ObjectReconstruction:
    """The scene reconstruction with its box and position replaced by CuTR's."""
    box = {
        "obb_aabb": cubify.obb_aabb,
        "obb_transform": cubify.obb_transform,
        "position": cubify.position,
        "initial_obb_aabb": cubify.obb_aabb,
        "initial_obb_transform": cubify.obb_transform,
        "initial_position": cubify.position,
    }
    if recon is not None:
        return recon.model_copy(update=box)
    return ObjectReconstruction(
        reconstruction_id=0,
        sequence_id=cubify.sequence_id,
        object_id=cubify.object_id,
        time_range_start_ns=0,
        time_range_end_ns=0,
        **box,
    )

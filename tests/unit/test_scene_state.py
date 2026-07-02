# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.base import ObjectReconstruction, SceneObject
from r3d.pipeline.stores.sqlite_store import SQLiteSceneStore


def _make_scene(tmpdir: str) -> tuple[SQLiteSceneStore, str]:
    """Create a scene store with one object and reconstruction."""
    seq_id = "seq001"
    scene_store = SQLiteSceneStore(Path(tmpdir) / "scene.db")

    scene_store.write_scene_object(
        SceneObject(
            sequence_id=seq_id,
            object_id=1,
            query_name="coffee mug",
            first_seen_ns=100,
            last_seen_ns=500,
        )
    )

    obb_aabb = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    scene_store.write_reconstruction(
        ObjectReconstruction(
            reconstruction_id=0,
            sequence_id=seq_id,
            object_id=1,
            time_range_start_ns=100,
            time_range_end_ns=500,
            obb_aabb=obb_aabb,
            obb_transform=np.eye(4),
            position=np.array([0.5, 1.0, 1.5]),
            initial_obb_aabb=obb_aabb,
            initial_obb_transform=np.eye(4),
            initial_position=np.array([0.5, 1.0, 1.5]),
            psnr=25.0,
            created_ns=1_000,
        )
    )

    return scene_store, seq_id


class TestSceneStateObjects(unittest.TestCase):
    def test_get_object_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            self.assertEqual(state.get_object_ids(), [1])
            scene_store.close()

    def test_get_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            obj = state.get_object(1)
            self.assertIsNotNone(obj)
            self.assertEqual(obj.query_name, "coffee mug")
            scene_store.close()

    def test_get_nonexistent_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            self.assertIsNone(state.get_object(999))
            scene_store.close()

    def test_resolve_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            obj = state.resolve_by_name("coffee mug")
            self.assertEqual(obj.object_id, 1)
            scene_store.close()

    def test_resolve_by_name_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            with self.assertRaises(RuntimeError):
                state.resolve_by_name("nonexistent")
            scene_store.close()

    def test_get_object_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            pos = state.get_object_position(1)
            self.assertIsNotNone(pos)
            np.testing.assert_allclose(pos, [0.5, 1.0, 1.5], atol=1e-10)
            scene_store.close()

    def test_get_object_bbox_3d(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            result = state.get_object_bbox_3d(1)
            self.assertIsNotNone(result)
            aabb, transform = result
            np.testing.assert_allclose(aabb, [0.0, 1.0, 0.0, 2.0, 0.0, 3.0], atol=1e-10)
            np.testing.assert_allclose(transform, np.eye(4), atol=1e-10)
            scene_store.close()

    def test_no_reconstruction_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store = SQLiteSceneStore(Path(tmpdir) / "scene.db")
            scene_store.write_scene_object(
                SceneObject(
                    sequence_id="seq001",
                    object_id=2,
                    query_name="plate",
                    first_seen_ns=0,
                    last_seen_ns=100,
                )
            )
            state = SceneState(scene_store, "seq001")
            self.assertIsNone(state.get_object_volume(2))
            self.assertIsNone(state.get_object_position(2))
            scene_store.close()

    def test_sequence_id_property(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            self.assertEqual(state.sequence_id, "seq001")
            scene_store.close()

    def test_tracked_objects_in_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_store, seq_id = _make_scene(tmpdir)
            state = SceneState(scene_store, seq_id)
            # Object 1 is visible from 100 to 500
            ids = state.get_tracked_object_ids_in_window(200, 400)
            self.assertEqual(ids, [1])
            # Outside window
            ids = state.get_tracked_object_ids_in_window(600, 700)
            self.assertEqual(ids, [])
            scene_store.close()


if __name__ == "__main__":
    unittest.main()

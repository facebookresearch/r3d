# Copyright (c) Meta Platforms, Inc. and affiliates.

"""CuTR baseline: object boxes from cubify.db instead of the scene."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from r3d.pipeline.eval.tool_use import _execute_tool
from r3d.pipeline.scene_state import ObjectNotDetectedError, SceneState
from r3d.pipeline.stores.base import ObjectReconstruction, SceneObject
from r3d.pipeline.stores.sqlite_store import (
    SQLiteCubifyStore,
    SQLiteFrameStore,
    SQLiteSceneStore,
)

SEQ = "seq001"
SCENE_AABB = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
CUTR_AABB = np.array([-0.1, 0.1, -0.2, 0.2, -0.3, 0.3])
CUTR_POSITION = np.array([4.0, 5.0, 6.0])


def _blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float64).tobytes()


def _make_scene(db_path: Path) -> SQLiteSceneStore:
    store = SQLiteSceneStore(db_path)
    for oid, name in ((1, "mug"), (2, "table")):
        store.write_scene_object(
            SceneObject(
                sequence_id=SEQ,
                object_id=oid,
                query_name=name,
                first_seen_ns=100,
                last_seen_ns=500,
            )
        )
        store.write_reconstruction(
            ObjectReconstruction(
                reconstruction_id=0,
                sequence_id=SEQ,
                object_id=oid,
                time_range_start_ns=100,
                time_range_end_ns=500,
                obb_aabb=SCENE_AABB,
                obb_transform=np.eye(4),
                position=np.array([0.5, 1.0, 1.5]),
                initial_obb_aabb=SCENE_AABB,
                initial_obb_transform=np.eye(4),
                initial_position=np.array([0.5, 1.0, 1.5]),
            )
        )
    return store


def _make_cubify_db(db_path: Path) -> None:
    """cubify.db as written by the CuTR runner: a box for object 1 only."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE cubify_object_results (
            sequence_id TEXT NOT NULL,
            object_id INTEGER NOT NULL,
            query_name TEXT NOT NULL,
            obb_aabb BLOB NOT NULL,
            obb_transform BLOB NOT NULL,
            position BLOB NOT NULL,
            num_detections INTEGER NOT NULL,
            aggregation_method TEXT NOT NULL,
            PRIMARY KEY (sequence_id, object_id)
        )
    """)
    conn.execute(
        "INSERT INTO cubify_object_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            SEQ,
            1,
            "mug",
            _blob(CUTR_AABB),
            _blob(np.eye(4)),
            _blob(CUTR_POSITION),
            7,
            "highest_score",
        ),
    )
    conn.commit()
    conn.close()


class TestCutrBoxes(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.scene_store = _make_scene(tmp / "scene.db")
        _make_cubify_db(tmp / "cubify.db")
        self.cubify = SQLiteCubifyStore(tmp / "cubify.db")
        self.frames = SQLiteFrameStore(tmp / "frames.db")

    def tearDown(self) -> None:
        self.scene_store.close()
        self.cubify.close()
        self.frames.close()
        self._tmp.cleanup()

    def _state(self, allow_fallback: bool) -> SceneState:
        return SceneState(
            self.scene_store,
            SEQ,
            volume_method="bbox",
            cubify_store=self.cubify,
            allow_cutr_fallback=allow_fallback,
        )

    def _tool(self, state: SceneState, name: str, args: list[str]) -> str:
        return _execute_tool(name, args, state, self.frames, SEQ, 100, 500)

    def test_cutr_box_replaces_scene_box(self) -> None:
        state = self._state(allow_fallback=True)
        aabb, _ = state.get_object_bbox_3d(1)
        np.testing.assert_array_equal(aabb, CUTR_AABB)
        np.testing.assert_array_equal(state.get_object_position(1), CUTR_POSITION)
        self.assertAlmostEqual(state.get_object_volume(1), 0.2 * 0.4 * 0.6)
        self.assertEqual(
            self._tool(state, "get_object_size", ["1"]),
            "width=0.20m, height=0.40m, depth=0.60m",
        )

    def test_fallback_uses_scene_box(self) -> None:
        state = self._state(allow_fallback=True)
        aabb, _ = state.get_object_bbox_3d(2)
        np.testing.assert_array_equal(aabb, SCENE_AABB)

    def test_no_fallback_reports_not_detected(self) -> None:
        state = self._state(allow_fallback=False)
        with self.assertRaises(ObjectNotDetectedError):
            state.get_best_reconstruction(2)
        self.assertEqual(
            self._tool(state, "get_position", ["2"]),
            "Error: object not detected: object 2 was not detected by the 3D box "
            "detector (CuTR); no geometry is available for it.",
        )
        self.assertEqual(
            self._tool(state, "list_objects", []),
            "Tracked objects (2):\n  ID 1: mug\n  ID 2: table",
        )

    def test_no_fallback_requires_cubify_store(self) -> None:
        with self.assertRaises(ValueError):
            SceneState(self.scene_store, SEQ, allow_cutr_fallback=False)

    def test_missing_cubify_db_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SQLiteCubifyStore(Path(self._tmp.name) / "absent.db")


if __name__ == "__main__":
    unittest.main()

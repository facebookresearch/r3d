# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the gt_oracle tool backend in r3d.pipeline.eval.tool_use."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from r3d.pipeline.eval.gt_oracle import GTOracle
from r3d.pipeline.eval.tool_use import execute_oracle_tool
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.base import SceneObject
from r3d.pipeline.stores.sqlite_store import SQLiteSceneStore

SEQ = "seq001"
T_END = 100

_SCHEMA = """
CREATE TABLE oracle_objects (
    sequence_id TEXT, query_name TEXT, adt_instance_id INTEGER,
    adt_instance_name TEXT, category TEXT, obb_aabb BLOB, obb_transform BLOB,
    volume_m3 REAL
);
CREATE TABLE oracle_device_poses (
    sequence_id TEXT, timestamp_ns INTEGER, T_scene_device BLOB
);
"""


def _translation(x: float, y: float, z: float) -> bytes:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T.tobytes()


def _write_oracle(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    mug_aabb = np.array([-0.05, 0.05, 0.0, 0.2, -0.05, 0.05]).tobytes()
    table_aabb = np.array([-0.5, 0.5, -0.4, 0.4, -0.3, 0.3]).tobytes()
    conn.executemany(
        "INSERT INTO oracle_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (SEQ, "mug", 1, "Mug_01", "cup", mug_aabb, _translation(1, 0.5, 2), 3.5e-4),
            (SEQ, "table", 2, "Table_01", "table", table_aabb, _translation(3, 0.4, 2), None),
        ],
    )
    conn.execute(
        "INSERT INTO oracle_device_poses VALUES (?, ?, ?)",
        (SEQ, T_END, _translation(1.0, 1.5, 0.0)),
    )
    conn.commit()
    conn.close()


class TestGTOracleTools(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        _write_oracle(root / "gt_oracle.db")
        self.oracle = GTOracle(root / "gt_oracle.db")
        self.scene_store = SQLiteSceneStore(root / "scene.db")
        for oid, name in [(3, "mug"), (7, "table")]:
            self.scene_store.write_scene_object(
                SceneObject(
                    sequence_id=SEQ,
                    object_id=oid,
                    query_name=name,
                    first_seen_ns=0,
                    last_seen_ns=T_END,
                )
            )
        self.scene = SceneState(self.scene_store, SEQ)

    def tearDown(self) -> None:
        self.scene_store.close()
        self._tmpdir.cleanup()

    def _call(self, tool: str, args: list[str]) -> str:
        return execute_oracle_tool(tool, args, self.scene, self.oracle, SEQ, 0, T_END)

    def test_tool_outputs(self) -> None:
        self.assertEqual(
            self._call("list_objects", []),
            "Tracked objects (2):\n  ID 3: mug\n  ID 7: table",
        )
        self.assertEqual(self._call("get_distance", ["3", "7"]), "1.45 meters")
        self.assertEqual(self._call("get_position", ["3"]), "(1.00, 2.00, 0.60) meters")
        self.assertEqual(self._call("get_my_position", []), "(1.00, 0.00, 1.50) meters")
        self.assertEqual(self._call("get_distance_from_me", ["3"]), "2.11 meters")
        self.assertEqual(
            self._call("get_object_size", ["3"]),
            "width=0.10m, height=0.20m, depth=0.10m",
        )

    def test_volume(self) -> None:
        self.assertEqual(
            self._call("get_object_volume", ["3"]),
            "estimated_volume=0.000350 cubic meters (0.350 liters), "
            "computed from the object's rescaled 3D mesh",
        )
        self.assertEqual(
            self._call("get_object_volume", ["7"]),
            "Error: could not compute rescaled mesh volume for object 7.",
        )

    def test_errors(self) -> None:
        self.assertEqual(self._call("get_position", ["9"]), "Error: no object with ID 9.")
        self.assertEqual(
            self._call("get_distance", ["3"]),
            "Error: get_distance requires two object IDs.",
        )

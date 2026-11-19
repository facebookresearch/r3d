# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from r3d.data_gen.utils.annotation_schema import (
    Annotation,
    EvalMetric,
    EvalMode,
    IdentityLayer,
    QueryLayer,
    QuestionType,
    ReferencedObject,
    ReleaseType,
)
from r3d.pipeline.eval.prompts import ANSWER_FORMAT_INSTRUCTIONS
from r3d.pipeline.eval.scene_text import (
    build_scene_text_messages,
    messages_to_chat,
    run_scene_text,
    SceneTableBuilder,
)
from r3d.pipeline.eval.tool_use import _execute_tool
from r3d.pipeline.eval.vlm import VLMClient
from r3d.pipeline.frame_data import CameraIntrinsics, DepthSource, FrameData
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.base import (
    MeshStore,
    ObjectMesh,
    ObjectReconstruction,
    SceneObject,
)
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore, SQLiteSceneStore
from r3d.types import ChatRole, Message

SEQ = "seq001"
START_NS = 1_000
END_NS = 2_000
MUG_ID = 1
TABLE_ID = 2
LAMP_ID = 3
GHOST_ID = 4


def _write_cup_glb(path: Path) -> None:
    """Open-top 10 cm box with 1 cm walls; opening along +y (canonical up)."""
    slabs = [
        ((0.10, 0.01, 0.10), (0.0, -0.045, 0.0)),
        ((0.01, 0.10, 0.10), (-0.045, 0.0, 0.0)),
        ((0.01, 0.10, 0.10), (0.045, 0.0, 0.0)),
        ((0.10, 0.10, 0.01), (0.0, 0.0, -0.045)),
        ((0.10, 0.10, 0.01), (0.0, 0.0, 0.045)),
    ]
    parts = []
    for extents, center in slabs:
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(center)
        parts.append(box)
    trimesh.util.concatenate(parts).export(str(path))


class _FakeMeshStore(MeshStore):
    def __init__(self, paths: dict[str, str]) -> None:
        self._paths = paths

    def write_mesh(self, mesh: ObjectMesh) -> None:
        raise NotImplementedError

    def get_mesh(self, sequence_id: str, object_name: str) -> ObjectMesh | None:
        raise NotImplementedError

    def get_mesh_abs_path(self, sequence_id: str, object_name: str) -> str | None:
        if sequence_id != SEQ or object_name not in self._paths:
            return None
        return self._paths[object_name]

    def get_all_meshes(self) -> list[ObjectMesh]:
        raise NotImplementedError

    def get_all_sequence_ids(self) -> list[str]:
        return [SEQ]

    def close(self) -> None:
        pass


class _RecordingVLM(VLMClient):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def query(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("scene_text must use query_multiturn")

    def query_multiturn(
        self, messages: list[Message], model: str, max_tokens: int = 16384
    ) -> str:
        self.calls.append(messages)
        return "ANSWER: 1.23"


def _yaw_transform(yaw_rad: float, translation: list[float]) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    T[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    T[:3, 3] = translation
    return T


def _add_object(
    store: SQLiteSceneStore, oid: int, name: str, first_ns: int, last_ns: int
) -> None:
    store.write_scene_object(
        SceneObject(
            sequence_id=SEQ,
            object_id=oid,
            query_name=name,
            first_seen_ns=first_ns,
            last_seen_ns=last_ns,
        )
    )


def _add_recon(
    store: SQLiteSceneStore, oid: int, aabb: list[float], T: np.ndarray
) -> None:
    obb_aabb = np.array(aabb)
    position = T[:3, 3].copy()
    store.write_reconstruction(
        ObjectReconstruction(
            reconstruction_id=oid,
            sequence_id=SEQ,
            object_id=oid,
            time_range_start_ns=START_NS,
            time_range_end_ns=END_NS,
            obb_aabb=obb_aabb,
            obb_transform=T,
            position=position,
            initial_obb_aabb=obb_aabb,
            initial_obb_transform=T,
            initial_position=position,
            psnr=25.0,
            created_ns=1,
        )
    )


def _make_scene(tmpdir: Path) -> SQLiteSceneStore:
    store = SQLiteSceneStore(tmpdir / "scene.db")
    _add_object(store, MUG_ID, "coffee mug", 500, 1_500)
    _add_object(store, TABLE_ID, "dining table", 1_200, 3_000)
    _add_object(store, LAMP_ID, "floor lamp", 2_500, 3_000)
    _add_object(store, GHOST_ID, "ghost vase", 1_000, 2_000)
    mug_aabb = [-0.05, 0.05, -0.05, 0.05, -0.05, 0.05]
    _add_recon(store, MUG_ID, mug_aabb, _yaw_transform(0.0, [0.5, 1.0, 1.5]))
    table_aabb = [-0.8, 0.8, -0.375, 0.375, -0.45, 0.45]
    _add_recon(store, TABLE_ID, table_aabb, _yaw_transform(0.6, [2.0, 0.375, -1.0]))
    _add_recon(store, LAMP_ID, mug_aabb, _yaw_transform(0.0, [5.0, 0.8, 5.0]))
    return store


def _make_frames(tmpdir: Path) -> SQLiteFrameStore:
    store = SQLiteFrameStore(tmpdir / "frames.db")
    T_sd = np.eye(4)
    T_sd[:3, 3] = [1.234, 1.567, -0.891]
    store.write_frame_data(
        SEQ,
        FrameData(
            timestamp_ns=END_NS,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_map=np.ones((4, 4), dtype=np.float32),
            depth_source=DepthSource.GROUND_TRUTH,
            intrinsics=CameraIntrinsics(
                fx=2.0, fy=2.0, cx=2.0, cy=2.0, width=4, height=4
            ),
            T_scene_device=T_sd,
            T_device_camera=np.eye(4),
        ),
    )
    return store


def _make_annotation() -> Annotation:
    return Annotation(
        annotation_id="scene-text-001",
        identity_layer=IdentityLayer(
            sequence_id=SEQ,
            release_type=ReleaseType.FULL,
            referenced_objects=[
                ReferencedObject(
                    object_id="obj_001",
                    adt_instance_id=1,
                    adt_instance_name="CoffeeMug",
                    prototype_name="CoffeeMug",
                    canonical_name="coffee mug",
                    is_dynamic=False,
                ),
            ],
        ),
        query_layer=QueryLayer(
            question_type=QuestionType.GLOBAL_HOW_FAR,
            question_text="How far is the coffee mug from the dining table?",
            gt_answer="1.0",
            gt_answer_type="float",
            gt_answer_unit="meters",
            eval_mode=EvalMode.DETERMINISTIC,
            eval_metric=EvalMetric.PERCENTAGE_ERROR,
            query_timestamp_ns_start=START_NS,
            query_timestamp_ns_end=END_NS,
        ),
    )


def _parse_rows(table: str) -> dict[int, dict[str, str]]:
    """Parse '  ID <id>: <name> | position: ... | size: ... | volume: ...' rows."""
    rows: dict[int, dict[str, str]] = {}
    for line in table.splitlines()[1:-1]:
        head, *cells = line.strip().split(" | ")
        oid_str, name = head.removeprefix("ID ").split(": ", 1)
        fields = {"name": name}
        for cell in cells:
            label, value = cell.split(": ", 1)
            fields[label] = value
        rows[int(oid_str)] = fields
    return rows


class TestSceneTextTable(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self._tmp.name)
        cup_path = tmpdir / "cup.glb"
        _write_cup_glb(cup_path)
        self.scene_store = _make_scene(tmpdir)
        self.frame_store = _make_frames(tmpdir)
        self.scene = SceneState(
            self.scene_store,
            SEQ,
            mesh_store=_FakeMeshStore({"coffee mug": str(cup_path)}),
        )
        self.builder = SceneTableBuilder(self.frame_store)

    def tearDown(self) -> None:
        self.scene_store.close()
        self.frame_store.close()
        self._tmp.cleanup()

    def _tool(self, name: str, args: list[str]) -> str:
        return _execute_tool(
            name, args, self.scene, self.frame_store, SEQ, START_NS, END_NS
        )

    def test_objects_match_list_objects(self) -> None:
        table, num_objects = self.builder.build(self.scene, START_NS, END_NS)
        listed = self._tool("list_objects", []).splitlines()
        self.assertEqual(table.splitlines()[0], listed[0])
        expected = {
            int(line.strip().removeprefix("ID ").split(": ", 1)[0]): line.split(
                ": ", 1
            )[1]
            for line in listed[1:]
        }
        rows = _parse_rows(table)
        self.assertEqual({oid: r["name"] for oid, r in rows.items()}, expected)
        self.assertEqual(num_objects, 3)
        self.assertNotIn(LAMP_ID, rows)

    def test_fields_identical_to_tool_outputs(self) -> None:
        table, _ = self.builder.build(self.scene, START_NS, END_NS)
        rows = _parse_rows(table)
        tools = {
            "position": "get_position",
            "size": "get_object_size",
            "volume": "get_object_volume",
        }
        for oid, fields in rows.items():
            for label, tool in tools.items():
                self.assertEqual(fields[label], self._tool(tool, [str(oid)]))

    def test_my_position_identical_to_tool(self) -> None:
        table, _ = self.builder.build(self.scene, START_NS, END_NS)
        last = table.splitlines()[-1]
        self.assertEqual(
            last, f"Your current position: {self._tool('get_my_position', [])}"
        )
        self.assertEqual(last, "Your current position: (1.23, -0.89, 1.57) meters")

    def test_known_values_and_conventions(self) -> None:
        rows = _parse_rows(self.builder.build(self.scene, START_NS, END_NS)[0])
        self.assertEqual(rows[MUG_ID]["position"], "(0.50, 1.50, 1.00) meters")
        self.assertEqual(
            rows[TABLE_ID]["size"], "width=1.60m, height=0.75m, depth=0.90m"
        )
        self.assertTrue(rows[MUG_ID]["volume"].startswith("estimated_volume="))
        self.assertEqual(
            rows[TABLE_ID]["volume"],
            f"Error: could not compute rescaled mesh volume for object {TABLE_ID}.",
        )
        self.assertEqual(
            rows[GHOST_ID]["position"], "Error: object has no reconstruction."
        )

    def test_no_pairwise_quantities(self) -> None:
        table, _ = self.builder.build(self.scene, START_NS, END_NS)
        self.assertNotIn("distance", table.lower())

    def test_cached_rows_are_stable(self) -> None:
        first = self.builder.build(self.scene, START_NS, END_NS)
        second = self.builder.build(self.scene, START_NS, END_NS)
        self.assertEqual(first, second)

    def test_empty_window_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.builder.build(self.scene, 10_000, 20_000)

    def test_run_scene_text_single_turn(self) -> None:
        vlm = _RecordingVLM()
        result = run_scene_text(
            _make_annotation(), self.scene, self.builder, vlm, "m"
        )
        self.assertEqual(result.response, "ANSWER: 1.23")
        self.assertEqual(result.num_objects, 3)
        self.assertEqual(len(vlm.calls), 1)
        system, user = vlm.calls[0]
        self.assertEqual(system.role, ChatRole.SYSTEM)
        self.assertTrue(system.text.endswith(ANSWER_FORMAT_INSTRUCTIONS))
        self.assertNotIn("TOOL_CALL", system.text)
        self.assertIn(self.builder.build(self.scene, START_NS, END_NS)[0], user.text)
        self.assertIn("How far is the coffee mug", user.text)


class TestSceneTextMessages(unittest.TestCase):
    def test_messages_to_chat(self) -> None:
        messages = build_scene_text_messages(_make_annotation(), "TABLE")
        chat = messages_to_chat(messages)
        self.assertEqual([c["role"] for c in chat], ["system", "user"])
        self.assertTrue(chat[1]["content"].startswith("Scene:\nTABLE\n\nQuestion:"))


if __name__ == "__main__":
    unittest.main()

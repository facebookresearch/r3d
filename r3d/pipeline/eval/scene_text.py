# Copyright (c) Meta Platforms, Inc. and affiliates.

"""``scene_text`` strategy: the reconstructed scene given to the LLM as text.

Ablation of R3D's tool calling. Instead of letting the model query spatial
tools, the prompt contains one row per object that ``list_objects()`` returns
for the question window, holding the verbatim output of ``get_position``,
``get_object_size`` and ``get_object_volume`` for that object, plus the verbatim
output of ``get_my_position()``. Every value is produced by the tool dispatcher
itself, so units, rounding and the (x, z, y) print convention are identical to
what the tool-use model sees. Pairwise quantities (``get_distance``,
``get_distance_from_me``) are deliberately absent: the tools compute
closest-point distances between oriented boxes, and the text model has to infer
distances from centers and sizes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.prompts import ANSWER_FORMAT_INSTRUCTIONS
from r3d.pipeline.eval.tool_use import _execute_tool
from r3d.pipeline.eval.vlm import VLMClient
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore
from r3d.types import ChatRole, Message

SCENE_TEXT_STRATEGY: str = "scene_text"

SCENE_TEXT_INSTRUCTIONS: str = (
    """\
You are given a description of a 3D scene reconstructed from the camera's video.
It lists every tracked object with its numeric ID and name, followed by:
- position: the 3D position (x, y, z) of the object in meters, where z is the vertical (up) axis
- size: the dimensions (width, height, depth) of the object in meters
- volume: the estimated volume of the object in cubic meters and liters
It also gives your (the camera's) current 3D position (x, y, z) in meters, where z is the vertical (up) axis.

Use this scene description to answer the question.

"""
    + ANSWER_FORMAT_INSTRUCTIONS
)

_OBJECT_FIELDS: tuple[tuple[str, str], ...] = (
    ("position", "get_position"),
    ("size", "get_object_size"),
    ("volume", "get_object_volume"),
)


@dataclass(frozen=True)
class SceneTextResult:
    response: str
    messages: list[Message]
    num_objects: int


class SceneTableBuilder:
    """Formats the scene table from tool outputs, caching per-object rows.

    Benchmark objects are static, so an object's row depends only on
    (sequence_id, object_id); ``get_object_volume`` is expensive (mesh
    voxelization), hence the cache. Thread-safe.
    """

    def __init__(self, frame_store: SQLiteFrameStore) -> None:
        self._frame_store = frame_store
        self._rows: dict[tuple[str, int], str] = {}
        self._lock = threading.Lock()

    def _call(self, tool: str, args: list[str], scene: SceneState, ts: int) -> str:
        return _execute_tool(
            tool, args, scene, self._frame_store, scene.sequence_id, ts, ts
        )

    def object_row(self, scene: SceneState, object_id: int) -> str:
        key = (scene.sequence_id, object_id)
        with self._lock:
            if key in self._rows:
                return self._rows[key]
        name = scene.get_object_query_name(object_id)
        if name is None:
            raise RuntimeError(f"Object {object_id} has no query_name")
        cells = [f"ID {object_id}: {name}"]
        for label, tool in _OBJECT_FIELDS:
            cells.append(f"{label}: {self._call(tool, [str(object_id)], scene, 0)}")
        row = " | ".join(cells)
        with self._lock:
            self._rows[key] = row
        return row

    def build(self, scene: SceneState, start_ns: int, end_ns: int) -> tuple[str, int]:
        """Return (scene table text, number of objects) for a question window."""
        obj_ids = scene.get_tracked_object_ids_in_window(start_ns, end_ns)
        if not obj_ids:
            raise RuntimeError(
                f"No tracked objects in [{start_ns}, {end_ns}] for "
                f"{scene.sequence_id}"
            )
        rows = [f"  {self.object_row(scene, oid)}" for oid in obj_ids]
        my_pos = self._call("get_my_position", [], scene, end_ns)
        table = (
            f"Tracked objects ({len(rows)}):\n"
            + "\n".join(rows)
            + f"\nYour current position: {my_pos}"
        )
        return table, len(obj_ids)


def build_scene_text_messages(annotation: Annotation, table: str) -> list[Message]:
    question = annotation.query_layer.question_text
    text = (
        f"Scene:\n{table}\n\nQuestion: {question}\n\n"
        "Use the scene description above to work out the answer, then provide "
        "your final answer."
    )
    return [
        Message(role=ChatRole.SYSTEM, text=SCENE_TEXT_INSTRUCTIONS),
        Message(role=ChatRole.USER, text=text),
    ]


def messages_to_chat(messages: list[Message]) -> list[dict[str, str]]:
    """Text-only chat-template dicts, mirroring the vLLM client's conversion."""
    for msg in messages:
        if msg.attachments:
            raise ValueError("scene_text prompts must be text-only")
    return [{"role": msg.role.value, "content": msg.text} for msg in messages]


def run_scene_text(
    annotation: Annotation,
    scene: SceneState,
    builder: SceneTableBuilder,
    vlm: VLMClient,
    model: str,
) -> SceneTextResult:
    """Single-turn answer from the scene table (no tools)."""
    ql = annotation.query_layer
    table, num_objects = builder.build(
        scene, ql.query_timestamp_ns_start, ql.query_timestamp_ns_end
    )
    messages = build_scene_text_messages(annotation, table)
    response = vlm.query_multiturn(messages, model)
    return SceneTextResult(
        response=response, messages=messages, num_objects=num_objects
    )

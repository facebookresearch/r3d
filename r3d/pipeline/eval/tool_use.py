# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import logging
import re
import typing

import numpy as np
from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline import volume as volume_mod
from r3d.pipeline.eval.prompts import ANSWER_FORMAT_INSTRUCTIONS
from r3d.pipeline.eval.vlm import image_to_base64, VLMClient
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.base import ObjectReconstruction
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore
from r3d.types import ChatRole, Message, MessageAttachment, MessageAttachmentType

logger: logging.Logger = logging.getLogger(__name__)

TOOL_DESCRIPTIONS: str = (
    """\
You have access to spatial measurement tools. To use a tool, respond with exactly:
TOOL_CALL: tool_name(arg1, arg2)

IMPORTANT: First call list_objects() to discover available objects and their numeric IDs.
Then use those numeric IDs with the spatial tools below. Do NOT pass object names to
spatial tools — they only accept numeric IDs.

Available tools:

Resolution (use first):
- list_objects(): Returns all tracked objects with numeric IDs and names.

Spatial (require numeric object IDs):
- get_distance(id1, id2): Returns the distance in meters between two objects
- get_position(id): Returns the 3D position (x, y, z) of an object in meters, where z is the vertical (up) axis
- get_my_position(): Returns your (the camera's) current 3D position (x, y, z) in meters, where z is the vertical (up) axis
- get_distance_from_me(id): Returns the distance in meters from the camera to an object
- get_object_size(id): Returns the dimensions (width, height, depth) of an object in meters
- get_object_volume(id): Returns the estimated volume of an object in cubic meters and liters

You may only call ONE tool per response. If you attempt to call multiple tools,
only the first will be executed. After using tools to gather information,
provide your final answer. Do NOT call any more tools once you have enough
information to answer.

"""
    + ANSWER_FORMAT_INSTRUCTIONS
)

_TOOL_CALL_PATTERN: re.Pattern[str] = re.compile(
    r"TOOL_CALL\s*:\s*(\w+)\(\s*(.*?)\s*\)", re.DOTALL
)


def _invert_rigid(T: np.ndarray) -> np.ndarray:
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -(T[:3, :3].T @ T[:3, 3])
    return T_inv


def _closest_point_on_obb(point: np.ndarray, recon: ObjectReconstruction) -> np.ndarray:
    T_obj_scene = _invert_rigid(recon.obb_transform)
    pt_local = (T_obj_scene @ np.append(point, 1.0))[:3]
    a = recon.obb_aabb
    lo = np.array([a[0], a[2], a[4]])
    hi = np.array([a[1], a[3], a[5]])
    clipped = np.clip(pt_local, lo, hi)
    return (recon.obb_transform @ np.append(clipped, 1.0))[:3]


def _obb_distance(r1: ObjectReconstruction, r2: ObjectReconstruction) -> float:
    c1, c2 = r1.position, r2.position
    p1 = _closest_point_on_obb(c2, r1)
    p2 = _closest_point_on_obb(c1, r2)
    if np.allclose(p1, c2) or np.allclose(p2, c1):
        return 0.0
    return float(np.linalg.norm(p1 - p2))


def _get_camera_position(
    frame_store: SQLiteFrameStore, sequence_id: str, timestamp_ns: int
) -> np.ndarray:
    frame = frame_store.load_frame(sequence_id, timestamp_ns)
    T_scene_device = frame.T_scene_device
    return T_scene_device[:3, 3].copy()


def _parse_args(raw: str) -> list[str]:
    return [a.strip().strip("\"'") for a in raw.split(",") if a.strip()]


_ID_ERROR = (
    "Error: expected numeric object ID, got '{raw}'. "
    "Call list_objects() first to get IDs."
)


def _resolve_id(scene: SceneState, raw: str) -> tuple[int, str | None]:
    try:
        oid = int(raw)
    except ValueError:
        return -1, _ID_ERROR.format(raw=raw)
    obj = scene.get_object(oid)
    if obj is None:
        return -1, f"Error: no object with ID {oid}."
    return oid, None


def _tool_get_distance(args: list[str], scene: SceneState) -> str:
    if len(args) < 2:
        return "Error: get_distance requires two object IDs."
    oid1, err1 = _resolve_id(scene, args[0])
    if err1:
        return err1
    oid2, err2 = _resolve_id(scene, args[1])
    if err2:
        return err2
    r1 = scene.get_best_reconstruction(oid1)
    r2 = scene.get_best_reconstruction(oid2)
    if r1 is None or r2 is None:
        return "Error: one or both objects have no reconstruction."
    return f"{_obb_distance(r1, r2):.2f} meters"


def _tool_get_position(args: list[str], scene: SceneState) -> str:
    if len(args) < 1:
        return "Error: get_position requires an object ID."
    oid, err = _resolve_id(scene, args[0])
    if err:
        return err
    pos = scene.get_object_position(oid)
    if pos is None:
        return "Error: object has no reconstruction."
    return f"({pos[0]:.2f}, {pos[2]:.2f}, {pos[1]:.2f}) meters"


def _tool_get_distance_from_me(
    args: list[str],
    scene: SceneState,
    frame_store: SQLiteFrameStore,
    sequence_id: str,
    ts: int,
) -> str:
    if len(args) < 1:
        return "Error: get_distance_from_me requires an object ID."
    oid, err = _resolve_id(scene, args[0])
    if err:
        return err
    r = scene.get_best_reconstruction(oid)
    if r is None:
        return "Error: object has no reconstruction."
    cam_pos = _get_camera_position(frame_store, sequence_id, ts)
    closest = _closest_point_on_obb(cam_pos, r)
    return f"{float(np.linalg.norm(cam_pos - closest)):.2f} meters"


def _tool_get_object_size(args: list[str], scene: SceneState) -> str:
    if len(args) < 1:
        return "Error: get_object_size requires an object ID."
    oid, err = _resolve_id(scene, args[0])
    if err:
        return err
    bbox = scene.get_object_bbox_3d(oid)
    if bbox is None:
        return "Error: object has no reconstruction."
    aabb, _ = bbox
    w, h, d = aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]
    return f"width={w:.2f}m, height={h:.2f}m, depth={d:.2f}m"


def _tool_get_object_volume(
    args: list[str],
    scene: SceneState,
) -> str:
    if len(args) < 1:
        return "Error: get_object_volume requires an object ID."
    oid, err = _resolve_id(scene, args[0])
    if err:
        return err
    est_vol = scene.get_object_volume(oid)
    if est_vol is None:
        return "Error: object has no reconstruction."
    liters = est_vol * 1000
    recon = scene.get_best_reconstruction(oid)
    obb_vol = volume_mod.bbox_volume(recon.obb_aabb) if recon is not None else est_vol
    return (
        f"estimated_volume={est_vol:.6f} cubic meters ({liters:.3f} liters), "
        f"bounding_box_volume={obb_vol:.6f} cubic meters"
    )


def _tool_list_objects(scene: SceneState, start_ns: int, end_ns: int) -> str:
    obj_ids = scene.get_tracked_object_ids_in_window(start_ns, end_ns)
    lines = []
    for oid in obj_ids:
        name = scene.get_object_query_name(oid)
        if name is None:
            raise RuntimeError(f"Object {oid} has no query_name")
        lines.append(f"  ID {oid}: {name}")
    return f"Tracked objects ({len(lines)}):\n" + "\n".join(lines)


def _execute_tool(
    tool_name: str,
    args: list[str],
    scene: SceneState,
    frame_store: SQLiteFrameStore,
    sequence_id: str,
    timestamp_ns_start: int,
    timestamp_ns_end: int,
) -> str:
    if tool_name == "get_distance":
        return _tool_get_distance(args, scene)
    if tool_name == "get_position":
        return _tool_get_position(args, scene)
    if tool_name == "get_my_position":
        pos = _get_camera_position(frame_store, sequence_id, timestamp_ns_end)
        return f"({pos[0]:.2f}, {pos[2]:.2f}, {pos[1]:.2f}) meters"
    if tool_name == "get_distance_from_me":
        return _tool_get_distance_from_me(
            args, scene, frame_store, sequence_id, timestamp_ns_end
        )
    if tool_name == "list_objects":
        return _tool_list_objects(scene, timestamp_ns_start, timestamp_ns_end)
    if tool_name == "get_object_size":
        return _tool_get_object_size(args, scene)
    if tool_name == "get_object_volume":
        return _tool_get_object_volume(args, scene)
    return f"Error: unknown tool '{tool_name}'. Call list_objects() to see available tools."


def _build_initial_messages(
    annotation: Annotation,
    images: list[np.ndarray],
) -> list[Message]:
    question = annotation.query_layer.question_text
    text = f"Question: {question}\n\nUse the tools above to gather the information you need, then provide your final answer."

    attachments = [
        MessageAttachment(
            type=MessageAttachmentType.BASE64_IMAGE,
            data=image_to_base64(img),
            mime="image/jpeg",
        )
        for img in images
    ]
    user_msg = Message(
        role=ChatRole.USER,
        text=text,
        attachments=attachments if images else [],
    )

    return [
        Message(role=ChatRole.SYSTEM, text=TOOL_DESCRIPTIONS),
        user_msg,
    ]


def _handle_tool_turn(
    response: str,
    turn: int,
    messages: list[Message],
    tool_log: list[dict],
    history_text: list[str],
    vlm: VLMClient,
    model: str,
    scene: SceneState,
    frame_store: SQLiteFrameStore,
    sequence_id: str,
    timestamp_ns_start: int,
    timestamp_ns_end: int,
    log_fn: typing.Callable[[str], None],
) -> str | None:
    match = _TOOL_CALL_PATTERN.search(response)
    if match is None:
        log_fn(f"    [turn {turn + 1}] No tool call, done.")
        return None

    tool_name = match.group(1)
    args = _parse_args(match.group(2))
    log_fn(f"    [turn {turn + 1}] Tool: {tool_name}({', '.join(args)})")

    result = _execute_tool(
        tool_name,
        args,
        scene,
        frame_store,
        sequence_id,
        timestamp_ns_start,
        timestamp_ns_end,
    )
    log_fn(f"    [turn {turn + 1}] Result: {result}")
    tool_log.append({"tool": tool_name, "args": args, "result": result})

    messages.append(Message(role=ChatRole.AI, text=response))
    messages.append(Message(role=ChatRole.USER, text=f"Tool result: {result}"))
    history_text.append(f"Tool result: {result}")

    log_fn(f"    [turn {turn + 1}] Querying VLM...")
    new_response = vlm.query_multiturn(messages, model)
    log_fn(f"    [turn {turn + 1}] VLM: {new_response[:300]}")
    history_text.append(f"Assistant: {new_response}")
    return new_response


def _run_tool_loop(
    messages: list[Message],
    vlm: VLMClient,
    model: str,
    scene: SceneState,
    frame_store: SQLiteFrameStore,
    sequence_id: str,
    timestamp_ns_start: int,
    timestamp_ns_end: int,
    max_turns: int,
    log_fn: typing.Callable[[str], None],
) -> tuple[str, list[dict]]:
    tool_log: list[dict] = []
    history_text: list[str] = []

    log_fn("    [turn 0] Querying VLM...")
    response = vlm.query_multiturn(messages, model)
    log_fn(f"    [turn 0] VLM: {response[:300]}")
    history_text.append(f"Assistant: {response}")

    for turn in range(max_turns):
        new_response = _handle_tool_turn(
            response,
            turn,
            messages,
            tool_log,
            history_text,
            vlm,
            model,
            scene,
            frame_store,
            sequence_id,
            timestamp_ns_start,
            timestamp_ns_end,
            log_fn,
        )
        if new_response is None:
            break
        response = new_response

    return "\n".join(history_text), tool_log


def run_tool_use(
    annotation: Annotation,
    images: list[np.ndarray],
    scene: SceneState,
    frame_store: SQLiteFrameStore,
    sequence_id: str,
    vlm: VLMClient,
    model: str,
    max_turns: int = 10,
    log_lines: list[str] | None = None,
) -> tuple[str, list[dict]]:
    if log_lines is not None:
        log_fn = log_lines.append
    else:
        log_fn = logger.info

    messages = _build_initial_messages(annotation, images)

    return _run_tool_loop(
        messages=messages,
        vlm=vlm,
        model=model,
        scene=scene,
        frame_store=frame_store,
        sequence_id=sequence_id,
        timestamp_ns_start=annotation.query_layer.query_timestamp_ns_start,
        timestamp_ns_end=annotation.query_layer.query_timestamp_ns_end,
        max_turns=max_turns,
        log_fn=log_fn,
    )

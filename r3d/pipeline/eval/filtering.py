# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Annotation filtering for eval pipelines.

Shared by generate_responses.py and eval_volume.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.config import EvalConfig
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.sqlite_store import SQLiteSceneStore

logger: logging.Logger = logging.getLogger(__name__)


def get_scene_state(
    cache: dict[str, SceneState],
    scene_store: SQLiteSceneStore,
    sequence_id: str,
    mesh_store: Any = None,
) -> SceneState:
    if sequence_id not in cache:
        cache[sequence_id] = SceneState(
            scene_store,
            sequence_id,
            mesh_store=mesh_store,
        )
    return cache[sequence_id]


def _filter_tracked(
    annotations: list[Annotation],
    scene_store: SQLiteSceneStore,
    scene_states: dict[str, SceneState],
) -> list[Annotation]:
    before = len(annotations)
    filtered = []
    for a in annotations:
        state = get_scene_state(
            scene_states,
            scene_store,
            a.identity_layer.sequence_id,
        )
        tracked_names = {obj.query_name.lower() for obj in state.get_all_objects()}
        if all(
            ref.canonical_name.lower() in tracked_names
            for ref in a.identity_layer.referenced_objects
        ):
            filtered.append(a)
    logger.info(
        f"Filtered to {len(filtered)}/{before} annotations "
        f"(all referenced objects tracked)"
    )
    return filtered


def _load_seg_eval(
    seg_eval_path: str,
) -> dict[str, list]:
    if seg_eval_path.startswith("manifold://"):
        raise ValueError(
            f"Manifold paths are not supported in OSS mode: {seg_eval_path}. "
            f"Please provide a local file path."
        )
    with open(seg_eval_path) as f:
        return json.load(f)


def _build_wrong_object_ratios(
    seg_data: dict[str, list],
) -> dict[tuple[str, str], float]:
    ratios: dict[tuple[str, str], float] = {}
    for seq_id, objects in seg_data.items():
        for obj in objects:
            tracked = obj["num_tracked_frames"]
            ratio = obj["num_wrong_object_frames"] / tracked if tracked > 0 else 0.0
            ratios[(seq_id, obj["query_name"].lower())] = ratio
    return ratios


def _build_mean_ious(
    seg_data: dict[str, list],
) -> dict[tuple[str, str], float]:
    ious: dict[tuple[str, str], float] = {}
    for seq_id, objects in seg_data.items():
        for obj in objects:
            ious[(seq_id, obj["query_name"].lower())] = obj["mean_iou"]
    return ious


def _filter_wrong_object(
    annotations: list[Annotation],
    ratios: dict[tuple[str, str], float],
    max_ratio: float,
) -> list[Annotation]:
    before = len(annotations)
    filtered = []
    for a in annotations:
        seq_id = a.identity_layer.sequence_id
        passes = True
        for ref in a.identity_layer.referenced_objects:
            key = (seq_id, ref.canonical_name.lower())
            if key not in ratios:
                passes = False
                break
            if ratios[key] >= max_ratio:
                passes = False
                break
        if passes:
            filtered.append(a)
    logger.info(
        f"Filtered to {len(filtered)}/{before} annotations "
        f"(wrong-object ratio < {max_ratio})"
    )
    return filtered


def _filter_min_iou(
    annotations: list[Annotation],
    ious: dict[tuple[str, str], float],
    min_iou: float,
) -> list[Annotation]:
    before = len(annotations)
    filtered = []
    for a in annotations:
        seq_id = a.identity_layer.sequence_id
        passes = True
        for ref in a.identity_layer.referenced_objects:
            key = (seq_id, ref.canonical_name.lower())
            if key not in ious:
                passes = False
                break
            if ious[key] < min_iou:
                passes = False
                break
        if passes:
            filtered.append(a)
    logger.info(
        f"Filtered to {len(filtered)}/{before} annotations (mean IoU >= {min_iou})"
    )
    return filtered


def filter_annotations(
    annotations: list[Annotation],
    config: EvalConfig,
    scene_store: SQLiteSceneStore,
    scene_states: dict[str, SceneState],
) -> list[Annotation]:
    if config.question_types is not None:
        annotations = [
            a
            for a in annotations
            if a.query_layer.question_type.value in config.question_types
        ]
    if config.require_tracked_objects:
        annotations = _filter_tracked(annotations, scene_store, scene_states)
    if config.segmentation_eval_path is not None:
        seg_data = _load_seg_eval(config.segmentation_eval_path)
        ratios = _build_wrong_object_ratios(seg_data)
        annotations = _filter_wrong_object(
            annotations, ratios, config.max_wrong_object_ratio
        )
        ious = _build_mean_ious(seg_data)
        annotations = _filter_min_iou(annotations, ious, config.min_segmentation_iou)
    if config.max_annotations is not None:
        annotations = annotations[: config.max_annotations]
    return annotations

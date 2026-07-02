# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Generate R3D (tool_use) VLM responses for spatial AI evaluation.

The model answers each question by calling spatial tools over the 3D scene.
For a plain RGB (+ optional mask overlay) baseline instead, use
``r3d.scripts.rgb_overlay_evals``.

Usage:
    python -m r3d.pipeline.scripts.generate_responses \
      --hf-model Qwen/Qwen3-VL-8B-Instruct \
      --dataset facebook/r3d-bench \
      --scene-db /tmp/eval/scene.db \
      --frames-dir /tmp/eval \
      --output-dir /tmp/eval/responses
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.config import EvalConfig, parse_eval_config
from r3d.pipeline.eval.filtering import filter_annotations, get_scene_state
from r3d.pipeline.eval.hf_vlm import HFVLMClient
from r3d.pipeline.eval.responses import Response, ResponseStore
from r3d.pipeline.eval.tool_use import run_tool_use
from r3d.pipeline.eval.vlm import VLMClient
from r3d.pipeline.frame_data import CameraIntrinsics
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore, SQLiteSceneStore
from r3d.utils.logging import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class AnnotationGroup:
    sequence_id: str
    start_ns: int
    end_ns: int
    annotations: list[Annotation] = field(default_factory=list)


def _group_annotations(annotations: list[Annotation]) -> list[AnnotationGroup]:
    groups: dict[tuple[str, int, int], AnnotationGroup] = {}
    for ann in annotations:
        seq_id = ann.identity_layer.sequence_id
        ql = ann.query_layer
        key = (seq_id, ql.query_timestamp_ns_start or 0, ql.query_timestamp_ns_end or 0)
        if key not in groups:
            groups[key] = AnnotationGroup(
                sequence_id=seq_id, start_ns=key[1], end_ns=key[2]
            )
        groups[key].annotations.append(ann)
    return list(groups.values())


def _load_frames_for_window(
    sequence_id: str,
    start_ns: int,
    end_ns: int,
    frame_store: SQLiteFrameStore,
    image_size: int,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[int], list[np.ndarray], CameraIntrinsics
]:
    _MAX_FRAMES = 16
    all_ts = frame_store.get_all_timestamps(sequence_id)
    window_ts = [ts for ts in all_ts if start_ns <= ts <= end_ns]
    if not window_ts:
        raise RuntimeError(f"No frames in [{start_ns}, {end_ns}] for {sequence_id}")
    if len(window_ts) > _MAX_FRAMES:
        indices = np.linspace(0, len(window_ts) - 1, _MAX_FRAMES, dtype=int)
        window_ts = [window_ts[i] for i in indices]
    frame_datas = [frame_store.load_frame(sequence_id, ts) for ts in window_ts]
    frames = [cv2.resize(fd.rgb, (image_size, image_size)) for fd in frame_datas]
    depths = [
        cv2.resize(
            fd.depth_map, (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )
        for fd in frame_datas
    ]
    poses = [fd.T_scene_device @ fd.T_device_camera for fd in frame_datas]
    intrinsics = frame_datas[0].intrinsics
    return frames, depths, window_ts, poses, intrinsics


def _process_tool_use_annotation(
    annotation: Annotation,
    frames: list[np.ndarray],
    sequence_id: str,
    frame_store: SQLiteFrameStore,
    scene_state: SceneState,
    vlm: VLMClient,
    config: EvalConfig,
) -> Response:
    """Process a single annotation with the tool_use strategy."""
    t_start = time.monotonic()
    log_buf: list[str] = []

    images = [] if config.no_images else frames
    response_text, tool_log = run_tool_use(
        annotation,
        images,
        scene_state,
        frame_store,
        sequence_id,
        vlm,
        config.model,
        log_lines=log_buf,
    )

    header = (
        f"=== [{annotation.annotation_id[:8]}] "
        f"{annotation.query_layer.question_type.value}: "
        f"{annotation.query_layer.question_text[:60]} ==="
    )
    logger.info(header)
    for line in log_buf:
        logger.info(line)
    logger.info(f"    Response: {response_text[:200]}")
    logger.info(f"    GT: {annotation.query_layer.gt_answer}")
    logger.info("=" * len(header))

    latency_s = time.monotonic() - t_start
    return Response(
        annotation_id=annotation.annotation_id,
        model=config.model,
        strategy="tool_use",
        response=response_text,
        tool_call_log=tool_log,
        created_ns=int(time.time() * 1e9),
        latency_s=latency_s,
    )


def _resolve_local_db(path: str, filename: str) -> Path:
    """Resolve a local database path (directory or file)."""
    p = Path(path)
    if p.is_dir():
        return p / filename
    return p


def main() -> None:
    config = parse_eval_config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir)

    if config.backend == "vllm":
        from r3d.pipeline.eval.vllm_client import VLLMClient

        vlm = VLLMClient(
            model_name=config.hf_model,
            data_parallel_size=config.data_parallel_size,
            max_model_len=config.max_model_len,
        )
    else:
        vlm = HFVLMClient(model_name=config.hf_model)

    scene_db = _resolve_local_db(config.scene_db, "scene.db")
    frames_db = Path(config.frames_dir) / "frames.db"
    if not frames_db.exists():
        raise RuntimeError(f"frames.db not found at {frames_db}")

    from r3d.pipeline.hf_dataset import load_annotation_store, load_mesh_store

    logger.info(f"Loading annotations from HF dataset: {config.dataset}")
    ann_store = load_annotation_store(config.dataset)
    scene_store = SQLiteSceneStore(scene_db, read_only=True)
    frame_store = SQLiteFrameStore(frames_db, read_only=True)

    logger.info(f"Loading meshes from HF dataset: {config.dataset}")
    mesh_store = load_mesh_store(config.dataset)

    scene_states: dict[str, SceneState] = {}
    all_annotations = ann_store.get_all_annotations()

    if config.annotation_ids_path:
        with open(config.annotation_ids_path) as f:
            allowed_ids = {line.strip() for line in f if line.strip()}
        before = len(all_annotations)
        all_annotations = [a for a in all_annotations if a.annotation_id in allowed_ids]
        logger.info(
            f"Filtered to {len(all_annotations)}/{before} annotations "
            f"(annotation ID list)"
        )

    annotations = filter_annotations(all_annotations, config, scene_store, scene_states)
    scene_states.clear()
    logger.info(
        f"Evaluating {len(annotations)} annotations (tool_use), model={config.model}"
    )

    response_store = ResponseStore(output_dir / "responses.db")
    groups = _group_annotations(annotations)
    logger.info(f"Grouped into {len(groups)} frame-sharing groups")

    all_tasks: list[
        tuple[Annotation, list[np.ndarray], list[int], str, SceneState]
    ] = []
    for group in groups:
        pending = [
            a for a in group.annotations if response_store.get(a.annotation_id) is None
        ]
        if not pending:
            continue

        needs_frames = not config.no_images
        if needs_frames:
            frames, _depth_maps, timestamps_ns, _poses, _intrinsics = (
                _load_frames_for_window(
                    group.sequence_id,
                    group.start_ns,
                    group.end_ns,
                    frame_store,
                    config.image_size,
                )
            )
        else:
            frames, timestamps_ns = [], []

        logger.info(
            f"  Group [{group.sequence_id}] [{group.start_ns}, {group.end_ns}]: "
            f"{len(frames)} frames, {len(pending)} pending annotations"
        )

        scene_state = get_scene_state(
            scene_states,
            scene_store,
            group.sequence_id,
            mesh_store=mesh_store,
        )
        for ann in pending:
            all_tasks.append(
                (ann, frames, timestamps_ns, group.sequence_id, scene_state)
            )

    def _process_one(task: tuple) -> Response:
        ann, frames, timestamps_ns, seq_id, scene_state = task
        return _process_tool_use_annotation(
            ann,
            frames,
            seq_id,
            frame_store,
            scene_state,
            vlm,
            config,
        )

    concurrency = config.concurrency
    total = 0
    try:
        if concurrency <= 1 or config.data_parallel_size <= 1:
            for task in all_tasks:
                resp = _process_one(task)
                response_store.write(resp)
                total += 1
                logger.info(f"    DONE {task[0].annotation_id} ({resp.latency_s:.1f}s)")
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                future_to_ann = {
                    pool.submit(_process_one, task): task[0] for task in all_tasks
                }
                for future in concurrent.futures.as_completed(future_to_ann):
                    ann = future_to_ann[future]
                    resp = future.result()
                    response_store.write(resp)
                    total += 1
                    logger.info(f"    DONE {ann.annotation_id} ({resp.latency_s:.1f}s)")

        logger.info(f"Wrote {total} responses")
    finally:
        if hasattr(vlm, "shutdown"):
            vlm.shutdown()
        response_store.close()
        ann_store.close()
        scene_store.close()
        frame_store.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()

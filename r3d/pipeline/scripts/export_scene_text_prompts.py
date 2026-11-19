# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Export ``scene_text`` prompts to JSONL for offline inference.

Builds exactly the prompts ``generate_responses --strategy scene_text`` would
send (annotations and SAM3D meshes from the R3D-Bench HF dataset, scene and
frames from local SQLite stores), and counts their tokens with the served
model's tokenizer. With ``--validate-responses-db`` every tool result logged by
a tool_use run is compared against the scene table cells; any mismatch is an
error (the table must carry the same values the tools returned).

Usage:
    python -m r3d.pipeline.scripts.export_scene_text_prompts \
      --dataset facebook/r3d-bench --annotation-ids ids.txt \
      --scene-db scene/scene.db --frames-dir frames \
      --tokenizer Qwen/Qwen3.6-35B-A3B --output prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path

from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.config import DEFAULT_DATASET
from r3d.pipeline.eval.prompt_stats import PromptTokenCounter
from r3d.pipeline.eval.scene_text import (
    build_scene_text_messages,
    messages_to_chat,
    SCENE_TEXT_STRATEGY,
    SceneTableBuilder,
)
from r3d.pipeline.hf_dataset import load_annotation_store, load_mesh_store
from r3d.pipeline.scene_state import SceneState
from r3d.pipeline.stores.base import MeshStore
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore, SQLiteSceneStore
from r3d.utils.logging import setup_logging

logger: logging.Logger = logging.getLogger(__name__)

_OBJECT_TOOL_CELLS: dict[str, str] = {
    "get_position": "position",
    "get_object_size": "size",
    "get_object_volume": "volume",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="R3D-Bench HF repo")
    p.add_argument("--annotation-ids", required=True)
    p.add_argument("--scene-db", required=True)
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--tokenizer", required=True, help="HF model id or local dir")
    p.add_argument("--output", required=True)
    p.add_argument("--validate-responses-db", default=None)
    return p.parse_args()


def _load_annotations(dataset: str, ids_path: str) -> list[Annotation]:
    ids = {line.strip() for line in Path(ids_path).read_text().splitlines()}
    ids.discard("")
    store = load_annotation_store(dataset)
    anns = [a for a in store.get_all_annotations() if a.annotation_id in ids]
    store.close()
    found = {a.annotation_id for a in anns}
    if found != ids:
        raise RuntimeError(f"{len(ids - found)} annotation IDs not in {dataset}")
    return sorted(anns, key=lambda a: a.annotation_id)


class _Scenes:
    def __init__(self, scene_db: str, mesh_store: MeshStore) -> None:
        self.store = SQLiteSceneStore(Path(scene_db), read_only=True)
        self.mesh_store = mesh_store
        self._states: dict[str, SceneState] = {}

    def get(self, sequence_id: str) -> SceneState:
        if sequence_id not in self._states:
            self._states[sequence_id] = SceneState(
                self.store, sequence_id, mesh_store=self.mesh_store
            )
        return self._states[sequence_id]


def _export(
    anns: list[Annotation],
    scenes: _Scenes,
    builder: SceneTableBuilder,
    counter: PromptTokenCounter,
    output: Path,
) -> None:
    with open(output, "w") as f:
        for i, ann in enumerate(anns):
            ql = ann.query_layer
            scene = scenes.get(ann.identity_layer.sequence_id)
            table, num_objects = builder.build(
                scene, ql.query_timestamp_ns_start, ql.query_timestamp_ns_end
            )
            chat = messages_to_chat(build_scene_text_messages(ann, table))
            record = {
                "annotation_id": ann.annotation_id,
                "question_type": ql.question_type.value,
                "strategy": SCENE_TEXT_STRATEGY,
                "num_objects": num_objects,
                "prompt_tokens": counter.prompt_tokens(chat),
                "messages": chat,
            }
            f.write(json.dumps(record) + "\n")
            if i % 200 == 0:
                logger.info(f"  exported {i + 1}/{len(anns)}")
    logger.info(f"Wrote {len(anns)} prompts to {output}")


def _row_cells(row: str) -> dict[str, str]:
    _head, *cells = row.split(" | ")
    return dict(cell.split(": ", 1) for cell in cells)


def _expected_for_entry(
    entry: dict, table: str, scene: SceneState, builder: SceneTableBuilder
) -> str | None:
    """Table text the logged tool result must equal (None: not in the table)."""
    tool, args = entry["tool"], entry["args"]
    if tool == "list_objects":
        lines = table.splitlines()[:-1]
        return "\n".join([lines[0]] + [ln.split(" | ")[0] for ln in lines[1:]])
    if tool == "get_my_position":
        return table.splitlines()[-1].removeprefix("Your current position: ")
    if tool not in _OBJECT_TOOL_CELLS or not args or not args[0].isdigit():
        return None
    if scene.get_object(int(args[0])) is None:
        return None
    row = builder.object_row(scene, int(args[0]))
    return _row_cells(row)[_OBJECT_TOOL_CELLS[tool]]


def _validate(
    responses_db: str,
    anns: list[Annotation],
    scenes: _Scenes,
    builder: SceneTableBuilder,
) -> None:
    by_id = {a.annotation_id: a for a in anns}
    conn = sqlite3.connect(responses_db)
    rows = conn.execute("SELECT annotation_id, tool_call_log FROM responses")
    counts: Counter[tuple[str, bool]] = Counter()
    mismatches: list[tuple[str, str, str, str]] = []
    for aid, log in rows:
        ann = by_id[aid]
        ql = ann.query_layer
        scene = scenes.get(ann.identity_layer.sequence_id)
        table, _ = builder.build(
            scene, ql.query_timestamp_ns_start, ql.query_timestamp_ns_end
        )
        for entry in json.loads(log) if log else []:
            expected = _expected_for_entry(entry, table, scene, builder)
            if expected is None:
                continue
            ok = expected == entry["result"]
            counts[(entry["tool"], ok)] += 1
            if not ok:
                mismatches.append((aid, entry["tool"], entry["result"], expected))
    conn.close()
    for key, n in sorted(counts.items()):
        logger.info(f"  validate {key[0]} match={key[1]}: {n}")
    if mismatches:
        for m in mismatches[:10]:
            logger.error(f"  MISMATCH {m}")
        raise RuntimeError(f"{len(mismatches)} tool results differ from the table")
    logger.info("Validation passed: every logged tool result equals the table")


def main() -> None:
    args = _parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output.parent)
    anns = _load_annotations(args.dataset, args.annotation_ids)
    scenes = _Scenes(args.scene_db, load_mesh_store(args.dataset))
    frame_store = SQLiteFrameStore(Path(args.frames_dir) / "frames.db", read_only=True)
    builder = SceneTableBuilder(frame_store)
    if args.validate_responses_db:
        _validate(args.validate_responses_db, anns, scenes, builder)
    _export(anns, scenes, builder, PromptTokenCounter(args.tokenizer), output)
    frame_store.close()
    scenes.store.close()
    scenes.mesh_store.close()


if __name__ == "__main__":
    main()

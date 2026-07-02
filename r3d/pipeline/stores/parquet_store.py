# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Parquet-backed read-only stores for the R3D-Bench dataset.

These load the parquet configs published at facebook/r3d-bench (qa_annotations,
segmentations, meshes) directly — no SQLite intermediary. Each store builds
in-memory indexes at construction; the dataset is small enough to fit in RAM.

Writer methods raise NotImplementedError: parquet assets are immutable inputs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from r3d.data_gen.utils.annotation_schema import (
    Annotation,
    DisambiguationContext,
    DisambiguationLayer,
    DisambiguationMethod,
    EvalMetric,
    EvalMode,
    IdentityLayer,
    QueryLayer,
    QuestionType,
    ReferencedObject,
    ReleaseType,
)
from r3d.pipeline.segmentation import FrameSegmentation, ObjectSegmentation
from r3d.pipeline.stores.base import (
    AnnotationStore,
    MeshStore,
    ObjectMesh,
    SegmentationStore,
)
from r3d.utils.rle import decode_rle_to_mask


def _read_rows(parquet_path: str) -> list[dict[str, Any]]:
    return pq.read_table(parquet_path).to_pylist()


class ParquetAnnotationStore(AnnotationStore):
    """Read-only AnnotationStore backed by the qa_annotations parquet config."""

    def __init__(self, parquet_path: str) -> None:
        self._rows = _read_rows(parquet_path)

    def _to_annotation(self, r: dict[str, Any]) -> Annotation:
        refs = [
            ReferencedObject(
                object_id=o["object_id"],
                adt_instance_id=o["adt_instance_id"],
                adt_instance_name=o["adt_instance_name"],
                prototype_name=o["prototype_name"],
                canonical_name=o["canonical_name"],
                adt_mesh_path=o["adt_mesh_path"],
                is_dynamic=bool(o["is_dynamic"]),
                reference_frame_idx=o["reference_frame_idx"],
                reference_timestamp_ns=o["reference_timestamp_ns"],
            )
            for o in (r["referenced_objects"] or [])
        ]
        identity_layer = IdentityLayer(
            sequence_id=r["sequence_id"],
            release_type=ReleaseType(r["release_type"]),
            referenced_objects=refs,
        )
        query_layer = QueryLayer(
            question_type=QuestionType(r["question_type"]),
            question_text=r["question_text"],
            gt_answer=r["gt_answer"],
            gt_answer_type=r["gt_answer_type"],
            gt_answer_unit=r["gt_answer_unit"],
            gt_computation_method=r["gt_computation_method"],
            eval_mode=EvalMode(r["eval_mode"]),
            eval_metric=EvalMetric(r["eval_metric"]),
            query_timestamp_ns_start=r["timestamp_ns_start"],
            query_timestamp_ns_end=r["timestamp_ns_end"],
        )
        disambiguation_layer = DisambiguationLayer(
            method=DisambiguationMethod(r["disambiguation_method"]),
            disambiguation_context=DisambiguationContext(
                spatial_description=r["spatial_description"],
                temporal_description=r["temporal_description"],
            ),
        )
        return Annotation(
            annotation_id=r["annotation_id"],
            identity_layer=identity_layer,
            disambiguation_layer=disambiguation_layer,
            query_layer=query_layer,
        )

    def get_all_annotations(self) -> list[Annotation]:
        return [self._to_annotation(r) for r in self._rows]

    def get_annotations_by_sequence(self, sequence_id: str) -> list[Annotation]:
        return [
            self._to_annotation(r)
            for r in self._rows
            if r["sequence_id"] == sequence_id
        ]

    def get_all_sequence_ids(self) -> list[str]:
        return sorted({r["sequence_id"] for r in self._rows})

    def get_object_sequence_pairs(self) -> list[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for r in self._rows:
            for o in r["referenced_objects"] or []:
                pairs.add((o["canonical_name"], r["sequence_id"]))
        return sorted(pairs)

    def write_annotation(self, annotation: Annotation) -> None:
        raise NotImplementedError("ParquetAnnotationStore is read-only")

    def write_gt_bbox(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("ParquetAnnotationStore is read-only")

    def get_gt_bboxes(
        self, annotation_id: str, object_position: int
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("gt_bboxes is a separate config; not loaded here")

    def flush_gt_bboxes(self) -> None:
        pass

    def get_nearest_gt_bbox(
        self, annotation_id: str, object_position: int, timestamp_ns: int
    ) -> dict[str, Any] | None:
        raise NotImplementedError("gt_bboxes is a separate config; not loaded here")

    def close(self) -> None:
        pass


class ParquetSegmentationStore(SegmentationStore):
    """Read-only SegmentationStore backed by the segmentations parquet config."""

    def __init__(self, parquet_path: str) -> None:
        rows = _read_rows(parquet_path)
        # index: (sequence_id, timestamp_ns) -> list[row]; and helper maps
        self._by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._queries: dict[str, set[str]] = {}
        self._query_ts: dict[tuple[str, str], set[int]] = {}
        self._obj_queries: dict[tuple[str, int], set[str]] = {}
        for r in rows:
            seq = r["sequence_id"]
            ts = r["timestamp_ns"]
            qn = r["query_name"]
            self._by_frame.setdefault((seq, ts), []).append(r)
            self._queries.setdefault(seq, set()).add(qn)
            self._query_ts.setdefault((seq, qn), set()).add(ts)
            self._obj_queries.setdefault((seq, r["object_id"]), set()).add(qn)

    def _to_object(self, r: dict[str, Any]) -> ObjectSegmentation:
        rle = json.loads(r["mask_rle"])
        bbox = r["bbox_2d"]
        bbox_2d = (
            np.array(json.loads(bbox), dtype=np.float64) if bbox is not None else None
        )
        return ObjectSegmentation(
            object_id=r["object_id"],
            query_name=r["query_name"],
            bbox_2d=bbox_2d,
            mask=decode_rle_to_mask(rle),
            mask_rle=rle,
            score=r["score"],
            obj_ptr=None,
            min_depth_m=r["min_depth_m"],
        )

    def get_segmentation(
        self, sequence_id: str, timestamp_ns: int, query_name: str | None = None
    ) -> FrameSegmentation:
        rows = self._by_frame.get((sequence_id, timestamp_ns), [])
        if query_name is not None:
            rows = [r for r in rows if r["query_name"] == query_name]
        objects = {r["object_id"]: self._to_object(r) for r in rows}
        return FrameSegmentation(timestamp_ns=timestamp_ns, objects=objects)

    def get_all_query_names(self, sequence_id: str) -> list[str]:
        return sorted(self._queries.get(sequence_id, set()))

    def get_query_names_for_object(self, sequence_id: str, object_id: int) -> list[str]:
        return sorted(self._obj_queries.get((sequence_id, object_id), set()))

    def get_segmented_timestamps(self, sequence_id: str, query_name: str) -> list[int]:
        return sorted(self._query_ts.get((sequence_id, query_name), set()))

    def get_all_sequence_ids(self) -> list[str]:
        return sorted(self._queries.keys())

    def get_pending_timestamps(self, sequence_id: str, query_name: str) -> list[int]:
        return []

    def register_query(
        self, sequence_id: str, query_name: str, timestamps: list[int]
    ) -> None:
        raise NotImplementedError("ParquetSegmentationStore is read-only")

    def write_segmentation(
        self, sequence_id: str, timestamp_ns: int, obj_seg: ObjectSegmentation
    ) -> None:
        raise NotImplementedError("ParquetSegmentationStore is read-only")

    def mark_segmented(
        self, sequence_id: str, timestamp_ns: int, query_name: str
    ) -> None:
        raise NotImplementedError("ParquetSegmentationStore is read-only")

    def close(self) -> None:
        pass


class ParquetMeshStore(MeshStore):
    """Read-only MeshStore backed by the meshes parquet config (GLB bytes embedded).

    GLB geometry is materialized to a temp directory on demand so downstream
    code that expects a file path (volume tool) works unchanged.
    """

    def __init__(self, parquet_path: str) -> None:
        rows = _read_rows(parquet_path)
        self._by_obj: dict[tuple[str, str], dict[str, Any]] = {
            (r["sequence_id"], r["object_name"]): r for r in rows
        }
        self._tmp = Path(tempfile.mkdtemp(prefix="r3d_meshes_"))
        self._materialized: dict[tuple[str, str], str] = {}

    def _to_mesh(self, r: dict[str, Any], mesh_path: str) -> ObjectMesh:
        return ObjectMesh(
            sequence_id=r["sequence_id"],
            object_name=r["object_name"],
            annotation_id=r["annotation_id"],
            adt_instance_name=r["adt_instance_name"],
            mesh_path=mesh_path,
            source_timestamp_ns=r["source_timestamp_ns"],
            num_vertices=r["num_vertices"],
            num_faces=r["num_faces"],
            metric_scale_x=r["metric_scale_x"],
            metric_scale_y=r["metric_scale_y"],
            metric_scale_z=r["metric_scale_z"],
            created_ns=0,
        )

    def get_mesh_abs_path(self, sequence_id: str, object_name: str) -> str | None:
        key = (sequence_id, object_name)
        if key not in self._by_obj:
            return None
        if key not in self._materialized:
            safe = f"{sequence_id}__{object_name}".replace("/", "_").replace(" ", "_")
            path = self._tmp / f"{safe}.glb"
            path.write_bytes(self._by_obj[key]["glb"])
            self._materialized[key] = str(path)
        return self._materialized[key]

    def get_mesh(self, sequence_id: str, object_name: str) -> ObjectMesh | None:
        key = (sequence_id, object_name)
        if key not in self._by_obj:
            return None
        path = self.get_mesh_abs_path(sequence_id, object_name)
        return self._to_mesh(self._by_obj[key], path)

    def get_all_meshes(self) -> list[ObjectMesh]:
        return [
            self._to_mesh(r, self.get_mesh_abs_path(seq, obj))
            for (seq, obj), r in self._by_obj.items()
        ]

    def get_all_sequence_ids(self) -> list[str]:
        return sorted({seq for (seq, _) in self._by_obj})

    def write_mesh(self, mesh: ObjectMesh) -> None:
        raise NotImplementedError("ParquetMeshStore is read-only")

    def close(self) -> None:
        pass

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""V2 question generation pipeline.

Clean-break v2 orchestrator.  Outputs Annotation natively -- no
_PartialAnnotationSample, no string-answer round-tripping, no v1
backward compatibility.

Key design: disambiguation is orthogonal to question type.
All objects are resolved to ObjectRefs *before* generators run.

Usage:
    python -m r3d.data_gen.generate_qa \\
        --sequences /path/to/seq1 /path/to/seq2 --output-dir ./output
"""

from __future__ import annotations

import argparse
import importlib
import logging
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from projectaria_tools.core.stream_id import StreamId
from r3d.data_gen.generators.base import GeneratorConfig, get_obj_name, ObjectRef
from r3d.data_gen.generators.complex import generate_volume_estimation
from r3d.data_gen.generators.disambiguation import resolve_object_references
from r3d.data_gen.generators.distance import generate_how_far, generate_how_far_from_me
from r3d.data_gen.generators.multihop import (
    generate_gap_fit,
    generate_how_much_longer_dim,
    generate_how_much_taller,
    generate_nearest_from_set,
    generate_pour_leftover,
    generate_pour_room_left,
    generate_top_higher,
    generate_total_fly_distance,
    generate_total_walk_distance,
    generate_which_longer_dim,
    generate_which_taller,
)
from r3d.data_gen.generators.size import generate_how_long
from r3d.data_gen.generators.visualize import render_annotation_video
from r3d.data_gen.utils.annotation_schema import (
    Annotation,
    DisambiguationMethod,
    GenerationManifest,
    make_deterministic_id,
    QuestionType,
    ReleaseType,
)
from r3d.data_gen.utils.scene_helpers import get_video_object_info
from r3d.data_gen.utils.sequence import get_valid_timestamps, load_sequence_data
from r3d.pipeline.stores.sqlite_store import SQLiteAnnotationStore, SQLiteFrameStore

logger: logging.Logger = logging.getLogger(__name__)

# Import from filter.global using importlib since 'global' is a Python keyword
_filter_global_module = importlib.import_module("r3d.data_gen.filter.global")
filter_global_objects = _filter_global_module.filter_global_objects
get_globally_unique_objects = _filter_global_module.get_globally_unique_objects
get_multi_instance_groups = _filter_global_module.get_multi_instance_groups


# Default camera stream -- RGB camera
DEFAULT_STREAM_ID = StreamId("214-1")


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

# Type alias for generator functions.  Each generator receives a common
# set of arguments and returns a list of Annotation.
GeneratorFn = Callable[..., list[Annotation]]


def _register_generators() -> dict[QuestionType, GeneratorFn]:
    """Build the generator registry mapping QuestionType -> generator function."""
    return {
        # Distance
        QuestionType.GLOBAL_HOW_FAR: generate_how_far,
        QuestionType.GLOBAL_HOW_FAR_FROM_ME: generate_how_far_from_me,
        # Size
        QuestionType.GLOBAL_HOW_LONG: generate_how_long,
        # Complex reasoning
        QuestionType.VOLUME_ESTIMATION: generate_volume_estimation,
        # Multi-hop reasoning
        QuestionType.GAP_FIT: generate_gap_fit,
        QuestionType.NEAREST_FROM_SET: generate_nearest_from_set,
        QuestionType.TOTAL_WALK_DISTANCE: generate_total_walk_distance,
        QuestionType.TOTAL_FLY_DISTANCE: generate_total_fly_distance,
        QuestionType.WHICH_TALLER: generate_which_taller,
        QuestionType.WHICH_LONGER_DIM: generate_which_longer_dim,
        QuestionType.HOW_MUCH_TALLER: generate_how_much_taller,
        QuestionType.HOW_MUCH_LONGER_DIM: generate_how_much_longer_dim,
        QuestionType.TOP_HIGHER: generate_top_higher,
        QuestionType.POUR_ROOM_LEFT: generate_pour_room_left,
        QuestionType.POUR_LEFTOVER: generate_pour_leftover,
    }


GENERATOR_REGISTRY = _register_generators()

# Question types whose ground truth is computed from the ADT object-library
# meshes at generation time (functional/cavity volume).
MESH_REQUIRING_TYPES: frozenset[QuestionType] = frozenset(
    {
        QuestionType.VOLUME_ESTIMATION,
        QuestionType.POUR_ROOM_LEFT,
        QuestionType.POUR_LEFTOVER,
    }
)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_v2_annotations(
    gt_provider: object,
    sequence_name: str,
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    question_types: list[QuestionType] | None = None,
    disambiguation_method: DisambiguationMethod | None = None,
    require_unique_names: bool = False,
    object_filter: str = "all",
    min_fov_degrees: float = 2.0,
    max_distance: float = 4.0,
    min_visibility_ratio: float = 0.3,
    min_visible_frames: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list[Annotation], list[ObjectRef]]:
    """Generate v2 annotations for a single sequence.

    Pipeline:
    1. Build VideoObjectInfo (reuses existing infrastructure).
    2. Filter objects (visibility, distance, FOV).
    3. Build unique/multi-instance groups.
    4. Resolve disambiguation -> list[ObjectRef].
    5. For each question type: call the registered generator.

    Returns:
        (annotations, object_refs) tuple.
    """
    if question_types is None:
        question_types = list(GENERATOR_REGISTRY.keys())

    # 1. Build VideoObjectInfo
    if verbose:
        logger.info(f"Building object info for {sequence_name}...")
    all_video_objects = get_video_object_info(gt_provider, timestamps_ns, stream_id)

    # 2. Filter objects
    if verbose:
        logger.info(f"Filtering objects ({len(all_video_objects)} total)...")
    filtered_objects = filter_global_objects(
        objects=all_video_objects,
        gt_provider=gt_provider,
        timestamps_ns=timestamps_ns,
        stream_id=stream_id,
        filter_fps=1.0,
        unique=False,
        min_fov_degrees=min_fov_degrees,
        max_distance=max_distance,
        require_2d_bbox=True,
        min_visibility_ratio=min_visibility_ratio,
        min_visible_frames=min_visible_frames,
    )
    if verbose:
        logger.info(f"  {len(filtered_objects)} objects pass filter")

    if not filtered_objects:
        return [], []

    if require_unique_names:
        name_counts = Counter(get_obj_name(obj) for obj in filtered_objects)
        before = len(filtered_objects)
        filtered_objects = [
            obj for obj in filtered_objects if name_counts[get_obj_name(obj)] == 1
        ]
        if verbose:
            logger.info(
                f"  {before - len(filtered_objects)} dropped for duplicate names, "
                f"{len(filtered_objects)} remain"
            )

    if not filtered_objects:
        return [], []

    if object_filter == "static":
        before = len(filtered_objects)
        filtered_objects = [obj for obj in filtered_objects if obj.is_static]
        if verbose:
            logger.info(
                f"  Object filter 'static': {before} -> {len(filtered_objects)} objects"
            )
    elif object_filter == "has-dynamic":
        before = len(filtered_objects)
        filtered_objects = [obj for obj in filtered_objects if not obj.is_static]
        if verbose:
            logger.info(
                f"  Object filter 'has-dynamic': {before} -> {len(filtered_objects)} objects"
            )
    elif object_filter != "all":
        raise ValueError(f"Unknown object_filter: {object_filter}")

    if not filtered_objects:
        return [], []

    # 3. Build groups
    objects_dict = {obj.instance_id: obj for obj in filtered_objects}
    unique_objects = get_globally_unique_objects(objects_dict)
    multi_instance_groups = get_multi_instance_groups(objects_dict)

    if verbose:
        logger.info(
            f"{len(unique_objects)} unique, "
            f"{len(multi_instance_groups)} multi-instance groups"
        )

    # 4. Resolve disambiguation
    if verbose:
        logger.info("Resolving object references...")
    all_refs = resolve_object_references(
        objects=filtered_objects,
        unique_objects=unique_objects,
        multi_instance_groups=multi_instance_groups,
        gt_provider=gt_provider,
        timestamps_ns=timestamps_ns,
        stream_id=stream_id,
        allowed_method=disambiguation_method,
    )
    if verbose:
        method_counts = Counter(r.disambiguation_method.value for r in all_refs)
        logger.info(f"  {len(all_refs)} ObjectRefs: {dict(method_counts)}")

    # 5. Generate questions
    all_annotations: list[Annotation] = []

    for qt in question_types:
        gen_fn = GENERATOR_REGISTRY[qt]
        if gen_fn is None:
            raise ValueError(f"No generator registered for {qt.value}")

        rng = random.Random(f"{seed}_{sequence_name}_{qt.value}")

        annotations = gen_fn(
            gt_provider=gt_provider,
            object_refs=all_refs,
            all_refs=all_refs,
            timestamps_ns=timestamps_ns,
            stream_id=stream_id,
            config=config,
            rng=rng,
        )
        all_annotations.extend(annotations)

        if verbose:
            logger.info(f"{qt.value}: {len(annotations)} questions")

    return all_annotations, all_refs


# ---------------------------------------------------------------------------
# Pointing mask persistence
# ---------------------------------------------------------------------------


def _populate_pointing_references(
    annotations: list[Annotation],
    all_refs: list[ObjectRef],
) -> None:
    """Set each pointing object's reference frame (index + timestamp).

    The pointing mask is not stored; it is derived at use time from the ADT
    ground-truth segmentation for the object (adt_instance_id) at its
    reference timestamp.
    """
    ref_by_instance: dict[int, ObjectRef] = {}
    for ref in all_refs:
        existing = ref_by_instance.get(ref.obj.instance_id)
        if existing is None or (
            ref.pointing_mask is not None and existing.pointing_mask is None
        ):
            ref_by_instance[ref.obj.instance_id] = ref

    count = 0
    for ann in annotations:
        if ann.disambiguation_layer.method != DisambiguationMethod.POINTING:
            continue

        for ref_obj in ann.identity_layer.referenced_objects:
            ref = ref_by_instance.get(ref_obj.adt_instance_id)
            if ref is None or ref.pointing_mask is None:
                raise RuntimeError(
                    f"Pointing annotation for '{ref_obj.canonical_name}' "
                    f"(instance {ref_obj.adt_instance_id}) has no mask"
                )

            if not ref.pointing_mask.any():
                raise RuntimeError(
                    f"Pointing mask for '{ref_obj.canonical_name}' "
                    f"(instance {ref_obj.adt_instance_id}) is all False -- "
                    f"object not visible at reference frame"
                )

            ref_obj.reference_frame_idx = ref.reference_frame_idx
            ref_obj.reference_timestamp_ns = ref.reference_timestamp_ns
            count += 1

    if count > 0:
        logger.info(f"Set pointing reference frames for {count} objects")


# ---------------------------------------------------------------------------
# GT bbox population
# ---------------------------------------------------------------------------


def _populate_gt_bboxes(
    store: SQLiteAnnotationStore,
    annotations: list[Annotation],
    gt_provider: object,
    timestamps_ns: list[int],
) -> None:
    """Write per-frame GT 3D bboxes for each referenced object in each annotation."""

    for ann in annotations:
        ql = ann.query_layer
        start_ns = ql.query_timestamp_ns_start or timestamps_ns[0]
        end_ns = ql.query_timestamp_ns_end or timestamps_ns[-1]
        window_ts = [ts for ts in timestamps_ns if start_ns <= ts <= end_ns]

        for obj_pos, ref_obj in enumerate(ann.identity_layer.referenced_objects):
            instance_id = ref_obj.adt_instance_id
            written = 0
            for ts in window_ts:
                bbox3d_with_dt = (
                    gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(ts)
                )
                if not bbox3d_with_dt.is_valid():
                    continue
                bboxes3d = bbox3d_with_dt.data()
                if instance_id not in bboxes3d:
                    continue

                bbox3d = bboxes3d[instance_id]
                obb_transform = bbox3d.transform_scene_object.to_matrix()
                obb_aabb = np.array([float(x) for x in bbox3d.aabb])

                store.write_gt_bbox(
                    ann.annotation_id,
                    obj_pos,
                    ts,
                    obb_aabb,
                    obb_transform,
                )
                written += 1

            store.flush_gt_bboxes()
            logger.info(
                f"  {ref_obj.canonical_name}: {written} GT bboxes "
                f"({len(window_ts)} frames)"
            )


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------


def _render_sequence_videos(
    annotations: list[Annotation],
    all_refs: list[ObjectRef],
    gt_provider: object,
    timestamps_ns: list[int],
    output_dir: Path,
    video_paths: dict[str, str],
) -> None:
    viz_dir = output_dir / "videos"
    logger.info(f"Rendering {len(annotations)} annotation videos to {viz_dir}")
    frame_cache: dict[int, np.ndarray] = {}
    ref_by_instance: dict[int, ObjectRef] = {}
    for ref in all_refs:
        existing = ref_by_instance.get(ref.obj.instance_id)
        if existing is None or (
            ref.pointing_mask is not None and existing.pointing_mask is None
        ):
            ref_by_instance[ref.obj.instance_id] = ref

    for ann_idx, ann in enumerate(annotations):
        logger.info(
            f"  Video {ann_idx + 1}/{len(annotations)}: "
            f"{ann.query_layer.question_type.value} {ann.annotation_id[:8]}"
        )
        ann_refs = []
        for ref_obj in ann.identity_layer.referenced_objects:
            ref = ref_by_instance.get(ref_obj.adt_instance_id)
            if ref is None:
                ref = next(
                    (
                        r
                        for r in all_refs
                        if r.obj.instance_id == ref_obj.adt_instance_id
                    ),
                    None,
                )
            if ref is not None:
                ann_refs.append(ref)

        if not ann_refs:
            raise RuntimeError(
                f"No matching ObjectRefs for annotation {ann.annotation_id}"
            )

        video_path = render_annotation_video(
            annotation=ann,
            refs=ann_refs,
            gt_provider=gt_provider,
            timestamps_ns=timestamps_ns,
            stream_id=DEFAULT_STREAM_ID,
            output_dir=viz_dir,
            frame_cache=frame_cache,
        )

        if video_path is not None:
            video_paths[ann.annotation_id] = str(video_path)
            logger.info(f"  {ann.annotation_id[:8]}: {video_path}")

    logger.info(f"Rendered {len(video_paths)} videos to {viz_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2 question generation pipeline for R3D dataset"
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        required=True,
        help="Local paths to ADT sequence directories.",
    )
    parser.add_argument(
        "--frames-db",
        type=str,
        default=None,
        help="Path to frames.db. "
        "When provided, timestamps are loaded from frames.db instead of "
        "the full ADT sequence. This ensures annotations reference exact "
        "frame timestamps that exist in the pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for annotations",
    )
    parser.add_argument(
        "--object-library",
        type=str,
        default=None,
        help="Path to the local ADT object library directory containing "
        "{instance_name}/3d-asset.glb per object. Required for volume-based "
        "question types (volume_estimation, pour_room_left, pour_leftover).",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        help="Question types to generate (default: all)",
    )
    parser.add_argument(
        "--disambiguation-method",
        type=str,
        default=None,
        help="Disambiguation method to use (one of: global, pointing). "
        "All annotations will use this method.",
    )
    parser.add_argument(
        "--require-unique-names",
        action="store_true",
        help="Only keep objects whose canonical name is unique in the scene. "
        "Objects sharing a name with another are dropped.",
    )
    parser.add_argument(
        "--object-filter",
        choices=["all", "static", "has-dynamic"],
        default="all",
        help="Filter objects by movement status. "
        "'static': only objects that did not move. "
        "'has-dynamic': only objects that moved (>= 10cm displacement). "
        "(default: all)",
    )
    parser.add_argument(
        "--questions-per-type",
        type=int,
        default=5,
        help="Maximum questions per type (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--release-type",
        choices=["full", "lite"],
        default="full",
        help="ADT release type (default: full)",
    )
    parser.add_argument(
        "--min-fov-degrees",
        type=float,
        default=2.0,
        help="Minimum angular size filter (default: 2.0)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=4.0,
        help="Maximum distance filter in meters (default: 4.0)",
    )
    parser.add_argument(
        "--min-visibility-ratio",
        type=float,
        default=0.3,
        help="Minimum 2D bbox visibility ratio (default: 0.3)",
    )
    parser.add_argument(
        "--min-visible-frames",
        type=int,
        default=1,
        help="Minimum number of sampled frames where an object must be visible "
        "(default: 1). Higher values filter out briefly-seen objects.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render visualization videos into output-dir/videos/",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args(argv)


def _load_timestamps_from_frames_db(db_path: Path, sequence_id: str) -> list[int]:
    """Load timestamps for a specific sequence from a local frames.db."""
    store = SQLiteFrameStore(db_path)
    timestamps = store.get_all_timestamps(sequence_id)
    store.close()
    logger.info(f"[{sequence_id}] Loaded {len(timestamps)} timestamps from {db_path}")
    return timestamps


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args(argv)

    question_types: list[QuestionType] | None = None
    if args.question_types:
        question_types = [QuestionType(qt) for qt in args.question_types]

    effective_types = question_types or list(GENERATOR_REGISTRY.keys())
    if args.object_library is None:
        needs_mesh = sorted(
            qt.value for qt in effective_types if qt in MESH_REQUIRING_TYPES
        )
        if needs_mesh:
            raise SystemExit(
                f"--object-library is required for volume-based question types "
                f"({', '.join(needs_mesh)}). Provide the ADT object library "
                f"directory, or restrict --question-types to exclude them."
            )

    disambiguation_method: DisambiguationMethod | None = None
    if args.disambiguation_method:
        disambiguation_method = DisambiguationMethod(args.disambiguation_method)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "annotations.db"
    store = SQLiteAnnotationStore(db_path)

    all_annotations: list[Annotation] = []
    all_sequence_names: list[str] = []
    video_paths: dict[str, str] = {}

    frames_db_local: Path | None = None
    if args.frames_db:
        frames_db_local = Path(args.frames_db)
        if frames_db_local.is_dir():
            frames_db_local = frames_db_local / "frames.db"
        if not frames_db_local.exists():
            raise RuntimeError(f"frames.db not found at {frames_db_local}")

    for sequence_path in args.sequences:
        sequence_name = Path(sequence_path).name

        logger.info(f"=== Processing sequence: {sequence_name} ===")
        all_sequence_names.append(sequence_name)

        gt_provider, _data_paths = load_sequence_data(
            sequence_path, verbose=not args.quiet
        )

        if frames_db_local is not None:
            timestamps_ns = _load_timestamps_from_frames_db(
                frames_db_local, sequence_name
            )
        else:
            timestamps_ns = get_valid_timestamps(gt_provider, DEFAULT_STREAM_ID)
        logger.info(f"[{sequence_name}] {len(timestamps_ns)} timestamps")

        config = GeneratorConfig(
            sequence_name=sequence_name,
            release_type=ReleaseType(args.release_type),
            questions_per_type=args.questions_per_type,
            seed=args.seed,
            object_library_path=args.object_library,
        )

        annotations, all_refs = generate_v2_annotations(
            gt_provider=gt_provider,
            sequence_name=sequence_name,
            timestamps_ns=timestamps_ns,
            stream_id=DEFAULT_STREAM_ID,
            config=config,
            question_types=question_types,
            disambiguation_method=disambiguation_method,
            require_unique_names=args.require_unique_names,
            object_filter=args.object_filter,
            min_fov_degrees=args.min_fov_degrees,
            max_distance=args.max_distance,
            min_visibility_ratio=args.min_visibility_ratio,
            min_visible_frames=args.min_visible_frames,
            seed=args.seed,
            verbose=not args.quiet,
        )

        _populate_pointing_references(annotations, all_refs)

        for ann in annotations:
            store.write_annotation(ann)
        all_annotations.extend(annotations)

        logger.info("Populating GT 3D bboxes...")
        _populate_gt_bboxes(store, annotations, gt_provider, timestamps_ns)

        if args.visualize and annotations:
            _render_sequence_videos(
                annotations,
                all_refs,
                gt_provider,
                timestamps_ns,
                output_dir,
                video_paths,
            )

        logger.info(f"[{sequence_name}] {len(annotations)} annotations generated")

    type_counts: dict[str, int] = Counter()
    method_counts: dict[str, int] = Counter()
    for ann in all_annotations:
        type_counts[ann.query_layer.question_type.value] += 1
        method_counts[ann.disambiguation_layer.method.value] += 1

    manifest_rng = random.Random(
        f"{args.seed}_manifest_{'_'.join(sorted(all_sequence_names))}"
    )
    manifest = GenerationManifest(
        manifest_id=make_deterministic_id(manifest_rng),
        commit_hash="unknown",
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        sequences=all_sequence_names,
        settings={
            "questions_per_type": args.questions_per_type,
            "seed": args.seed,
            "min_fov_degrees": args.min_fov_degrees,
            "max_distance": args.max_distance,
            "min_visibility_ratio": args.min_visibility_ratio,
            "release_type": args.release_type,
            "object_filter": args.object_filter,
        },
        annotation_count=len(all_annotations),
        annotations_by_question_type=dict(type_counts),
        annotations_by_disambiguation_method=dict(method_counts),
    )
    store.write_manifest(manifest)
    store.close()
    logger.info(
        f"Wrote {len(all_annotations)} annotations from "
        f"{len(all_sequence_names)} sequences to {db_path}"
    )


if __name__ == "__main__":
    main()

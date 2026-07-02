# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Multi-hop spatial reasoning question generators.

These question types require 3+ tool calls and multi-step reasoning
to answer -- comparing sizes, computing paths, combining position
and dimension data, or reasoning about volumes.
"""

from __future__ import annotations

import itertools
import math
import random

_NEAR_EQUAL_THRESHOLD_M = 0.05

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.position import (
    distance_between_objects,
    distance_from_aria,
    get_aria_position_at_timestamp,
    get_object_3d_info,
    get_object_3d_position,
)
from r3d.data_gen.generators.base import (
    build_referenced_object,
    GeneratorConfig,
    get_obj_name,
    ObjectRef,
)
from r3d.data_gen.utils.annotation_schema import (
    Annotation,
    DisambiguationLayer,
    EvalMetric,
    EvalMode,
    IdentityLayer,
    make_deterministic_id,
    QueryLayer,
    QuestionType,
)
from r3d.data_gen.utils.functional_volume import compute_functional_volume
from r3d.data_gen.utils.mesh_utils import get_object_mesh_path
from r3d.data_gen.utils.scene_helpers import find_frame_with_object
from r3d.data_gen.utils.volume_categories import is_volume_eligible


_GAP_FIT_TEMPLATES: list[str] = [
    "Will {obj} fit in the space between {ref1} and {ref2}?",
    "Is there enough room for {obj} between {ref1} and {ref2}?",
    "Could I place {obj} in the gap between {ref1} and {ref2}?",
    "Would {obj} fit between {ref1} and {ref2}?",
]


def generate_gap_fit(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'Will X fit between Y and Z?' questions.

    Sidesteps orientation ambiguity: gap >= max(dims) -> Yes,
    gap < min(dims) -> No, otherwise skip.
    """
    annotations: list[Annotation] = []

    refs = [r for r in object_refs if get_obj_name(r.obj) is not None]
    if len(refs) < 3:
        return annotations

    triples = list(itertools.combinations(refs, 3))
    rng.shuffle(triples)

    for ref_obj, ref_a, ref_b in triples:
        if len(annotations) >= config.questions_per_type:
            break

        gap = distance_between_objects(ref_a.obj, ref_b.obj)
        if gap is None or gap < 0.3 or gap > 3.0:
            continue

        ts = find_frame_with_object(
            gt_provider, ref_obj.obj.instance_id, timestamps_ns, stream_id
        )
        info = get_object_3d_info(gt_provider, ref_obj.obj.instance_id, ts)
        if info is None:
            continue

        dims = info["dimensions"]
        sorted_dims = sorted([dims["width"], dims["height"], dims["depth"]])
        max_dim = sorted_dims[2]
        min_dim = sorted_dims[0]

        if gap >= max_dim:
            gt_answer = "yes"
        elif gap < min_dim:
            gt_answer = "no"
        else:
            continue

        template = rng.choice(_GAP_FIT_TEMPLATES)
        question = template.format(
            obj=ref_obj.text_description,
            ref1=ref_a.text_description,
            ref2=ref_b.text_description,
        )

        annotations.append(
            Annotation(
                annotation_id=make_deterministic_id(rng),
                identity_layer=IdentityLayer(
                    sequence_id=config.sequence_name,
                    release_type=config.release_type,
                    referenced_objects=[
                        build_referenced_object(ref_obj.obj, config),
                        build_referenced_object(ref_a.obj, config),
                        build_referenced_object(ref_b.obj, config),
                    ],
                ),
                disambiguation_layer=DisambiguationLayer(
                    method=ref_obj.disambiguation_method,
                    disambiguation_context=ref_obj.disambiguation_context,
                ),
                query_layer=QueryLayer(
                    question_type=QuestionType.GAP_FIT,
                    question_text=question,
                    gt_answer=gt_answer,
                    gt_answer_type="bool",
                    eval_mode=EvalMode.DETERMINISTIC,
                    eval_metric=EvalMetric.ACCURACY,
                    query_timestamp_ns_start=timestamps_ns[0],
                    query_timestamp_ns_end=timestamps_ns[-1],
                ),
            )
        )

    return annotations


_NEAREST_TEMPLATES_2: list[str] = [
    "Which is closer to me, {obj1} or {obj2}?",
    "Which of these is nearer to me: {obj1} or {obj2}?",
    "Between {obj1} and {obj2}, which one am I closer to?",
]

_NEAREST_TEMPLATES_3: list[str] = [
    "Which is closest to me: {obj1}, {obj2}, or {obj3}?",
    "Out of {obj1}, {obj2}, and {obj3}, which is nearest to me?",
    "Among {obj1}, {obj2}, and {obj3}, which one am I closest to?",
]

_NEAREST_TEMPLATES_4: list[str] = [
    "Which is closest to me: {obj1}, {obj2}, {obj3}, or {obj4}?",
    "Out of {obj1}, {obj2}, {obj3}, and {obj4}, which is nearest to me?",
    "Among {obj1}, {obj2}, {obj3}, and {obj4}, which one am I closest to?",
]

_NEAREST_TEMPLATES = {
    2: _NEAREST_TEMPLATES_2,
    3: _NEAREST_TEMPLATES_3,
    4: _NEAREST_TEMPLATES_4,
}


def generate_nearest_from_set(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'Which of X, Y, Z is nearest to me?' questions."""
    annotations: list[Annotation] = []

    refs = [r for r in object_refs if get_obj_name(r.obj) is not None]
    if len(refs) < 2:
        return annotations

    aria_pos = get_aria_position_at_timestamp(gt_provider, timestamps_ns[-1])
    if aria_pos is None:
        return annotations

    for set_size in [3, 2, 4]:
        if len(refs) < set_size:
            continue

        combos = list(itertools.combinations(refs, set_size))
        rng.shuffle(combos)

        for combo in combos:
            if len(annotations) >= config.questions_per_type:
                break

            distances: list[tuple[ObjectRef, float]] = []
            for ref in combo:
                d = distance_from_aria(ref.obj, aria_pos)
                if d is None:
                    break
                distances.append((ref, d))

            if len(distances) != set_size:
                continue

            distances.sort(key=lambda x: x[1])
            nearest_ref, nearest_dist = distances[0]
            runner_up_dist = distances[1][1]

            if runner_up_dist - nearest_dist < 0.3:
                continue

            nearest_name = get_obj_name(nearest_ref.obj)
            assert nearest_name is not None

            presentation_order = list(distances)
            rng.shuffle(presentation_order)
            obj_names = {
                f"obj{i + 1}": r.text_description
                for i, (r, _) in enumerate(presentation_order)
            }
            templates = _NEAREST_TEMPLATES[set_size]
            template = rng.choice(templates)
            question = template.format(**obj_names)

            annotations.append(
                Annotation(
                    annotation_id=make_deterministic_id(rng),
                    identity_layer=IdentityLayer(
                        sequence_id=config.sequence_name,
                        release_type=config.release_type,
                        referenced_objects=[
                            build_referenced_object(r.obj, config) for r, _ in distances
                        ],
                    ),
                    disambiguation_layer=DisambiguationLayer(
                        method=nearest_ref.disambiguation_method,
                        disambiguation_context=nearest_ref.disambiguation_context,
                    ),
                    query_layer=QueryLayer(
                        question_type=QuestionType.NEAREST_FROM_SET,
                        question_text=question,
                        gt_answer=nearest_name,
                        gt_answer_type="str",
                        eval_mode=EvalMode.DETERMINISTIC,
                        eval_metric=EvalMetric.ACCURACY,
                        query_timestamp_ns_start=timestamps_ns[0],
                        query_timestamp_ns_end=timestamps_ns[-1],
                    ),
                )
            )

    return annotations


_WALK_TEMPLATES: dict[int, list[str]] = {
    2: [
        "If I walk to the center of {obj1}, then to the center of {obj2}, how much horizontal distance have I traveled, in meters?",
        "What is the total horizontal walking distance from me to the center of {obj1} and then to the center of {obj2}, in meters?",
        "How far would I walk horizontally if I go to the center of {obj1} first, then to the center of {obj2}, in meters?",
    ],
    3: [
        "If I walk to the center of {obj1}, then to the center of {obj2}, then to the center of {obj3}, how much horizontal distance have I traveled, in meters?",
        "What is the total horizontal walking distance from me to the center of {obj1}, then the center of {obj2}, then the center of {obj3}, in meters?",
        "How far would I walk horizontally visiting the centers of {obj1}, {obj2}, and {obj3} in that order, in meters?",
    ],
}

_FLY_TEMPLATES: dict[int, list[str]] = {
    2: [
        "If a bug flies from me to the center of {obj1}, then to the center of {obj2}, how much total distance has it traveled, in meters?",
        "What is the total straight-line distance from me to the center of {obj1} and then to the center of {obj2}, in meters?",
        "How far in total would something travel in a straight line from me to the center of {obj1}, then to the center of {obj2}, in meters?",
    ],
    3: [
        "If a bug flies from me to the center of {obj1}, then to the center of {obj2}, then to the center of {obj3}, how much total distance has it traveled, in meters?",
        "What is the total straight-line distance from me to the center of {obj1}, then the center of {obj2}, then the center of {obj3}, in meters?",
        "How far in total would something fly from me to the center of {obj1}, then the center of {obj2}, then the center of {obj3}, in meters?",
    ],
}


def _horizontal_distance(pos_a: dict[str, float], pos_b: dict[str, float]) -> float:
    dx = pos_a["x"] - pos_b["x"]
    dz = pos_a["z"] - pos_b["z"]
    return math.sqrt(dx * dx + dz * dz)


def _euclidean_distance_3d(pos_a: dict[str, float], pos_b: dict[str, float]) -> float:
    dx = pos_a["x"] - pos_b["x"]
    dy = pos_a["y"] - pos_b["y"]
    dz = pos_a["z"] - pos_b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _compute_path_total(
    gt_provider: AriaDigitalTwinDataProvider,
    combo: tuple[ObjectRef, ...],
    camera_pos: dict[str, float],
    timestamps_ns: list[int],
    stream_id: StreamId,
    distance_fn: object,
    min_leg: float,
) -> float | None:
    positions: list[dict[str, float]] = []
    for ref in combo:
        ts = find_frame_with_object(
            gt_provider, ref.obj.instance_id, timestamps_ns, stream_id
        )
        pos = get_object_3d_position(gt_provider, ref.obj.instance_id, ts)
        if pos is None:
            return None
        positions.append(pos)

    assert callable(distance_fn)
    legs = [distance_fn(camera_pos, positions[0])]
    for i in range(len(positions) - 1):
        legs.append(distance_fn(positions[i], positions[i + 1]))

    if any(leg < min_leg for leg in legs):
        return None
    total = sum(legs)
    if total < 2.0 or total > 15.0:
        return None
    return total


def _generate_path_distance(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
    question_type: QuestionType,
    templates: dict[int, list[str]],
    distance_fn: object,
    gt_unit: str,
    min_leg: float,
) -> list[Annotation]:
    annotations: list[Annotation] = []

    refs = [r for r in object_refs if get_obj_name(r.obj) is not None]
    if len(refs) < 2:
        return annotations

    aria_pos = get_aria_position_at_timestamp(gt_provider, timestamps_ns[-1])
    if aria_pos is None:
        return annotations
    camera_pos = {
        "x": float(aria_pos[0]),
        "y": float(aria_pos[1]),
        "z": float(aria_pos[2]),
    }

    for n_waypoints in [2, 3]:
        if len(refs) < n_waypoints:
            continue

        combos = list(itertools.combinations(refs, n_waypoints))
        rng.shuffle(combos)

        for combo in combos:
            if len(annotations) >= config.questions_per_type:
                break

            total = _compute_path_total(
                gt_provider,
                combo,
                camera_pos,
                timestamps_ns,
                stream_id,
                distance_fn,
                min_leg,
            )
            if total is None:
                continue

            obj_names = {f"obj{i + 1}": r.text_description for i, r in enumerate(combo)}
            template = rng.choice(templates[n_waypoints])
            question = template.format(**obj_names)

            annotations.append(
                Annotation(
                    annotation_id=make_deterministic_id(rng),
                    identity_layer=IdentityLayer(
                        sequence_id=config.sequence_name,
                        release_type=config.release_type,
                        referenced_objects=[
                            build_referenced_object(r.obj, config) for r in combo
                        ],
                    ),
                    disambiguation_layer=DisambiguationLayer(
                        method=combo[0].disambiguation_method,
                        disambiguation_context=combo[0].disambiguation_context,
                    ),
                    query_layer=QueryLayer(
                        question_type=question_type,
                        question_text=question,
                        gt_answer=str(round(total, 2)),
                        gt_answer_type="float",
                        gt_answer_unit=gt_unit,
                        eval_mode=EvalMode.DETERMINISTIC,
                        eval_metric=EvalMetric.PERCENTAGE_ERROR,
                        query_timestamp_ns_start=timestamps_ns[0],
                        query_timestamp_ns_end=timestamps_ns[-1],
                    ),
                )
            )

    return annotations


def generate_total_walk_distance(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate horizontal path distance questions."""
    return _generate_path_distance(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.TOTAL_WALK_DISTANCE,
        templates=_WALK_TEMPLATES,
        distance_fn=_horizontal_distance,
        gt_unit="meters",
        min_leg=1.0,
    )


def generate_total_fly_distance(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 3D path distance questions."""
    return _generate_path_distance(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.TOTAL_FLY_DISTANCE,
        templates=_FLY_TEMPLATES,
        distance_fn=_euclidean_distance_3d,
        gt_unit="meters",
        min_leg=1.0,
    )


def _generate_size_comparison(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
    question_type: QuestionType,
    templates: list[str],
    extract_metric: object,
    gt_answer_type: str,
    gt_answer_unit: str | None,
    eval_metric: EvalMetric,
    format_gt: object,
) -> list[Annotation]:
    annotations: list[Annotation] = []

    refs = [r for r in object_refs if get_obj_name(r.obj) is not None]
    if len(refs) < 2:
        return annotations

    pairs = list(itertools.combinations(refs, 2))
    rng.shuffle(pairs)

    for ref_a, ref_b in pairs:
        if len(annotations) >= config.questions_per_type:
            break

        ts_a = find_frame_with_object(
            gt_provider, ref_a.obj.instance_id, timestamps_ns, stream_id
        )
        ts_b = find_frame_with_object(
            gt_provider, ref_b.obj.instance_id, timestamps_ns, stream_id
        )
        info_a = get_object_3d_info(gt_provider, ref_a.obj.instance_id, ts_a)
        info_b = get_object_3d_info(gt_provider, ref_b.obj.instance_id, ts_b)
        if info_a is None or info_b is None:
            continue

        assert callable(extract_metric)
        val_a = extract_metric(info_a)
        val_b = extract_metric(info_b)

        if abs(val_a - val_b) < _NEAR_EQUAL_THRESHOLD_M:
            continue

        if val_a >= val_b:
            bigger_ref, smaller_ref = ref_a, ref_b
            bigger_val, smaller_val = val_a, val_b
        else:
            bigger_ref, smaller_ref = ref_b, ref_a
            bigger_val, smaller_val = val_b, val_a

        assert callable(format_gt)
        gt_answer = format_gt(bigger_ref, smaller_ref, bigger_val, smaller_val)

        template = rng.choice(templates)
        question = template.format(
            obj1=bigger_ref.text_description
            if gt_answer_type == "float"
            else ref_a.text_description,
            obj2=smaller_ref.text_description
            if gt_answer_type == "float"
            else ref_b.text_description,
        )

        annotations.append(
            Annotation(
                annotation_id=make_deterministic_id(rng),
                identity_layer=IdentityLayer(
                    sequence_id=config.sequence_name,
                    release_type=config.release_type,
                    referenced_objects=[
                        build_referenced_object(ref_a.obj, config),
                        build_referenced_object(ref_b.obj, config),
                    ],
                ),
                disambiguation_layer=DisambiguationLayer(
                    method=ref_a.disambiguation_method,
                    disambiguation_context=ref_a.disambiguation_context,
                ),
                query_layer=QueryLayer(
                    question_type=question_type,
                    question_text=question,
                    gt_answer=gt_answer,
                    gt_answer_type=gt_answer_type,
                    gt_answer_unit=gt_answer_unit,
                    eval_mode=EvalMode.DETERMINISTIC,
                    eval_metric=eval_metric,
                    query_timestamp_ns_start=timestamps_ns[0],
                    query_timestamp_ns_end=timestamps_ns[-1],
                ),
            )
        )

    return annotations


def _extract_height(info: dict) -> float:
    return info["dimensions"]["height"]


def _extract_max_dim(info: dict) -> float:
    d = info["dimensions"]
    return max(d["width"], d["height"], d["depth"])


def _format_comparison_gt(
    bigger_ref: ObjectRef,
    smaller_ref: ObjectRef,
    bigger_val: float,
    smaller_val: float,
) -> str:
    name = get_obj_name(bigger_ref.obj)
    assert name is not None
    return name


def _format_difference_gt(
    bigger_ref: ObjectRef,
    smaller_ref: ObjectRef,
    bigger_val: float,
    smaller_val: float,
) -> str:
    return str(round(bigger_val - smaller_val, 3))


_WHICH_TALLER_TEMPLATES: list[str] = [
    "Which is taller, {obj1} or {obj2}?",
    "Which of these is taller: {obj1} or {obj2}?",
    "Between {obj1} and {obj2}, which one is taller?",
]

_WHICH_LONGER_TEMPLATES: list[str] = [
    "Which is longer in its longest dimension, {obj1} or {obj2}?",
    "Which has a greater longest dimension: {obj1} or {obj2}?",
    "Between {obj1} and {obj2}, which one is bigger in its longest dimension?",
]

_HOW_MUCH_TALLER_TEMPLATES: list[str] = [
    "How much taller is {obj1} than {obj2}, in meters?",
    "By how much does {obj1} exceed {obj2} in height, in meters?",
    "By how many meters is {obj1} taller than {obj2}?",
]

_HOW_MUCH_LONGER_TEMPLATES: list[str] = [
    "How much longer is {obj1} than {obj2} in their longest dimensions, in meters?",
    "By what length does the longest dimension of {obj1} exceed the longest dimension of {obj2}, in meters?",
    "By how many meters does {obj1} exceed {obj2} in its longest dimension?",
]


def generate_which_taller(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'Which is taller, X or Y?' questions."""
    return _generate_size_comparison(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.WHICH_TALLER,
        templates=_WHICH_TALLER_TEMPLATES,
        extract_metric=_extract_height,
        gt_answer_type="str",
        gt_answer_unit=None,
        eval_metric=EvalMetric.ACCURACY,
        format_gt=_format_comparison_gt,
    )


def generate_which_longer_dim(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'Which is longer in its longest dimension?' questions."""
    return _generate_size_comparison(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.WHICH_LONGER_DIM,
        templates=_WHICH_LONGER_TEMPLATES,
        extract_metric=_extract_max_dim,
        gt_answer_type="str",
        gt_answer_unit=None,
        eval_metric=EvalMetric.ACCURACY,
        format_gt=_format_comparison_gt,
    )


def generate_how_much_taller(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'How much taller is X than Y?' questions."""
    return _generate_size_comparison(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.HOW_MUCH_TALLER,
        templates=_HOW_MUCH_TALLER_TEMPLATES,
        extract_metric=_extract_height,
        gt_answer_type="float",
        gt_answer_unit="meters",
        eval_metric=EvalMetric.PERCENTAGE_ERROR,
        format_gt=_format_difference_gt,
    )


def generate_how_much_longer_dim(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'How much longer is X than Y in longest dim?' questions."""
    return _generate_size_comparison(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.HOW_MUCH_LONGER_DIM,
        templates=_HOW_MUCH_LONGER_TEMPLATES,
        extract_metric=_extract_max_dim,
        gt_answer_type="float",
        gt_answer_unit="meters",
        eval_metric=EvalMetric.PERCENTAGE_ERROR,
        format_gt=_format_difference_gt,
    )


_TOP_HIGHER_TEMPLATES: list[str] = [
    "Is the top of {obj1} higher up than the top of {obj2}?",
    "Is the top of {obj1} above the top of {obj2}?",
]


def generate_top_higher(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'Is the top of X higher than the top of Y?' questions."""
    annotations: list[Annotation] = []

    refs = [r for r in object_refs if get_obj_name(r.obj) is not None]
    if len(refs) < 2:
        return annotations

    pairs = list(itertools.combinations(refs, 2))
    rng.shuffle(pairs)

    for ref_a, ref_b in pairs:
        if len(annotations) >= config.questions_per_type:
            break

        ts_a = find_frame_with_object(
            gt_provider, ref_a.obj.instance_id, timestamps_ns, stream_id
        )
        ts_b = find_frame_with_object(
            gt_provider, ref_b.obj.instance_id, timestamps_ns, stream_id
        )

        pos_a = get_object_3d_position(gt_provider, ref_a.obj.instance_id, ts_a)
        pos_b = get_object_3d_position(gt_provider, ref_b.obj.instance_id, ts_b)
        info_a = get_object_3d_info(gt_provider, ref_a.obj.instance_id, ts_a)
        info_b = get_object_3d_info(gt_provider, ref_b.obj.instance_id, ts_b)
        if pos_a is None or pos_b is None or info_a is None or info_b is None:
            continue

        top_a = pos_a["y"] + info_a["dimensions"]["height"] / 2.0
        top_b = pos_b["y"] + info_b["dimensions"]["height"] / 2.0

        if abs(top_a - top_b) < _NEAR_EQUAL_THRESHOLD_M:
            continue

        gt_answer = "yes" if top_a > top_b else "no"

        template = rng.choice(_TOP_HIGHER_TEMPLATES)
        question = template.format(
            obj1=ref_a.text_description,
            obj2=ref_b.text_description,
        )

        annotations.append(
            Annotation(
                annotation_id=make_deterministic_id(rng),
                identity_layer=IdentityLayer(
                    sequence_id=config.sequence_name,
                    release_type=config.release_type,
                    referenced_objects=[
                        build_referenced_object(ref_a.obj, config),
                        build_referenced_object(ref_b.obj, config),
                    ],
                ),
                disambiguation_layer=DisambiguationLayer(
                    method=ref_a.disambiguation_method,
                    disambiguation_context=ref_a.disambiguation_context,
                ),
                query_layer=QueryLayer(
                    question_type=QuestionType.TOP_HIGHER,
                    question_text=question,
                    gt_answer=gt_answer,
                    gt_answer_type="bool",
                    eval_mode=EvalMode.DETERMINISTIC,
                    eval_metric=EvalMetric.ACCURACY,
                    query_timestamp_ns_start=timestamps_ns[0],
                    query_timestamp_ns_end=timestamps_ns[-1],
                ),
            )
        )

    return annotations


_POUR_ROOM_TEMPLATES: list[str] = [
    "If I fill {source} with water and pour it into {target}, how much room is left in {target}, in liters?",
    "If I fill {source} completely with water and pour all of it into {target}, how many liters of space remain in {target}?",
    "After pouring a full {source} of water into {target}, how much empty space is left in {target}, in liters?",
]

_POUR_LEFTOVER_TEMPLATES: list[str] = [
    "If I fill {source} with water and pour it into {target}, how much water is left in {source}, in liters?",
    "If I fill {source} completely with water and pour into {target} until it is full, how many liters remain in {source}?",
    "After filling {source} with water and pouring into {target} until {target} is full, how many liters are left over in {source}?",
]


def _compute_pour_gt(
    ref_a: ObjectRef,
    ref_b: ObjectRef,
    source_larger: bool,
    object_library_path: str | None,
) -> tuple[ObjectRef, ObjectRef, float] | None:
    vol_a = compute_functional_volume(
        get_object_mesh_path(ref_a.obj.name, object_library_path)
    )
    vol_b = compute_functional_volume(
        get_object_mesh_path(ref_b.obj.name, object_library_path)
    )

    if vol_a < 0.00005 or vol_b < 0.00005:
        return None
    if vol_a == vol_b:
        return None

    if source_larger:
        source_ref, target_ref = (ref_a, ref_b) if vol_a > vol_b else (ref_b, ref_a)
        gt_liters = round((max(vol_a, vol_b) - min(vol_a, vol_b)) * 1000, 3)
    else:
        source_ref, target_ref = (ref_a, ref_b) if vol_a < vol_b else (ref_b, ref_a)
        gt_liters = round((max(vol_a, vol_b) - min(vol_a, vol_b)) * 1000, 3)

    if gt_liters < 0.05:
        return None
    return source_ref, target_ref, gt_liters


def _generate_pour(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
    question_type: QuestionType,
    templates: list[str],
    source_larger: bool,
) -> list[Annotation]:
    annotations: list[Annotation] = []

    eligible = [r for r in object_refs if is_volume_eligible(r.obj.category)]
    if len(eligible) < 2:
        return annotations

    pairs = list(itertools.combinations(eligible, 2))
    rng.shuffle(pairs)

    for ref_a, ref_b in pairs:
        if len(annotations) >= config.questions_per_type:
            break

        result = _compute_pour_gt(
            ref_a, ref_b, source_larger, config.object_library_path
        )
        if result is None:
            continue
        source_ref, target_ref, gt_liters = result

        template = rng.choice(templates)
        question = template.format(
            source=source_ref.text_description,
            target=target_ref.text_description,
        )

        annotations.append(
            Annotation(
                annotation_id=make_deterministic_id(rng),
                identity_layer=IdentityLayer(
                    sequence_id=config.sequence_name,
                    release_type=config.release_type,
                    referenced_objects=[
                        build_referenced_object(source_ref.obj, config),
                        build_referenced_object(target_ref.obj, config),
                    ],
                ),
                disambiguation_layer=DisambiguationLayer(
                    method=source_ref.disambiguation_method,
                    disambiguation_context=source_ref.disambiguation_context,
                ),
                query_layer=QueryLayer(
                    question_type=question_type,
                    question_text=question,
                    gt_answer=str(gt_liters),
                    gt_answer_type="float",
                    gt_answer_unit="liters",
                    eval_mode=EvalMode.DETERMINISTIC,
                    eval_metric=EvalMetric.PERCENTAGE_ERROR,
                    query_timestamp_ns_start=timestamps_ns[0],
                    query_timestamp_ns_end=timestamps_ns[-1],
                ),
            )
        )

    return annotations


def generate_pour_room_left(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'pour X into Y, how much room left in Y?' questions."""
    return _generate_pour(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.POUR_ROOM_LEFT,
        templates=_POUR_ROOM_TEMPLATES,
        source_larger=False,
    )


def generate_pour_leftover(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'pour X into Y, how much left in X?' questions."""
    return _generate_pour(
        gt_provider,
        object_refs,
        all_refs,
        timestamps_ns,
        stream_id,
        config,
        rng,
        question_type=QuestionType.POUR_LEFTOVER,
        templates=_POUR_LEFTOVER_TEMPLATES,
        source_larger=True,
    )

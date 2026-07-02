# Copyright (c) Meta Platforms, Inc. and affiliates.

"""V2 distance question generators.

Generates GLOBAL_HOW_FAR and GLOBAL_HOW_FAR_FROM_ME questions.
Disambiguation is handled by the ObjectRef layer -- generators just
use ``ref.text_description``.
"""

from __future__ import annotations

import random

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.position import (
    distance_between_objects,
    distance_from_aria,
    get_aria_position_at_timestamp,
)
from r3d.data_gen.generators.base import (
    build_referenced_object,
    GeneratorConfig,
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
from r3d.data_gen.utils.scene_helpers import is_likely_held

# Min/max distance filters (meters)
MIN_DISTANCE = 0.2
MAX_DISTANCE = 10.0

_HOW_FAR_TEMPLATES: list[str] = [
    "How far is {obj1} from {obj2}, in meters?",
    "What is the distance between {obj1} and {obj2}, in meters?",
    "How many meters apart are {obj1} and {obj2}?",
]

_HOW_FAR_FROM_ME_TEMPLATES: list[str] = [
    "How far is {obj} from me, in meters?",
    "What is the distance from me to {obj}, in meters?",
    "How many meters away is {obj} from where I am?",
    "How far away from me is {obj}, in meters?",
]


def _build_annotation(
    ref: ObjectRef,
    question_type: QuestionType,
    question_text: str,
    gt_answer: float,
    config: GeneratorConfig,
    timestamps_ns: list[int],
    rng: random.Random,
    partner_ref: ObjectRef | None = None,
) -> Annotation:
    """Build a distance Annotation.

    DisambiguationLayer reflects the primary (first) object's method.
    Partner objects' disambiguation is encoded in their text_description
    within the question text.
    """
    referenced_objects = [build_referenced_object(ref.obj, config)]
    if partner_ref is not None:
        referenced_objects.append(build_referenced_object(partner_ref.obj, config))

    return Annotation(
        annotation_id=make_deterministic_id(rng),
        identity_layer=IdentityLayer(
            sequence_id=config.sequence_name,
            release_type=config.release_type,
            referenced_objects=referenced_objects,
        ),
        disambiguation_layer=DisambiguationLayer(
            method=ref.disambiguation_method,
            disambiguation_context=ref.disambiguation_context,
        ),
        query_layer=QueryLayer(
            question_type=question_type,
            question_text=question_text,
            gt_answer=str(gt_answer),
            gt_answer_type="float",
            gt_answer_unit="meters",
            gt_computation_method="surface_to_surface",
            eval_mode=EvalMode.DETERMINISTIC,
            eval_metric=EvalMetric.PERCENTAGE_ERROR,
            query_timestamp_ns_start=timestamps_ns[0],
            query_timestamp_ns_end=timestamps_ns[-1],
        ),
    )


def generate_how_far(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'How far is X from Y?' questions."""
    annotations: list[Annotation] = []
    used_pairs: set[tuple[int, int]] = set()

    refs = list(object_refs)
    pairs = [
        (refs[i], refs[j]) for i in range(len(refs)) for j in range(i + 1, len(refs))
    ]
    rng.shuffle(pairs)

    for ref1, ref2 in pairs:
        if len(annotations) >= config.questions_per_type:
            break

        pair_key = (
            min(ref1.obj.instance_id, ref2.obj.instance_id),
            max(ref1.obj.instance_id, ref2.obj.instance_id),
        )
        if pair_key in used_pairs:
            continue

        dist = distance_between_objects(ref1.obj, ref2.obj)
        if dist is None or dist < MIN_DISTANCE or dist > MAX_DISTANCE:
            continue

        gt_answer = round(dist, 3)
        template = rng.choice(_HOW_FAR_TEMPLATES)
        question = template.format(
            obj1=ref1.text_description,
            obj2=ref2.text_description,
        )

        annotations.append(
            _build_annotation(
                ref=ref1,
                question_type=QuestionType.GLOBAL_HOW_FAR,
                question_text=question,
                gt_answer=gt_answer,
                config=config,
                timestamps_ns=timestamps_ns,
                rng=rng,
                partner_ref=ref2,
            )
        )
        used_pairs.add(pair_key)

    return annotations


def generate_how_far_from_me(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'How far is X from me?' questions."""
    annotations: list[Annotation] = []

    aria_position = get_aria_position_at_timestamp(gt_provider, timestamps_ns[-1])
    if aria_position is None:
        return []

    refs = list(object_refs)
    rng.shuffle(refs)

    for ref in refs:
        if len(annotations) >= config.questions_per_type:
            break

        # Skip held objects
        if is_likely_held(gt_provider, ref.obj.instance_id, timestamps_ns[-1]):
            continue

        dist = distance_from_aria(ref.obj, aria_position)
        if dist is None:
            continue

        gt_answer = round(dist, 3)
        template = rng.choice(_HOW_FAR_FROM_ME_TEMPLATES)
        question = template.format(obj=ref.text_description)

        annotations.append(
            _build_annotation(
                ref=ref,
                question_type=QuestionType.GLOBAL_HOW_FAR_FROM_ME,
                question_text=question,
                gt_answer=gt_answer,
                config=config,
                timestamps_ns=timestamps_ns,
                rng=rng,
            )
        )

    return annotations

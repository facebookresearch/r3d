# Copyright (c) Meta Platforms, Inc. and affiliates.

"""V2 size question generators.

Generates GLOBAL_HOW_LONG questions.
"""

from __future__ import annotations

import random

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
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
from r3d.data_gen.utils.scene_helpers import (
    find_frame_with_object,
    get_longest_dimension_at_timestamp,
)

# Minimum longest dimension (meters) -- skip tiny objects
MIN_LONGEST_DIM = 0.1

_HOW_LONG_TEMPLATES: list[str] = [
    "How long is {obj} in its longest dimension, in meters?",
    "What is the longest dimension of {obj}, in meters?",
    "How big is {obj} along its longest axis, in meters?",
]


def generate_how_long(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'How long is X in its longest dimension?' questions."""
    annotations: list[Annotation] = []

    refs = list(object_refs)
    rng.shuffle(refs)

    for ref in refs:
        if len(annotations) >= config.questions_per_type:
            break

        ts = find_frame_with_object(
            gt_provider, ref.obj.instance_id, timestamps_ns, stream_id
        )

        longest_dim = get_longest_dimension_at_timestamp(gt_provider, ref.obj, ts)
        if longest_dim is None or longest_dim < MIN_LONGEST_DIM:
            continue

        gt_answer = round(longest_dim, 3)
        template = rng.choice(_HOW_LONG_TEMPLATES)
        question = template.format(obj=ref.text_description)

        annotations.append(
            Annotation(
                annotation_id=make_deterministic_id(rng),
                identity_layer=IdentityLayer(
                    sequence_id=config.sequence_name,
                    release_type=config.release_type,
                    referenced_objects=[build_referenced_object(ref.obj, config)],
                ),
                disambiguation_layer=DisambiguationLayer(
                    method=ref.disambiguation_method,
                    disambiguation_context=ref.disambiguation_context,
                ),
                query_layer=QueryLayer(
                    question_type=QuestionType.GLOBAL_HOW_LONG,
                    question_text=question,
                    gt_answer=str(gt_answer),
                    gt_answer_type="float",
                    gt_answer_unit="meters",
                    gt_computation_method="bbox_longest_dimension",
                    eval_mode=EvalMode.DETERMINISTIC,
                    eval_metric=EvalMetric.PERCENTAGE_ERROR,
                    query_timestamp_ns_start=timestamps_ns[0],
                    query_timestamp_ns_end=timestamps_ns[-1],
                ),
            )
        )

    return annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.

"""V2 complex reasoning question generators.

Generates VOLUME_ESTIMATION questions.
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
from r3d.data_gen.utils.functional_volume import compute_functional_volume
from r3d.data_gen.utils.mesh_utils import get_object_mesh_path
from r3d.data_gen.utils.scene_helpers import find_frame_with_object
from r3d.data_gen.utils.volume_categories import is_volume_eligible

_VOLUME_TEMPLATES: list[str] = [
    "What is the volume of {obj}, in liters?",
    "How much liquid can {obj} hold, in liters?",
    "What is the capacity of {obj}, in liters?",
]


def generate_volume_estimation(
    gt_provider: AriaDigitalTwinDataProvider,
    object_refs: list[ObjectRef],
    all_refs: list[ObjectRef],
    timestamps_ns: list[int],
    stream_id: StreamId,
    config: GeneratorConfig,
    rng: random.Random,
) -> list[Annotation]:
    """Generate 'What is the volume of this X?' questions.

    Only generates questions for objects whose ADT category is
    volume-eligible (cups, jars, vases, bowls, etc.). GT volume is
    computed as *functional capacity* (how much liquid the container
    can hold) via voxel-based cavity detection on the GLB mesh.

    """
    annotations: list[Annotation] = []

    eligible_refs = [r for r in object_refs if is_volume_eligible(r.obj.category)]
    rng.shuffle(eligible_refs)

    for ref in eligible_refs:
        if len(annotations) >= config.questions_per_type:
            break

        find_frame_with_object(
            gt_provider, ref.obj.instance_id, timestamps_ns, stream_id
        )

        glb_path = get_object_mesh_path(ref.obj.name, config.object_library_path)
        functional_volume_m3 = compute_functional_volume(glb_path)
        if functional_volume_m3 < 0.00005:
            continue
        gt_volume_liters = round(functional_volume_m3 * 1000, 3)

        template = rng.choice(_VOLUME_TEMPLATES)
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
                    question_type=QuestionType.VOLUME_ESTIMATION,
                    question_text=question,
                    gt_answer=str(gt_volume_liters),
                    gt_answer_type="float",
                    gt_answer_unit="liters",
                    gt_computation_method="mesh_cavity_voxelization",
                    eval_mode=EvalMode.DETERMINISTIC,
                    eval_metric=EvalMetric.PERCENTAGE_ERROR,
                    query_timestamp_ns_start=timestamps_ns[0],
                    query_timestamp_ns_end=timestamps_ns[-1],
                ),
            )
        )

    return annotations

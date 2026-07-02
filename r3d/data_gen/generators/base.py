# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Generator base abstractions.

Core types shared by all question generators:

- ``ObjectRef``: An object + how to reference it in question text.
  Disambiguation is determined *before* generators run, so every
  generator receives pre-resolved ObjectRefs and never needs to know
  about uniqueness or disambiguation logic.

- ``GeneratorConfig``: Shared settings for all generators.

- ``build_referenced_object``: Converts a ``VideoObjectInfo`` into the
  ``ReferencedObject`` Pydantic model used in ``Annotation``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from r3d.data_gen.extractor.object_info import VideoObjectInfo
from r3d.data_gen.utils.annotation_schema import (
    DisambiguationContext,
    DisambiguationMethod,
    ReferencedObject,
    ReleaseType,
)
from r3d.data_gen.utils.name_mapping import get_natural_name_v0p3


def get_obj_name(obj: VideoObjectInfo) -> str | None:
    """Get the v0p3 natural name for a VideoObjectInfo."""
    adt_name = obj.object_info.name if hasattr(obj, "object_info") else None
    if adt_name is None:
        return obj.natural_name
    return get_natural_name_v0p3(adt_name)


# ---------------------------------------------------------------------------
# ObjectRef -- the key abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectRef:
    """An object + how to reference it in question text.

    Every generator receives ``list[ObjectRef]``.  The ``text_description``
    is the phrase to use in the question (e.g. "the mug", "the mug near
    the couch", "this mug [pointing reference]").  The disambiguation
    fields are copied verbatim into the output ``DisambiguationLayer``.

    This makes disambiguation *orthogonal* to question type: adding a
    new disambiguation method only requires updating
    ``disambiguation.py``; adding a new question type only requires
    adding a generator.
    """

    obj: VideoObjectInfo
    disambiguation_method: DisambiguationMethod
    disambiguation_context: DisambiguationContext
    text_description: str
    reference_frame_idx: int | None = None
    reference_timestamp_ns: int | None = None
    pointing_mask: Any = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# GeneratorConfig -- shared settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorConfig:
    """Shared configuration for all v2 generators."""

    sequence_name: str
    release_type: ReleaseType
    questions_per_type: int
    seed: int
    object_library_path: str | None = None


# ---------------------------------------------------------------------------
# ReferencedObject construction
# ---------------------------------------------------------------------------


def _derive_prototype_name(instance_name: str) -> str:
    """Derive ADT prototype name from instance name.

    Strips trailing ``_NN`` numeric suffix (e.g. ``_01``, ``_02``).
    If no suffix is found, returns the name unchanged.

    Examples:
        "CoffeeCanisterSmall_01" -> "CoffeeCanisterSmall"
        "KeyboardLogitech" -> "KeyboardLogitech"
    """
    m = re.match(r"^(.+?)_(\d+)$", instance_name)
    if m:
        return m.group(1)
    return instance_name


def build_referenced_object(
    obj: VideoObjectInfo,
    config: GeneratorConfig,
) -> ReferencedObject:
    """Build a ReferencedObject from a VideoObjectInfo.

    Args:
        obj: The video object info to convert.
        config: Generator config (provides sequence_name, mesh_base_path).

    Returns:
        Fully populated ReferencedObject.

    Raises:
        ValueError: If the object has no natural name.
    """
    natural_name = get_natural_name_v0p3(obj.name)
    if natural_name is None:
        raise ValueError(
            f"Object {obj.name} (id={obj.instance_id}) has no natural name"
        )

    object_id = f"{config.sequence_name}_{obj.instance_id}"
    prototype_name = _derive_prototype_name(obj.name)

    return ReferencedObject(
        object_id=object_id,
        adt_instance_id=obj.instance_id,
        adt_instance_name=obj.name,
        prototype_name=prototype_name,
        canonical_name=natural_name,
        is_dynamic=not obj.is_static,
    )

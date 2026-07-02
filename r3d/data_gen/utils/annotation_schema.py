# Copyright (c) Meta Platforms, Inc. and affiliates.

"""R3D Dataset v2 -- Unified Annotation Schema.

Three-layer annotation schema for spatial reasoning evaluation:

1. **Identity Layer**: Which objects, which sequence, what release type.
2. **Disambiguation Layer**: How multi-instance ambiguity is resolved.
3. **Query Layer**: The question, expected answer, and evaluation method.

This is a clean break from v1. No backward compatibility with
_PartialAnnotationSample / AnnotationSample.

Uses pydantic v2 API.
"""

from __future__ import annotations

import random
import uuid
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReleaseType(str, Enum):
    """ADT release type determines which question types are valid."""

    FULL = "full"
    LITE = "lite"


class DisambiguationMethod(str, Enum):
    """How a multi-instance object is disambiguated."""

    GLOBAL = "global"
    POINTING = "pointing"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


class EvalMode(str, Enum):
    """How the answer is evaluated."""

    DETERMINISTIC = "deterministic"


class EvalMetric(str, Enum):
    """Specific evaluation metric."""

    PERCENTAGE_ERROR = "percentage_error"
    ACCURACY = "accuracy"


class QuestionType(str, Enum):
    """All v2 question types.

    Disambiguation is orthogonal -- encoded in DisambiguationLayer, not here.
    _UNIQUE/_RECENTLY_MOVED suffixes and _DISAMBIGUATED_* types are gone.
    """

    # Distance
    GLOBAL_HOW_FAR = "global_how_far"
    GLOBAL_HOW_FAR_FROM_ME = "global_how_far_from_me"

    # Size
    GLOBAL_HOW_LONG = "global_how_long"

    # Complex reasoning
    VOLUME_ESTIMATION = "volume_estimation"

    # Multi-hop reasoning
    GAP_FIT = "gap_fit"
    NEAREST_FROM_SET = "nearest_from_set"
    TOTAL_WALK_DISTANCE = "total_walk_distance"
    TOTAL_FLY_DISTANCE = "total_fly_distance"
    WHICH_TALLER = "which_taller"
    WHICH_LONGER_DIM = "which_longer_dim"
    HOW_MUCH_TALLER = "how_much_taller"
    HOW_MUCH_LONGER_DIM = "how_much_longer_dim"
    TOP_HIGHER = "top_higher"
    POUR_ROOM_LEFT = "pour_room_left"
    POUR_LEFTOVER = "pour_leftover"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ReferencedObject(BaseModel):
    """A single object referenced by an annotation."""

    object_id: str = Field(
        description="Stable unique ID for this object within the dataset"
    )
    adt_instance_id: int = Field(description="ADT numeric instance ID")
    adt_instance_name: str = Field(
        description="ADT instance name (e.g. 'CoffeeCanisterSmall_01')"
    )
    prototype_name: str = Field(
        description="ADT prototype name (e.g. 'CoffeeCanisterSmall')"
    )
    canonical_name: str = Field(
        description="Natural language name (e.g. 'small coffee canister')"
    )
    adt_mesh_path: Optional[str] = Field(
        default=None,
        description="Deprecated -- mesh is resolved from adt_instance_name at eval time",
    )
    is_dynamic: bool = Field(
        description="Whether this object moved during the sequence",
    )
    reference_frame_idx: Optional[int] = Field(
        default=None,
        description="Frame index of this object's pointing reference frame",
    )
    reference_timestamp_ns: Optional[int] = Field(
        default=None,
        description="Timestamp of this object's pointing reference frame",
    )

    @field_validator("canonical_name")
    @classmethod
    def canonical_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("canonical_name must not be empty")
        return v


class BBox3D(BaseModel):
    """Oriented 3D bounding box."""

    center: List[float] = Field(description="[x, y, z] in scene coordinates")
    dimensions: List[float] = Field(description="[width, height, depth] in meters")
    rotation: List[float] = Field(
        description="Quaternion [qw, qx, qy, qz] for OBB orientation",
    )

    @field_validator("center", "dimensions")
    @classmethod
    def must_be_3d(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError(f"Expected 3 elements, got {len(v)}")
        return v

    @field_validator("rotation")
    @classmethod
    def must_be_quaternion(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError(f"Expected 4 elements for quaternion, got {len(v)}")
        return v


class DisambiguationContext(BaseModel):
    """Context provided for disambiguating between duplicate objects.

    Which fields are populated depends on the disambiguation method.
    """

    spatial_description: Optional[str] = Field(
        default=None,
        description="Natural language spatial description (for spatial method)",
    )
    temporal_description: Optional[str] = Field(
        default=None,
        description="Natural language temporal reference (for temporal method)",
    )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class IdentityLayer(BaseModel):
    """Which sequence, which objects, what release type."""

    sequence_id: str = Field(description="ADT sequence identifier")
    release_type: ReleaseType = Field(description="ADT release type (full or lite)")
    referenced_objects: List[ReferencedObject] = Field(
        description="Objects referenced by this annotation (at least one)",
    )

    @field_validator("referenced_objects")
    @classmethod
    def must_have_objects(cls, v: List[ReferencedObject]) -> List[ReferencedObject]:
        if len(v) == 0:
            raise ValueError("referenced_objects must contain at least one object")
        return v


class DisambiguationLayer(BaseModel):
    """How multi-instance ambiguity is resolved (if applicable)."""

    method: DisambiguationMethod = Field(
        default=DisambiguationMethod.GLOBAL,
        description="Disambiguation method used",
    )
    disambiguation_context: DisambiguationContext = Field(
        default_factory=DisambiguationContext,
        description="Method-specific context",
    )

    @model_validator(mode="after")
    def validate_context_fields(self) -> DisambiguationLayer:  # noqa: B902
        """Validate that method-specific context fields are present."""
        method = self.method
        ctx = self.disambiguation_context

        if method == DisambiguationMethod.POINTING:
            pass
        elif method == DisambiguationMethod.SPATIAL:
            if not isinstance(ctx, DisambiguationContext):
                raise ValueError("SPATIAL requires disambiguation_context")
            if not ctx.spatial_description:
                raise ValueError("SPATIAL disambiguation requires spatial_description")
        elif method == DisambiguationMethod.TEMPORAL:
            if not isinstance(ctx, DisambiguationContext):
                raise ValueError("TEMPORAL requires disambiguation_context")
            if not ctx.temporal_description:
                raise ValueError(
                    "TEMPORAL disambiguation requires temporal_description"
                )

        return self


class QueryLayer(BaseModel):
    """The question, expected answer, and how to evaluate it."""

    question_type: QuestionType
    question_text: str = Field(description="The question posed to the model")
    gt_answer: str = Field(
        description="Ground truth answer as a string. Numeric answers are stored "
        "as their string representation (e.g. '1.23').",
    )
    gt_answer_type: str = Field(
        description="Semantic type of gt_answer: 'bool', 'float', or 'str'.",
    )
    gt_answer_unit: Optional[str] = Field(
        default=None,
        description="Unit for numeric answers (e.g. 'meters', 'cubic_meters')",
    )
    gt_computation_method: Optional[str] = Field(
        default=None,
        description="How GT was computed (e.g. 'surface_to_surface', 'bbox_center')",
    )
    eval_mode: EvalMode
    eval_metric: EvalMetric
    query_timestamp_ns_start: int = Field(
        description="Start of temporal window (for single-frame queries, equals end)",
    )
    query_timestamp_ns_end: int = Field(
        description="End of temporal window (for single-frame queries, equals start)",
    )

    @field_validator("question_text")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question_text must not be empty")
        return v


# ---------------------------------------------------------------------------
# Top-level Annotation
# ---------------------------------------------------------------------------


def make_deterministic_id(rng: random.Random) -> str:
    """Generate a UUID4-format ID from a seeded RNG for reproducibility."""
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


class Annotation(BaseModel):
    """A single v2 annotation sample.

    This is the atomic unit of the R3D Dataset v2. Each annotation
    contains all four layers, though some layers may have default/empty
    values depending on the question type.
    """

    annotation_id: str = Field(description="Unique annotation identifier")
    version: Literal["2.0"] = "2.0"

    identity_layer: IdentityLayer
    disambiguation_layer: DisambiguationLayer = Field(
        default_factory=DisambiguationLayer,
    )
    query_layer: QueryLayer


# ---------------------------------------------------------------------------
# Generation Manifest
# ---------------------------------------------------------------------------


class GenerationManifest(BaseModel):
    """Metadata about how a batch of annotations was generated.

    Tracks reproducibility information per the CEO review acceptance.
    """

    manifest_id: str = Field(description="Unique manifest identifier")
    version: Literal["2.0"] = "2.0"
    commit_hash: str = Field(description="Sapling commit hash at generation time")
    generation_timestamp: str = Field(description="ISO 8601 timestamp")
    sequences: List[str] = Field(description="ADT sequence IDs processed")
    settings: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Generation settings (fps, thresholds, etc.)",
    )
    annotation_count: int = Field(description="Total annotations generated")
    annotations_by_question_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per question type",
    )
    annotations_by_disambiguation_method: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per disambiguation method",
    )
    skipped_objects: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Objects skipped with reasons",
    )

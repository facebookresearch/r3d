# Copyright (c) Meta Platforms, Inc. and affiliates.

"""V2 disambiguation -- resolve objects to ObjectRefs.

This module is the single place that decides *how* an object is
referenced in question text.  Generators never need to know about
uniqueness or disambiguation; they just use ``ref.text_description``.

Two supported disambiguation methods:

- **Global**: "the coffee mug" -- uses the canonical object name directly. Works for unique objects and as a fallback for multi-instance objects.
- **Pointing**: "this mug" -- reference frame with GT segmentation mask.

Helper functions also exist for building spatial and temporal refs, but
``resolve_object_references`` only dispatches to GLOBAL and POINTING.
"""

from __future__ import annotations

import numpy as np
from projectaria_tools.core import calibration as pat_calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.object_info import VideoObjectInfo
from r3d.data_gen.extractor.position import distance_between_objects
from r3d.data_gen.generators.base import get_obj_name, ObjectRef
from r3d.data_gen.utils.annotation_schema import (
    DisambiguationContext,
    DisambiguationMethod,
)


# ---------------------------------------------------------------------------
# Spatial disambiguation helpers
# ---------------------------------------------------------------------------


def find_closest_unique_reference(
    target: VideoObjectInfo,
    unique_objects: list[VideoObjectInfo],
) -> VideoObjectInfo | None:
    """Find the closest globally unique object to use as a spatial landmark.

    Args:
        target: The multi-instance object needing disambiguation.
        unique_objects: All globally unique objects in the scene.

    Returns:
        The closest unique object, or None if none available.
    """
    best_ref = None
    best_dist = float("inf")

    for ref in unique_objects:
        if ref.instance_id == target.instance_id:
            continue
        dist = distance_between_objects(target, ref)
        if dist is not None and dist < best_dist:
            best_dist = dist
            best_ref = ref

    return best_ref


def is_spatially_unambiguous(
    target: VideoObjectInfo,
    reference: VideoObjectInfo,
    siblings: list[VideoObjectInfo],
) -> bool:
    """Check that "the X near the Y" uniquely identifies target among siblings.

    Target must be strictly the closest sibling to the reference object.

    Args:
        target: The object we want to describe.
        reference: The unique reference landmark.
        siblings: All objects sharing the same normalized name as target.

    Returns:
        True if target is strictly the closest sibling to reference.
    """
    target_dist = distance_between_objects(target, reference)
    if target_dist is None:
        return False

    for sibling in siblings:
        if sibling.instance_id == target.instance_id:
            continue
        sib_dist = distance_between_objects(sibling, reference)
        if sib_dist is not None and sib_dist <= target_dist:
            return False

    return True


# ---------------------------------------------------------------------------
# Temporal disambiguation helpers
# ---------------------------------------------------------------------------


def is_temporally_unambiguous(
    target: VideoObjectInfo,
    siblings: list[VideoObjectInfo],
) -> bool:
    """Check that "the X that was recently moved" uniquely identifies target.

    Target must be moved AND be the only moved sibling.

    Args:
        target: The object we want to describe.
        siblings: All objects sharing the same normalized name as target.

    Returns:
        True if target is the only moved sibling.
    """
    if not target.was_moved:
        return False

    moved_siblings = [
        s for s in siblings if s.was_moved and s.instance_id != target.instance_id
    ]
    return len(moved_siblings) == 0


# ---------------------------------------------------------------------------
# ObjectRef construction per method
# ---------------------------------------------------------------------------


def _make_global_ref(obj: VideoObjectInfo) -> ObjectRef:
    """Build an ObjectRef for a globally unique object (no disambiguation)."""
    name = get_obj_name(obj)
    assert name is not None
    return ObjectRef(
        obj=obj,
        disambiguation_method=DisambiguationMethod.GLOBAL,
        disambiguation_context=DisambiguationContext(),
        text_description=f"the {name}",
    )


def _make_spatial_ref(
    target: VideoObjectInfo,
    reference: VideoObjectInfo,
) -> ObjectRef | None:
    """Build an ObjectRef using spatial disambiguation."""
    target_name = get_obj_name(target)
    ref_name = get_obj_name(reference)
    if not target_name or not ref_name:
        return None

    return ObjectRef(
        obj=target,
        disambiguation_method=DisambiguationMethod.SPATIAL,
        disambiguation_context=DisambiguationContext(
            spatial_description=f"the {target_name} near the {ref_name}",
        ),
        text_description=f"the {target_name} near the {ref_name}",
    )


def _make_temporal_ref(target: VideoObjectInfo) -> ObjectRef | None:
    """Build an ObjectRef using temporal disambiguation."""
    target_name = get_obj_name(target)
    if not target_name:
        return None

    assert target.movement_event is not None

    return ObjectRef(
        obj=target,
        disambiguation_method=DisambiguationMethod.TEMPORAL,
        disambiguation_context=DisambiguationContext(
            temporal_description=f"the {target_name} that was recently moved",
        ),
        text_description=f"the {target_name} that was recently moved",
    )


def _extract_object_mask(
    gt_provider: AriaDigitalTwinDataProvider,
    instance_id: int,
    timestamp_ns: int,
    stream_id: StreamId,
) -> np.ndarray | None:
    """Extract corrected binary mask for an object at a timestamp.

    Loads the segmentation image, undistorts with nearest-neighbor
    interpolation (to preserve integer instance IDs), rotates to
    upright orientation, and returns a boolean mask.

    Returns:
        Boolean mask (H, W) or None if segmentation unavailable.
    """
    seg_with_dt = gt_provider.get_segmentation_image_by_timestamp_ns(
        timestamp_ns, stream_id
    )
    if not seg_with_dt.is_valid():
        return None

    seg_image = seg_with_dt.data().to_numpy_array()
    if seg_image.ndim > 2:
        seg_image = seg_image[:, :, 0]

    # Undistort with nearest-neighbor to preserve integer IDs
    camera_calib = gt_provider.get_aria_camera_calibration(stream_id)
    if camera_calib is not None:
        pinhole = pat_calibration.get_linear_camera_calibration(
            int(camera_calib.get_image_size()[0]),
            int(camera_calib.get_image_size()[1]),
            camera_calib.get_focal_lengths()[0],
            "pinhole",
            camera_calib.get_transform_device_camera(),
        )
        seg_image = pat_calibration.distort_label_by_calibration(
            seg_image, pinhole, camera_calib
        )
        # Rotate 90 degrees CW to match upright RGB orientation
        seg_image = np.rot90(seg_image, k=3)

    return seg_image == int(instance_id)


def _make_pointing_ref(
    target: VideoObjectInfo,
    gt_provider: AriaDigitalTwinDataProvider,
    timestamps_ns: list[int],
    stream_id: StreamId,
) -> ObjectRef | None:
    """Build an ObjectRef using pointing disambiguation.

    Uses the first visibility-filtered timestamp where the object has a
    non-empty GT segmentation mask.
    """
    target_name = get_obj_name(target)
    if not target_name:
        return None

    if not target.visible_timestamps:
        raise RuntimeError(
            f"Object '{target_name}' (instance {target.instance_id}) "
            f"has no visible_timestamps -- it should not have passed filtering"
        )

    for ts in target.visible_timestamps:
        mask = _extract_object_mask(gt_provider, target.instance_id, ts, stream_id)
        if mask is not None and mask.any():
            frame_idx = timestamps_ns.index(ts)
            return ObjectRef(
                obj=target,
                disambiguation_method=DisambiguationMethod.POINTING,
                disambiguation_context=DisambiguationContext(),
                text_description=f"the {target_name}",
                reference_frame_idx=frame_idx,
                reference_timestamp_ns=ts,
                pointing_mask=mask,
            )

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def resolve_object_references(
    objects: list[VideoObjectInfo],
    unique_objects: list[VideoObjectInfo],
    multi_instance_groups: dict[str, list[VideoObjectInfo]],
    gt_provider: AriaDigitalTwinDataProvider,
    timestamps_ns: list[int],
    stream_id: StreamId,
    allowed_method: DisambiguationMethod | None = None,
) -> list[ObjectRef]:
    """Resolve all objects to ObjectRefs using a single disambiguation method.

    Args:
        objects: All objects in the scene.
        unique_objects: Objects whose normalized name appears exactly once.
        multi_instance_groups: Groups of 2+ objects sharing normalized name.
        gt_provider: ADT data provider for the sequence.
        timestamps_ns: All valid timestamps in the sequence.
        stream_id: Camera stream ID.
        allowed_method: Which disambiguation method to use. If None,
            defaults to GLOBAL.

    Returns:
        List of ObjectRefs, one per object.
    """
    if allowed_method is None:
        allowed_method = DisambiguationMethod.GLOBAL

    refs: list[ObjectRef] = []

    for obj in objects:
        if get_obj_name(obj) is None:
            continue

        if allowed_method == DisambiguationMethod.GLOBAL:
            refs.append(_make_global_ref(obj))
        elif allowed_method == DisambiguationMethod.POINTING:
            pointing_ref = _make_pointing_ref(
                obj, gt_provider, timestamps_ns, stream_id
            )
            if pointing_ref is not None:
                refs.append(pointing_ref)
        else:
            raise RuntimeError(f"Unsupported disambiguation method: {allowed_method}")

    return refs

# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Global-specific filtering functions.

This module provides filtering functions for global (video-based) question generation,
which filters objects across a temporal window.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataProvider
from r3d.data_gen.extractor.object_info import VideoObjectInfo
from r3d.data_gen.extractor.position import get_aria_position_at_timestamp
from r3d.data_gen.filter.uniqueness import normalize_name_for_uniqueness
from r3d.data_gen.filter.visibility import filter_objects

# Default filtering FPS (for checking filter conditions across temporal window)
DEFAULT_FILTER_FPS = 5.0

# Default filtering parameters
DEFAULT_MIN_FOV_DEGREES = 2.0
DEFAULT_MAX_DISTANCE = 4.0
DEFAULT_MIN_VISIBILITY_RATIO = 0.3


def filter_global_objects(
    objects: list[VideoObjectInfo],
    gt_provider: AriaDigitalTwinDataProvider,
    timestamps_ns: list[int],
    stream_id: StreamId,
    filter_fps: float = DEFAULT_FILTER_FPS,
    unique: bool = False,
    min_fov_degrees: float = DEFAULT_MIN_FOV_DEGREES,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    require_2d_bbox: bool = True,
    min_visibility_ratio: float = DEFAULT_MIN_VISIBILITY_RATIO,
    min_visible_frames: int = 1,
) -> list[VideoObjectInfo]:
    """Filter objects based on visibility criteria across a temporal window.

    An object passes the filter if it meets ALL per-frame conditions
    (FOV, distance, visibility ratio) in at least ``min_visible_frames``
    sampled frames.

    Args:
        objects: List of VideoObjectInfo objects to filter.
        gt_provider: Data provider for the sequence.
        timestamps_ns: All valid timestamps in the sequence.
        stream_id: Camera stream ID for bbox projection checks.
        filter_fps: FPS at which to sample frames for filtering.
        unique: If True, only include objects with unique natural names.
        min_fov_degrees: Minimum angular size (degrees) from viewer's position.
        max_distance: Maximum distance (meters) from viewer to object.
        require_2d_bbox: If True, require objects to have a visible 2D bbox.
        min_visibility_ratio: Minimum visibility ratio (0-1) for 2D bbox.
        min_visible_frames: Minimum number of sampled frames where the object
            must pass all conditions (default: 1 for backward compat).

    Returns:
        List of objects that pass filtering criteria in enough frames.
    """
    if not objects or not timestamps_ns:
        return []

    video_duration_ns = timestamps_ns[-1] - timestamps_ns[0]
    if video_duration_ns <= 0:
        return []

    sample_interval_ns = int(1e9 / filter_fps)

    sample_timestamps = []
    current_ts = timestamps_ns[0]
    while current_ts <= timestamps_ns[-1]:
        closest_ts = min(timestamps_ns, key=lambda t: abs(t - current_ts))
        if closest_ts not in sample_timestamps:
            sample_timestamps.append(closest_ts)
        current_ts += sample_interval_ns

    pass_timestamps: dict[int, list[int]] = {obj.instance_id: [] for obj in objects}

    for ts in sample_timestamps:
        viewer_position = get_aria_position_at_timestamp(gt_provider, ts)
        if viewer_position is None:
            continue

        filtered_at_ts = filter_objects(
            objects=objects,
            gt_provider=gt_provider,
            timestamp_ns=ts,
            stream_id=stream_id,
            viewer_position=viewer_position,
            unique=unique,
            min_fov_degrees=min_fov_degrees,
            max_distance=max_distance,
            require_2d_bbox=require_2d_bbox,
            min_visibility_ratio=min_visibility_ratio,
        )

        for obj in filtered_at_ts:
            pass_timestamps[obj.instance_id].append(ts)

    result = []
    for obj in objects:
        ts_list = pass_timestamps[obj.instance_id]
        if len(ts_list) >= min_visible_frames:
            result.append(replace(obj, visible_timestamps=ts_list))
    return result


def get_globally_unique_objects(
    objects: dict[int, VideoObjectInfo],
) -> list[VideoObjectInfo]:
    """Get objects that are globally unique across the entire scene.

    An object is globally unique if its normalized name appears exactly once
    across all objects in the scene.

    Args:
        objects: Dictionary of all objects in the scene.

    Returns:
        List of globally unique objects.
    """
    name_counts: Counter[str] = Counter()
    for obj in objects.values():
        natural = obj.natural_name
        if natural:
            normalized = normalize_name_for_uniqueness(natural)
            name_counts[normalized] += 1

    unique_objects = []
    for obj in objects.values():
        natural = obj.natural_name
        if natural:
            normalized = normalize_name_for_uniqueness(natural)
            if name_counts[normalized] == 1:
                unique_objects.append(obj)

    return unique_objects


def get_multi_instance_groups(
    objects: dict[int, VideoObjectInfo],
) -> dict[str, list[VideoObjectInfo]]:
    """Get groups of objects that share the same normalized name.

    This is the complement of get_globally_unique_objects(): it returns
    objects whose normalized name appears 2+ times in the scene, grouped
    by that normalized name.

    Args:
        objects: Dictionary of all objects in the scene.

    Returns:
        Dictionary mapping normalized name to list of objects (only groups
        with 2+ instances).
    """
    # Group objects by normalized name
    groups: dict[str, list[VideoObjectInfo]] = {}
    for obj in objects.values():
        natural = obj.natural_name
        if natural:
            normalized = normalize_name_for_uniqueness(natural)
            groups.setdefault(normalized, []).append(obj)

    # Keep only groups with 2+ instances
    return {name: objs for name, objs in groups.items() if len(objs) >= 2}

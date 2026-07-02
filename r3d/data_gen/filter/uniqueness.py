# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Uniqueness-based filtering for ADT objects.

This module provides functions to filter objects based on the uniqueness
of their natural language names.
"""

from __future__ import annotations

from collections import Counter

from r3d.data_gen.extractor.object_info import ObjectInfo


# Size modifiers that should be ignored when checking uniqueness
SIZE_MODIFIERS = {"large", "small", "medium", "big", "tiny", "huge"}


def normalize_name_for_uniqueness(natural_name: str) -> str:
    """Normalize a natural name for uniqueness checking.

    Removes size modifiers like "large", "small", etc. so that
    "large black vase" and "small black vase" are considered the same.

    Args:
        natural_name: The natural language name to normalize.

    Returns:
        Normalized name with size modifiers removed.
    """
    words = natural_name.lower().split()
    filtered_words = [w for w in words if w not in SIZE_MODIFIERS]
    return " ".join(filtered_words) if filtered_words else natural_name.lower()


def filter_unique_names(objects: list[ObjectInfo]) -> list[ObjectInfo]:
    """Filter to objects with unique natural names.

    Two objects with the same normalized name (after removing size modifiers)
    are considered duplicates and both are excluded.

    Args:
        objects: List of objects to filter.

    Returns:
        Objects whose normalized natural name appears exactly once.
    """
    name_counts = Counter(
        normalize_name_for_uniqueness(obj.natural_name)
        for obj in objects
        if obj.natural_name is not None
    )
    return [
        obj
        for obj in objects
        if obj.natural_name is not None
        and name_counts[normalize_name_for_uniqueness(obj.natural_name)] == 1
    ]

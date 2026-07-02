# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Category allowlist for volume-eligible objects.

Only objects in categories where "functional volume" (how much liquid
a container can hold) makes semantic sense are eligible for volume
estimation questions.
"""

from __future__ import annotations

VOLUME_CATEGORIES: frozenset[str] = frozenset(
    {
        "bottle",
        "bowl",
        "can",
        "container",
        "cup",
        "jar",
        "pet bowl",
        "pot",
        "vase",
    }
)


def is_volume_eligible(category: str) -> bool:
    """Check whether an object's ADT category is eligible for volume questions."""
    return category.lower() in VOLUME_CATEGORIES

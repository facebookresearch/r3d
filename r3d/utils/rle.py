# Copyright (c) Meta Platforms, Inc. and affiliates.

"""COCO-style RLE (Run-Length Encoding) for binary masks.

Canonical encode/decode used by all R3D subsystems.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def encode_mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask to COCO-style RLE.

    Args:
        mask: shape (H, W) boolean array.

    Returns:
        Dict with 'counts' (list[int]) and 'size' [H, W].
    """
    h, w = mask.shape
    flat = mask.flatten(order="F").astype(np.uint8)

    if len(flat) == 0:
        return {"counts": [0], "size": [h, w]}

    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(flat)]])
    counts = (run_ends - run_starts).tolist()

    if flat[0] == 1:
        counts.insert(0, 0)
    return {"counts": counts, "size": [h, w]}


def decode_rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    """Decode COCO-style RLE to binary mask.

    Args:
        rle: Dict with 'counts' (list[int]) and 'size' [H, W].

    Returns:
        Boolean array of shape (H, W).

    Raises:
        KeyError: If 'counts' or 'size' keys are missing.
        ValueError: If run-length counts exceed the mask dimensions.
    """
    h, w = rle["size"]
    counts = np.array(rle["counts"], dtype=np.int64)
    total = h * w
    if counts.sum() != total:
        raise ValueError(f"RLE counts sum to {counts.sum()}, expected {total}")
    flat = np.zeros(total, dtype=bool)
    ends = np.cumsum(counts)
    starts = np.concatenate([[0], ends[:-1]])
    for i in range(1, len(counts), 2):
        flat[starts[i] : ends[i]] = True
    return flat.reshape((h, w), order="F")

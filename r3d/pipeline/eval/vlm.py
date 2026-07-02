# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np
from r3d.types import Message
from r3d.utils.viz import draw_bbox_from_mask, track_color

logger: logging.Logger = logging.getLogger(__name__)

RETRYABLE_KEYWORDS: list[str] = [
    "throttl",
    "rate limit",
    "saturated",
    "overloaded",
    "temporarily unavailable",
    "timeout",
    "connection",
    "503",
    "429",
]


class VLMClient(ABC):
    @abstractmethod
    def query(
        self,
        prompt: str,
        images: list[np.ndarray],
        model: str,
        max_tokens: int = 16384,
        depth_maps: list[np.ndarray] | None = None,
        masks: list[list[np.ndarray | None]] | None = None,
        object_names: list[str] | None = None,
        poses: list[np.ndarray] | None = None,
        intrinsics: object | None = None,
        gt_answer_type: str = "float",
    ) -> str: ...

    @abstractmethod
    def query_multiturn(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 16384,
    ) -> str: ...


def _overlay_masks_on_images(
    images: list[np.ndarray],
    masks: list[list[np.ndarray | None]],
    object_names: list[str],
    blend: float = 0.4,
) -> list[np.ndarray]:
    n_objects = len(object_names)
    colors = [track_color(idx) for idx in range(n_objects)]
    names = [
        object_names[idx] if idx < len(object_names) else "" for idx in range(n_objects)
    ]

    blend_f32 = np.float32(blend)
    inv_blend_f32 = np.float32(1.0 - blend)
    result = []
    for i, frame in enumerate(images):
        if i >= len(masks) or not any(m is not None for m in masks[i]):
            result.append(frame)
            continue

        h, w = frame.shape[:2]
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        any_mask = np.zeros((h, w), dtype=np.uint8)

        for obj_idx, mask in enumerate(masks[i]):
            if mask is None:
                continue
            m = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
            any_mask = cv2.bitwise_or(any_mask, m)
            color_img[m > 0] = colors[obj_idx]

        blended_full = cv2.addWeighted(
            frame, float(inv_blend_f32), color_img, float(blend_f32), 0.0
        )
        out = np.where(any_mask[:, :, None], blended_full, frame)

        for obj_idx, mask in enumerate(masks[i]):
            if mask is None:
                continue
            draw_bbox_from_mask(
                out, mask, colors[obj_idx], names[obj_idx], inplace=True
            )

        result.append(out)
    return result


def image_to_base64(image: np.ndarray) -> str:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def prepare_images_b64(
    images: list[np.ndarray],
    masks: list[list[np.ndarray | None]] | None = None,
    object_names: list[str] | None = None,
) -> list[str]:
    """CPU-heavy: overlay masks + JPEG encode + base64. Safe for ProcessPoolExecutor."""
    query_images = images
    if masks is not None and object_names is not None:
        query_images = _overlay_masks_on_images(images, masks, object_names)
    return [image_to_base64(img) for img in query_images]

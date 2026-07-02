# Copyright (c) Meta Platforms, Inc. and affiliates.

"""HuggingFace transformers-based VLM client for local model inference.

Loads any HuggingFace model and runs inference directly on GPU.
Supports both vision-language and text-only models.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from r3d.pipeline.eval.vlm import VLMClient
from r3d.types import ChatRole, Message, MessageAttachmentType

logger: logging.Logger = logging.getLogger(__name__)

VISION_MODEL_TYPES: set[str] = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "llava",
    "llava_next",
    "llava_onevision",
    "idefics2",
    "idefics3",
    "paligemma",
}


def _numpy_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image)


def _b64_to_pil(b64_data: str) -> Image.Image:
    jpeg_bytes = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(jpeg_bytes))


class HFVLMClient(VLMClient):
    """VLM client using HuggingFace transformers directly."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ) -> None:
        from transformers import AutoConfig

        self._model_name = model_name
        self._device = device
        logger.info("Loading HF model: %s", model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self._is_vision = config.model_type in VISION_MODEL_TYPES
        logger.info("Model type: %s, vision=%s", config.model_type, self._is_vision)

        if self._is_vision:
            self._init_vision_model(model_name)
        else:
            self._init_text_model(model_name)

        self._model.eval()
        logger.info("HF model loaded: %s", model_name)

    def _init_vision_model(self, model_name: str) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as VisionModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as VisionModelClass

        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self._device != "cpu":
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        self._model = VisionModelClass.from_pretrained(model_name, **load_kwargs)

    def _init_text_model(self, model_name: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._processor = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self._device != "cpu":
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    def _build_chat_messages(
        self,
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        chat_messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        role_map = {
            ChatRole.SYSTEM: "system",
            ChatRole.USER: "user",
            ChatRole.AI: "assistant",
            ChatRole.ASSISTANT: "assistant",
        }
        for msg in messages:
            role = role_map[msg.role]
            has_images = msg.attachments and any(
                a.type == MessageAttachmentType.BASE64_IMAGE for a in msg.attachments
            )
            if has_images and self._is_vision:
                content: list[dict[str, Any]] = []
                for att in msg.attachments:
                    if att.type == MessageAttachmentType.BASE64_IMAGE:
                        content.append({"type": "image"})
                        images.append(_b64_to_pil(att.data))
                content.append({"type": "text", "text": msg.text})
                chat_messages.append({"role": role, "content": content})
            else:
                chat_messages.append({"role": role, "content": msg.text})
        return chat_messages, images

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
    ) -> str:
        if self._is_vision and images:
            content: list[dict[str, Any]] = []
            pil_images: list[Image.Image] = []
            for img in images:
                content.append({"type": "image"})
                pil_images.append(_numpy_to_pil(img))
            content.append({"type": "text", "text": prompt})
            chat_messages = [{"role": "user", "content": content}]
            return self._generate_vision(chat_messages, pil_images, max_tokens)
        chat_messages = [{"role": "user", "content": prompt}]
        return self._generate_text(chat_messages, max_tokens)

    def query_multiturn(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 16384,
    ) -> str:
        chat_messages, images = self._build_chat_messages(messages)
        if self._is_vision and images:
            return self._generate_vision(chat_messages, images, max_tokens)
        return self._generate_text(chat_messages, max_tokens)

    def _generate_vision(
        self,
        chat_messages: list[dict[str, Any]],
        images: list[Image.Image],
        max_tokens: int,
    ) -> str:
        text = self._processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        generated_ids = output_ids[0][input_len:]
        del inputs, output_ids
        torch.cuda.empty_cache()
        return self._processor.decode(generated_ids, skip_special_tokens=True)

    def _generate_text(
        self,
        chat_messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        generated_ids = output_ids[0][input_len:]
        del inputs, output_ids
        torch.cuda.empty_cache()
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

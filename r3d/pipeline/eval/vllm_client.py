# Copyright (c) Meta Platforms, Inc. and affiliates.

"""vLLM-based VLM client with multi-GPU data parallelism.

Spawns N worker processes (one per data-parallel rank), each running a
vLLM LLM instance on its assigned GPU(s). Queries are sticky-routed by
caller thread ID for prefix cache locality across tool_use turns.

Single-GPU usage (data_parallel_size=1) runs in-process with no workers.
"""

from __future__ import annotations

import base64
import io
import logging
import multiprocessing as mp
import os
import threading
from typing import Any

import numpy as np
from PIL import Image
from r3d.pipeline.eval.vlm import VLMClient
from r3d.types import ChatRole, Message, MessageAttachmentType

logger: logging.Logger = logging.getLogger(__name__)


def _numpy_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image)


def _b64_to_pil(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _build_generate_request(
    text: str, images: list[Image.Image]
) -> dict[str, Any]:
    """Build a vLLM generate request, attaching images as multi-modal data."""
    request: dict[str, Any] = {"prompt": text}
    if images:
        request["multi_modal_data"] = {"image": images}
    return request


def _worker_loop(
    rank: int,
    model_name: str,
    gpu_ids: list[int],
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_images_per_prompt: int | None,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": True,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if max_images_per_prompt is not None:
        llm_kwargs["limit_mm_per_prompt"] = {"image": max_images_per_prompt}
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    while True:
        item = request_queue.get()
        if item is None:
            break
        caller_id, chat_messages, max_tokens, images = item
        # No-thinking is the default; thinking models generate direct answers.
        text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = llm.generate([_build_generate_request(text, images)], params)
        response_queue.put((caller_id, outputs[0].outputs[0].text))


class VLLMClient(VLMClient):
    """vLLM client with optional multi-GPU data parallelism."""

    def __init__(
        self,
        model_name: str,
        data_parallel_size: int = 1,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        max_images_per_prompt: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._dp_size = data_parallel_size
        self._max_images_per_prompt = max_images_per_prompt

        if data_parallel_size <= 1:
            self._init_single(model_name, gpu_memory_utilization, max_model_len)
        else:
            self._init_multi(
                model_name, data_parallel_size, gpu_memory_utilization, max_model_len
            )

    def _init_single(
        self,
        model_name: str,
        gpu_memory_utilization: float,
        max_model_len: int | None,
    ) -> None:
        from vllm import LLM

        logger.info("Loading vLLM model: %s (dp=1)", model_name)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": True,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if self._max_images_per_prompt is not None:
            kwargs["limit_mm_per_prompt"] = {"image": self._max_images_per_prompt}
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self._workers = None
        logger.info("vLLM model loaded: %s", model_name)

    def _init_multi(
        self,
        model_name: str,
        data_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
    ) -> None:
        logger.info(
            "Starting vLLM workers: %s (dp=%d)",
            model_name,
            data_parallel_size,
        )
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_env:
            available_gpus = [int(g) for g in cuda_env.split(",")][:data_parallel_size]
        else:
            available_gpus = list(range(data_parallel_size))

        ctx = mp.get_context("spawn")
        self._request_queues: list[mp.Queue] = []
        self._response_queue: mp.Queue = ctx.Queue()
        self._workers: list[mp.Process] = []
        self._next_worker = 0
        self._thread_to_worker: dict[int, int] = {}
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, str] = {}
        self._lock = threading.Lock()

        self._dispatcher = threading.Thread(
            target=self._dispatch_responses, daemon=True
        )

        for rank in range(data_parallel_size):
            gpu_ids = [available_gpus[rank]]
            req_q: mp.Queue = ctx.Queue()
            self._request_queues.append(req_q)

            p = ctx.Process(
                target=_worker_loop,
                args=(
                    rank,
                    model_name,
                    gpu_ids,
                    gpu_memory_utilization,
                    max_model_len,
                    self._max_images_per_prompt,
                    req_q,
                    self._response_queue,
                ),
                daemon=False,
            )
            p.start()
            self._workers.append(p)
            logger.info("  Worker %d started (pid=%d, gpus=%s)", rank, p.pid, gpu_ids)

        self._dispatcher.start()
        self._llm = None
        self._tokenizer = None

    def _dispatch_responses(self) -> None:
        while True:
            try:
                item = self._response_queue.get(timeout=1.0)
            except Exception:
                if self._workers is None:
                    break
                continue
            caller_id, text = item
            with self._lock:
                self._results[caller_id] = text
                event = self._pending.get(caller_id)
            if event:
                event.set()

    def _get_worker_for_caller(self) -> int:
        tid = threading.get_ident()
        if tid not in self._thread_to_worker:
            worker = self._next_worker % self._dp_size
            self._next_worker += 1
            self._thread_to_worker[tid] = worker
        return self._thread_to_worker[tid]

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
        pil_images = [_numpy_to_pil(img) for img in images] if images else []
        if pil_images:
            content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_images]
            content.append({"type": "text", "text": prompt})
            chat_messages = [{"role": "user", "content": content}]
        else:
            chat_messages = [{"role": "user", "content": prompt}]
        return self._generate(chat_messages, max_tokens, pil_images)

    def query_multiturn(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 16384,
    ) -> str:
        chat_messages, images = self._build_chat_messages(messages)
        return self._generate(chat_messages, max_tokens, images)

    def _build_chat_messages(
        self,
        messages: list[Message],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        role_map = {
            ChatRole.USER: "user",
            ChatRole.ASSISTANT: "assistant",
            ChatRole.AI: "assistant",
            ChatRole.SYSTEM: "system",
        }
        chat_messages: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for msg in messages:
            role = role_map[msg.role]
            img_atts = [
                a
                for a in (msg.attachments or [])
                if a.type == MessageAttachmentType.BASE64_IMAGE
            ]
            if img_atts:
                content: list[dict[str, Any]] = [{"type": "image"} for _ in img_atts]
                content.append({"type": "text", "text": msg.text})
                chat_messages.append({"role": role, "content": content})
                images.extend(_b64_to_pil(a.data) for a in img_atts)
            else:
                chat_messages.append({"role": role, "content": msg.text})
        return chat_messages, images

    def _generate(
        self,
        chat_messages: list[dict[str, Any]],
        max_tokens: int,
        images: list[Image.Image],
    ) -> str:
        if self._workers is None:
            return self._generate_single(chat_messages, max_tokens, images)
        return self._generate_multi(chat_messages, max_tokens, images)

    def _generate_single(
        self,
        chat_messages: list[dict[str, Any]],
        max_tokens: int,
        images: list[Image.Image],
    ) -> str:
        from vllm import SamplingParams

        # No-thinking is the default; thinking models generate direct answers.
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        try:
            outputs = self._llm.generate(
                [_build_generate_request(text, images)], params
            )
        except Exception as e:
            if "context length" in str(e) or "input tokens" in str(e):
                logger.warning("Prompt too long, returning empty: %s", e)
                return ""
            raise
        return outputs[0].outputs[0].text

    def _generate_multi(
        self,
        chat_messages: list[dict[str, Any]],
        max_tokens: int,
        images: list[Image.Image],
    ) -> str:
        caller_id = threading.get_ident()
        worker_rank = self._get_worker_for_caller()
        event = threading.Event()
        with self._lock:
            self._pending[caller_id] = event
        self._request_queues[worker_rank].put(
            (caller_id, chat_messages, max_tokens, images)
        )
        event.wait()
        with self._lock:
            result = self._results.pop(caller_id)
            del self._pending[caller_id]
        return result

    def shutdown(self) -> None:
        if self._workers is None:
            return
        for q in self._request_queues:
            q.put(None)
        for w in self._workers:
            w.join(timeout=30)
            if w.is_alive():
                w.terminate()
        self._workers = None

# Copyright (c) Meta Platforms, Inc. and affiliates.

"""VLLMClient must raise, not return an empty answer, when generation fails."""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest
from typing import Any
from unittest import mock

from r3d.pipeline.eval.vllm_client import VLLMClient

_CONTEXT_ERROR = (
    "This model's maximum context length is 32768 tokens. However, you "
    "requested 40000 input tokens."
)


class _FakeTokenizer:
    def apply_chat_template(self, messages: list[Any], **kwargs: Any) -> str:
        return "prompt"


class _RaisingLLM:
    def generate(self, requests: list[Any], params: Any) -> list[Any]:
        raise ValueError(_CONTEXT_ERROR)


class _ErrorQueue:
    """Request queue whose worker always answers with an error."""

    def __init__(self, client: VLLMClient) -> None:
        self._client = client

    def put(self, item: tuple[Any, ...]) -> None:
        caller_id = item[0]
        with self._client._lock:
            self._client._results[caller_id] = (None, f"ValueError: {_CONTEXT_ERROR}")
            event = self._client._pending[caller_id]
        event.set()


def _fake_vllm_module() -> types.ModuleType:
    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kwargs: kwargs  # pyre-ignore[16]
    return mod


class TestVLLMClientFailsLoudly(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = sys.modules.get("vllm")
        sys.modules["vllm"] = _fake_vllm_module()

    def tearDown(self) -> None:
        if self._saved is None:
            del sys.modules["vllm"]
        else:
            sys.modules["vllm"] = self._saved

    def test_single_process_context_overflow_raises(self) -> None:
        client = VLLMClient.__new__(VLLMClient)
        client._workers = None
        client._llm = _RaisingLLM()
        client._tokenizer = _FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "maximum context length"):
            client.query("q", [], model="m")

    def test_data_parallel_worker_error_raises(self) -> None:
        client = VLLMClient.__new__(VLLMClient)
        client._workers = []
        client._dp_size = 1
        client._next_worker = 0
        client._thread_to_worker = {}
        client._pending = {}
        client._results = {}
        client._lock = threading.Lock()
        client._request_queues = [_ErrorQueue(client)]
        with self.assertRaisesRegex(RuntimeError, "maximum context length"):
            client.query("q", [], model="m")


class TestVLLMClientParallelism(unittest.TestCase):
    def test_tensor_parallel_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
            VLLMClient("m", tensor_parallel_size=0)

    def test_dp_times_tp_must_fit_visible_gpus(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}):
            with self.assertRaisesRegex(ValueError, "needs 4 GPUs"):
                VLLMClient("m", data_parallel_size=2, tensor_parallel_size=2)


if __name__ == "__main__":
    unittest.main()

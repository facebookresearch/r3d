# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Per-question prompt token accounting (sidecar ``prompt_stats.jsonl``).

Counts tokens with the served model's tokenizer and the same chat-template
arguments the vLLM client uses (``add_generation_prompt=True``,
``enable_thinking=False``).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import AutoTokenizer

PROMPT_STATS_FILENAME: str = "prompt_stats.jsonl"


@dataclass(frozen=True)
class PromptStats:
    annotation_id: str
    question_type: str
    strategy: str
    num_objects: int
    prompt_tokens: int
    completion_tokens: int


class PromptTokenCounter:
    def __init__(self, hf_model: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_model, trust_remote_code=True
        )
        self._lock = threading.Lock()

    def prompt_tokens(self, chat: list[dict[str, str]]) -> int:
        with self._lock:
            text = self._tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return self.text_tokens(text)

    def text_tokens(self, text: str) -> int:
        with self._lock:
            return len(self._tokenizer.encode(text, add_special_tokens=False))


class PromptStatsWriter:
    """Thread-safe JSONL appender, one record per answered question."""

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / PROMPT_STATS_FILENAME
        self._lock = threading.Lock()

    def write(self, stats: PromptStats) -> None:
        line = json.dumps(asdict(stats))
        with self._lock, open(self._path, "a") as f:
            f.write(line + "\n")

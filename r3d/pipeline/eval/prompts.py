# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

ANSWER_FORMAT_INSTRUCTIONS = """\
CRITICAL: When giving your final answer, format it exactly as follows:
- For distances/measurements: ANSWER: <number> (e.g. ANSWER: 2.46)
- For yes/no questions: ANSWER: yes or ANSWER: no
- For "which" questions (e.g. "which is taller"): ANSWER: <object name> (e.g. ANSWER: wooden fork)
Do NOT include units or explanations on the ANSWER line."""

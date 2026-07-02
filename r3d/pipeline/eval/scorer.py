# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import re
import time

from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.scores import Score

_NUMERIC_ANSWER_PATTERN: re.Pattern[str] = re.compile(
    r"ANSWER\s*:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE
)

_BARE_NUMERIC_PATTERN: re.Pattern[str] = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*$")

_BOOL_ANSWER_PATTERN: re.Pattern[str] = re.compile(
    r"ANSWER\s*:\s*(yes|no|true|false)\b", re.IGNORECASE
)

_BARE_BOOL_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(yes|no|true|false)\s*$", re.IGNORECASE
)

_STR_ANSWER_PATTERN: re.Pattern[str] = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)


def _parse_numeric_answer(response: str) -> float | None:
    matches = _NUMERIC_ANSWER_PATTERN.findall(response)
    if matches:
        return float(matches[-1])
    bare = _BARE_NUMERIC_PATTERN.match(response)
    if bare:
        return float(bare.group(1))
    return None


def _parse_bool_answer(response: str) -> bool | None:
    matches = _BOOL_ANSWER_PATTERN.findall(response)
    if matches:
        return matches[-1].lower() in ("yes", "true")
    bare = _BARE_BOOL_PATTERN.match(response)
    if bare:
        return bare.group(1).lower() in ("yes", "true")
    return None


def _parse_str_answer(response: str) -> str | None:
    matches = _STR_ANSWER_PATTERN.findall(response)
    if matches:
        raw = matches[-1]
    else:
        raw = response
    raw = raw.strip().strip("\"'").strip()
    raw = re.sub(r"^(the|a|an)\s+", "", raw, flags=re.IGNORECASE)
    raw = raw.rstrip(".,;:!?")
    return raw.lower() if raw else None


def _percentage_error(predicted: float, gt: float) -> float:
    if gt == 0:
        return abs(predicted) * 100.0
    return abs(predicted - gt) / abs(gt) * 100.0


def _make_parse_failed_score(
    annotation: Annotation,
    gt_answer: str,
    model: str,
    strategy: str,
    now_ns: int,
) -> Score:
    return Score(
        annotation_id=annotation.annotation_id,
        model=model,
        strategy=strategy,
        parsed_answer=None,
        gt_answer=gt_answer,
        percentage_error=None,
        accuracy=None,
        parse_failed=True,
        created_ns=now_ns,
    )


def score_response(
    annotation: Annotation,
    response: str,
    model: str,
    strategy: str,
) -> Score:
    gt_type = annotation.query_layer.gt_answer_type
    now_ns = int(time.time() * 1e9)

    if gt_type == "float":
        return _score_float(annotation, response, model, strategy, now_ns)
    elif gt_type == "bool":
        return _score_bool(annotation, response, model, strategy, now_ns)
    elif gt_type == "str":
        return _score_str(annotation, response, model, strategy, now_ns)
    else:
        raise RuntimeError(
            f"Annotation {annotation.annotation_id} has unsupported "
            f"gt_answer_type: {gt_type!r}"
        )


def _score_float(
    annotation: Annotation,
    response: str,
    model: str,
    strategy: str,
    now_ns: int,
) -> Score:
    gt_value = float(annotation.query_layer.gt_answer)
    parsed = _parse_numeric_answer(response)
    if parsed is None:
        return _make_parse_failed_score(
            annotation, str(gt_value), model, strategy, now_ns
        )
    pct_err = _percentage_error(parsed, gt_value)
    return Score(
        annotation_id=annotation.annotation_id,
        model=model,
        strategy=strategy,
        parsed_answer=str(parsed),
        gt_answer=str(gt_value),
        percentage_error=pct_err,
        accuracy=None,
        parse_failed=False,
        created_ns=now_ns,
    )


def _score_bool(
    annotation: Annotation,
    response: str,
    model: str,
    strategy: str,
    now_ns: int,
) -> Score:
    gt_bool = annotation.query_layer.gt_answer.lower() in ("true", "yes")
    parsed = _parse_bool_answer(response)
    if parsed is None:
        return _make_parse_failed_score(
            annotation, str(gt_bool), model, strategy, now_ns
        )
    return Score(
        annotation_id=annotation.annotation_id,
        model=model,
        strategy=strategy,
        parsed_answer=str(parsed),
        gt_answer=str(gt_bool),
        percentage_error=None,
        accuracy=100.0 if parsed == gt_bool else 0.0,
        parse_failed=False,
        created_ns=now_ns,
    )


def _get_referenced_object_names(annotation: Annotation) -> list[str]:
    return [
        ref.canonical_name.lower()
        for ref in annotation.identity_layer.referenced_objects
    ]


def _answer_matches_referenced_object(parsed: str, object_names: list[str]) -> bool:
    if not object_names:
        return True
    parsed_lower = parsed.lower().strip()
    if not parsed_lower:
        return False
    for name in object_names:
        if parsed_lower == name or parsed_lower in name or name in parsed_lower:
            return True
    return False


def _score_str(
    annotation: Annotation,
    response: str,
    model: str,
    strategy: str,
    now_ns: int,
) -> Score:
    gt_str = annotation.query_layer.gt_answer.lower().strip()
    parsed = _parse_str_answer(response)
    if parsed is None:
        return _make_parse_failed_score(annotation, gt_str, model, strategy, now_ns)
    obj_names = _get_referenced_object_names(annotation)
    if not _answer_matches_referenced_object(parsed, obj_names):
        return _make_parse_failed_score(annotation, gt_str, model, strategy, now_ns)
    return Score(
        annotation_id=annotation.annotation_id,
        model=model,
        strategy=strategy,
        parsed_answer=parsed,
        gt_answer=gt_str,
        percentage_error=None,
        accuracy=100.0 if parsed == gt_str else 0.0,
        parse_failed=False,
        created_ns=now_ns,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.scores import Score, ScoreStore

logger: logging.Logger = logging.getLogger(__name__)


_MRA_THRESHOLDS: list[float] = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

_FLOAT_ANSWER_TYPE = "float"
_ACCURACY_ANSWER_TYPES = ("bool", "str")


_PF_AS_INCORRECT_NOTE = (
    "Parse failures counted as incorrect (accuracy/MRA/within-X = 0); "
    "mean/median % error computed over parsed answers only."
)


_STAT_KEYS: list[str] = [
    "total",
    "parse_failures",
    "valid",
    "mean_percentage_error",
    "median_percentage_error",
    "mean_accuracy",
    "within_5_pct",
    "within_10_pct",
    "within_25_pct",
    "mra",
]


def generate_report(
    score_store: ScoreStore,
    annotations: list[Annotation],
    output_dir: Path,
    parse_failures_as_incorrect: bool = True,
) -> None:
    """Write eval_report.json and eval_summary.txt for the scores in the store.

    By default (``parse_failures_as_incorrect``, as in the paper) parse failures
    stay in the denominator and score 0: accuracy 0 for MC questions, and MRA 0 /
    not-within-threshold for float questions. With
    ``parse_failures_as_incorrect=False`` they are excluded from every metric.
    Mean and median % error have no defined value for an unparsed answer, so
    they are always computed over parsed answers only.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = score_store.get_all()
    ann_map = {a.annotation_id: a for a in annotations}
    answer_types = (
        _answer_types_by_qtype(scores, ann_map) if parse_failures_as_incorrect else None
    )

    by_qtype: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        ann = ann_map[s.annotation_id]
        qtype = ann.query_layer.question_type.value
        by_qtype[qtype].append(s)

    aggregate = _compute_stats(by_qtype, answer_types)

    by_seq_qtype: dict[str, dict[str, list[Score]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for s in scores:
        ann = ann_map[s.annotation_id]
        seq_id = ann.identity_layer.sequence_id
        qtype = ann.query_layer.question_type.value
        by_seq_qtype[seq_id][qtype].append(s)

    by_sequence = {
        seq_id: _compute_stats(qtypes, answer_types)
        for seq_id, qtypes in sorted(by_seq_qtype.items())
    }

    full_report: dict[str, object] = {
        "aggregate": aggregate,
        "by_sequence": by_sequence,
    }
    lines: list[str] = ["Eval Report", "=" * 60, ""]
    full_report["parse_failures_as_incorrect"] = parse_failures_as_incorrect
    if parse_failures_as_incorrect:
        lines.extend([_PF_AS_INCORRECT_NOTE, ""])
    lines.extend(_format_stats(aggregate))
    for seq_id, seq_stats in sorted(by_sequence.items()):
        lines.extend(["", f"--- {seq_id} ---", ""])
        lines.extend(_format_stats(seq_stats))

    _write_report_files(full_report, "\n".join(lines), output_dir)


def _write_report_files(
    full_report: dict[str, object], summary: str, output_dir: Path
) -> None:
    json_path = output_dir / "eval_report.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    text_path = output_dir / "eval_summary.txt"
    with open(text_path, "w") as f:
        f.write(summary)

    logger.info(f"Report written to {json_path}")
    logger.info(f"Summary written to {text_path}")
    logger.info("\n" + summary)


def _answer_types_by_qtype(
    scores: list[Score], ann_map: dict[str, Annotation]
) -> dict[str, str]:
    """Map each question type to its single gt_answer_type; raise if mixed."""
    result: dict[str, str] = {}
    for s in scores:
        ql = ann_map[s.annotation_id].query_layer
        qtype = ql.question_type.value
        gt_type = ql.gt_answer_type
        if gt_type != _FLOAT_ANSWER_TYPE and gt_type not in _ACCURACY_ANSWER_TYPES:
            raise RuntimeError(
                f"Annotation {s.annotation_id} has unsupported gt_answer_type "
                f"{gt_type!r}"
            )
        if result.setdefault(qtype, gt_type) != gt_type:
            raise RuntimeError(
                f"Question type {qtype!r} mixes gt_answer_types "
                f"{result[qtype]!r} and {gt_type!r}; cannot assign parse "
                "failures to a single metric"
            )
    return result


def _compute_stats(
    by_qtype: dict[str, list[Score]],
    answer_types: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Per-question-type stats.

    ``answer_types`` (qtype -> gt_answer_type) enables parse-failures-as-
    incorrect mode; ``None`` excludes parse failures from every metric.
    """
    result: dict[str, dict] = {}
    for qtype in sorted(by_qtype.keys()):
        qscores = by_qtype[qtype]
        parse_failures = sum(1 for s in qscores if s.parse_failed)
        valid = [s for s in qscores if not s.parse_failed]
        errors = [s.percentage_error for s in valid if s.percentage_error is not None]
        accuracies = [s.accuracy for s in valid if s.accuracy is not None]
        pf_acc, pf_float = 0, 0
        if answer_types is not None:
            if answer_types[qtype] == _FLOAT_ANSWER_TYPE:
                pf_float = parse_failures
            else:
                pf_acc = parse_failures
        stats = {
            "total": len(qscores),
            "parse_failures": parse_failures,
            "valid": len(valid),
        }
        stats.update(_error_stats(errors))
        stats.update(_scored_stats(accuracies, errors, pf_acc, pf_float))
        result[qtype] = {k: stats[k] for k in _STAT_KEYS}
        if answer_types is not None:
            result[qtype]["parse_failures_scored_as_incorrect"] = pf_acc + pf_float
    return result


def _error_stats(errors: list[float]) -> dict[str, float | None]:
    """Mean/median % error over parsed answers (never includes failures)."""
    return {
        "mean_percentage_error": sum(errors) / len(errors) if errors else None,
        "median_percentage_error": _median(errors) if errors else None,
    }


def _scored_stats(
    accuracies: list[float],
    errors: list[float],
    pf_acc: int,
    pf_float: int,
) -> dict[str, float | None]:
    """Accuracy, within-X and MRA; ``pf_*`` failures each contribute a 0.

    Failures only enlarge the denominator, so with ``pf_* == 0`` the floating
    point result is identical to excluding them.
    """
    n_acc = len(accuracies) + pf_acc
    n_err = len(errors) + pf_float

    def within(threshold: float) -> float | None:
        if not n_err:
            return None
        return sum(1 for e in errors if e <= threshold) / n_err * 100

    mra = (
        sum(
            sum(1 for theta in _MRA_THRESHOLDS if e < (1 - theta) * 100)
            / len(_MRA_THRESHOLDS)
            for e in errors
        )
        / n_err
        * 100
        if n_err
        else None
    )
    return {
        "mean_accuracy": sum(accuracies) / n_acc if n_acc else None,
        "within_5_pct": within(5.0),
        "within_10_pct": within(10.0),
        "within_25_pct": within(25.0),
        "mra": mra,
    }


def _format_stats(stats: dict[str, dict]) -> list[str]:
    lines: list[str] = []
    for qtype, data in sorted(stats.items()):
        lines.append(f"{qtype}: {data['valid']}/{data['total']} valid")
        if data["mean_accuracy"] is not None:
            lines.append(f"  Mean accuracy:  {data['mean_accuracy']:.0f}%")
        if data["mean_percentage_error"] is not None:
            lines.append(f"  Mean % error:   {data['mean_percentage_error']:.1f}%")
        if data["median_percentage_error"] is not None:
            lines.append(f"  Median % error: {data['median_percentage_error']:.1f}%")
        if data["within_5_pct"] is not None:
            lines.append(f"  Within 5%:      {data['within_5_pct']:.0f}%")
        if data["within_10_pct"] is not None:
            lines.append(f"  Within 10%:     {data['within_10_pct']:.0f}%")
        if data["within_25_pct"] is not None:
            lines.append(f"  Within 25%:     {data['within_25_pct']:.0f}%")
        if data["mra"] is not None:
            lines.append(f"  MRA:            {data['mra']:.1f}%")
        if data["parse_failures"] > 0:
            lines.append(f"  Parse failures: {data['parse_failures']}")
        lines.append("")
    return lines


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

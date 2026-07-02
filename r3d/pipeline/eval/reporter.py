# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from r3d.data_gen.utils.annotation_schema import Annotation
from r3d.pipeline.eval.scores import Score, ScoreStore

logger: logging.Logger = logging.getLogger(__name__)


def generate_report(
    score_store: ScoreStore,
    annotations: list[Annotation],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = score_store.get_all()
    ann_map = {a.annotation_id: a for a in annotations}

    by_qtype: dict[str, list[Score]] = defaultdict(list)
    for s in scores:
        ann = ann_map[s.annotation_id]
        qtype = ann.query_layer.question_type.value
        by_qtype[qtype].append(s)

    aggregate = _compute_stats(by_qtype)

    by_seq_qtype: dict[str, dict[str, list[Score]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for s in scores:
        ann = ann_map[s.annotation_id]
        seq_id = ann.identity_layer.sequence_id
        qtype = ann.query_layer.question_type.value
        by_seq_qtype[seq_id][qtype].append(s)

    by_sequence = {
        seq_id: _compute_stats(qtypes)
        for seq_id, qtypes in sorted(by_seq_qtype.items())
    }

    full_report = {"aggregate": aggregate, "by_sequence": by_sequence}

    lines: list[str] = ["Eval Report", "=" * 60, ""]
    lines.extend(_format_stats(aggregate))
    for seq_id, seq_stats in sorted(by_sequence.items()):
        lines.extend(["", f"--- {seq_id} ---", ""])
        lines.extend(_format_stats(seq_stats))

    json_path = output_dir / "eval_report.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    text_path = output_dir / "eval_summary.txt"
    summary = "\n".join(lines)
    with open(text_path, "w") as f:
        f.write(summary)

    logger.info(f"Report written to {json_path}")
    logger.info(f"Summary written to {text_path}")
    logger.info("\n" + summary)


def _compute_stats(
    by_qtype: dict[str, list[Score]],
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for qtype in sorted(by_qtype.keys()):
        qscores = by_qtype[qtype]
        total = len(qscores)
        parse_failures = sum(1 for s in qscores if s.parse_failed)
        valid = [s for s in qscores if not s.parse_failed]
        errors = [s.percentage_error for s in valid if s.percentage_error is not None]
        accuracies = [s.accuracy for s in valid if s.accuracy is not None]
        mean_err = sum(errors) / len(errors) if errors else None
        median_err = _median(errors) if errors else None
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else None
        within_5 = (
            sum(1 for e in errors if e <= 5.0) / len(errors) * 100 if errors else None
        )
        within_10 = (
            sum(1 for e in errors if e <= 10.0) / len(errors) * 100 if errors else None
        )
        within_25 = (
            sum(1 for e in errors if e <= 25.0) / len(errors) * 100 if errors else None
        )
        mra_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        mra = (
            sum(
                sum(1 for theta in mra_thresholds if e < (1 - theta) * 100)
                / len(mra_thresholds)
                for e in errors
            )
            / len(errors)
            * 100
            if errors
            else None
        )
        result[qtype] = {
            "total": total,
            "parse_failures": parse_failures,
            "valid": len(valid),
            "mean_percentage_error": mean_err,
            "median_percentage_error": median_err,
            "mean_accuracy": mean_acc,
            "within_5_pct": within_5,
            "within_10_pct": within_10,
            "within_25_pct": within_25,
            "mra": mra,
        }
    return result


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

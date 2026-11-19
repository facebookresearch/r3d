# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from r3d.data_gen.utils.annotation_schema import (
    Annotation,
    EvalMetric,
    EvalMode,
    IdentityLayer,
    QueryLayer,
    QuestionType,
    ReferencedObject,
    ReleaseType,
)
from r3d.pipeline.eval.reporter import _compute_stats, generate_report
from r3d.pipeline.eval.scores import Score, ScoreStore

MC = QuestionType.GAP_FIT.value
FLOAT = QuestionType.GLOBAL_HOW_FAR.value
ANSWER_TYPES: dict[str, str] = {MC: "bool", FLOAT: "float"}


def _score(
    ann_id: str,
    accuracy: float | None = None,
    pct_err: float | None = None,
    parse_failed: bool = False,
) -> Score:
    return Score(
        annotation_id=ann_id,
        model="m",
        strategy="s",
        parsed_answer=None if parse_failed else "x",
        gt_answer="x",
        percentage_error=pct_err,
        accuracy=accuracy,
        parse_failed=parse_failed,
        created_ns=0,
    )


def _mc_scores() -> list[Score]:
    return [
        _score("mc1", accuracy=100.0),
        _score("mc2", accuracy=100.0),
        _score("mc3", accuracy=0.0),
        _score("mc4", parse_failed=True),
    ]


def _float_scores() -> list[Score]:
    # 0% error -> MRA 1.0; 32% error -> passes theta 0.50..0.65 -> MRA 0.4.
    return [
        _score("f1", pct_err=0.0),
        _score("f2", pct_err=32.0),
        _score("f3", parse_failed=True),
    ]


def _annotation(ann_id: str, qtype: QuestionType, gt_type: str) -> Annotation:
    is_float = gt_type == "float"
    return Annotation(
        annotation_id=ann_id,
        identity_layer=IdentityLayer(
            sequence_id="seq001",
            release_type=ReleaseType.FULL,
            referenced_objects=[
                ReferencedObject(
                    object_id="obj_001",
                    adt_instance_id=1,
                    adt_instance_name="Mug",
                    prototype_name="Mug",
                    canonical_name="mug",
                    is_dynamic=False,
                )
            ],
        ),
        query_layer=QueryLayer(
            question_type=qtype,
            question_text="q?",
            gt_answer="1.0" if is_float else "True",
            gt_answer_type=gt_type,
            gt_answer_unit="meters" if is_float else None,
            eval_mode=EvalMode.DETERMINISTIC,
            eval_metric=(
                EvalMetric.PERCENTAGE_ERROR if is_float else EvalMetric.ACCURACY
            ),
            query_timestamp_ns_start=0,
            query_timestamp_ns_end=0,
        ),
    )


class TestComputeStatsDefault(unittest.TestCase):
    def test_mc_excludes_parse_failures(self) -> None:
        stats = _compute_stats({MC: _mc_scores()})[MC]
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["parse_failures"], 1)
        self.assertEqual(stats["valid"], 3)
        self.assertAlmostEqual(stats["mean_accuracy"], 200.0 / 3)
        self.assertIsNone(stats["mra"])
        self.assertNotIn("parse_failures_scored_as_incorrect", stats)

    def test_float_excludes_parse_failures(self) -> None:
        stats = _compute_stats({FLOAT: _float_scores()})[FLOAT]
        self.assertAlmostEqual(stats["mra"], 70.0)
        self.assertAlmostEqual(stats["within_5_pct"], 50.0)
        self.assertAlmostEqual(stats["within_25_pct"], 50.0)
        self.assertAlmostEqual(stats["within_10_pct"], 50.0)
        self.assertAlmostEqual(stats["mean_percentage_error"], 16.0)
        self.assertAlmostEqual(stats["median_percentage_error"], 16.0)
        self.assertIsNone(stats["mean_accuracy"])

    def test_all_failed_is_none(self) -> None:
        stats = _compute_stats({FLOAT: [_score("f", parse_failed=True)]})[FLOAT]
        self.assertIsNone(stats["mra"])
        self.assertIsNone(stats["within_5_pct"])


class TestComputeStatsPfAsIncorrect(unittest.TestCase):
    def test_mc_counts_failure_as_zero(self) -> None:
        stats = _compute_stats({MC: _mc_scores()}, ANSWER_TYPES)[MC]
        self.assertAlmostEqual(stats["mean_accuracy"], 50.0)
        self.assertEqual(stats["valid"], 3)
        self.assertEqual(stats["parse_failures"], 1)
        self.assertEqual(stats["parse_failures_scored_as_incorrect"], 1)
        self.assertIsNone(stats["mra"])

    def test_float_counts_failure_as_zero(self) -> None:
        stats = _compute_stats({FLOAT: _float_scores()}, ANSWER_TYPES)[FLOAT]
        self.assertAlmostEqual(stats["mra"], 140.0 / 3)
        self.assertAlmostEqual(stats["within_5_pct"], 100.0 / 3)
        self.assertAlmostEqual(stats["within_10_pct"], 100.0 / 3)
        self.assertAlmostEqual(stats["within_25_pct"], 100.0 / 3)
        self.assertIsNone(stats["mean_accuracy"])

    def test_percentage_error_excludes_failures(self) -> None:
        stats = _compute_stats({FLOAT: _float_scores()}, ANSWER_TYPES)[FLOAT]
        self.assertAlmostEqual(stats["mean_percentage_error"], 16.0)
        self.assertAlmostEqual(stats["median_percentage_error"], 16.0)

    def test_all_failed_is_zero(self) -> None:
        by_qtype = {FLOAT: [_score("f", parse_failed=True)]}
        stats = _compute_stats(by_qtype, ANSWER_TYPES)[FLOAT]
        self.assertEqual(stats["mra"], 0.0)
        self.assertEqual(stats["within_5_pct"], 0.0)
        self.assertIsNone(stats["mean_percentage_error"])

    def test_no_failures_matches_default(self) -> None:
        by_qtype = {
            MC: [s for s in _mc_scores() if not s.parse_failed],
            FLOAT: [s for s in _float_scores() if not s.parse_failed],
        }
        default = _compute_stats(by_qtype)
        flagged = _compute_stats(by_qtype, ANSWER_TYPES)
        for qtype, stats in default.items():
            flagged_stats = dict(flagged[qtype])
            self.assertEqual(flagged_stats.pop("parse_failures_scored_as_incorrect"), 0)
            self.assertEqual(flagged_stats, stats)


class TestGenerateReport(unittest.TestCase):
    def _run(self, pf_as_incorrect: bool | None) -> dict[str, object]:
        annotations = [
            _annotation(s.annotation_id, QuestionType.GAP_FIT, "bool")
            for s in _mc_scores()
        ] + [
            _annotation(s.annotation_id, QuestionType.GLOBAL_HOW_FAR, "float")
            for s in _float_scores()
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = ScoreStore(Path(tmp) / "scores.db")
            for s in _mc_scores() + _float_scores():
                store.write(s)
            out = Path(tmp) / "out"
            if pf_as_incorrect is None:
                generate_report(store, annotations, out)
            else:
                generate_report(store, annotations, out, pf_as_incorrect)
            store.close()
            with open(Path(tmp) / "out" / "eval_report.json") as f:
                return json.load(f)

    def test_default_counts_parse_failures_as_incorrect(self) -> None:
        self.assertEqual(
            self._run(pf_as_incorrect=None), self._run(pf_as_incorrect=True)
        )

    def test_excluded_report_layout(self) -> None:
        report = self._run(pf_as_incorrect=False)
        self.assertEqual(
            set(report.keys()),
            {"aggregate", "by_sequence", "parse_failures_as_incorrect"},
        )
        self.assertFalse(report["parse_failures_as_incorrect"])
        agg = report["aggregate"]
        self.assertAlmostEqual(agg[MC]["mean_accuracy"], 200.0 / 3)
        self.assertAlmostEqual(agg[FLOAT]["mra"], 70.0)

    def test_pf_as_incorrect_report(self) -> None:
        report = self._run(pf_as_incorrect=True)
        self.assertTrue(report["parse_failures_as_incorrect"])
        agg = report["aggregate"]
        self.assertAlmostEqual(agg[MC]["mean_accuracy"], 50.0)
        self.assertAlmostEqual(agg[FLOAT]["mra"], 140.0 / 3)
        seq = report["by_sequence"]["seq001"]
        self.assertAlmostEqual(seq[MC]["mean_accuracy"], 50.0)

    def test_mixed_answer_types_raise(self) -> None:
        annotations = [
            _annotation("a", QuestionType.GAP_FIT, "bool"),
            _annotation("b", QuestionType.GAP_FIT, "float"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = ScoreStore(Path(tmp) / "scores.db")
            store.write(_score("a", accuracy=100.0))
            store.write(_score("b", parse_failed=True))
            with self.assertRaises(RuntimeError):
                generate_report(store, annotations, Path(tmp) / "out", True)
            store.close()

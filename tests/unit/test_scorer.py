# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import unittest

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
from r3d.pipeline.eval.scorer import score_response


def _make_annotation(
    gt_answer: str = "0.45",
    gt_answer_type: str = "float",
    gt_answer_unit: str | None = "meters",
    question_type: QuestionType = QuestionType.GLOBAL_HOW_FAR,
    canonical_name: str = "test object",
) -> Annotation:
    eval_metric = (
        EvalMetric.ACCURACY
        if gt_answer_type in ("bool", "str")
        else EvalMetric.PERCENTAGE_ERROR
    )
    return Annotation(
        annotation_id="test-scorer-001",
        identity_layer=IdentityLayer(
            sequence_id="seq131",
            release_type=ReleaseType.FULL,
            referenced_objects=[
                ReferencedObject(
                    object_id="obj_001",
                    adt_instance_id=12345,
                    adt_instance_name="TestObject",
                    prototype_name="TestObject",
                    canonical_name=canonical_name,
                    is_dynamic=False,
                ),
            ],
        ),
        query_layer=QueryLayer(
            question_type=question_type,
            question_text="Test question?",
            gt_answer=gt_answer,
            gt_answer_type=gt_answer_type,
            gt_answer_unit=gt_answer_unit,
            eval_mode=EvalMode.DETERMINISTIC,
            eval_metric=eval_metric,
            query_timestamp_ns_start=1_000_000_000,
            query_timestamp_ns_end=1_000_000_000,
        ),
    )


class TestScoreBool(unittest.TestCase):
    def test_correct_yes(self) -> None:
        ann = _make_annotation(
            gt_answer="True",
            gt_answer_type="bool",
            gt_answer_unit=None,
            question_type=QuestionType.GAP_FIT,
        )
        score = score_response(ann, "ANSWER: yes", "model", "tool_use")
        self.assertFalse(score.parse_failed)
        self.assertEqual(score.accuracy, 100.0)
        self.assertIsNone(score.percentage_error)

    def test_wrong(self) -> None:
        ann = _make_annotation(
            gt_answer="True",
            gt_answer_type="bool",
            gt_answer_unit=None,
            question_type=QuestionType.GAP_FIT,
        )
        score = score_response(ann, "ANSWER: no", "model", "tool_use")
        self.assertFalse(score.parse_failed)
        self.assertEqual(score.accuracy, 0.0)

    def test_parse_failure(self) -> None:
        ann = _make_annotation(
            gt_answer="True",
            gt_answer_type="bool",
            gt_answer_unit=None,
            question_type=QuestionType.GAP_FIT,
        )
        score = score_response(ann, "I think maybe", "model", "tool_use")
        self.assertTrue(score.parse_failed)


class TestScoreStr(unittest.TestCase):
    def test_correct(self) -> None:
        ann = _make_annotation(
            gt_answer="wooden fork",
            gt_answer_type="str",
            gt_answer_unit=None,
            question_type=QuestionType.WHICH_TALLER,
            canonical_name="wooden fork",
        )
        score = score_response(ann, "ANSWER: the wooden fork", "model", "tool_use")
        self.assertFalse(score.parse_failed)
        self.assertEqual(score.accuracy, 100.0)
        self.assertIsNone(score.percentage_error)

    def test_wrong(self) -> None:
        ann = _make_annotation(
            gt_answer="wooden fork",
            gt_answer_type="str",
            gt_answer_unit=None,
            question_type=QuestionType.WHICH_TALLER,
            canonical_name="knife",
        )
        score = score_response(ann, "ANSWER: the knife", "model", "tool_use")
        self.assertFalse(score.parse_failed)
        self.assertEqual(score.accuracy, 0.0)

    def test_bare_answer_accepted(self) -> None:
        ann = _make_annotation(
            gt_answer="wooden fork",
            gt_answer_type="str",
            gt_answer_unit=None,
            question_type=QuestionType.WHICH_TALLER,
            canonical_name="wooden fork",
        )
        score = score_response(ann, "wooden fork", "model", "tool_use")
        self.assertFalse(score.parse_failed)
        self.assertEqual(score.accuracy, 100.0)

    def test_bare_wrong_answer(self) -> None:
        ann = _make_annotation(
            gt_answer="wooden fork",
            gt_answer_type="str",
            gt_answer_unit=None,
            question_type=QuestionType.WHICH_TALLER,
            canonical_name="wooden fork",
        )
        score = score_response(ann, "I'm not sure", "model", "tool_use")
        self.assertTrue(score.parse_failed)

    def test_empty_parse_failure(self) -> None:
        ann = _make_annotation(
            gt_answer="wooden fork",
            gt_answer_type="str",
            gt_answer_unit=None,
            question_type=QuestionType.WHICH_TALLER,
        )
        score = score_response(ann, "", "model", "tool_use")
        self.assertTrue(score.parse_failed)


class TestScoreFloat(unittest.TestCase):
    def test_existing_scoring_unchanged(self) -> None:
        ann = _make_annotation(gt_answer="2.0", gt_answer_type="float")
        score = score_response(ann, "ANSWER: 2.0", "model", "tool_use")
        self.assertFalse(score.parse_failed)
        self.assertAlmostEqual(score.percentage_error, 0.0)
        self.assertIsNone(score.accuracy)

    def test_percentage_error_computed(self) -> None:
        ann = _make_annotation(gt_answer="10.0", gt_answer_type="float")
        score = score_response(ann, "ANSWER: 12.0", "model", "tool_use")
        self.assertFalse(score.parse_failed)
        self.assertAlmostEqual(score.percentage_error, 20.0)

    def test_float_parse_failure(self) -> None:
        ann = _make_annotation(gt_answer="5.0", gt_answer_type="float")
        score = score_response(ann, "about five meters", "model", "tool_use")
        self.assertTrue(score.parse_failed)


if __name__ == "__main__":
    unittest.main()

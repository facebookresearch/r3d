# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

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
from r3d.pipeline.eval.responses import Response, ResponseStore
from r3d.pipeline.eval.scorer import score_response
from r3d.pipeline.eval.scores import ScoreStore
from r3d.pipeline.stores.sqlite_store import SQLiteAnnotationStore


def _make_float_annotation(
    annotation_id: str,
    gt_answer: str,
) -> Annotation:
    return Annotation(
        annotation_id=annotation_id,
        identity_layer=IdentityLayer(
            sequence_id="seq001",
            release_type=ReleaseType.FULL,
            referenced_objects=[
                ReferencedObject(
                    object_id="obj_01",
                    adt_instance_id=1111,
                    adt_instance_name="Mug_01",
                    prototype_name="Mug",
                    canonical_name="mug",
                    is_dynamic=False,
                ),
            ],
        ),
        query_layer=QueryLayer(
            question_type=QuestionType.GLOBAL_HOW_FAR,
            question_text="How far is the mug?",
            gt_answer=gt_answer,
            gt_answer_type="float",
            gt_answer_unit="meters",
            eval_mode=EvalMode.DETERMINISTIC,
            eval_metric=EvalMetric.PERCENTAGE_ERROR,
            query_timestamp_ns_start=1_000_000,
            query_timestamp_ns_end=1_000_000,
        ),
    )


def _make_bool_annotation(
    annotation_id: str,
    gt_answer: str,
) -> Annotation:
    return Annotation(
        annotation_id=annotation_id,
        identity_layer=IdentityLayer(
            sequence_id="seq001",
            release_type=ReleaseType.FULL,
            referenced_objects=[
                ReferencedObject(
                    object_id="obj_02",
                    adt_instance_id=2222,
                    adt_instance_name="Plate_01",
                    prototype_name="Plate",
                    canonical_name="plate",
                    is_dynamic=False,
                ),
            ],
        ),
        query_layer=QueryLayer(
            question_type=QuestionType.GAP_FIT,
            question_text="Does the plate fit?",
            gt_answer=gt_answer,
            gt_answer_type="bool",
            eval_mode=EvalMode.DETERMINISTIC,
            eval_metric=EvalMetric.ACCURACY,
            query_timestamp_ns_start=2_000_000,
            query_timestamp_ns_end=2_000_000,
        ),
    )


def _make_str_annotation(
    annotation_id: str,
    gt_answer: str,
) -> Annotation:
    return Annotation(
        annotation_id=annotation_id,
        identity_layer=IdentityLayer(
            sequence_id="seq001",
            release_type=ReleaseType.FULL,
            referenced_objects=[
                ReferencedObject(
                    object_id="obj_03",
                    adt_instance_id=3333,
                    adt_instance_name="Fork_01",
                    prototype_name="Fork",
                    canonical_name="fork",
                    is_dynamic=False,
                ),
            ],
        ),
        query_layer=QueryLayer(
            question_type=QuestionType.WHICH_TALLER,
            question_text="Which is taller?",
            gt_answer=gt_answer,
            gt_answer_type="str",
            eval_mode=EvalMode.DETERMINISTIC,
            eval_metric=EvalMetric.ACCURACY,
            query_timestamp_ns_start=3_000_000,
            query_timestamp_ns_end=3_000_000,
        ),
    )


class TestGenerateScoresIntegration(unittest.TestCase):
    def test_end_to_end_scoring(self) -> None:
        """Create synthetic annotations + responses, score them, verify scores.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # 1. Create annotations.db
            ann_store = SQLiteAnnotationStore(tmppath / "annotations.db")
            annotations = [
                _make_float_annotation("ann-float-01", "2.0"),
                _make_bool_annotation("ann-bool-01", "True"),
                _make_str_annotation("ann-str-01", "fork"),
            ]
            for ann in annotations:
                ann_store.write_annotation(ann)

            loaded = ann_store.get_all_annotations()
            self.assertEqual(len(loaded), 3)

            # 2. Create responses.db
            resp_store = ResponseStore(tmppath / "responses.db")
            responses = [
                Response(
                    annotation_id="ann-float-01",
                    model="test-model",
                    strategy="tool_use",
                    response="ANSWER: 2.0",
                    tool_call_log=None,
                    created_ns=1_000_000,
                ),
                Response(
                    annotation_id="ann-bool-01",
                    model="test-model",
                    strategy="tool_use",
                    response="ANSWER: yes",
                    tool_call_log=None,
                    created_ns=2_000_000,
                ),
                Response(
                    annotation_id="ann-str-01",
                    model="test-model",
                    strategy="tool_use",
                    response="ANSWER: the fork",
                    tool_call_log=None,
                    created_ns=3_000_000,
                ),
            ]
            for resp in responses:
                resp_store.write(resp)

            # 3. Score responses (mimicking generate_scores.main logic)
            ann_map = {a.annotation_id: a for a in loaded}
            score_store = ScoreStore(tmppath / "scores.db")

            for resp in resp_store.get_all():
                annotation = ann_map[resp.annotation_id]
                score = score_response(
                    annotation, resp.response, resp.model, resp.strategy
                )
                score_store.write(score)

            # 4. Verify scores.db
            scores = score_store.get_all()
            self.assertEqual(len(scores), 3)

            score_map = {s.annotation_id: s for s in scores}

            # Float: exact match -> 0% error
            float_score = score_map["ann-float-01"]
            self.assertFalse(float_score.parse_failed)
            self.assertAlmostEqual(float_score.percentage_error, 0.0)
            self.assertIsNone(float_score.accuracy)

            # Bool: "yes" matches "True" -> 100% accuracy
            bool_score = score_map["ann-bool-01"]
            self.assertFalse(bool_score.parse_failed)
            self.assertEqual(bool_score.accuracy, 100.0)
            self.assertIsNone(bool_score.percentage_error)

            # Str: "the fork" -> "fork" (article stripped) matches "fork" -> 100%
            str_score = score_map["ann-str-01"]
            self.assertFalse(str_score.parse_failed)
            self.assertEqual(str_score.accuracy, 100.0)
            self.assertIsNone(str_score.percentage_error)

            # Verify scores.db file exists
            self.assertTrue((tmppath / "scores.db").exists())

            resp_store.close()
            score_store.close()
            ann_store.close()

    def test_parse_failure_scoring(self) -> None:
        """Verify that unparseable responses produce parse_failed scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            ann_store = SQLiteAnnotationStore(tmppath / "annotations.db")
            ann = _make_float_annotation("ann-fail-01", "5.0")
            ann_store.write_annotation(ann)

            resp_store = ResponseStore(tmppath / "responses.db")
            resp_store.write(
                Response(
                    annotation_id="ann-fail-01",
                    model="test-model",
                    strategy="tool_use",
                    response="I don't know the answer",
                    tool_call_log=None,
                    created_ns=1_000_000,
                )
            )

            loaded = ann_store.get_all_annotations()
            ann_map = {a.annotation_id: a for a in loaded}
            score_store = ScoreStore(tmppath / "scores.db")

            for resp in resp_store.get_all():
                score = score_response(
                    ann_map[resp.annotation_id],
                    resp.response,
                    resp.model,
                    resp.strategy,
                )
                score_store.write(score)

            scores = score_store.get_all()
            self.assertEqual(len(scores), 1)
            self.assertTrue(scores[0].parse_failed)
            self.assertIsNone(scores[0].percentage_error)
            self.assertIsNone(scores[0].accuracy)

            resp_store.close()
            score_store.close()
            ann_store.close()

    def test_percentage_error_computed_correctly(self) -> None:
        """Verify percentage error for a non-exact float response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            ann_store = SQLiteAnnotationStore(tmppath / "annotations.db")
            ann = _make_float_annotation("ann-pct-01", "10.0")
            ann_store.write_annotation(ann)

            resp_store = ResponseStore(tmppath / "responses.db")
            resp_store.write(
                Response(
                    annotation_id="ann-pct-01",
                    model="test-model",
                    strategy="tool_use",
                    response="ANSWER: 12.0",
                    tool_call_log=None,
                    created_ns=1_000_000,
                )
            )

            loaded = ann_store.get_all_annotations()
            ann_map = {a.annotation_id: a for a in loaded}
            score_store = ScoreStore(tmppath / "scores.db")

            for resp in resp_store.get_all():
                score = score_response(
                    ann_map[resp.annotation_id],
                    resp.response,
                    resp.model,
                    resp.strategy,
                )
                score_store.write(score)

            scores = score_store.get_all()
            self.assertEqual(len(scores), 1)
            self.assertFalse(scores[0].parse_failed)
            # |12 - 10| / |10| * 100 = 20%
            self.assertAlmostEqual(scores[0].percentage_error, 20.0)

            resp_store.close()
            score_store.close()
            ann_store.close()


if __name__ == "__main__":
    unittest.main()

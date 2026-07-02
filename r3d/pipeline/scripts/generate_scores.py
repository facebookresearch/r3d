# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Score VLM responses and generate a summary report.

Loads annotations from the R3D-Bench HF dataset + responses.db, scores each
response, writes scores.db, and prints a summary report.

Usage:
    python -m r3d.pipeline.scripts.generate_scores \
      --dataset facebook/r3d-bench \
      --responses-db /tmp/eval/responses/responses.db \
      --output-dir /tmp/eval/scores
"""

from __future__ import annotations

import logging
from pathlib import Path

from r3d.pipeline.eval.config import parse_score_config
from r3d.pipeline.eval.reporter import generate_report
from r3d.pipeline.eval.responses import ResponseStore
from r3d.pipeline.eval.scorer import score_response
from r3d.pipeline.eval.scores import ScoreStore
from r3d.utils.logging import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_local_db(path: str, filename: str) -> Path:
    """Resolve a local database path (directory or file)."""
    p = Path(path)
    if p.is_dir():
        return p / filename
    return p


def main() -> None:
    config = parse_score_config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir)

    from r3d.pipeline.hf_dataset import load_annotation_store

    logger.info(f"Loading annotations from HF dataset: {config.dataset}")
    ann_store = load_annotation_store(config.dataset)
    annotations = ann_store.get_all_annotations()
    ann_map = {a.annotation_id: a for a in annotations}

    resp_db = _resolve_local_db(config.responses_db, "responses.db")
    response_store = ResponseStore(resp_db)
    score_store = ScoreStore(output_dir / "scores.db")

    responses = response_store.get_all()
    logger.info(f"Scoring {len(responses)} responses")

    for resp in responses:
        annotation = ann_map[resp.annotation_id]
        score = score_response(annotation, resp.response, resp.model, resp.strategy)
        score_store.write(score)
        if score.parse_failed:
            status = "FAIL (parse)"
        elif score.accuracy is not None:
            status = f"accuracy={score.accuracy:.0f}%"
        elif score.percentage_error is not None:
            status = f"error={score.percentage_error:.1f}%"
        else:
            status = "no metric"
        logger.info(
            f"  {resp.annotation_id}: "
            f"{annotation.query_layer.question_type.value} -> {status}"
        )

    generate_report(score_store, annotations, output_dir)

    response_store.close()
    score_store.close()
    ann_store.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()

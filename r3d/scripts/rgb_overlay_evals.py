# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Standalone RGB (+ SAM3 mask overlay) evaluation on R3D-Bench.

A self-contained baseline: no 3D scene, no tools. It loads RGB frames (and, with
--overlay, the referenced objects' SAM3 masks), sends them to a VLM, parses the
answer, and scores against the ground truth. Shares its core with the main
pipeline via r3d.pipeline.eval.rgb.

Usage:
    python -m r3d.scripts.rgb_overlay_evals \
        --dataset facebook/r3d-bench \
        --frames-dir $ASSETS/frames \
        --model qwen3-vl-8b --hf-model Qwen/Qwen3-VL-8B-Instruct \
        --overlay --output-dir /tmp/rgb_overlay
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from r3d.pipeline.eval.config import DEFAULT_DATASET
from r3d.pipeline.eval.hf_vlm import HFVLMClient
from r3d.pipeline.eval.reporter import generate_report
from r3d.pipeline.eval.responses import ResponseStore
from r3d.pipeline.eval.rgb import run_rgb_overlay_eval
from r3d.pipeline.eval.scorer import score_response
from r3d.pipeline.eval.scores import ScoreStore
from r3d.pipeline.hf_dataset import load_annotation_store, load_segmentation_store
from r3d.pipeline.stores.sqlite_store import SQLiteFrameStore
from r3d.utils.logging import setup_logging

logger: logging.Logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone RGB(+overlay) eval on R3D-Bench"
    )
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--frames-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--hf-model", type=str, required=True)
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], default="vllm")
    p.add_argument("--data-parallel-size", type=int, default=1)
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent in-flight requests (feeds the vLLM DP replicas).",
    )
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay referenced objects' SAM3 masks on the frames.",
    )
    p.add_argument("--annotation-ids", type=str, default=None)
    p.add_argument("--max-annotations", type=int, default=None)
    p.add_argument("--score", action="store_true", help="Score after generation.")
    return p


def _make_vlm(args: argparse.Namespace):
    if args.backend == "vllm":
        from r3d.pipeline.eval.vllm_client import VLLMClient

        return VLLMClient(
            model_name=args.hf_model,
            data_parallel_size=args.data_parallel_size,
            max_model_len=args.max_model_len,
        )
    return HFVLMClient(model_name=args.hf_model)


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir)

    frames_db = Path(args.frames_dir) / "frames.db"
    if not frames_db.exists():
        raise RuntimeError(f"frames.db not found at {frames_db}")
    frame_store = SQLiteFrameStore(frames_db, read_only=True)

    ann_store = load_annotation_store(args.dataset)
    seg_store = load_segmentation_store(args.dataset) if args.overlay else None

    annotations = ann_store.get_all_annotations()
    if args.annotation_ids:
        keep = {ln.strip() for ln in open(args.annotation_ids) if ln.strip()}
        annotations = [a for a in annotations if a.annotation_id in keep]
    if args.max_annotations is not None:
        annotations = annotations[: args.max_annotations]
    logger.info(f"Evaluating {len(annotations)} annotations (overlay={args.overlay})")

    vlm = _make_vlm(args)
    response_store = ResponseStore(output_dir / "responses.db")

    def _process(ann):
        return run_rgb_overlay_eval(
            ann,
            ann.identity_layer.sequence_id,
            frame_store,
            vlm,
            args.model,
            args.image_size,
            seg_store=seg_store,
        )

    if args.concurrency > 1 and args.data_parallel_size > 1:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [pool.submit(_process, a) for a in annotations]
            for future in concurrent.futures.as_completed(futures):
                response_store.write(future.result())
    else:
        for ann in annotations:
            response_store.write(_process(ann))
    logger.info(f"Wrote {len(annotations)} responses")

    if args.score:
        score_store = ScoreStore(output_dir / "scores.db")
        for resp in response_store.get_all():
            ann = next(
                (a for a in annotations if a.annotation_id == resp.annotation_id), None
            )
            if ann is not None:
                score_store.write(
                    score_response(ann, resp.response, resp.model, resp.strategy)
                )
        generate_report(score_store, annotations, output_dir)
        logger.info("Scoring complete.")

    response_store.close()


if __name__ == "__main__":
    main()

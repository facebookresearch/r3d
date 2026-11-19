# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import argparse
from dataclasses import dataclass

DEFAULT_DATASET: str = "facebook/r3d-bench"
VOLUME_METHODS: tuple[str, ...] = ("mesh_rescaled", "bbox")


@dataclass(frozen=True)
class EvalConfig:
    dataset: str
    scene_db: str
    frames_dir: str
    output_dir: str
    model: str
    hf_model: str
    backend: str
    data_parallel_size: int
    image_size: int
    require_tracked_objects: bool
    max_annotations: int | None
    question_types: list[str] | None
    concurrency: int
    segmentation_eval_path: str | None
    max_wrong_object_ratio: float
    min_segmentation_iou: float
    annotation_ids_path: str | None
    no_images: bool
    use_process_pool: bool
    max_model_len: int | None
    volume_method: str
    tool_backend: str = "scene"
    gt_oracle_db: str | None = None
    strategy: str = "tool_use"
    cubify_db: str | None = None
    no_cutr_fallback: bool = False
    tensor_parallel_size: int = 1


@dataclass(frozen=True)
class ScoreConfig:
    dataset: str
    responses_db: str
    output_dir: str
    parse_failures_as_incorrect: bool = True


def _build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VLM responses for spatial AI eval"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="HF dataset repo for R3D-Bench (parquet: annotations, segmentations, "
        f"meshes). Default: {DEFAULT_DATASET}.",
    )
    parser.add_argument("--scene-db", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="Model name/path (HuggingFace Hub ID or local dir).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="vllm",
        help="Inference backend (default: vllm).",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of model replicas for data parallelism (vllm only, default: 1).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs per model replica (vllm only, default: 1). Uses "
        "data-parallel-size x tensor-parallel-size GPUs.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--require-tracked-objects", action="store_true")
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=None,
        help="Limit to the first N annotations for faster debugging.",
    )
    parser.add_argument("--question-types", type=str, nargs="+", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--segmentation-eval",
        type=str,
        default=None,
        help="Path to segmentation_eval.json for wrong-object filtering.",
    )
    parser.add_argument(
        "--max-wrong-object-ratio",
        type=float,
        default=0.2,
        help="Drop annotations where any object has wrong-object ratio >= this.",
    )
    parser.add_argument(
        "--min-segmentation-iou",
        type=float,
        default=0.5,
        help="Drop annotations where any object has mean IoU below this.",
    )
    parser.add_argument(
        "--annotation-ids",
        type=str,
        default=None,
        help="Path to annotation_ids.txt (one ID per line) to restrict eval set.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip sending images to the model (text-only tool_use).",
    )
    parser.add_argument("--use-process-pool", action="store_true")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model sequence length (vLLM only, default: model's max).",
    )
    parser.add_argument(
        "--volume-method",
        type=str,
        choices=list(VOLUME_METHODS),
        default="mesh_rescaled",
        help="get_object_volume backend: mesh_rescaled (default, SAM3D mesh "
        "cavity volume rescaled to the OBB) or bbox (OBB box volume, no SAM3D).",
    )
    parser.add_argument(
        "--tool-backend",
        type=str,
        choices=["scene", "gt_oracle"],
        default="scene",
        help="Where tool values come from: 'scene' (reconstructed scene) or "
        "'gt_oracle' (ADT ground truth from --gt-oracle-db; same tools, object "
        "IDs and output formats).",
    )
    parser.add_argument(
        "--gt-oracle-db",
        type=str,
        default=None,
        help="Path to gt_oracle.db (required with --tool-backend gt_oracle).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["tool_use", "scene_text"],
        default="tool_use",
        help="tool_use: multi-turn spatial tool calling (R3D). scene_text: "
        "no tools; the tool outputs for every object are given as a text table "
        "in a single-turn prompt (ablation).",
    )
    parser.add_argument(
        "--cubify-db",
        type=str,
        default=None,
        help="Path to cubify.db: take every object's box and position from the "
        "CuTR detector instead of the scene (the CuTR baseline; pair with "
        "--volume-method bbox).",
    )
    parser.add_argument(
        "--no-cutr-fallback",
        action="store_true",
        help="With --cubify-db: objects without a CuTR box make the spatial "
        "tools return 'object not detected' instead of the scene box.",
    )
    return parser


def _args_to_eval_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        dataset=args.dataset,
        scene_db=args.scene_db,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        model=args.model,
        hf_model=args.hf_model,
        backend=args.backend,
        data_parallel_size=args.data_parallel_size,
        image_size=args.image_size,
        require_tracked_objects=args.require_tracked_objects,
        max_annotations=args.max_annotations,
        question_types=args.question_types,
        concurrency=args.concurrency,
        segmentation_eval_path=args.segmentation_eval,
        max_wrong_object_ratio=args.max_wrong_object_ratio,
        min_segmentation_iou=args.min_segmentation_iou,
        annotation_ids_path=args.annotation_ids,
        no_images=args.no_images,
        use_process_pool=args.use_process_pool,
        max_model_len=args.max_model_len,
        volume_method=args.volume_method,
        tool_backend=args.tool_backend,
        gt_oracle_db=args.gt_oracle_db,
        strategy=args.strategy,
        cubify_db=args.cubify_db,
        no_cutr_fallback=args.no_cutr_fallback,
        tensor_parallel_size=args.tensor_parallel_size,
    )


def parse_eval_config() -> EvalConfig:
    parser = _build_eval_parser()
    args = parser.parse_args()
    if args.tool_backend == "gt_oracle" and not args.gt_oracle_db:
        parser.error("--gt-oracle-db is required for --tool-backend gt_oracle")
    if args.no_cutr_fallback and not args.cubify_db:
        parser.error("--no-cutr-fallback requires --cubify-db")
    return _args_to_eval_config(args)


def parse_score_config() -> ScoreConfig:
    parser = argparse.ArgumentParser(
        description="Score VLM responses for spatial AI eval"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"HF dataset repo for R3D-Bench annotations. Default: {DEFAULT_DATASET}.",
    )
    parser.add_argument("--responses-db", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--exclude-parse-failures",
        action="store_true",
        help="Exclude parse failures from every metric. By default they count "
        "as incorrect (accuracy/MRA/within-X = 0), as in the paper.",
    )
    args = parser.parse_args()
    return ScoreConfig(
        dataset=args.dataset,
        responses_db=args.responses_db,
        output_dir=args.output_dir,
        parse_failures_as_incorrect=not args.exclude_parse_failures,
    )

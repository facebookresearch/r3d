# R3D: Quantitative 3D Spatial Reasoning for Egocentric Wearables

![R3D teaser](assets/teaser.png)

This is the official repository for [**R3D: Quantitative 3D Spatial Reasoning for Egocentric Wearables**](https://arxiv.org/abs/2607.02921).

This repository includes:
- [**R3D-Bench**](https://huggingface.co/datasets/facebook/r3d-bench): a benchmark evaluation for quantitative 3D Spatial Reasoning using natural egocentric RGB-D video.
- **R3D**: a system for 3D spatial reasoning with tool calling.

The repository contains:
- Setup scripts for the 2,950 annotations of R3D-Bench.
- Code used for generating R3D-Bench.
- Code used for evaluating on R3D-Bench.

## Table of Contents

- [Installation](#installation)
- [Data Setup](#data-setup)
  - [R3D-Bench (Hugging Face)](#r3d-bench-hugging-face)
  - [Obtain ADT Sequences](#obtain-adt-sequences)
  - [Extract Frames](#extract-frames)
- [Quick Start - Running Your Model On R3D-Bench](#quick-start---running-your-model-on-r3d-bench)
  - [Standard VLM (RGB Only)](#standard-vlm-rgb-only)
  - [Dataset Tutorial](#dataset-tutorial)
- [Running R3D Evaluation on R3D-Bench](#running-r3d-evaluation-on-r3d-bench)
  - [Step 1: Build Scene](#step-1-build-scene)
  - [Step 2: Generate Responses](#step-2-generate-responses)
  - [Step 3: Generate Scores](#step-3-generate-scores)
  - [Step 4: Generate Eval Table](#step-4-generate-eval-table)
- [Generating Additional Samples](#generating-additional-samples)
- [Scripts](#scripts)
- [Key Flags](#key-flags)
- [Evaluation Entry Points](#evaluation-entry-points)
- [License](#license)
- [Citation](#citation)

## Installation

```bash
git clone https://github.com/facebookresearch/r3d
cd r3d
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

This installs everything, including PyTorch and vLLM (the default inference backend). The `--extra-index-url` pulls the CUDA build of PyTorch/vLLM — change `cu128` to match your CUDA toolkit.

Requirements: Python 3.10+, CUDA-capable GPU.

## Data Setup

### R3D-Bench (Hugging Face)

The R3D-Bench data lives on Hugging Face Hub at [`facebook/r3d-bench`](https://huggingface.co/datasets/facebook/r3d-bench) as a Parquet, split into these configs:

- `qa_annotations` — the 2,950 Q&A annotations in R3D-Bench
- `segmentations` — per-object per-video SAM3 segmentation masks used in evaluations
- `meshes` — per-object 3D reconstructed meshes from SAM3D, used in evaluations
- `gt_bboxes` — optional ground-truth oriented-box trajectories from ADT that the reference answers were derived from (for debugging; evaluations do not use them)

Every script takes a `--dataset` flag (default `facebook/r3d-bench`) and fetches only the configs it needs on first use via `huggingface_hub`, caching them under `$HF_HOME`. To use a local mirror or fork instead, pass `--dataset <org>/<repo>`.

R3D-Bench requires downloading Aria Digital Twin, described in the next step.

### Obtain ADT Sequences

R3D-Bench is built on the 57 [Aria Digital Twin (ADT)](https://www.projectaria.com/datasets/adt/) sequences listed in [`sequences.txt`](sequences.txt). Request access on the ADT page; you will receive a personal CDN file (a JSON of time-limited download URLs). Point `CDN_FILE` at your copy.

Download all 57 sequences required by R3D-Bench with every data type (RGB, depth, segmentation, SLAM pose, ground truth, etc.) using the `aria_dataset_downloader` bundled with `projectaria-tools` (installed above):

```bash
export ADT_ROOT=/path/to/ADT
export CDN_FILE=/path/to/your_ADT_download_urls.json

aria_dataset_downloader \
    --cdn_file "$CDN_FILE" \
    --output_folder "$ADT_ROOT" \
    --data_types 0 1 2 3 4 5 6 7 8 9 \
    --sequence_names $(cat sequences.txt)
```

Each sequence is ~6–7 GB across all data types (~370 GB total). This creates one subfolder per sequence under `$ADT_ROOT`.

### Extract Frames
R3D-Bench mandates that RGB frames are extracted at 3FPS and 512x512 resolution. Extract them from ADT sequences and place them in a folder with this script:

```bash
# Find all sequence directories
SEQ_PATHS=$(find $ADT_ROOT -maxdepth 1 -mindepth 1 -type d | sort | tr '\n' ' ')

python -m r3d.pipeline.scripts.populate_frames \
    --sequence-paths $SEQ_PATHS \
    --output-dir $ASSETS/frames \
    --fps 3.0 \
    --image-size 512 \
    --num-workers 16
```

This produces:
- `frames/frames.db` — frame metadata (timestamps, intrinsics, poses)
- `frames/frame_data/` — RGB and depth PNGs

## Quick Start - Running Your Model On R3D-Bench

### Standard VLM (RGB Only)

This command is primarily for demonstration purposes. RGB-only models do not usually perform very well on R3D-Bench, but we include this example command so that researchers can modify it for their own purposes.

To run a standard VLM (RGB-only, no depth inputs) on R3D-Bench, use `r3d.scripts.rgb_overlay_evals`. Add `--overlay` to additionally paint the referenced objects' SAM3 masks onto the frames, and `--score` to score immediately after generation. `--max-frames` caps how many frames from each question's window are sent to the model. In our paper evaluations we do not limit the frames sent to models. In this example command, we set the value to 16 for computational efficiency. Removing the flag will result in all frames being sent to the model. You may need to set `--max-model-len` accordingly. Here is the command:

```bash
python -m r3d.scripts.rgb_overlay_evals \
    --dataset facebook/r3d-bench \
    --frames-dir $ASSETS/frames \
    --model qwen3-vl-8b --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --backend vllm --data-parallel-size 8 \
    --max-frames 16 \
    --overlay --score \
    --output-dir /tmp/r3d/rgb_overlay
```

Modify this evaluation command to run your own VLM that accepts depth and pose as input.

The 16-frame example above is not a paper configuration. The paper's RGB rows (Table 2) send every frame in the question's window with SAM3 overlays and count parse failures as incorrect.

For reference, the example above produces the following on the 2,950 R3D-Bench questions (6 parse failures, counted as wrong). Multiple-choice types report mean accuracy; distance and volume types report Mean Relative Accuracy (MRA). OVERALL is the mean over all 15 question types. All values are percentages; higher is better.

| Question type | Metric | RGB 8B |
|---|:---:|---:|
| `gap_fit` | Acc | 62.3 |
| `nearest_from_set` | Acc | 55.9 |
| `top_higher` | Acc | 65.4 |
| `which_taller` | Acc | 68.4 |
| `which_longer_dim` | Acc | 64.4 |
| **Multiple-choice avg** | Acc | **63.3** |
| `global_how_far` | MRA | 39.8 |
| `global_how_far_from_me` | MRA | 28.8 |
| `global_how_long` | MRA | 54.4 |
| `how_much_taller` | MRA | 29.6 |
| `how_much_longer_dim` | MRA | 37.8 |
| `total_fly_distance` | MRA | 21.5 |
| `total_walk_distance` | MRA | 16.9 |
| **Distance avg** | MRA | **32.7** |
| `volume_estimation` | MRA | 22.0 |
| `pour_leftover` | MRA | 0.0 |
| `pour_room_left` | MRA | 3.5 |
| **Volume avg** | MRA | **8.5** |
| **OVERALL (all 15)** | — | **38.1** |

vLLM's batching makes greedy decoding slightly nondeterministic: a second run of the same command matched 2,701 of the 2,950 responses exactly and also scored 38.1 overall.

### Dataset Tutorial

`r3d.scripts.dataset_tutorial` is a runnable walkthrough that doubles as worked example code — it reuses the same pipeline utilities the R3D eval below uses (HF parquet stores, depth-lifting + OBB fit from `build_scene`, mesh rescaling from `volume`, and the viz/video helpers). It runs five demos (select a subset with `--demos`):

- `annotation` — load a QA annotation and print its fields
- `mesh` — load a SAM3D mesh and save it to disk
- `pointing` — visualize a pointing annotation: overlay the object's ADT ground-truth mask on its single reference frame (one per object per video). Requires `--adt-root`.
- `scene_video` — overlay all SAM3 masks across a sequence into an MP4, captioned with the question + GT answer
- `depth_lift` — depth-lift one volumetric object and save its 3D point cloud (`.ply`), mask, fitted 3D bounding box (drawn + `.npz`), and the SAM3D mesh rescaled to that box (`.glb`)

```bash
python -m r3d.scripts.dataset_tutorial \
    --dataset facebook/r3d-bench \
    --frames-dir $ASSETS/frames \
    --output-dir /tmp/r3d_tutorial \
    --sequence Apartment_release_clean_seq131_M1292 \
    --object mug \
    --adt-root $ADT_ROOT   # only needed for the pointing demo
```

## Running R3D Evaluation on R3D-Bench
Running official R3D evaluation and scoring runs in 4 steps. All intermediate outputs are stored in SQLite databases.

### Step 1: Build Scene

Depth-lift segmentation masks to 3D, assign each point to an object by multi-view
plurality vote (matching the production reconstruction), filter outliers with KNN, and
fit gravity-aligned oriented bounding boxes:

```bash
python -m r3d.pipeline.scripts.build_scene \
    --dataset facebook/r3d-bench \
    --frames-dir $ASSETS/frames \
    --output-dir $ASSETS/scene \
    --depth-stride 4 \
    --knn-k 6 \
    --mask-erosion 0 \
    --max-points-per-object 50000
```

This produces:
- `scene/scene.db` — per-object 3D bounding boxes, visibility, and coverage
- `scene/object_points.db` — per-object 3D point clouds

#### Scene-construction ablations

The scene-construction ablation of the paper (Appendix, Tables 12 and 13) uses these `build_scene` flags, followed by `generate_responses` with the `--volume-method` shown:

| Boxes | Vote | KNN | Volume | `build_scene` flags | `generate_responses` flag |
|---|:---:|:---:|---|---|---|
| First frame | | | Box | `--box-mode first_frame --no-multiview-vote --no-knn` | `--volume-method bbox` |
| Median | | | Box | `--box-mode median --no-multiview-vote --no-knn` | `--volume-method bbox` |
| Fused | | | Box | `--no-multiview-vote --no-knn` | `--volume-method bbox` |
| Fused | ✓ | | Box | `--no-knn` | `--volume-method bbox` |
| Fused | | ✓ | Box | `--no-multiview-vote` | `--volume-method bbox` |
| Fused | ✓ | ✓ | Box | (none) | `--volume-method bbox` |
| Fused | ✓ | ✓ | SAM3D | (none) | (none; `mesh_rescaled` is the default) |

### Step 2: Generate Responses

Run a VLM on R3D-Bench with vLLM, which provides fast inference with automatic KV cache reuse across tool-use turns.

```bash
# Qwen3-VL-8B on 8 GPUs (8-way data parallel)
python -m r3d.pipeline.scripts.generate_responses \
    --dataset facebook/r3d-bench \
    --scene-db $ASSETS/scene/scene.db \
    --frames-dir $ASSETS/frames \
    --model qwen3-vl-8b \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --backend vllm \
    --no-images \
    --data-parallel-size 8 \
    --concurrency 8 \
    --max-model-len 131072 \
    --output-dir /tmp/r3d/responses_8b
```

```bash
# Qwen3.6-35B on 2 GPUs (tensor parallel), the configuration used for the paper results
# Note: set CUDA_VISIBLE_DEVICES to control which GPUs are used.
CUDA_VISIBLE_DEVICES=0,1 python -m r3d.pipeline.scripts.generate_responses \
    --dataset facebook/r3d-bench \
    --scene-db $ASSETS/scene/scene.db \
    --frames-dir $ASSETS/frames \
    --model qwen36-35b \
    --hf-model Qwen/Qwen3.6-35B-A3B \
    --backend vllm \
    --no-images \
    --data-parallel-size 1 \
    --tensor-parallel-size 2 \
    --concurrency 32 \
    --max-model-len 131072 \
    --output-dir /tmp/r3d/responses_35b
```

### Step 3: Generate Scores

Score the responses and produce per-question-type metrics:

```bash
python -m r3d.pipeline.scripts.generate_scores \
    --dataset facebook/r3d-bench \
    --responses-db /tmp/r3d/responses_8b/responses.db \
    --output-dir /tmp/r3d/scores_8b
```

This produces:
- `scores.db` — per-annotation scores
- `eval_report.json` — aggregate and per-sequence metrics
- `eval_summary.txt` — human-readable summary

Unparseable answers (parse failures) count as wrong by default, as in the paper's main results. Add `--exclude-parse-failures` to exclude them from the metrics instead.

### Step 4: Generate Eval Table

Compare results across models in a formatted table:

```bash
python -m r3d.pipeline.scripts.generate_eval_table \
    --reports \
      "R3D 8B:/tmp/r3d/scores_8b/eval_report.json" \
      "R3D 35B:/tmp/r3d/scores_35b/eval_report.json" \
    --output /tmp/r3d/eval_results.txt
```

This produces a table with per-category accuracy (MC), MRA (distance/volume), within-threshold accuracy, and mean % error, plus grouped averages and overall scores.

Expected per-question-type results on the 2,950 questions of R3D-Bench (R3D 8B = `Qwen3-VL-8B-Instruct`, R3D 35B = `Qwen3.6-35B-A3B`), produced with this code (vLLM, `--no-images`, `--max-model-len 131072`; 8B on one GPU, 35B with tensor parallel 2) and scored with the default protocol (parse failures counted as wrong), as in the paper's main results. With `--exclude-parse-failures`, OVERALL is 52.5 (8B) and 60.5 (35B). Multiple-choice types report **mean accuracy**; distance and volume types report **Mean Relative Accuracy (MRA)**. OVERALL is the mean over all 15 question types. All values are percentages; higher is better.

| Question type | Metric | R3D 8B | R3D 35B |
|---|:---:|---:|---:|
| `gap_fit` | Acc | 70.5 | 80.3 |
| `nearest_from_set` | Acc | 62.4 | 79.0 |
| `top_higher` | Acc | 57.6 | 75.6 |
| `which_taller` | Acc | 68.0 | 80.4 |
| `which_longer_dim` | Acc | 68.0 | 75.6 |
| **Multiple-choice avg** | Acc | **65.3** | **78.2** |
| `global_how_far` | MRA | 58.1 | 55.6 |
| `global_how_far_from_me` | MRA | 54.4 | 74.4 |
| `global_how_long` | MRA | 59.0 | 62.9 |
| `how_much_taller` | MRA | 48.6 | 52.6 |
| `how_much_longer_dim` | MRA | 38.7 | 41.1 |
| `total_fly_distance` | MRA | 35.2 | 51.9 |
| `total_walk_distance` | MRA | 16.5 | 32.1 |
| **Distance avg** | MRA | **44.4** | **53.0** |
| `volume_estimation` | MRA | 34.1 | 39.3 |
| `pour_leftover` | MRA | 20.8 | 25.1 |
| `pour_room_left` | MRA | 24.4 | 23.3 |
| **Volume avg** | MRA | **26.4** | **29.2** |
| **OVERALL (all 15)** | — | **47.8** | **56.6** |

The paper's Table 2 rows were generated with the Hugging Face transformers backend (`--backend hf`, greedy) on GB200 GPUs and score 47.6 (8B) and 55.6 (35B). Scoring those responses with this code reproduces them exactly, and tool outputs are identical wherever the two runs make the same call. The vLLM numbers above differ because the backend's numerics take greedy decoding down different paths; with vLLM, 35B reruns are bit-identical and 8B reruns vary by about 0.6.

## Generating Additional Samples

Most users do not need this — R3D-Bench is published on Hugging Face and downloaded automatically. This section is for generating additional Q&A samples (e.g. new sequences or more questions per type) using the same production pipeline. Note that this does not include the question balancing and filtering steps described in Appendix B of our paper.

### Obtain ADT Object Meshes

Volume ground truth is computed from the ADT objects' 3D meshes. These are not part of the per-sequence ADT download — they come from the [Digital Twin Catalog (DTC)](https://www.projectaria.com/datasets/dtc/). Request access there; you will receive a DTC objects CDN file. Download the ADT-release object meshes with the `dtc_object_downloader` bundled with `projectaria-tools`:

```bash
export DTC_CDN_FILE=/path/to/your_DTC_objects_ADT_download_urls.json
export ADT_OBJECT_LIBRARY=$HOME/adt_object_library

dtc_object_downloader -r ADT -c "$DTC_CDN_FILE" -o "$ADT_OBJECT_LIBRARY"
```

This writes one `3d-asset.glb` per object at `$ADT_OBJECT_LIBRARY/{object_name}/3d-asset.glb` (~400 objects).

One object, `Flask`, is not part of the `ADT` release and so is missing from the command above. You must download this one mesh separately from the full DTC catalog CDN file (`DTC_objects_all_download_urls.json`, also from the DTC access page):

```bash
export DTC_CDN_ALL=/path/to/your_DTC_objects_all_download_urls.json

dtc_object_downloader -c "$DTC_CDN_ALL" -o "$ADT_OBJECT_LIBRARY" --objects Flask -k 3d-asset_glb -x __nomatch__
```

This adds `$ADT_OBJECT_LIBRARY/Flask/3d-asset.glb`, completing the object library.

### Generate the Q&A annotations

With the frames (Extract Frames, above) and the object library in place, regenerate the annotations with the exact production settings via [`scripts/generate_dataset.sh`](scripts/generate_dataset.sh):

```bash
export ADT_ROOT=/path/to/ADT
export ADT_OBJECT_LIBRARY=$HOME/adt_object_library
export ASSETS=/path/to/r3d_assets   # must contain frames/frames.db

./scripts/generate_dataset.sh /tmp/r3d/annotations
```

## Scripts

The [`scripts/`](scripts/) directory wraps the commands used to build and evaluate R3D-Bench for the paper. The eval scripts reproduce the paper's R3D results within run-to-run noise (see the results table above).

| Script | What it runs |
|--------|--------------|
| `run_populate_frames.sh` | Extract frames from ADT sequences |
| `generate_dataset.sh` | Generate Q&A annotations with production filters (static objects, pointing disambiguation, min-fov 6.0, min-visibility 0.5, 100/type) |
| `run_build_scene.sh` | Build the 3D scene (depth-lift + gravity-aligned OBB) |
| `run_eval_8b_full.sh` | R3D eval — Qwen3-VL-8B, `tool_use` + `mesh_rescaled`, no images, DP=8 |
| `run_eval_35b.sh` | R3D eval — Qwen3.6-35B-A3B, TP=2 |
| `run_rgb_baseline.sh` | RGB baseline — Qwen3-VL-8B, images + SAM3 mask overlay |
| `run_eval_8b_quick.sh` | 100-annotation smoke test (not a paper number) |
| `run_tests.sh` | Unit tests |

## Key Flags

| Flag | Description |
|------|-------------|
| `--dataset ORG/REPO` | Hugging Face R3D-Bench repo (default: `facebook/r3d-bench`) |
| `--num-workers N` | Parallel frame extraction workers (frame extraction) |
| `--backend vllm\|hf` | Inference backend (Step 2, default: vllm) |
| `--data-parallel-size N` | Number of model replicas (vLLM only) |
| `--max-model-len N` | Maximum context length in tokens (vLLM only, required for large models) |
| `--max-annotations N` | Limit annotations for quick testing |
| `--require-tracked-objects` | Skip annotations with untracked objects (not used for the paper results) |
| `--no-images` | Text-only (no frames sent to model). Required for tool-use runs, as in the paper. |
| `--concurrency N` | Concurrent annotation processing (default: 8) |
| `--question-types TYPE [TYPE ...]` | Restrict to specific question types |
| `--strategy tool_use\|scene_text` | `tool_use` (default): R3D tool calling. `scene_text`: no-tools ablation; every object's tool outputs are given as a text table in one prompt, and per-question token counts go to `prompt_stats.jsonl` |
| `--cubify-db PATH` | CuTR baseline: object boxes and positions come from the CuTR detector's `cubify.db` instead of the scene (use with `--volume-method bbox`) |
| `--no-cutr-fallback` | With `--cubify-db`: objects CuTR did not detect return `Error: object not detected` instead of the scene box |
| `--volume-method mesh_rescaled\|bbox` | `get_object_volume` backend: `mesh_rescaled` (default): SAM3D mesh cavity volume rescaled to the box. `bbox`: box volume, no SAM3D |
| `--tool-backend scene\|gt_oracle` | `scene` (default): tools read the reconstructed scene. `gt_oracle`: the same tools return ADT ground truth from `--gt-oracle-db` |
| `--tensor-parallel-size N` | GPUs per model replica (vLLM only) |
| `--annotation-ids PATH` | Restrict the run to the annotation IDs in a file (one per line) |
| `--exclude-parse-failures` | `generate_scores`: exclude parse failures from the metrics instead of counting them as wrong (score 0, the default, as in the paper's main results) |

## Evaluation Entry Points

| Entry point | Description |
|-------------|-------------|
| `r3d.pipeline.scripts.generate_responses` | R3D tool-use: multi-turn, model calls spatial tools (get_distance, get_position, get_object_volume, etc.) over the reconstructed 3D scene |
| `r3d.pipeline.scripts.export_scene_text_prompts` | Writes the `scene_text` prompts (with token counts) to JSONL for offline inference; `--validate-responses-db` checks every tool result logged by a tool-use run against the table |
| `r3d.scripts.rgb_overlay_evals` | RGB baseline: single-turn, model sees multi-frame images, optionally with `--overlay` segmentation masks |

## License

This project is licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE).

## Citation

When using our work, please cite as follows:

```bibtex
@misc{horton2026r3dquantitative3dspatial,
      title={R3D: Quantitative 3D Spatial Reasoning for Egocentric Wearables},
      author={Maxwell Horton and Wei Lu and Quan Tran and Yury Astashonok and Kirmani Ahmed and Babak Damavandi and Anuj Kumar and Xiao Zhang and Seungwhan Moon},
      year={2026},
      eprint={2607.02921},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.02921},
}
```

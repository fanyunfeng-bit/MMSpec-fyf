# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMSpec is a benchmark for evaluating speculative decoding methods on vision-language models (VLMs). It covers 10 speculative decoding methods across two target models (Qwen2.5-VL-7B-Instruct and LLaVA-1.5-7B), with 600 multimodal samples spanning 6 task categories.

## Key Commands

### Installation
```bash
pip install -r requirements.txt
# Requires Python 3.10+, transformers==4.51.3, torch==2.7.0
# Optional: deepspeed for training EAGLE3/MSD
```

### Running Evaluations
All commands run from project root. Scripts accept `testmini` (60 samples) or `test` (600 samples):
```bash
bash scripts/Qwen2.5-VL-7B/eval_baseline_mmspec.sh testmini
bash scripts/Qwen2.5-VL-7B/eval_vispec_mmspec.sh test
bash scripts/LLaVA-1.5-7B/eval_msd_mmspec.sh test
```

Direct Python invocation (all eval scripts use `-m` module syntax):
```bash
python -m evaluation.eval_baseline_mmspec --base-model-path <path> --data-folder dataset/MMSpec/testmini --answer-file results/out.jsonl --model-id baseline --temperature 0 --max-new-token 1024
```

### Summarizing Results
After eval runs finish, aggregate per-method metrics (MAT, walltime, speedup vs. baseline) into a CSV:
```bash
bash scripts/summarize_metrics.sh <model> [split] [temperature] [method] [extra args...]
# e.g.
bash scripts/summarize_metrics.sh Qwen2.5-VL-7B test 0 all --group-by topic
bash scripts/summarize_metrics.sh Qwen2.5-VL-7B test 0 msd      # baseline included automatically for speedup
```
CSV is written into `results/<model>/mmspec_<split>/` next to the raw JSONL. Underlying module: `evaluation.summarize_metrics`. Per-query/per-round breakdowns: `evaluation.per_query_round_stats`; layer-wise L2 probing: `evaluation.layer_l2_probe`.

### Training Draft Models
```bash
# EAGLE 1/2 (Accelerate)
accelerate launch --multi_gpu --mixed_precision=bf16 -m train.main_stage1 --basepath <model_path>
# EAGLE3 (DeepSpeed)
deepspeed -m train.main_eagle3_stage2 --basepath <model_path>
# MSD (DeepSpeed)
deepspeed -m train.msd.main_deepspeed --deepspeed_config train/msd/ds_config.json --basepath <model_path>
```

Training scripts and SLURM jobs are in `train/scripts/`. See `train/README.md` for full details.

## Architecture

### Evaluation Pipeline
Every eval script follows the same flow:
1. **Load model** -- architecture-aware: detects Qwen2.5-VL vs LLaVA via `config.architectures[0]`
2. **Load dataset** -- `load_mmspec_data()` reads `dataset/MMSpec/{split}/mmspec.jsonl` + PIL images
3. **Multi-turn loop** -- iterates turns per sample, maintaining conversation history via `build_prompt()`
4. **Generate** -- calls method-specific generation (e.g., `model.specgenerate()`, `baseline_forward()`)
5. **Time tracking** -- `TimeBreakdownTracker` uses PyTorch forward hooks to separate draft vs target time; wall time uses `torch.cuda.synchronize()` barriers
6. **Save results** -- JSONL with per-turn metrics: `wall_time`, `draft_time`, `target_time`, `new_tokens`, `acceptance_length`
7. **Resume support** -- `load_existing_ids()` skips already-evaluated samples; `reorg_answer_file()` deduplicates

### Method Implementations (`method/`)
Each method folder (eagle, eagle2, eagle3, eagle_SAGE, vispec, msd, medusa, sam, lookahead, pld, recycling) contains:
- `spec_model*.py` -- `SpecModel` class with `from_pretrained()` and generation method (`specgenerate`/`msdgenerate`)
- `cnets*.py` -- draft architecture/heads
- `kv_cache.py` -- KV cache management
- `modeling_*.py` -- model-specific wrappers for Qwen2.5-VL and LLaVA

ViSkip variants (`vispec_vskip`, `msd_vskip`, `sam_vskip`) add vision-aware skip logic on top of base methods.

### Key Shared Utilities
- `evaluation/utils.py` -- dataset loading, prompt building (multi-turn aware), output processing, result saving, sanity checks
- `evaluation/time_breakdown.py` -- `TimeBreakdownTracker` with module-level forward hooks; fallback wrapping for `specgenerate`/`msdgenerate` methods

### Results Format
```
results/<model_name>/mmspec_<split>/<method>-temperature-<temp>.jsonl
```
Each JSONL record contains per-turn arrays: `wall_time`, `draft_time`, `target_time`, `new_tokens`, `acceptance_length`.

### Dataset Format
```
dataset/MMSpec/{test|testmini}/mmspec.jsonl  -- samples with id, image filename, turns[], category, topic
dataset/MMSpec/{test|testmini}/images/       -- corresponding image files
```
Topics: general vqa, text vqa, image captioning, chart understanding, complex reasoning pro, multi-turn conversation.

## Important Patterns

- All Python scripts use `-m` module invocation from project root (e.g., `python -m evaluation.eval_vispec_mmspec`)
- Speculative method parameters (depth, top_k, total_token, threshold) are set in the shell scripts under `scripts/`
- Model/checkpoint paths are hardcoded in shell scripts -- update `BASE_MODEL` and `SPEC_MODEL` variables
- Multi-turn samples have multiple entries in `turns[]`; the eval loop processes them sequentially with conversation history

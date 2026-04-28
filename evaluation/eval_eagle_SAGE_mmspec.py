"""SAGE (on EAGLE) speculative decoding evaluation for MMSpec unified dataset.

SAGE = Sink-Aware Generative Enhancement. Identifies visual sink tokens by L2
norm of the vision encoder's pre-projector output, and repositions them to the
front of the visual sub-sequence before the draft's first forward.

Usage:
  python -m evaluation.eval_eagle_SAGE_mmspec \\
      --base-model-path <path> --spec-model-path <path> \\
      --data-folder dataset/MMSpec/testmini \\
      --answer-file results/.../sage-*.jsonl \\
      --sage-threshold-mode topk_ratio --sage-threshold-value 0.1
"""
import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from method.eagle.kv_cache import initialize_past_key_values  # noqa: F401
from method.eagle.utils import *  # noqa: F401,F403  — includes prepare_logits_processor
from method.eagle_SAGE.spec_model import SageSpecModel as SpecModel
from method.eagle_SAGE.visual_processor import VisualProcessor
from method.eagle_SAGE.text_importance_probe import TextImportanceProbe
from evaluation.utils import (
    load_mmspec_data,
    build_prompt,
    process_output,
    save_result,
    reorg_answer_file,
    load_existing_ids,
    get_common_args,
    get_num_turns,
    run_sanity_check,
    iter_eval_samples,
)
from evaluation.time_breakdown import build_time_breakdown_tracker


@torch.inference_mode()
def evaluate(args):
    """Run SAGE-on-EAGLE speculative decoding evaluation on MMSpec dataset."""
    # Load model (SageSpecModel inherits from_pretrained from EagleSpecModel)
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()

    # Wire the VisualProcessor pipeline (SinkDetector + SinkFirstRepositioner).
    model.visual_processor = VisualProcessor.from_args(
        args, spatial_merge_size=model.vision_hook.spatial_merge_size
    )
    model.sage_debug = bool(getattr(args, "sage_debug", False))

    # Wire the TextImportanceProbe (TGVC-style top-K diagnostic).
    if getattr(args, "ti_enable", False):
        model.text_importance_probe = TextImportanceProbe(
            model.base_model, layer_idx=int(args.ti_layer)
        )
        model.text_importance_topk = int(args.ti_topk)
        print(
            f"[TI-Probe] enabled at layer={args.ti_layer} topk={args.ti_topk} "
            f"(prints per prefill + per verify; cumulative H_t)"
        )

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)  # noqa: F405
    else:
        logits_processor = None

    model.eval()
    time_tracker = build_time_breakdown_tracker(model)
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(
        f"[SAGE] mode={args.sage_threshold_mode} value={args.sage_threshold_value} "
        f"min_sinks={args.sage_min_sinks} "
        f"enable_sink={args.sage_enable_sink_detection} "
        f"enable_reposition={args.sage_enable_repositioning}"
    )
    print(f"[SAGE] vision_hook arch={model.vision_hook.arch} "
          f"spatial_merge_size={model.vision_hook.spatial_merge_size} "
          f"drop_cls={model.vision_hook.drop_cls}")

    # Load data
    data = load_mmspec_data(args.data_folder)
    print(f"Loaded {len(data)} samples")
    if getattr(args, "sanity", False):
        run_sanity_check(args, model, tokenizer, data)
        return

    # Warmup
    print("Warming up...")
    for _ in range(3):
        torch.manual_seed(0)
        model_inputs = build_prompt(data[0], args)
        output_ids, new_token, idx = model.specgenerate(
            **model_inputs,
            temperature=args.temperature,
            max_new_tokens=min(args.max_new_token, 128),
            log=True,
        )
    print("Warmup done")

    # Evaluate
    existing_ids = load_existing_ids(args.answer_file)
    print(f"Skipping {len(existing_ids)} already-evaluated samples")
    for d in tqdm(
        iter_eval_samples(data, args.batch_size),
        total=len(data),
        desc=f"Evaluating (bs={args.batch_size})",
    ):
        if d["id"] in existing_ids:
            continue
        choices = []
        num_turns = get_num_turns(d)

        for i in range(args.num_choices):
            torch.manual_seed(i)

            turn_outputs = []
            turn_idxs = []
            turn_new_tokens = []
            turn_wall_times = []
            turn_acceptance_lengths = []
            turn_draft_times = []
            turn_target_times = []
            conversation_history = []

            for turn_idx in range(num_turns):
                model_inputs = build_prompt(
                    d,
                    args,
                    turn_idx=turn_idx,
                    conversation_history=conversation_history if turn_idx > 0 else None,
                )
                input_len = model_inputs["input_ids"].shape[1]

                torch.cuda.synchronize()
                start_time = time.time()
                time_tracker.reset()
                # Tag the upcoming specgenerate call so probe prints carry sample_id.
                model._sage_sample_id = f"{d['id']}-turn{turn_idx}"

                output_ids, new_token, idx, accp_len = model.specgenerate(
                    **model_inputs,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_token,
                    log=True,
                    return_acceptance_len=True,
                )

                torch.cuda.synchronize()
                total_time = time.time() - start_time
                draft_time, target_time = time_tracker.snapshot()

                output = process_output(output_ids, tokenizer, input_len)

                turn_outputs.append(output)
                turn_idxs.append(int(idx))
                turn_new_tokens.append(int(new_token))
                turn_wall_times.append(total_time)
                turn_acceptance_lengths.append(accp_len)
                turn_draft_times.append(draft_time)
                turn_target_times.append(target_time)

                turns = d.get("turns", [d.get("prompt", "")])
                user_msg = turns[turn_idx] if turn_idx < len(turns) else turns[0]
                conversation_history.append((user_msg, output))

            choices.append({
                "index": i,
                "turns": turn_outputs,
                "idxs": turn_idxs,
                "new_tokens": turn_new_tokens,
                "wall_time": turn_wall_times,
                "acceptance_length": turn_acceptance_lengths,
                "draft_time": turn_draft_times,
                "target_time": turn_target_times,
            })

        save_result(args.answer_file, d, args.model_id, choices)

    reorg_answer_file(args.answer_file)
    print(f"Results saved to {args.answer_file}")


def main():
    parser = argparse.ArgumentParser(
        description="SAGE (on EAGLE) evaluation for MMSpec"
    )
    parser = get_common_args(parser)

    # EAGLE-inherited args
    parser.add_argument(
        "--total-token",
        type=int,
        default=30,
        help="Total tokens for speculative decoding",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth for speculative decoding",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k for speculative decoding",
    )

    # SAGE-specific args
    parser.add_argument(
        "--sage-threshold-mode",
        type=str,
        default="topk_ratio",
        choices=["abs", "topk_ratio"],
        help="Sink threshold mode: 'abs' uses absolute L2 norm, "
             "'topk_ratio' picks top K fraction by norm.",
    )
    parser.add_argument(
        "--sage-threshold-value",
        type=float,
        default=0.1,
        help="topk_ratio: fraction in (0, 1]; abs: absolute norm value.",
    )
    parser.add_argument(
        "--sage-min-sinks",
        type=int,
        default=0,
        help="Minimum number of sinks (topk_ratio mode). "
             "Useful when images have few visual tokens.",
    )
    parser.add_argument(
        "--sage-enable-sink-detection",
        type=lambda s: s.lower() not in ("false", "0", "no"),
        default=True,
        help="Enable sink detection stage (default True).",
    )
    parser.add_argument(
        "--sage-enable-repositioning",
        type=lambda s: s.lower() not in ("false", "0", "no"),
        default=True,
        help="Enable sink-first repositioning stage (default True).",
    )
    parser.add_argument(
        "--sage-debug",
        action="store_true",
        default=False,
        help="Print per-call SAGE diagnostics (num_visual, num_sinks).",
    )

    # Text-importance probe (TGVC-style top-K visual tokens by avg attention from text)
    parser.add_argument(
        "--ti-enable",
        action="store_true",
        default=False,
        help="Enable TextImportanceProbe: at chosen LLM layer, print top-K "
             "visual tokens by average cross-modal attention from text. "
             "Prints once after prefill and once after each verify (cumulative).",
    )
    parser.add_argument(
        "--ti-layer",
        type=int,
        default=16,
        help="0-indexed target LLM decoder layer at which to compute text "
             "importance (default 16; LLaVA-1.5-7B has 32, Qwen2.5-VL-7B has 28).",
    )
    parser.add_argument(
        "--ti-topk",
        type=int,
        default=10,
        help="Number of top visual tokens (by alpha) to print per call.",
    )

    args = parser.parse_args()

    # Set model for prompt building
    args.model = args.base_model_path
    args.model_id = (
        args.model_id
        + f"-sage-{args.sage_threshold_mode}-{args.sage_threshold_value}"
        + f"-temperature-{args.temperature}"
    )

    # Set answer file
    if not args.answer_file:
        args.answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {args.answer_file}")
    print("Method: SAGE on EAGLE (visual sink repositioning)")

    evaluate(args)


if __name__ == "__main__":
    main()

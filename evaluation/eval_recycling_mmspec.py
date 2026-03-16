"""Token Recycling evaluation for MMSpec unified dataset.

Usage:
python -m evaluation.eval_recycling_mmspec --base-model-path <path> --data-folder <path>
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

from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.utils import *
from method.recycling.spec_model_recycling import SpecModel
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
    """Run Token Recycling evaluation on MMSpec dataset."""
    # Load model
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()
    
    model.eval()
    time_tracker = build_time_breakdown_tracker(model)
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
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
            matrix_top_k=args.matrix_top_k,
            draft_len=args.draft_len,
        )
    print("Warmup done")
    
    # Evaluate
    total_tokens = 0
    total_time = 0.0
    total_acceptance_lengths = []
    
    existing_ids = load_existing_ids(args.answer_file)
    print(f"Skipping {len(existing_ids)} already-evaluated samples")
    for sample_idx, d in enumerate(
        tqdm(
            iter_eval_samples(data, args.batch_size),
            total=len(data),
            desc=f"Evaluating (bs={args.batch_size})",
        )
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
                model_inputs = build_prompt(d, args, turn_idx=turn_idx, conversation_history=conversation_history if turn_idx > 0 else None)
                input_len = model_inputs["input_ids"].shape[1]
                
                torch.cuda.synchronize()
                start_time = time.time()
                time_tracker.reset()
                
                output_ids, new_token, idx, accp_len = model.specgenerate(
                    **model_inputs,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_token,
                    log=True,
                    return_acceptance_len=True,
                    matrix_top_k=args.matrix_top_k,
                    draft_len=args.draft_len,
                )
                
                torch.cuda.synchronize()
                wall_time = time.time() - start_time
                draft_time, target_time = time_tracker.snapshot()
                
                output = process_output(output_ids, tokenizer, input_len)
                
                turn_outputs.append(output)
                turn_idxs.append(int(idx))
                turn_new_tokens.append(int(new_token))
                turn_wall_times.append(wall_time)
                turn_acceptance_lengths.append(accp_len)
                turn_draft_times.append(draft_time)
                turn_target_times.append(target_time)
                
                # Accumulate stats
                total_tokens += int(new_token)
                total_time += wall_time
                total_acceptance_lengths.extend(accp_len)
                
                # Print per-sample stats
                tokens_per_sec = new_token / wall_time if wall_time > 0 else 0
                avg_accept = sum(accp_len) / len(accp_len) if accp_len else 0
                # print(f"\n[EVAL] Sample {sample_idx}/{len(data)} turn {turn_idx}: "
                #       f"new_tokens={new_token}, time={wall_time:.2f}s, "
                #       f"speed={tokens_per_sec:.1f} tok/s, "
                #       f"avg_accept_len={avg_accept:.3f}")
                
                turns = d.get("turns", [d.get("prompt", "")])
                conversation_history.append((turns[turn_idx], output))
            
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
    
    # # Print overall stats
    # print(f"\n{'=' * 60}")
    # print(f"[EVAL] OVERALL STATS")
    # print(f"{'=' * 60}")
    # overall_speed = total_tokens / total_time if total_time > 0 else 0
    # print(f"[EVAL] Total tokens: {total_tokens}")
    # print(f"[EVAL] Total time: {total_time:.2f}s")
    # print(f"[EVAL] Overall speed: {overall_speed:.2f} tokens/s")
    # if total_acceptance_lengths:
    #     avg_accept = sum(total_acceptance_lengths) / len(total_acceptance_lengths)
    #     print(f"[EVAL] Avg acceptance length: {avg_accept:.4f}")
    #     # Acceptance distribution
    #     from collections import Counter
    #     accept_dist = Counter(total_acceptance_lengths)
    #     total_count = len(total_acceptance_lengths)
    #     print(f"[EVAL] Acceptance distribution:")
    #     for k in sorted(accept_dist.keys()):
    #         pct = 100 * accept_dist[k] / total_count
    #         print(f"[EVAL]   {k}: {accept_dist[k]} ({pct:.1f}%)")
    # print(f"{'=' * 60}")
    
    reorg_answer_file(args.answer_file)
    # print(f"Results saved to {args.answer_file}")


def main():
    parser = argparse.ArgumentParser(description="Token Recycling speculative decoding evaluation for MMSpec")
    parser = get_common_args(parser)
    
    # Token Recycling specific args
    parser.add_argument(
        "--matrix-top-k",
        type=int,
        default=8,
        help="Number of candidate tokens to store per token in adjacency matrix",
    )
    parser.add_argument(
        "--draft-len",
        type=int,
        default=10,
        help="Maximum chain draft length",
    )
    
    args = parser.parse_args()
    
    # Set model for prompt building
    args.model = args.base_model_path
    args.model_id = args.model_id + "-Recycling-temperature-" + str(args.temperature)
    
    # Set answer file
    if not args.answer_file:
        args.answer_file = f"{args.bench_name}/{args.model_id}.jsonl"
    
    print(f"Output to {args.answer_file}")
    print("Method: Token Recycling")
    print(f"Recycling params: matrix_top_k={args.matrix_top_k}, draft_len={args.draft_len}")
    
    evaluate(args)


if __name__ == "__main__":
    main()

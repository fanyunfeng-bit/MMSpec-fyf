import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from method.medusa.kv_cache import initialize_past_key_values
from method.medusa.utils import *
from method.medusa.spec_model_medusa import SpecModel
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
    # print("\n[DEBUG] === Loading Model ===")
    # print(f"[DEBUG] base_model_path: {args.base_model_path}")
    # print(f"[DEBUG] spec_model_path: {args.spec_model_path}")
    # print(f"[DEBUG] total_token={args.total_token}, depth={args.depth}, top_k={args.top_k}")
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

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)
    else:
        logits_processor = None

    model.eval()
    time_tracker = build_time_breakdown_tracker(model)
    # print("[DEBUG] model.training:", model.training)
    # print("[DEBUG] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # print(f"[DEBUG] spec_layer type: {type(model.spec_layer).__name__}")
    # print(f"[DEBUG] spec_layer.medusa heads: {model.spec_layer.medusa}")
    # print(f"[DEBUG] spec_layer.hidden_size: {model.spec_layer.hidden_size}")
    # print(f"[DEBUG] spec_layer.vocab_size: {model.spec_layer.vocab_size}")
    # print(f"[DEBUG] spec_layer.total_tokens: {model.spec_layer.total_tokens}")
    # print(f"[DEBUG] spec_layer.depth: {model.spec_layer.depth}")
    # print(f"[DEBUG] spec_layer.top_k: {model.spec_layer.top_k}")
    # print(f"[DEBUG] base_model arch: {model.base_model.config.architectures}")

    data = load_mmspec_data(args.data_folder)
    # print(f"\n[DEBUG] === Dataset ===")
    # print(f"[DEBUG] data_folder: {args.data_folder}")
    print(f"Loaded {len(data)} samples")
    if getattr(args, "sanity", False):
        run_sanity_check(args, model, tokenizer, data)
        return
    # if len(data) > 0:
    #     d0 = data[0]
    #     print(f"[DEBUG] Sample 0 keys: {list(d0.keys())}")
    #     print(f"[DEBUG] Sample 0 id: {d0.get('question_id', d0.get('id', 'N/A'))}")
    #     turns0 = d0.get('turns', [d0.get('prompt', '')])
    #     print(f"[DEBUG] Sample 0 num_turns: {len(turns0)}")
    #     print(f"[DEBUG] Sample 0 turn[0] (first 80 chars): {str(turns0[0])[:80]}")

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

    # total_tokens_generated = 0
    # total_time_spent = 0.0
    # total_acceptance_lengths = []
    # total_steps_count = 0
    # print(f"\n[DEBUG] === Starting Evaluation ({len(data)} samples) ===")

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

                # avg_accp = sum(accp_len) / len(accp_len) if accp_len else 0
                # accept_rate = (sum(accp_len) / (len(accp_len) * model.spec_layer.total_tokens)) if accp_len else 0
                # print(f"[DEBUG] Sample {sample_idx} turn {turn_idx}: "
                #       f"new_token={new_token}, steps={idx}, "
                #       f"time={total_time:.3f}s, "
                #       f"tokens/s={new_token/total_time:.1f}, "
                #       f"avg_accept_len={avg_accp:.2f}, "
                #       f"accept_rate={accept_rate:.4f}, "
                #       f"accept_len_list={accp_len}, "
                #       f"output[:80]={output[:80]}")

                # total_tokens_generated += int(new_token)
                # total_time_spent += total_time
                # total_acceptance_lengths.extend(accp_len)
                # total_steps_count += int(idx) + 1

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
    # print(f"\n[DEBUG] === Final Statistics ===")
    # print(f"[DEBUG] Total samples: {len(data)}")
    # print(f"[DEBUG] Total tokens generated: {total_tokens_generated}")
    # print(f"[DEBUG] Total decoding steps: {total_steps_count}")
    # print(f"[DEBUG] Total wall time: {total_time_spent:.2f}s")
    # if total_time_spent > 0:
    #     print(f"[DEBUG] Overall tokens/s: {total_tokens_generated / total_time_spent:.2f}")
    # if total_acceptance_lengths:
    #     avg_al = sum(total_acceptance_lengths) / len(total_acceptance_lengths)
    #     max_possible = model.spec_layer.total_tokens
    #     overall_accept_rate = avg_al / max_possible if max_possible > 0 else 0
    #     print(f"[DEBUG] Overall avg acceptance length: {avg_al:.3f}")
    #     print(f"[DEBUG] Overall acceptance rate: {overall_accept_rate:.4f} (avg_accept_len / max_draft_tokens={max_possible})")
    #     print(f"[DEBUG] Overall avg tokens per step: {total_tokens_generated / total_steps_count:.3f}" if total_steps_count > 0 else "")
    #     # Distribution of acceptance lengths
    #     from collections import Counter
    #     accp_counter = Counter(total_acceptance_lengths)
    #     print(f"[DEBUG] Acceptance length distribution: {dict(sorted(accp_counter.items()))}")
    print(f"Results saved to {args.answer_file}")


def main():
    parser = argparse.ArgumentParser(description="Medusa evaluation for MMSpec")
    parser = get_common_args(parser)

    parser.add_argument(
        "--total-token",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    args.model = args.base_model_path
    args.model_id = args.model_id + "-medusa-temperature-" + str(args.temperature)

    if not args.answer_file:
        args.answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {args.answer_file}")
    print("Method: Medusa (Speculative Decoding)")

    evaluate(args)


if __name__ == "__main__":
    main()

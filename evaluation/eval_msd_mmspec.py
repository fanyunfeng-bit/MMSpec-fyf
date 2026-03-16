"""MSD (Multimodal Speculative Decoding) evaluation for MMSpec unified dataset.

Usage:
python -m evaluation.eval_msd_mmspec --base-model-path <path> --msd-model-path <path> --data-folder <path>
"""

import argparse
import os
import sys
import time

import torch
from tqdm import tqdm
from PIL import Image

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from method.msd.ea_model import EaModel
from method.msd.utils import temp_cache
from evaluation.utils import (
    load_mmspec_data,
    build_prompt,
    process_output,
    save_result,
    reorg_answer_file,
    load_existing_ids,
    get_num_turns,
    run_sanity_check,
    iter_eval_samples,
)
from evaluation.time_breakdown import build_time_breakdown_tracker


def _assert_non_fsdp_checkpoint(msd_model_path: str):
    """Fail fast if only FSDP-only checkpoint files are provided."""
    if not os.path.isdir(msd_model_path):
        return

    fsdp_candidates = (
        "pytorch_model_fsdp.bin",
        "model_fsdp.safetensors",
        "fsdp_model.bin",
    )
    std_candidates = ("pytorch_model.bin", "model.safetensors")

    has_fsdp = any(os.path.exists(os.path.join(msd_model_path, f)) for f in fsdp_candidates)
    has_std = any(os.path.exists(os.path.join(msd_model_path, f)) for f in std_candidates)

    if has_fsdp and not has_std:
        raise RuntimeError(
            "Found FSDP-only MSD checkpoint, but FSDP is disabled for fair comparison. "
            "Please provide standard weights (pytorch_model.bin or model.safetensors)."
        )


@torch.inference_mode()
def evaluate(args):
    """Run MSD evaluation on MMSpec dataset."""
    # Enable MSD mode
    temp_cache.use_msd = args.use_msd
    if not args.allow_fsdp_checkpoint:
        _assert_non_fsdp_checkpoint(args.msd_model_path)
    
    # Load model
    print(f"Loading MSD model from {args.msd_model_path}...")
    loaded = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.msd_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
    )
    if isinstance(loaded, tuple):
        if len(loaded) == 4:
            _, model, _, _ = loaded
        elif len(loaded) == 2:
            model, _ = loaded
        else:
            model = loaded[0]
    else:
        model = loaded
    
    model.eval()
    time_tracker = build_time_breakdown_tracker(model)
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Attention implementation:", args.attn_implementation)

    model_cfg = getattr(getattr(model, "base_model", model), "config", None)
    image_token_id = None
    if model_cfg is not None:
        image_token_id = getattr(model_cfg, "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(model_cfg, "image_token_id", None)

    def _to_msd_input_ids(ids: torch.Tensor) -> torch.Tensor:
        # HF-LLaVA path in MSD builds EA -200 placeholder internally.
        if getattr(model, "_is_hf_llava", False):
            return ids

        # Qwen2-VL/Qwen2.5-VL: keep native image token layout (151652/151655...)
        # because MSD's Qwen path consumes the original sequence format directly.
        model_type = getattr(model_cfg, "model_type", None) if model_cfg is not None else None
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            return ids

        # MSD draft code expects a single -200 placeholder for each image block.
        if image_token_id is None:
            return ids
        mapped_rows = []
        for row in ids:
            new_row = []
            in_image_block = False
            for token in row.tolist():
                if token == image_token_id:
                    if not in_image_block:
                        new_row.append(-200)
                        in_image_block = True
                    # skip subsequent image tokens in the same contiguous block
                else:
                    in_image_block = False
                    new_row.append(token)
            mapped_rows.append(new_row)
        mapped = torch.tensor(mapped_rows, dtype=ids.dtype, device=ids.device)
        return mapped
    
    # Load data
    args.model = args.base_model_path
    data = load_mmspec_data(args.data_folder)
    print(f"Loaded {len(data)} samples")
    if getattr(args, "sanity", False):
        run_sanity_check(args, model, model.get_tokenizer(), data)
        return

    # Warmup
    print("Warming up...")
    for _ in range(3):
        torch.manual_seed(0)
        model_inputs = build_prompt(data[0], args)
        input_ids = model_inputs["input_ids"]
        gen_input_ids = _to_msd_input_ids(input_ids)
        inputs_embeds, attention_mask = model.get_inputs_embeds(
            input_ids=input_ids,
            pixel_values=model_inputs.get("pixel_values", None),
            image_sizes=[data[0]["image"].size],
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            attention_mask=model_inputs.get("attention_mask", None),
        )
        if args.use_msd:
            _ = model.msdgenerate(
                gen_input_ids,
                inputs_embeds=inputs_embeds,
                temperature=args.temperature,
                max_new_tokens=min(args.max_new_token, 50),
            )
        else:
            _ = model.naivegenerate(
                gen_input_ids,
                inputs_embeds=inputs_embeds,
                temperature=args.temperature,
                max_new_tokens=min(args.max_new_token, 50),
            )
    print("Warmup done")
    
    # Reset counters
    model.acclen = 0
    model.accnum = 0
    
    # Evaluate
    total_time = 0
    total_tokens = 0
    
    existing_ids = load_existing_ids(args.answer_file)
    print(f"Skipping {len(existing_ids)} already-evaluated samples")
    for d in tqdm(iter_eval_samples(data, args.batch_size), total=len(data), desc=f"Evaluating (bs={args.batch_size})"):
        if d["id"] in existing_ids:
            continue
        choices = []
        num_turns = get_num_turns(d)
        
        for i in range(args.num_choices):
            torch.manual_seed(i)
            
            # For multi-turn, collect results across all turns
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
                input_ids = model_inputs["input_ids"]
                gen_input_ids = _to_msd_input_ids(input_ids)
                input_len = gen_input_ids.shape[1]
                
                # Get input embeds
                inputs_embeds, attention_mask = model.get_inputs_embeds(
                    input_ids=input_ids,
                    pixel_values=model_inputs.get("pixel_values", None),
                    image_sizes=[d["image"].size],
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    attention_mask=model_inputs.get("attention_mask", None),
                )
                
                torch.cuda.synchronize()
                start_time = time.time()
                time_tracker.reset()
                
                if args.use_msd:
                    output_ids, new_tokens, idx = model.msdgenerate(
                        gen_input_ids,
                        inputs_embeds=inputs_embeds,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_token,
                        log=True,
                    )
                else:
                    output_ids, new_tokens, idx = model.naivegenerate(
                        gen_input_ids,
                        inputs_embeds=inputs_embeds,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_token,
                        log=True,
                    )
                
                torch.cuda.synchronize()
                dec_time = time.time() - start_time
                draft_time, target_time = time_tracker.snapshot()
                
                total_time += dec_time
                total_tokens += new_tokens
                
                output = process_output(output_ids, model.get_tokenizer(), input_len)
                
                # Calculate acceptance length if MSD
                if args.use_msd and model.accnum > 0:
                    avg_accept_len = model.acclen / model.accnum
                else:
                    avg_accept_len = 1.0
                
                # Store results for this turn
                turn_outputs.append(output)
                turn_idxs.append(int(idx))
                turn_new_tokens.append(int(new_tokens))
                turn_wall_times.append(dec_time)
                turn_acceptance_lengths.append(float(avg_accept_len))
                turn_draft_times.append(draft_time)
                turn_target_times.append(target_time)
                
                # Add to conversation history for next turn
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
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total samples: {len(data)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens/s: {total_tokens / total_time:.2f}")
    if args.use_msd and model.accnum > 0:
        print(f"Average acceptance length: {model.acclen / model.accnum:.2f}")
    
    reorg_answer_file(args.answer_file)
    print(f"Results saved to {args.answer_file}")


def main():
    parser = argparse.ArgumentParser(description="MSD evaluation for MMSpec")
    
    # Model paths
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to base LLaVA model (e.g., liuhaotian/llava-v1.5-7b)",
    )
    parser.add_argument(
        "--msd-model-path",
        type=str,
        required=True,
        help="Path to MSD speculative head model (e.g., lucylyn/MSD-LLaVA1.5-7B)",
    )
    
    # Data and output
    parser.add_argument(
        "--data-folder",
        type=str,
        default="dataset/MMSpec",
        help="Path to MMSpec dataset folder",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        help="Output answer file path",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="msd-llava",
    )
    
    # Generation params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size setting (used for unified scheduling across methods).",
    )
    
    # MSD params
    parser.add_argument("--sanity", action="store_true", default=False, help="Sanity-check mode: load model, log details, smoke test, then exit")
    parser.add_argument("--use-msd", action="store_true", help="Enable MSD speculative decoding")
    parser.add_argument("--total-token", type=int, default=-1, help="Total draft tokens (-1 for auto)")
    parser.add_argument("--depth", type=int, default=5, help="Decoding tree depth")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for tree generation")
    
    # LLaVA params
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1", help="Conversation template")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="HF attention implementation. Default eager to disable SDPA for fair comparison.",
    )
    parser.add_argument(
        "--allow-fsdp-checkpoint",
        action="store_true",
        help="Allow FSDP-only speculative-head checkpoints. Disabled by default for fair comparison.",
    )
    
    args = parser.parse_args()
    
    # Set answer file
    if not args.answer_file:
        result_dir = "results/mmspec_test"
        msd_suffix = "-msd" if args.use_msd else "-baseline"
        args.answer_file = f"{result_dir}/msd-llava{msd_suffix}-temperature-{args.temperature}.jsonl"
    
    args.model_id = args.model_id + ("-msd" if args.use_msd else "-baseline") + f"-temperature-{args.temperature}"
    
    print(f"Output to {args.answer_file}")
    
    evaluate(args)


if __name__ == "__main__":
    main()

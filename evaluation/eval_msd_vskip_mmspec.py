"""MSD-vskip (visual-attention gated MSD) evaluation for MMSpec.

Usage:
python -m evaluation.eval_msd_vskip_mmspec --base-model-path <path> --msd-model-path <path> --data-folder <path>
"""

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

from method.msd_vskip.ea_model import EaModel
from method.msd.utils import temp_cache
from evaluation.utils import (
    get_num_turns,
    load_mmspec_data,
    process_output,
    reorg_answer_file,
    load_existing_ids,
    save_result,
    build_prompt,
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
    temp_cache.use_msd = args.use_msd

    if not args.allow_fsdp_checkpoint:
        _assert_non_fsdp_checkpoint(args.msd_model_path)

    print(f"Loading MSD-vskip model from {args.msd_model_path}...")
    model_load_kwargs = dict(
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if args.attn_implementation:
        model_load_kwargs["attn_implementation"] = args.attn_implementation
    loaded = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.msd_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        **model_load_kwargs,
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
    print(
        "MSD-vskip config:",
        f"skip_when_visual={not args.disable_visual_skip},",
        f"visual_attn_threshold={args.visual_attn_threshold},",
        f"visual_probe_only={args.visual_probe_only},",
        f"allow_fsdp_checkpoint={args.allow_fsdp_checkpoint}",
        f"attn_implementation={args.attn_implementation}",
    )

    model_cfg = getattr(getattr(model, "base_model", model), "config", None)
    image_token_id = None
    if model_cfg is not None:
        image_token_id = getattr(model_cfg, "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(model_cfg, "image_token_id", None)

    def _to_msd_input_ids(ids: torch.Tensor) -> torch.Tensor:
        if getattr(model, "_is_hf_llava", False):
            return ids

        model_type = getattr(model_cfg, "model_type", None) if model_cfg is not None else None
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            return ids

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
                else:
                    in_image_block = False
                    new_row.append(token)
            mapped_rows.append(new_row)
        return torch.tensor(mapped_rows, dtype=ids.dtype, device=ids.device)

    args.model = args.base_model_path
    data = load_mmspec_data(args.data_folder)
    print(f"Loaded {len(data)} samples")

    print("Warming up...")
    for _ in range(3):
        torch.manual_seed(0)
        model_inputs = build_prompt(data[0], args)
        input_ids = model_inputs["input_ids"]
        gen_input_ids = _to_msd_input_ids(input_ids)
        inputs_embeds, _ = model.get_inputs_embeds(
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
                skip_when_visual=(not args.disable_visual_skip),
                visual_attn_threshold=args.visual_attn_threshold,
                no_op_on_visual_hit=args.visual_probe_only,
                visual_probe_interval=args.visual_probe_interval,
            )
        else:
            _ = model.naivegenerate(
                gen_input_ids,
                inputs_embeds=inputs_embeds,
                temperature=args.temperature,
                max_new_tokens=min(args.max_new_token, 50),
            )
    print("Warmup done")

    model.acclen = 0
    model.accnum = 0

    total_time = 0.0
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

                inputs_embeds, _ = model.get_inputs_embeds(
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
                        skip_when_visual=(not args.disable_visual_skip),
                        visual_attn_threshold=args.visual_attn_threshold,
                        no_op_on_visual_hit=args.visual_probe_only,
                        visual_probe_interval=args.visual_probe_interval,
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
                if args.use_msd and model.accnum > 0:
                    avg_accept_len = model.acclen / model.accnum
                else:
                    avg_accept_len = 1.0

                turn_outputs.append(output)
                turn_idxs.append(int(idx))
                turn_new_tokens.append(int(new_tokens))
                turn_wall_times.append(dec_time)
                turn_acceptance_lengths.append(float(avg_accept_len))
                turn_draft_times.append(draft_time)
                turn_target_times.append(target_time)

                turns = d.get("turns", [d.get("prompt", "")])
                user_msg = turns[turn_idx] if turn_idx < len(turns) else turns[0]
                conversation_history.append((user_msg, output))

            choices.append(
                {
                    "index": i,
                    "turns": turn_outputs,
                    "idxs": turn_idxs,
                    "new_tokens": turn_new_tokens,
                    "wall_time": turn_wall_times,
                    "acceptance_length": turn_acceptance_lengths,
                    "draft_time": turn_draft_times,
                    "target_time": turn_target_times,
                }
            )

        save_result(args.answer_file, d, args.model_id, choices)

    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total samples: {len(data)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    if total_time > 0:
        print(f"Tokens/s: {total_tokens / total_time:.2f}")
    if args.use_msd and model.accnum > 0:
        print(f"Average acceptance length: {model.acclen / model.accnum:.2f}")

    reorg_answer_file(args.answer_file)
    print(f"Results saved to {args.answer_file}")


def main():
    parser = argparse.ArgumentParser(description="MSD-vskip evaluation for MMSpec")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--msd-model-path", type=str, required=True)
    parser.add_argument("--data-folder", type=str, default="dataset/MMSpec")
    parser.add_argument("--answer-file", type=str, help="Output answer file path")
    parser.add_argument("--model-id", type=str, default="msd-vskip")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size setting (used for unified scheduling across methods).",
    )
    parser.add_argument("--use-msd", action="store_true")
    parser.add_argument("--total-token", type=int, default=-1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument(
        "--disable-visual-skip",
        action="store_true",
        help="Disable vskip gating and run vanilla MSD everywhere.",
    )
    parser.add_argument(
        "--visual-attn-threshold",
        type=float,
        default=0.35,
        help="If visual attention ratio >= threshold, force one target-only step.",
    )
    parser.add_argument(
        "--visual-probe-only",
        action="store_true",
        help="Only probe visual attention; do not change decoding behavior on gate hit.",
    )
    parser.add_argument(
        "--visual-probe-interval",
        type=int,
        default=1,
        help="Probe visual attention every N speculative steps (default: 1).",
    )
    parser.add_argument(
        "--allow-fsdp-checkpoint",
        action="store_true",
        help="Allow FSDP-only speculative-head checkpoints. Disabled by default for fair comparison.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="HF attention implementation. Default eager to disable SDPA for fair comparison.",
    )
    args = parser.parse_args()

    if not args.answer_file:
        result_dir = "results/mmspec_test"
        msd_suffix = "-msd-vskip" if args.use_msd else "-baseline"
        args.answer_file = (
            f"{result_dir}/msd-vskip{msd_suffix}-temperature-{args.temperature}.jsonl"
        )

    args.model_id = (
        args.model_id
        + ("-msd-vskip" if args.use_msd else "-baseline")
        + f"-temperature-{args.temperature}"
    )

    print(f"Output to {args.answer_file}")
    evaluate(args)


if __name__ == "__main__":
    main()

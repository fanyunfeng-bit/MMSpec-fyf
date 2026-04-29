"""MSD + SAGE (visual-token compression on the draft) evaluation for MMSpec.

This is a thin wrapper around `eval_msd_mmspec.py` that loads
`SageEaModel` instead of plain `EaModel`, wires the SAGE visual processor
(L2-norm sink detector + TI-Prob compressor), and forwards the per-turn
sample id so log lines are easy to grep.

Usage:
  python -m evaluation.eval_msd_SAGE_mmspec \\
      --base-model-path llava-hf/llava-1.5-7b-hf \\
      --msd-model-path  Cloudriver/MSD-LLaVA1.5-7B \\
      --data-folder dataset/MMSpec/testmini \\
      --answer-file results/.../msd_SAGE-temperature-0.jsonl \\
      --use-msd \\
      --sage-enable-compressor \\
      --sage-threshold-mode abs --sage-threshold-value 100 \\
      --sage-compressor-layer 16 --sage-compressor-topk 10
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

from method.msd.utils import temp_cache
from method.msd_SAGE.sage_ea_model import SageEaModel
from method.eagle_SAGE.visual_processor import VisualProcessor
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
            "Found FSDP-only MSD checkpoint, but FSDP is disabled. "
            "Provide standard weights (pytorch_model.bin or model.safetensors)."
        )


@torch.inference_mode()
def evaluate(args):
    temp_cache.use_msd = args.use_msd
    if not args.allow_fsdp_checkpoint:
        _assert_non_fsdp_checkpoint(args.msd_model_path)

    print(f"Loading SageEaModel from {args.msd_model_path}...")
    loaded = SageEaModel.from_pretrained(
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

    if not getattr(model, "_is_hf_llava", False):
        raise NotImplementedError(
            "msd_SAGE currently supports only the HF-LLaVA branch "
            "(LlavaForConditionalGeneration). Got a non-HF backbone."
        )

    model.eval()
    time_tracker = build_time_breakdown_tracker(model)

    # Resolve image_token_id from base_model.config (LLaVA: image_token_index).
    cfg = model.base_model.config
    image_token_id = getattr(cfg, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(cfg, "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError("Could not resolve image_token_id from base_model.config")

    # Build the SAGE pipeline (sink detector + compressor; repositioner OFF).
    visual_processor = VisualProcessor.from_args(
        args,
        spatial_merge_size=1,  # LLaVA-1.5 has no spatial merger
        base_model=model.base_model,
        image_token_id=image_token_id,
    )
    model.setup_sage(
        visual_processor=visual_processor,
        image_token_id=image_token_id,
        debug=bool(getattr(args, "sage_debug", False)),
    )

    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Attention implementation:", args.attn_implementation)
    print(
        f"[SAGE-MSD] mode={args.sage_threshold_mode} value={args.sage_threshold_value} "
        f"min_sinks={args.sage_min_sinks} "
        f"enable_sink={args.sage_enable_sink_detection} "
        f"enable_compressor={args.sage_enable_compressor} "
        f"compressor_layer={args.sage_compressor_layer} "
        f"compressor_topk={args.sage_compressor_topk}"
    )
    print(
        f"[SAGE-MSD] vision_hook arch={model._sage_vision_hook.arch} "
        f"drop_cls={model._sage_vision_hook.drop_cls}"
    )

    # ---- Helpers (verbatim from upstream eval_msd_mmspec) ----
    model_cfg = getattr(getattr(model, "base_model", model), "config", None)

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
    total_time = 0
    total_tokens = 0

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
                model._sage_sample_id = f"{d['id']}-turn{turn_idx}"

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
    parser = argparse.ArgumentParser(description="MSD + SAGE evaluation for MMSpec")

    # Model paths
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--msd-model-path", type=str, required=True)

    # Data and output
    parser.add_argument("--data-folder", type=str, default="dataset/MMSpec")
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--model-id", type=str, default="msd_SAGE")

    # Generation params
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)

    # MSD params
    parser.add_argument("--sanity", action="store_true", default=False)
    parser.add_argument("--use-msd", action="store_true")
    parser.add_argument("--total-token", type=int, default=-1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)

    # LLaVA params
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument(
        "--attn-implementation", type=str, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--allow-fsdp-checkpoint", action="store_true")

    # SAGE params (mirror eagle_SAGE flags)
    parser.add_argument(
        "--sage-threshold-mode", type=str, default="abs",
        choices=["abs", "topk_ratio"],
    )
    parser.add_argument("--sage-threshold-value", type=float, default=100.0)
    parser.add_argument("--sage-min-sinks", type=int, default=0)
    parser.add_argument(
        "--sage-enable-sink-detection",
        type=lambda s: s.lower() not in ("false", "0", "no"),
        default=True,
    )
    parser.add_argument(
        "--sage-enable-repositioning",
        type=lambda s: s.lower() not in ("false", "0", "no"),
        default=False,
        help="Repositioner is unused under compression; kept for parity with eagle_SAGE.",
    )
    parser.add_argument(
        "--sage-enable-compressor",
        type=lambda s: s.lower() not in ("false", "0", "no"),
        default=True,
    )
    parser.add_argument("--sage-compressor-layer", type=int, default=16)
    parser.add_argument("--sage-compressor-topk", type=int, default=10)
    parser.add_argument("--sage-debug", action="store_true", default=False)

    args = parser.parse_args()

    if not args.answer_file:
        result_dir = "results/mmspec_test"
        msd_suffix = "-msd" if args.use_msd else "-baseline"
        args.answer_file = f"{result_dir}/msd_SAGE{msd_suffix}-temperature-{args.temperature}.jsonl"

    args.model_id = (
        args.model_id
        + ("-msd" if args.use_msd else "-baseline")
        + f"-sage-{args.sage_threshold_mode}-{args.sage_threshold_value}"
        + f"-temperature-{args.temperature}"
    )

    print(f"Output to {args.answer_file}")
    evaluate(args)


if __name__ == "__main__":
    main()

"""SAGE utilities — wraps EAGLE's initialize_tree to inject VisualProcessor.

`sage_initialize_tree` replaces `method.eagle.utils.initialize_tree` with one
extra step between target prefill and draft topK_genrate:
  - Build a VisualContext from (hidden_states, input_ids, position_ids,
    visual_mask, pre_projector_features).
  - Run the VisualProcessor pipeline (sink detection + repositioning).
  - Pass the reordered tensors plus reordered position_ids into topK_genrate.

All other eagle utilities (tree_decoding, update_inference_inputs,
evaluate_posterior, generate_candidates, reset_tree_mode, prepare_logits_processor)
are re-exported verbatim so callers can `from method.eagle_SAGE.utils import *`.
"""
from __future__ import annotations

import torch

# Re-export EAGLE utilities so SAGE is a drop-in replacement.
from method.eagle.utils import (  # noqa: F401
    TOPK,
    Timer,
    prepare_logits_processor,
    pad_path,
    generate_tree_buffers,
    initialize_tree0,
    initialize_tree,
    reset_tree_mode,
    reset_past_key_values,
    generate_candidates,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
)

from .visual_processor import VisualContext


@torch.no_grad()
def sage_initialize_tree(
    input_ids,
    model,
    past_key_values,
    logits_processor,
    inputs_embeds=None,
    embed_weights=None,
    image_mask=None,
    visual_processor=None,
    visual_mask_1d=None,
    pre_projector_features=None,
    target_position_ids=None,
    debug: bool = False,
    sample_id=None,
    **kwargs,
):
    """SAGE-aware replacement for eagle.utils.initialize_tree.

    Parameters beyond the original:
      - visual_processor: VisualProcessor instance, or None to skip SAGE hook.
      - visual_mask_1d: [L] bool indicating visual-token positions in prefill.
      - pre_projector_features: vision encoder output before projector.
      - target_position_ids: position_ids tensor to feed (after reordering) into
        the draft's first forward. If None, defaults to arange(L_prefill).
      - debug: if True, prints a brief per-call diagnostic line.

    Returns: same tuple shape as eagle.utils.initialize_tree, i.e.
      (draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
       orig_logits, hidden_states, sampled_token).
    """
    outputs, orig, hidden_states = model(
        input_ids,
        past_key_values=past_key_values,
        output_orig=True,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    # Pop pre-projector vision features AFTER the model forward. For LLaVA-1.5,
    # the parent specgenerate has no pre-fire branch — the vision_tower fires
    # only inside this model() call — so popping earlier returns the previous
    # sample's stale features. For Qwen2.5-VL and LLaVA-Next, the parent does
    # pre-fire visual; the model() above passes inputs_embeds (already populated
    # with image features) so base_model.forward does NOT re-fire the vision
    # encoder, and last_features stays as the parent's pre-call capture.
    vh = getattr(model, "vision_hook", None)
    if vh is not None:
        popped = vh.pop()
        if popped is not None:
            pre_projector_features = popped

    # Sample the first target token (unchanged from eagle.utils.initialize_tree).
    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # --- SAGE HOOK: reposition visual tokens before draft's first forward ---
    spec_position_ids = None
    hs_len = hidden_states.shape[1]

    use_sage = (
        visual_processor is not None
        and visual_mask_1d is not None
        and pre_projector_features is not None
        and bool(visual_mask_1d.any())
    )
    if use_sage:
        device = hidden_states.device
        # Draft-side position_ids: arange(L_prefill), shape [1, L_prefill].
        # The draft is LLaMA-style 1D RoPE (regardless of target's M-RoPE for
        # Qwen2.5-VL). The repositioner permutes these so that each reordered
        # token keeps its ORIGINAL position encoding — this is the "positions
        # travel with tokens" invariant. In effect, RoPE rotation at draft
        # sequence-position i = rotation at original target-position perm[i].
        if target_position_ids is None:
            pids = torch.arange(hs_len, device=device, dtype=torch.long)[None, :]
        else:
            pids = target_position_ids.to(device)

        # Ensure visual_mask aligns with the prefill length.
        vmask = visual_mask_1d.to(device)
        if vmask.numel() != hs_len:
            vmask = (
                vmask[:hs_len]
                if vmask.numel() > hs_len
                else torch.nn.functional.pad(vmask, (0, hs_len - vmask.numel()))
            )

        # Qwen2.5-VL: hook on visual.blocks[-1] captures features in
        # window-permuted order (the visual tower applies reverse_indices only
        # AFTER the merger). Compute the reverse permutation here so downstream
        # stages can convert per-post-merger-token quantities back to spatial
        # (LLM-input) order. None for LLaVA (no window permutation).
        window_reverse_indices = None
        arch_name = model.base_model.config.architectures[0]
        image_grid_thw = kwargs.get("image_grid_thw")
        if (
            arch_name == "Qwen2_5_VLForConditionalGeneration"
            and image_grid_thw is not None
        ):
            try:
                window_index, _ = model.base_model.visual.get_window_index(
                    image_grid_thw
                )
                window_reverse_indices = torch.argsort(window_index).to(device)
            except Exception:
                window_reverse_indices = None

        ctx = VisualContext(
            hidden_states=hidden_states,
            input_ids=input_ids[:, :hs_len].clone(),
            position_ids=pids,
            visual_mask=vmask,
            pre_projector_features=pre_projector_features,
            arch=arch_name,
            window_reverse_indices=window_reverse_indices,
        )
        ctx.meta["sample_id"] = sample_id
        ctx = visual_processor.run(ctx)

        hidden_states = ctx.hidden_states
        # spec_position_ids is length L_prefill, matching the shifted draft
        # input in topK_genrate (input_ids[:, 1:] yields length L_prefill when
        # its caller-fed length is L_prefill+1). Do NOT slice or append here.
        spec_position_ids = ctx.position_ids

        # input_ids_for_draft = compressed/reordered prefix + the newly sampled
        # token. ctx.input_ids may be shorter than the original prefix if a
        # VisualCompressor stage ran.
        input_ids_for_draft = torch.cat(
            (ctx.input_ids, input_ids[:, hs_len:]), dim=1
        )

        # If a VisualCompressor stage produced a keep_mask, stash it on the
        # draft so that subsequent topK_genrate calls (from update_inference_inputs,
        # which re-pass the full original input_ids) can compress to match the
        # draft's stable_kv layout. original_prefix_len = pre-compression hs_len.
        keep_full = ctx.meta.get("sage_keep_mask_full")
        if keep_full is not None and hasattr(model, "spec_layer"):
            model.spec_layer._sage_keep_mask = keep_full.to(hidden_states.device)
            model.spec_layer._sage_original_prefix_len = int(hs_len)

        if debug:
            nv = ctx.meta.get("num_visual", 0)
            ns = ctx.meta.get("num_sinks", 0)
            mode = (
                visual_processor.stages[0].mode
                if visual_processor.stages and hasattr(visual_processor.stages[0], "mode")
                else "-"
            )
            comp_len = ctx.meta.get("compressed_len")
            n_kept_v = ctx.meta.get("num_visual_after")
            n_sinks_kept = ctx.meta.get("num_sinks_kept")
            n_ti_kept = ctx.meta.get("num_ti_kept")
            extra = ""
            if comp_len is not None:
                extra = (
                    f" kept_visual={n_kept_v} (sinks={n_sinks_kept}+ti={n_ti_kept})"
                    f" compressed_len={comp_len}"
                )
            print(
                f"[SAGE] prefill_len={hs_len} visual={nv} sinks={ns} "
                f"(mode={mode}){extra}"
            )
    else:
        input_ids_for_draft = input_ids

    try:
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
            model.spec_layer.topK_genrate(
                hidden_states,
                input_ids_for_draft,
                model.base_model.lm_head,
                logits_processor,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                position_ids=spec_position_ids,
            )
        )
    except AttributeError:
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
            model.spec_layer.topK_genrate(
                hidden_states,
                input_ids_for_draft,
                model.base_model.language_model.lm_head,
                logits_processor,
                inputs_embeds=inputs_embeds,
                image_mask=image_mask,
                position_ids=spec_position_ids,
            )
        )

    return (
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        orig,
        hidden_states,
        token,
    )

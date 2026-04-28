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

        ctx = VisualContext(
            hidden_states=hidden_states,
            input_ids=input_ids[:, :hs_len].clone(),
            position_ids=pids,
            visual_mask=vmask,
            pre_projector_features=pre_projector_features,
            arch=model.base_model.config.architectures[0],
        )
        ctx = visual_processor.run(ctx)

        hidden_states = ctx.hidden_states
        # spec_position_ids is length L_prefill, matching the shifted draft
        # input in topK_genrate (input_ids[:, 1:] yields length L_prefill when
        # its caller-fed length is L_prefill+1). Do NOT slice or append here.
        spec_position_ids = ctx.position_ids

        # input_ids_for_draft = reordered prefix + the newly sampled token.
        input_ids_for_draft = torch.cat(
            (ctx.input_ids, input_ids[:, hs_len:]), dim=1
        )

        if debug:
            nv = ctx.meta.get("num_visual", 0)
            ns = ctx.meta.get("num_sinks", 0)
            mode = (
                visual_processor.stages[0].mode
                if visual_processor.stages and hasattr(visual_processor.stages[0], "mode")
                else "-"
            )
            print(
                f"[SAGE] prefill_len={hs_len} visual={nv} sinks={ns} (mode={mode})"
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

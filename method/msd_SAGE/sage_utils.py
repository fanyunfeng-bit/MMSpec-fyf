"""SAGE versions of MSD's HF-LLaVA tree initialization and update helpers.

These are minimal-diff clones of `method.msd.utils.initialize_tree_hf` and
`update_inference_inputs_hf`. The only behavioral change is a SAGE compression
step right before `model.ea_layer.topK_genrate`:

  * On prefill (first call): build a [P] bool keep mask over the FULL prefix
    (text positions kept; visual positions reduced to sinks ∪ TI-Prob top-k),
    then compress `hidden_states` and `ea_inputs_embeds[:, :P, :]` accordingly.
    Stash the mask + meta on `model.ea_layer` so subsequent calls reuse it.

  * On verify (subsequent calls): the wrapper compresses `ea_inputs_embeds`
    again — its visual prefix is still in the original layout because the
    update helper only appends accepted-token embeddings to the tail.

Everything outside the draft is untouched: target prefill, target's KV cache,
tree decoding, evaluate_posterior, sampling, accept/reject — all reuse the
upstream MSD code.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from method.msd.utils import temp_cache, evaluate_posterior  # noqa: F401
from method.eagle_SAGE.visual_processor import VisualContext


# Set SAGE_MSD_TRIM_DEBUG=1 to print before/after lengths of every trim.
import os as _os
_TRIM_DEBUG = _os.environ.get("SAGE_MSD_TRIM_DEBUG", "0") == "1"


def _trim_draft_stable_kv(model) -> None:
    """Drop the `depth * top_k` speculative tokens that MSD's `topK_genrate`
    appends to `model.ea_layer.stable_kv` during its tree-expansion loop.

    Why this is needed under SAGE compression
    -----------------------------------------
    `topK_genrate` does `self.stable_kv = past_key_values` BEFORE its depth
    loop, but the loop extends `past_key_values` in place (HF-cache style),
    so by the time the function returns `stable_kv` actually contains
    `(committed prefix) + (depth * top_k speculative tokens)`.

    Vanilla MSD masks this on the next call via the arithmetic
        kv_len = stable_kv_len - image_extra_tokens
    where `image_extra_tokens = 576` (HF-LLaVA visual span). The 576 happens
    to be large enough that `kv_len` stays below `input_ids.shape[1]`, so
    `use_stable_kv` stays True and the draft incrementally encodes only the
    new accepted tokens.

    Under SAGE compression, `image_extra_tokens = K` (e.g. 16). `kv_len`
    overshoots `input_ids.shape[1]`, MSD's `use_stable_kv=False` branch
    fires, and the draft falls into the from-scratch path with
    `hidden_states (length A+1)` against `inputs_embeds (length C+A+2)` —
    a shape mismatch that triggers a CUDA OOB inside the depth loop.

    Trimming back to `length - depth*top_k` after each topK_genrate keeps
    the draft cache equal to "committed prefix only", which is what the
    next round's incremental forward expects. This does NOT touch the
    target's KV cache (a separate DynamicCache); the target's verify path
    and accept/reject logic are unchanged → speculative decoding stays
    lossless.
    """
    skv = getattr(model.ea_layer, "stable_kv", None)
    if skv is None:
        return
    drop = int(getattr(model.ea_layer, "depth", 0)) * int(
        getattr(model.ea_layer, "top_k", 0)
    )
    if drop <= 0:
        return
    new_kv = []
    trimmed_any = False
    before_len = None
    for layer_kv in skv:
        if (
            isinstance(layer_kv, (tuple, list))
            and len(layer_kv) >= 2
            and torch.is_tensor(layer_kv[0])
            and layer_kv[0].dim() >= 3
            and layer_kv[0].shape[2] > drop
        ):
            cur = layer_kv[0].shape[2]
            if before_len is None:
                before_len = cur
            keep = cur - drop
            k = layer_kv[0][:, :, :keep, :].contiguous()
            v = layer_kv[1][:, :, :keep, :].contiguous()
            new_kv.append((k, v))
            trimmed_any = True
        else:
            new_kv.append(layer_kv)
    if trimmed_any:
        model.ea_layer.stable_kv = tuple(new_kv)
        if _TRIM_DEBUG:
            print(
                f"[SAGE-MSD trim] stable_kv {before_len} -> {before_len - drop} "
                f"(dropped depth*top_k = {drop})",
                flush=True,
            )


def _build_visual_mask_1d(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Boolean [L] mask marking visual placeholder positions in the FULL prefill input_ids."""
    return (input_ids[0] == image_token_id).to(torch.bool)


@torch.no_grad()
def _compress_for_draft(
    *,
    model,
    visual_processor,
    hidden_states: torch.Tensor,
    ea_inputs_embeds: torch.Tensor,
    full_input_ids: torch.Tensor,
    image_token_id: int,
    sample_id: Optional[str] = None,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the SAGE pipeline on a (hidden_states, ea_inputs_embeds) pair and
    return their compressed versions.

    The visual_mask is built from the FULL prefill input_ids (length P,
    pre-collapse). hidden_states must have length P; ea_inputs_embeds must
    have length P + suffix (suffix = sample token, possibly + accepted
    tokens). Only the first P positions are compressed.

    Side effects:
      Stashes `model.ea_layer._sage_keep_mask` (bool [P]) and
      `model.ea_layer._sage_original_prefix_len` (int) so subsequent calls
      can reuse the same compression decision.
    """
    P = int(hidden_states.shape[1])
    visual_mask_full = _build_visual_mask_1d(full_input_ids, image_token_id).to(
        hidden_states.device
    )
    if visual_mask_full.numel() < P:
        visual_mask_full = torch.nn.functional.pad(
            visual_mask_full, (0, P - visual_mask_full.numel())
        )
    elif visual_mask_full.numel() > P:
        visual_mask_full = visual_mask_full[:P]

    # Pop pre-projector vision features (vision_tower hook captured them
    # during the just-finished target prefill).
    pre_feats = None
    vh = getattr(model, "_sage_vision_hook", None)
    if vh is not None:
        pre_feats = vh.pop()

    pids = torch.arange(P, device=hidden_states.device, dtype=torch.long)[None, :]
    ctx = VisualContext(
        hidden_states=hidden_states,
        input_ids=full_input_ids[:, :P].clone(),
        position_ids=pids,
        visual_mask=visual_mask_full,
        pre_projector_features=pre_feats,
        arch=model.base_model.config.architectures[0],
        window_reverse_indices=None,  # LLaVA: no window permutation
    )
    ctx.meta["sample_id"] = sample_id
    ctx = visual_processor.run(ctx)

    keep_full = ctx.meta.get("sage_keep_mask_full")
    if keep_full is None:
        # Compressor disabled (e.g. only sink detector ran). No compression.
        return hidden_states, ea_inputs_embeds

    keep_full = keep_full.to(hidden_states.device)
    # Compress hidden_states (length P → C).
    new_hidden = hidden_states.index_select(1, torch.nonzero(keep_full).squeeze(-1))

    # Compress ea_inputs_embeds first-P slice; keep the suffix (sample / accepted tokens) intact.
    keep_emb = keep_full.to(ea_inputs_embeds.device)
    prefix_emb = ea_inputs_embeds[:, :P, :]
    suffix_emb = ea_inputs_embeds[:, P:, :]
    compressed_prefix = prefix_emb[:, keep_emb, :]
    new_inputs_embeds = torch.cat([compressed_prefix, suffix_emb], dim=1)

    # Stash for subsequent topK_genrate calls.
    model.ea_layer._sage_keep_mask = keep_full
    model.ea_layer._sage_original_prefix_len = P

    if debug:
        nv = ctx.meta.get("num_visual", 0)
        ns = ctx.meta.get("num_sinks", 0)
        n_kept_v = ctx.meta.get("num_visual_after")
        n_sinks_kept = ctx.meta.get("num_sinks_kept")
        n_ti_kept = ctx.meta.get("num_ti_kept")
        comp_len = ctx.meta.get("compressed_len")
        print(
            f"[SAGE-MSD] prefill_len={P} visual={nv} sinks={ns} "
            f"kept_visual={n_kept_v} (sinks={n_sinks_kept}+ti={n_ti_kept}) "
            f"compressed_len={comp_len}",
            flush=True,
        )
    return new_hidden, new_inputs_embeds


@torch.no_grad()
def _compress_subsequent(
    *,
    model,
    ea_inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    """Apply the cached keep_mask to ea_inputs_embeds on subsequent calls.

    The first P positions of ea_inputs_embeds are still in the original
    visual-expanded layout (update_inference_inputs_hf only APPENDS to
    ea_inputs_embeds, so the prefix is unchanged across rounds).
    """
    keep_full = getattr(model.ea_layer, "_sage_keep_mask", None)
    P = getattr(model.ea_layer, "_sage_original_prefix_len", None)
    if keep_full is None or P is None:
        return ea_inputs_embeds
    if ea_inputs_embeds.shape[1] < P:
        return ea_inputs_embeds  # already compressed or no visual block
    keep_emb = keep_full.to(ea_inputs_embeds.device)
    prefix_emb = ea_inputs_embeds[:, :P, :]
    suffix_emb = ea_inputs_embeds[:, P:, :]
    compressed_prefix = prefix_emb[:, keep_emb, :]
    return torch.cat([compressed_prefix, suffix_emb], dim=1)


@torch.no_grad()
def sage_initialize_tree_hf(
    input_ids,
    model,
    past_key_values,
    logits_processor,
    inputs_embeds=None,
    *,
    visual_processor=None,
    image_token_id: int = 32000,
    sample_id: Optional[str] = None,
    debug: bool = False,
):
    """Drop-in replacement for `method.msd.utils.initialize_tree_hf` with
    visual-token compression injected before the draft's first forward.
    """
    # Reset draft mask state per generation.
    model.ea_layer._sage_keep_mask = None
    model.ea_layer._sage_original_prefix_len = None

    # ---- Target prefill ----
    if inputs_embeds is not None:
        outputs = model.base_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
    else:
        outputs = model.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
    orig = outputs.logits
    hidden_states = outputs.hidden_states[-1]

    # Sample first token from target.
    if logits_processor is not None:
        logits = logits_processor(None, orig[:, -1])
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])[None, None]

    # ---- Build ea_input_ids / ea_inputs_embeds (verbatim from msd.utils) ----
    input_seq_len = input_ids.shape[1]
    image_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(image_pos) > 0 and inputs_embeds is not None:
        image_start = image_pos[0].item()
        image_end = image_start
        while (
            image_end + 1 < input_seq_len
            and input_ids[0, image_end + 1].item() == image_token_id
        ):
            image_end += 1
        num_image_tokens = image_end - image_start + 1

        before = input_ids[0, :image_start]
        after = input_ids[0, image_end + 1 :]
        marker = torch.tensor([-200], dtype=input_ids.dtype, device=input_ids.device)
        ea_input_ids = torch.cat([before, marker, after]).unsqueeze(0)

        ea_input_ids_with_token = torch.cat([ea_input_ids, token], dim=1)

        text_before = model.ea_layer.embed_tokens(before.unsqueeze(0))
        text_after = model.ea_layer.embed_tokens(after.unsqueeze(0))
        token_embed = model.ea_layer.embed_tokens(token)
        image_embeds = inputs_embeds[
            0, image_start : image_start + num_image_tokens, :
        ].unsqueeze(0)

        ea_inputs_embeds = torch.cat(
            [text_before, image_embeds, text_after, token_embed], dim=1
        )
    else:
        ea_input_ids = input_ids.clone()
        ea_input_ids_with_token = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        ea_inputs_embeds = None

    # ---- SAGE compression (the only deviation from upstream MSD) ----
    if visual_processor is not None and ea_inputs_embeds is not None and len(image_pos) > 0:
        hidden_states, ea_inputs_embeds = _compress_for_draft(
            model=model,
            visual_processor=visual_processor,
            hidden_states=hidden_states,
            ea_inputs_embeds=ea_inputs_embeds,
            full_input_ids=input_ids,
            image_token_id=image_token_id,
            sample_id=sample_id,
            debug=debug,
        )

    # ---- Draft topK_genrate ----
    lm_head = model._get_lm_head()
    temp_cache.use_msd = True
    try:
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
            model.ea_layer.topK_genrate(
                hidden_states,
                ea_input_ids_with_token,
                lm_head,
                logits_processor,
                ea_inputs_embeds,
            )
        )
    finally:
        temp_cache.use_msd = False

    return (
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        orig,
        hidden_states,
        token,
        outputs.past_key_values,
        ea_input_ids,
        ea_inputs_embeds,  # NOTE: this is the COMPRESSED variant if SAGE ran
    )


@torch.no_grad()
def sage_update_inference_inputs_hf(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    logits_processor,
    new_token,
    past_key_values,  # DynamicCache
    model,
    hidden_state_new,
    sample_p,
    ea_input_ids,
    ea_inputs_embeds,  # may be COMPRESSED already
):
    """Drop-in replacement for `method.msd.utils.update_inference_inputs_hf`.

    Same KV-cache update logic; only difference is that we re-compress
    `ea_inputs_embeds` before the next topK_genrate call so the draft sees a
    consistent layout. NOTE: `ea_inputs_embeds` arriving here may already be
    compressed (from sage_initialize_tree_hf); subsequent appends grow the
    suffix, but the prefix length is still `_sage_original_prefix_len` so the
    `_compress_subsequent` helper is a no-op when the prefix is already short
    (it checks `>= P`).
    """
    prev_input_len = input_ids.shape[1]

    if (
        len(past_key_values.key_cache) > 0
        and past_key_values.key_cache[0] is not None
    ):
        kv_total_len = past_key_values.key_cache[0].shape[2]
        tree_len = hidden_state_new.shape[1]
        kv_len_before_tree = kv_total_len - tree_len
    else:
        kv_len_before_tree = prev_input_len

    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + kv_len_before_tree
    )

    input_ids = torch.cat(
        [
            input_ids,
            candidates[None, best_candidate, : accept_length + 1].to(input_ids.device),
        ],
        dim=-1,
    )
    ea_input_ids = torch.cat(
        [
            ea_input_ids,
            candidates[None, best_candidate, : accept_length + 1].to(ea_input_ids.device),
        ],
        dim=-1,
    )

    # Truncate the target's KV cache to keep only original + accepted tree tokens.
    for layer_idx in range(len(past_key_values.key_cache)):
        if past_key_values.key_cache[layer_idx] is not None:
            key = past_key_values.key_cache[layer_idx]
            value = past_key_values.value_cache[layer_idx]
            orig_keys = key[:, :, :kv_len_before_tree, :]
            orig_values = value[:, :, :kv_len_before_tree, :]
            tree_keys = key[:, :, select_indices.to(key.device), :]
            tree_values = value[:, :, select_indices.to(value.device), :]
            past_key_values.key_cache[layer_idx] = torch.cat(
                [orig_keys, tree_keys], dim=2
            )
            past_key_values.value_cache[layer_idx] = torch.cat(
                [orig_values, tree_values], dim=2
            )

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[
        :, best_candidate, : accept_length + 1
    ]

    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]

    # Append accepted-token embeddings to ea_inputs_embeds (matches upstream).
    if ea_inputs_embeds is not None:
        accepted_tokens = candidates[best_candidate, : accept_length + 1].to(input_ids.device)
        accepted_embeds = model.ea_layer.embed_tokens(accepted_tokens.unsqueeze(0))
        ea_inputs_embeds = torch.cat([ea_inputs_embeds, accepted_embeds], dim=1)

    lm_head = model._get_lm_head()
    temp_cache.use_msd = True
    try:
        # SAGE: enforce compression before each draft call (no-op if not active).
        ea_inputs_embeds_for_draft = _compress_subsequent(
            model=model, ea_inputs_embeds=ea_inputs_embeds
        )
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = (
            model.ea_layer.topK_genrate(
                accept_hidden_state_new,
                input_ids=torch.cat(
                    (ea_input_ids, token.to(ea_input_ids.device)), dim=1
                ),
                head=lm_head,
                logits_processor=logits_processor,
                inputs_embeds=ea_inputs_embeds_for_draft,
            )
        )
    finally:
        temp_cache.use_msd = False

    new_token += accept_length + 1
    # NOTE: we return the "uncompressed-prefix" ea_inputs_embeds so subsequent
    # rounds keep growing the suffix from the same starting point. Compression
    # is re-applied each round inside this helper. If we returned the compressed
    # version, `_compress_subsequent` would still be a no-op (same prefix
    # length), but we'd lose the ability to change the keep_mask later.
    return (
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
        None,
        token,
        past_key_values,
        ea_input_ids,
        ea_inputs_embeds,
    )

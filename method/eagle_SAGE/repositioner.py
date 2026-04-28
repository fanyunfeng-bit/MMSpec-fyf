"""Repositioner: within the contiguous visual sub-sequence, move sink tokens
to the front while preserving relative order of sinks and of non-sinks.

The permutation is applied jointly to:
  - hidden_states  [B, L, H]
  - input_ids      [B, L]
  - position_ids   [B, L] (1D RoPE) or [3, B, L] (Qwen2.5-VL M-RoPE)

The invariant of interest for the draft's self-attention is that each token's
RoPE/positional encoding *travels with the token*. Since RoPE is applied per
token using `position_ids[i]`, a pure index permutation preserves each token's
rotation; only the causal-mask reachability changes (which is the intended
effect of repositioning).
"""
from __future__ import annotations

import torch


class SinkFirstRepositioner:
    stage_name = "repositioner"

    @torch.no_grad()
    def process(self, ctx):
        if ctx.sink_mask is None or not bool(ctx.sink_mask.any()):
            return ctx

        device = ctx.visual_mask.device
        L = ctx.visual_mask.numel()

        all_idx = torch.arange(L, device=device)
        visual_idx = all_idx[ctx.visual_mask]            # positions of visual tokens
        sink_within = ctx.sink_mask[ctx.visual_mask]     # [V] bool
        sink_pos = visual_idx[sink_within]               # ascending
        nonsink_pos = visual_idx[~sink_within]           # ascending

        # Assume the visual block is contiguous (MMSpec prompts satisfy this).
        # For robustness, assert contiguity up front; raise early if violated.
        if visual_idx.numel() > 0:
            expected = torch.arange(
                int(visual_idx[0].item()),
                int(visual_idx[-1].item()) + 1,
                device=device,
            )
            if visual_idx.numel() != expected.numel() or not torch.equal(visual_idx, expected):
                raise RuntimeError(
                    "SinkFirstRepositioner: visual tokens are not contiguous. "
                    "Current implementation assumes a single contiguous visual "
                    "block; generalize to per-block repositioning if needed."
                )

        perm = all_idx.clone()
        v_start = int(visual_idx[0].item())
        v_end = int(visual_idx[-1].item()) + 1
        k = int(sink_pos.numel())
        perm[v_start : v_start + k] = sink_pos
        perm[v_start + k : v_end] = nonsink_pos

        ctx.reorder_indices = perm
        ctx.hidden_states = ctx.hidden_states.index_select(1, perm)
        ctx.input_ids = ctx.input_ids.index_select(1, perm)
        pids = ctx.position_ids
        if pids is not None:
            if pids.dim() == 2:     # [B, L]
                ctx.position_ids = pids.index_select(1, perm)
            elif pids.dim() == 3:   # [3, B, L] for Qwen2.5-VL M-RoPE
                ctx.position_ids = pids.index_select(2, perm)
            else:
                raise ValueError(
                    f"SinkFirstRepositioner: unsupported position_ids dim {pids.dim()}"
                )
        # Note: visual_mask is unchanged (positions are still visual; only their
        # contents were reshuffled within the block). sink_mask is also unchanged
        # in the full-L frame of reference — but downstream stages that care about
        # "which post-reorder position is a sink" should derive from the now-
        # reordered layout:  sink_mask_new[i] = sink_mask[perm[i]].
        ctx.meta["v_start"] = v_start
        ctx.meta["v_end"] = v_end
        ctx.meta["num_sinks_repositioned"] = k
        return ctx

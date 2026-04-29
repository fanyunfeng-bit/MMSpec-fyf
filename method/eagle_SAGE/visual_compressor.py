"""VisualCompressor — drop non-selected visual tokens before draft forward.

Pipeline contract (runs AFTER L2NormSinkDetector):
  Inputs:  ctx.sink_mask (set by L2NormSinkDetector)
           ctx.visual_mask, ctx.input_ids, ctx.hidden_states, ctx.position_ids
  Action:  Compute TI-Prob top-k visual tokens at a target LLM layer; UNION with
           sinks (if TI-Prob picks overlap with sinks, push down ranks).
           Drop all other visual tokens; keep text unchanged. Visual tokens that
           survive retain their ORIGINAL position_ids (1D arange over the prefill).
  Outputs: compressed ctx.input_ids/hidden_states/position_ids/visual_mask/sink_mask
           ctx.meta["sage_keep_mask_full"] — bool [original_prefix_len], True for
             every position kept. Picked up by `sage_initialize_tree` and stored
             on `model.spec_layer` so subsequent topK_genrate calls can compress
             too (the draft KV cache is short → input prefix must match length).

Hook lifecycle:
  Owns its own forward hook on a target LLM decoder layer (same layer the user
  used in TextImportanceProbe). Only the most recent capture is read; the hook
  is overwritten by every later forward but `process()` only runs once per
  specgenerate, immediately after target prefill.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


class VisualCompressor:
    stage_name = "visual_compressor"

    def __init__(
        self,
        base_model,
        layer_idx: int,
        topk: int,
        image_token_id: int,
        video_token_id: Optional[int] = None,
    ):
        self.layer_idx = int(layer_idx)
        self.topk = int(topk)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id) if video_token_id is not None else None

        if hasattr(base_model, "language_model"):
            lm = base_model.language_model
        else:
            lm = base_model
        layers = lm.model.layers
        if not 0 <= self.layer_idx < len(layers):
            raise IndexError(
                f"VisualCompressor: layer_idx={self.layer_idx} out of [0,{len(layers)})"
            )
        self._layer = layers[self.layer_idx]
        self._latest_hidden: Optional[torch.Tensor] = None
        self._handle = self._layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        if isinstance(h, torch.Tensor):
            self._latest_hidden = h.detach()

    def remove(self):
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None

    def __del__(self):
        try:
            self.remove()
        except Exception:
            pass

    @torch.no_grad()
    def _compute_ti_alpha(
        self,
        input_ids_full: torch.Tensor,
        visual_mask_full: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute TI-Prob alpha over visual tokens at the hooked layer.

        alpha[i] = mean_t softmax_i( H_t @ H_v.T / sqrt(D) )
        Text positions are everything AFTER the LAST visual token (matches
        TextImportanceProbe's H_t_initial definition at prefill).
        Returns alpha [V] or None if hidden state unavailable.
        """
        if self._latest_hidden is None:
            return None
        h = self._latest_hidden[0]  # [L, D]
        L = min(int(h.shape[0]), int(visual_mask_full.shape[0]))
        h = h[:L]
        vmask = visual_mask_full[:L].to(h.device)

        if not bool(vmask.any()):
            return None
        H_v = h[vmask].float()  # [V, D]
        # Text = positions after last visual.
        last_visual = int(torch.nonzero(vmask, as_tuple=False)[-1].item())
        if last_visual + 1 >= L:
            return None  # no text after visual block — cannot compute TI
        H_t = h[last_visual + 1 :].float()  # [N_t, D]

        D = H_v.shape[-1]
        scores = (H_t @ H_v.T) / math.sqrt(D)         # [N_t, V]
        A = torch.softmax(scores, dim=-1)             # softmax over V
        alpha = A.mean(dim=0)                         # [V]
        return alpha

    @torch.no_grad()
    def process(self, ctx):
        vmask = ctx.visual_mask
        if vmask is None or not bool(vmask.any()):
            return ctx
        if ctx.sink_mask is None:
            sink_mask_full = torch.zeros_like(vmask)
        else:
            sink_mask_full = ctx.sink_mask

        sink_within_v = sink_mask_full[vmask]                          # [V]
        alpha = self._compute_ti_alpha(ctx.input_ids[0], vmask)        # [V] or None

        # Align everything onto vmask's device — with device_map="auto" the
        # hooked layer's hidden states may live on a different GPU than the
        # masks in ctx.
        target_device = vmask.device
        if alpha is None:
            ti_within_v = torch.zeros_like(sink_within_v)
        else:
            alpha = alpha.to(target_device)
            available = ~sink_within_v
            available_alpha = alpha.clone()
            available_alpha[~available] = float("-inf")
            k = min(int(self.topk), int(available.sum().item()))
            ti_within_v = torch.zeros_like(sink_within_v)
            if k > 0:
                top = torch.topk(available_alpha, k=k)
                ti_within_v[top.indices] = True

        keep_within_v = sink_within_v | ti_within_v                    # [V]
        # Build full-prefix keep mask: keep all non-visual + selected visual.
        keep_full = (~vmask).clone()
        vis_idx_full = torch.nonzero(vmask, as_tuple=False).squeeze(-1)  # [V]
        keep_full[vis_idx_full[keep_within_v]] = True
        keep_idx = torch.nonzero(keep_full, as_tuple=False).squeeze(-1)  # sorted ascending

        # Compress payload tensors. index_select needs the index on the same
        # device as the tensor.
        hs_dev = ctx.hidden_states.device
        ctx.hidden_states = ctx.hidden_states.index_select(1, keep_idx.to(hs_dev))
        ids_dev = ctx.input_ids.device
        ctx.input_ids = ctx.input_ids.index_select(1, keep_idx.to(ids_dev))
        if ctx.position_ids is not None:
            pids_dev = ctx.position_ids.device
            kept_for_pids = keep_idx.to(pids_dev)
            if ctx.position_ids.dim() == 2:
                ctx.position_ids = ctx.position_ids.index_select(1, kept_for_pids)
            elif ctx.position_ids.dim() == 3:
                ctx.position_ids = ctx.position_ids.index_select(2, kept_for_pids)
            else:
                raise ValueError(
                    f"VisualCompressor: unsupported position_ids dim "
                    f"{ctx.position_ids.dim()}"
                )

        # Update masks to compressed layout (stay on original device).
        ctx.visual_mask = vmask[keep_idx]
        ctx.sink_mask = sink_mask_full[keep_idx]

        # Stash for the draft model: full-prefix-length bool mask saying which
        # original positions are kept. Cnets uses this to reconstruct compressed
        # input_ids in subsequent topK_genrate calls.
        ctx.meta["sage_keep_mask_full"] = keep_full
        ctx.meta["compressed_len"] = int(keep_idx.numel())
        ctx.meta["num_visual_after"] = int(ctx.visual_mask.sum().item())
        ctx.meta["num_sinks_kept"] = int(sink_within_v.sum().item())
        ctx.meta["num_ti_kept"] = int(ti_within_v.sum().item())
        return ctx

"""Visual sink detection based on L2 norm of pre-projector ViT features.

Sets `ctx.sink_mask` (a [L] bool subset of `ctx.visual_mask`) from
`ctx.pre_projector_features`. Supports two threshold modes:
  - "abs":        norms >= threshold_value
  - "topk_ratio": top ceil(threshold_value * V) tokens by norm
"""
from __future__ import annotations

import torch


class L2NormSinkDetector:
    """Pick visual sink tokens by L2 norm of the vision encoder's pre-projector output.

    For Qwen2.5-VL, pre-projector tokens are more numerous than post-projector
    tokens (spatial merger groups s×s patches into one). Per-post-merger-token
    sink score is the MAX norm over its pre-merger group, which emphasizes the
    "outlier" sink within the group.
    """

    stage_name = "sink_detector"

    def __init__(
        self,
        mode: str = "topk_ratio",
        value: float = 0.1,
        min_sinks: int = 0,
        spatial_merge_size: int = 1,
    ):
        if mode not in ("abs", "topk_ratio"):
            raise ValueError(f"Unknown threshold mode: {mode!r}")
        self.mode = mode
        self.value = float(value)
        self.min_sinks = int(min_sinks)
        self.spatial_merge_size = int(spatial_merge_size)

    @torch.no_grad()
    def process(self, ctx):
        vmask = ctx.visual_mask
        # No visual tokens → no sinks.
        if vmask is None or not bool(vmask.any()):
            ctx.sink_mask = (
                torch.zeros_like(vmask) if vmask is not None else None
            )
            ctx.meta["num_visual"] = 0
            ctx.meta["num_sinks"] = 0
            return ctx

        feats = ctx.pre_projector_features
        if feats is None:
            ctx.sink_mask = torch.zeros_like(vmask)
            ctx.meta["num_visual"] = int(vmask.sum().item())
            ctx.meta["num_sinks"] = 0
            return ctx

        # [B, V_pre, D] → [V_pre, D]
        if feats.dim() == 3:
            feats = feats[0]
        # L2 norm per pre-merger token, computed in float32 for stability.
        norms = feats.float().pow(2).sum(dim=-1).sqrt()  # [V_pre]

        # Aggregate pre-merger norms to per-post-merger-token score.
        if self.spatial_merge_size > 1:
            group = self.spatial_merge_size ** 2
            if norms.numel() % group != 0:
                raise RuntimeError(
                    f"L2NormSinkDetector: pre-projector features ({norms.numel()}) "
                    f"not divisible by group size {group} (spatial_merge_size="
                    f"{self.spatial_merge_size}). Check hook target."
                )
            norms = norms.view(-1, group).max(dim=-1).values  # [V_post]
        # print(f"[SAGE norms] V={norms.numel()} min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f} values={norms.tolist()}")
        count = (norms > 100).sum().item()
        print(f"[SAGE norms] 大于100的数量: {count}")
        # Positions in the sequence that are visual tokens.
        vis_idx = torch.nonzero(vmask, as_tuple=False).squeeze(-1)  # [V_post]
        if vis_idx.numel() != norms.numel():
            raise RuntimeError(
                f"L2NormSinkDetector: visual-positions count ({vis_idx.numel()}) "
                f"!= post-merger feature count ({norms.numel()}). "
                f"This usually means the vision encoder hook captured the wrong "
                f"module or CLS-handling is inconsistent."
            )

        # Threshold selection.
        if self.mode == "abs":
            sink_local = norms >= self.value                           # [V]
        else:  # topk_ratio
            k = int(round(self.value * vis_idx.numel()))
            k = max(self.min_sinks, k)
            k = max(0, min(k, vis_idx.numel()))
            if k == 0:
                sink_local = torch.zeros_like(norms, dtype=torch.bool)
            else:
                top_idx = torch.topk(norms, k=k).indices
                sink_local = torch.zeros_like(norms, dtype=torch.bool)
                sink_local[top_idx] = True

        # Scatter back to full-sequence mask.
        sink_full = torch.zeros_like(vmask)
        sink_full[vis_idx[sink_local]] = True
        ctx.sink_mask = sink_full
        ctx.meta["num_visual"] = int(vmask.sum().item())
        ctx.meta["num_sinks"] = int(sink_full.sum().item())
        return ctx

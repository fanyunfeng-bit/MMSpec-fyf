"""Pluggable pipeline that runs between target prefill output and draft forward.

Pipeline contract (order of stages):
  1. SinkDetector       — sets ctx.sink_mask from ctx.pre_projector_features.
  2. VisualRepositioner — permutes ctx.hidden_states / input_ids / position_ids
                          so sinks lead within the visual sub-sequence.
  3. [FUTURE] VisualCompressor — drops non-sink visual tokens (mutates lengths).
                                 Must run AFTER repositioning; nothing after it
                                 may assume original lengths.
  4. [FUTURE] DraftGate — queried via should_draft(); does NOT mutate ctx.

Each stage implements `.process(ctx) -> ctx`. Adding a future stage is a single
`register_stage(stage)` call plus wiring in `from_args`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class VisualContext:
    """Shared state between pipeline stages."""

    hidden_states: torch.Tensor
    """[B, L, H] from target, last-layer (EAGLE1) or concat (EAGLE3)."""

    input_ids: torch.Tensor
    """[B, L], same L as hidden_states."""

    position_ids: torch.Tensor | None
    """[B, L] (1D RoPE) or [3, B, L] (Qwen2.5-VL M-RoPE). May be None."""

    visual_mask: torch.Tensor
    """[L] bool — True where token is a visual (image/video) token."""

    pre_projector_features: torch.Tensor
    """Vision encoder output BEFORE projector.
    Shape: [V_pre, D_vit] or [B=1, V_pre, D_vit]. For Qwen2.5-VL, V_pre is the
    pre-merger token count (= V_post * spatial_merge_size**2)."""

    sink_mask: torch.Tensor | None = None
    """[L] bool, subset of visual_mask. Set by SinkDetector."""

    reorder_indices: torch.Tensor | None = None
    """[L] long permutation applied by VisualRepositioner."""

    arch: str = ""
    """base_model.config.architectures[0]."""

    meta: dict = field(default_factory=dict)
    """Free-form counters and debug info."""


class VisualProcessor:
    """Ordered pipeline of plug-in stages."""

    def __init__(self):
        self.stages: list = []
        self.gate = None  # Future: DraftGate

    def register_stage(self, stage):
        """Append a stage. DraftGate-like stages are stored separately."""
        if hasattr(stage, "should_draft"):
            self.gate = stage
        else:
            self.stages.append(stage)

    @torch.no_grad()
    def run(self, ctx: VisualContext) -> VisualContext:
        for stage in self.stages:
            ctx = stage.process(ctx)
        return ctx

    def should_draft(self, ctx: VisualContext, **kwargs: Any) -> bool:
        if self.gate is None:
            return True
        return bool(self.gate.should_draft(ctx, **kwargs))

    @classmethod
    def from_args(cls, args, spatial_merge_size: int = 1) -> "VisualProcessor":
        """Build a processor from argparse flags.

        Activates SinkDetector + SinkFirstRepositioner by default. Future
        plug-ins can be wired here guarded by their own `--sage-enable-*` flags.
        """
        from .repositioner import SinkFirstRepositioner
        from .sink_detector import L2NormSinkDetector

        vp = cls()
        if getattr(args, "sage_enable_sink_detection", True):
            vp.register_stage(
                L2NormSinkDetector(
                    mode=args.sage_threshold_mode,
                    value=args.sage_threshold_value,
                    min_sinks=getattr(args, "sage_min_sinks", 0),
                    spatial_merge_size=spatial_merge_size,
                )
            )
        if getattr(args, "sage_enable_repositioning", True):
            vp.register_stage(SinkFirstRepositioner())
        return vp

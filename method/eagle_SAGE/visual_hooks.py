"""Vision-tower forward hook to capture pre-projector visual features.

Per "To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language
Models", visual sink tokens are most cleanly identified at the vision encoder's
output *before* the projector / merger, i.e. in ViT hidden-dim space.

This module registers a forward hook on the architecture-specific module:
  - Qwen2.5-VL: hooks `base_model.visual.blocks[-1]` (last ViT block; upstream of
    the `merger` that projects & spatially-merges into the LLM's hidden dim).
  - LLaVA-1.5/NeXT: hooks `base_model.vision_tower` (upstream of the
    `multi_modal_projector`). CLS is dropped when the model uses the default
    vision_feature_select_strategy.

Each call into the vision encoder populates `self.last_features`. The consumer
invokes `pop()` to retrieve and clear.
"""
from __future__ import annotations

import torch


class VisionTowerHook:
    """Forward hook that captures the vision encoder's output before the projector."""

    def __init__(self, base_model):
        self.arch = base_model.config.architectures[0]
        self.last_features: torch.Tensor | None = None
        self.handle = None

        if self.arch == "Qwen2_5_VLForConditionalGeneration":
            self.hook_target = base_model.visual.blocks[-1]
            self.spatial_merge_size = self._resolve_qwen_spatial_merge_size(base_model)
            self.drop_cls = False
        elif self.arch in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        ):
            self.hook_target = base_model.vision_tower
            self.spatial_merge_size = 1
            strategy = getattr(
                base_model.config, "vision_feature_select_strategy", "default"
            )
            self.drop_cls = (strategy == "default")
        else:
            raise NotImplementedError(
                f"SAGE VisionTowerHook: unsupported architecture '{self.arch}'"
            )

        self.handle = self.hook_target.register_forward_hook(self._hook)

    @staticmethod
    def _resolve_qwen_spatial_merge_size(base_model) -> int:
        """Find spatial_merge_size across known attribute locations."""
        visual = getattr(base_model, "visual", None)
        if visual is not None and hasattr(visual, "spatial_merge_size"):
            return int(visual.spatial_merge_size)
        cfg = base_model.config
        if hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "spatial_merge_size"):
            return int(cfg.vision_config.spatial_merge_size)
        if hasattr(cfg, "spatial_merge_size"):
            return int(cfg.spatial_merge_size)
        # Sensible fallback for Qwen2.5-VL
        return 2

    def _hook(self, module, inputs, output):
        # LLaVA vision_tower returns BaseModelOutputWithPooling; use last_hidden_state.
        # Qwen2.5-VL block output is a raw tensor.
        if hasattr(output, "last_hidden_state"):
            feats = output.last_hidden_state
        else:
            feats = output
        if not isinstance(feats, torch.Tensor):
            # Unrecognized shape; store nothing.
            self.last_features = None
            return
        if self.drop_cls and feats.dim() == 3 and feats.shape[1] > 1:
            feats = feats[:, 1:, :]
        self.last_features = feats
        # self.last_features = feats.detach().clone()
        # print(f"[VHook] shape={tuple(feats.shape)} norm_mean={feats.float().pow(2).sum(-1).sqrt().mean().item():.3f}")

    def pop(self) -> torch.Tensor | None:
        out = self.last_features
        self.last_features = None
        return out

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __del__(self):
        # Best-effort cleanup; ignore exceptions at GC time.
        try:
            self.remove()
        except Exception:
            pass

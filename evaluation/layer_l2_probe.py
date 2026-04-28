"""Diagnostic probe: per-visual-token L2 distance between layer input and output.

Forward-hook-based probe that attaches to specified LlamaDecoderLayers (or any
decoder layer whose forward returns a tuple whose first element is the updated
`hidden_states [B, L, H]`). For each hooked layer, on each forward call, records
the per-token L2 distance ||layer_output - layer_input||_2.

Usage pattern (one `reset()` per `model.generate()` invocation):

    probe = LayerL2Probe(base_model, layer_indices=[3, 17, 22])
    probe.reset()
    model.generate(**inputs, ...)
    probe.print_topk(inputs["input_ids"], image_token_id=32000, topk=10)
    # ... repeat reset/generate/print_topk per sample ...
    probe.remove()

Only the FIRST forward with `seq_len > 1` is treated as the "prefill" capture.
Subsequent decode-step forwards with `seq_len == 1` are ignored — for LLaVA
with HF generate(), visual tokens live in the KV cache after prefill and are
not re-forwarded at decode time.
"""
from __future__ import annotations

from typing import Iterable

import torch


class LayerL2Probe:
    """Attach forward hooks on chosen decoder layers; capture per-token L2
    distance between each layer's input hidden_states and output hidden_states.
    """

    def __init__(self, base_model, layer_indices: Iterable[int]):
        self.layer_indices = list(layer_indices)
        self._captures: dict[int, list[torch.Tensor]] = {}
        self._handles: list = []

        # Resolve language model: HF LLaVA wraps it under `.language_model`.
        lm = (
            base_model.language_model
            if hasattr(base_model, "language_model")
            else base_model
        )

        # Validate and attach.
        num_layers = len(lm.model.layers)
        for idx in self.layer_indices:
            if idx < 0 or idx >= num_layers:
                raise IndexError(
                    f"LayerL2Probe: layer_idx={idx} out of range [0, {num_layers})"
                )
            layer = lm.model.layers[idx]
            self._handles.append(
                layer.register_forward_hook(self._make_hook(idx))
            )

    def _make_hook(self, idx: int):
        def hook(module, inputs, outputs):
            # `inputs` is the positional-args tuple; first arg is hidden_states.
            hin = inputs[0]
            # `outputs` from LlamaDecoderLayer is a tuple whose [0] is the
            # updated hidden_states. Other forms (bare Tensor, ModelOutput)
            # are covered for robustness.
            if isinstance(outputs, tuple):
                hout = outputs[0]
            elif isinstance(outputs, torch.Tensor):
                hout = outputs
            else:
                # Likely a ModelOutput-like object; fall back to attribute.
                hout = getattr(outputs, "last_hidden_state", None)
                if hout is None:
                    return
            # Per-token L2 in float32 for numerical stability.
            l2 = (hout.float() - hin.float()).pow(2).sum(dim=-1).sqrt()  # [B, L]
            self._captures.setdefault(idx, []).append(l2.detach().cpu())

        return hook

    def reset(self) -> None:
        """Clear all captured tensors. Call once per model.generate() invocation."""
        self._captures = {}

    def get_prefill(self) -> dict[int, torch.Tensor]:
        """Return the first forward's L2 [L] tensor per layer (batch=1).

        "Prefill" = first hook firing with `seq_len > 1` since the last reset().
        Returns a dict {layer_idx: [L] float tensor}.
        """
        out: dict[int, torch.Tensor] = {}
        for idx in self.layer_indices:
            for l2 in self._captures.get(idx, []):
                if l2.dim() >= 2 and l2.shape[1] > 1:
                    out[idx] = l2[0]  # assume batch=1
                    break
        return out

    def print_topk(
        self,
        input_ids: torch.Tensor,
        image_token_id: int,
        topk: int = 10,
        sample_id: str | None = None,
    ) -> None:
        """Print top-K visual tokens (by L2 distance) per hooked layer.

        Each row shows:
          rank | visual_idx (0..V-1 within visual block) | seq_pos (abs) | l2
        """
        dists = self.get_prefill()
        if not dists:
            print(f"[Probe] sample={sample_id} no prefill capture (no seq_len>1 forward)")
            return

        ids = input_ids[0].detach().cpu()
        visual_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]  # [V]
        if visual_positions.numel() == 0:
            print(
                f"[Probe] sample={sample_id} no visual tokens "
                f"(image_token_id={image_token_id}) in input_ids"
            )
            return

        V = int(visual_positions.numel())
        for layer_idx in sorted(dists.keys()):
            l2_vec = dists[layer_idx]                      # [L]
            l2_visual = l2_vec[visual_positions]           # [V]
            k = min(topk, V)
            top = torch.topk(l2_visual, k=k)
            print(f"[Probe] sample={sample_id} layer={layer_idx} (V={V}):")
            for rank in range(k):
                vis_idx = int(top.indices[rank])
                seq_pos = int(visual_positions[vis_idx])
                l2_val = float(top.values[rank])
                print(
                    f"  rank={rank:2d}  visual_idx={vis_idx:4d}  "
                    f"seq_pos={seq_pos:5d}  l2={l2_val:.4f}"
                )

    def remove(self) -> None:
        """Unregister all hooks. Safe to call multiple times."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def __del__(self):
        # Best-effort cleanup; ignore exceptions at GC time.
        try:
            self.remove()
        except Exception:
            pass

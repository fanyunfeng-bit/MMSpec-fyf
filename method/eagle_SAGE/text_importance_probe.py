"""TextImportanceProbe — TGVC-inspired diagnostic for eagle_SAGE.

For each visual token, compute its "text importance" = average attention
received from text tokens, using cross-modal attention at a chosen LLM layer.

Formula (from VisionTrim TGVC; text-query → visual-key direction):

    A_{t,i} = softmax_i( H^t_t · H^v_i / sqrt(D) )    ∈ R^{N_t × N_v}
    α_i    = (1 / N_t) Σ_t A_{t,i}                    ∈ R^{N_v}     (Σ_i α_i = 1)
    top_K  = argtopk_i α_i

NOTE: softmax is over the VISUAL key dimension (axis=-1 of A).  Each text token
distributes its attention across visual tokens; α_i is the average "share" each
visual token receives from text. If you accidentally softmax over the text dim,
every row sums to 1 ⇒ α_i = 1/N_t is constant ⇒ useless.

Where:
  - H^v ∈ R^{N_v × D}  : layer-ℓ hidden states of visual tokens (frozen at prefill;
                         visual tokens never re-flow through LLM after prefill)
  - H^t ∈ R^{N_t × D}  : layer-ℓ hidden states of TEXT tokens, defined as
                         (a) all positions AFTER the visual block at prefill, plus
                         (b) all accepted-generated tokens accumulated across verifies

The probe fires:
  1. After prefill (`on_prefill`) — populates H_v and H_t_initial; prints "verify=0".
  2. After EACH verify (`on_verify_accepted`) — slices accepted positions out of the
     just-completed tree forward's layer-ℓ output, appends to H_t_generated,
     recomputes α and prints with verify=N (1-indexed).
"""
from __future__ import annotations

import math
from typing import Optional

import torch


class TextImportanceProbe:
    """Hook one LLM decoder layer; track H_v + accumulating H_t across verifies."""

    def __init__(self, base_model, layer_idx: int):
        self.layer_idx = int(layer_idx)
        self._handle = None

        # State (per specgenerate; reset() clears)
        self._latest_hidden: Optional[torch.Tensor] = None  # [B, L, D] from last forward
        self.H_v: Optional[torch.Tensor] = None             # [N_v, D]   frozen at prefill
        self.H_t_initial: Optional[torch.Tensor] = None     # [N_t0, D]  text after visual at prefill
        self.H_t_generated: list = []                        # list of [n_i, D]; appended per verify
        self.D: Optional[int] = None

        # Resolve language model (LLaVA wraps it; Qwen2.5-VL nests as model)
        if hasattr(base_model, "language_model"):
            lm = base_model.language_model
        else:
            lm = base_model

        layers = lm.model.layers
        if not 0 <= self.layer_idx < len(layers):
            raise IndexError(
                f"TextImportanceProbe: layer_idx={self.layer_idx} out of [0,{len(layers)})"
            )
        self.layer = layers[self.layer_idx]
        self._handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        """Capture the layer's output hidden_states from EVERY forward (prefill + verify trees)."""
        if isinstance(outputs, tuple):
            h = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            h = outputs
        else:
            return
        self._latest_hidden = h.detach()

    def reset(self) -> None:
        """Call once at the start of each specgenerate()."""
        self._latest_hidden = None
        self.H_v = None
        self.H_t_initial = None
        self.H_t_generated = []
        self.D = None

    @torch.no_grad()
    def on_prefill(
        self,
        input_ids: torch.Tensor,
        image_token_id: int,
        sample_id: Optional[str] = None,
        topk: int = 10,
    ) -> None:
        """Called after the prefill forward. Splits the latest capture into
        H_v (visual positions) and H_t_initial (text positions AFTER last visual).
        """
        if self._latest_hidden is None:
            return
        h = self._latest_hidden[0]  # [L, D]
        ids = input_ids[0].to(h.device)
        L = min(int(h.shape[0]), int(ids.shape[0]))
        h = h[:L]
        ids = ids[:L]

        visual_mask = (ids == image_token_id)
        if not bool(visual_mask.any()):
            print(f"[TI-Probe] sample={sample_id} no visual tokens (image_token_id={image_token_id})")
            return

        visual_positions = visual_mask.nonzero(as_tuple=True)[0]
        last_visual = int(visual_positions[-1])

        self.H_v = h[visual_mask].clone()                   # [N_v, D]
        if last_visual + 1 < L:
            self.H_t_initial = h[last_visual + 1 :].clone() # [N_t0, D]
        else:
            self.H_t_initial = h.new_empty((0, h.shape[-1]))
        self.D = int(h.shape[-1])

        self._compute_and_print(topk=topk, sample_id=sample_id, verify_step=0)

    @torch.no_grad()
    def on_verify_accepted(
        self,
        select_indices: torch.Tensor,
        sample_id: Optional[str] = None,
        topk: int = 10,
        verify_step: Optional[int] = None,
    ) -> None:
        """Called after each verify (after update_inference_inputs).
        select_indices: [accept_length+1] long tensor — positions WITHIN the tree
        that were accepted (i.e., retrieve_indices[best_candidate, :accept_length+1]).
        """
        if self._latest_hidden is None or self.H_v is None:
            return
        # _latest_hidden is the tree forward's output [1, tree_size, D]
        h = self._latest_hidden[0]  # [tree_size, D]
        idx = select_indices.to(h.device).long()
        # Defensive bound check
        idx = idx[(idx >= 0) & (idx < h.shape[0])]
        if idx.numel() == 0:
            return
        accepted_h = h[idx].clone()  # [accept_length+1, D]
        self.H_t_generated.append(accepted_h)

        self._compute_and_print(topk=topk, sample_id=sample_id, verify_step=verify_step)

    @torch.no_grad()
    def _compute_and_print(self, topk: int, sample_id, verify_step) -> None:
        if self.H_v is None or self.D is None:
            return

        parts = []
        if self.H_t_initial is not None and self.H_t_initial.numel() > 0:
            parts.append(self.H_t_initial)
        parts.extend(self.H_t_generated)
        if not parts:
            print(
                f"[TI-Probe] sample={sample_id} verify={verify_step} "
                f"layer={self.layer_idx} (no text yet)"
            )
            return

        H_v = self.H_v.float()
        H_t = torch.cat(parts, dim=0).to(H_v.device).float()
        # text-query → visual-key attention: scores [N_t, N_v]
        scores = (H_t @ H_v.T) / math.sqrt(self.D)
        A = torch.softmax(scores, dim=-1)                   # softmax over V dim
        alpha = A.mean(dim=0)                               # [N_v], avg over text queries

        k = min(int(topk), int(alpha.numel()))
        top = torch.topk(alpha, k=k)
        print(
            f"[TI-Probe] sample={sample_id} verify={verify_step} "
            f"layer={self.layer_idx} N_v={int(H_v.shape[0])} N_t={int(H_t.shape[0])} top{k}:"
        )
        for r in range(k):
            print(
                f"  rank={r:2d}  visual_idx={int(top.indices[r]):4d}  "
                f"alpha={float(top.values[r]):.6f}"
            )

    def remove(self) -> None:
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

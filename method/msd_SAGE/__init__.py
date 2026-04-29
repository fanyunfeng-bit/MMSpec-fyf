"""msd_SAGE: MSD speculative decoding with visual-token compression for the
draft head.

Builds on top of `method.msd.ea_model.EaModel` for HF LLaVA. Reuses the SAGE
primitives from `method.eagle_SAGE` (vision-tower forward hook, L2-norm sink
detector, TI-Prob compressor) and applies them to MSD's draft head:

  - At each prefill, detect visual sinks (L2 norm on pre-projector ViT output)
    and TI-Prob top-k visual tokens at a target LLM layer.
  - Compress `hidden_states` and `ea_inputs_embeds` so the draft sees only the
    union of (sinks, TI top-k) visual positions, in original spatial order.
  - The same keep-mask is reused on every subsequent topK_genrate call, so the
    draft's `stable_kv` stays self-consistent across verify rounds.

LLaVA-1.5 path only for now (HF DynamicCache branch). Non-HF / Qwen2-VL paths
fall back to vanilla MSD.
"""

from .sage_ea_model import SageEaModel

__all__ = ["SageEaModel"]

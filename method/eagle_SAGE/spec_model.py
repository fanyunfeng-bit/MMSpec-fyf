"""SageSpecModel: EAGLE SpecModel with SAGE hooks.

Extends `method.eagle.spec_model.SpecModel` with:
  1. VisionTowerHook: captures the vision encoder's pre-projector output during
     the target's forward pass.
  2. SageModel: draft that accepts `position_ids` in `topK_genrate`. We swap
     the concrete class of `self.spec_layer` AFTER the parent __init__ loads
     the weights (zero-cost, weights are shared by identity).
  3. specgenerate override: monkey-patches `method.eagle.spec_model.initialize_tree`
     within the call, replacing it with a closure that invokes
     `sage_initialize_tree` with SAGE-specific arguments. All other phases
     (target KV cache, tree verification, update_inference_inputs) are
     inherited unchanged.
"""
from __future__ import annotations

import sys
from typing import Optional

import torch

from method.eagle.spec_model import SpecModel as EagleSpecModel
from method.eagle.cnets import Model as EagleModel

from .cnets import SageModel
from .utils import sage_initialize_tree
from .visual_hooks import VisionTowerHook


class SageSpecModel(EagleSpecModel):
    """EAGLE-based SpecModel with SAGE visual sink repositioning."""

    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        spec_model_path,
        total_token,
        depth,
        top_k,
        threshold,
        spec_layer_state_dict,
        is_eagle3_ckpt: bool = False,
    ):
        super().__init__(
            base_model=base_model,
            base_model_name_or_path=base_model_name_or_path,
            spec_model_path=spec_model_path,
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
            spec_layer_state_dict=spec_layer_state_dict,
            is_eagle3_ckpt=is_eagle3_ckpt,
        )

        # Upgrade the spec_layer class to SageModel so topK_genrate accepts
        # position_ids. Only valid when the base class is EagleModel (i.e., not
        # is_eagle3_ckpt). For EAGLE3 checkpoints, leave spec_layer unchanged;
        # SAGE is not supported there and sage_initialize_tree will fall back
        # to calling topK_genrate without position_ids (caught in utils.py).
        if not is_eagle3_ckpt and isinstance(self.spec_layer, EagleModel):
            self.spec_layer.__class__ = SageModel

        # Install a forward hook on the vision tower to capture pre-projector
        # visual features. Active for the lifetime of this instance.
        self.vision_hook = VisionTowerHook(self.base_model)

        # VisualProcessor is wired by the caller (typically the eval script)
        # after from_pretrained. Default: None → SAGE no-op (behaves as EAGLE).
        self.visual_processor = None
        self.sage_debug = False

        # Diagnostic probe (TGVC-style top-K text-importance for visual tokens).
        # When set, runs at prefill and after every verify (update_inference_inputs).
        self.text_importance_probe = None       # TextImportanceProbe instance or None
        self.text_importance_topk = 10          # K for top-K printing
        self._sage_sample_id = None             # set by eval script per-turn for log labels

    # ------------------------------------------------------------------ helpers

    def _build_visual_mask_1d(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Boolean mask [L] marking visual-token positions in the prefill.

        Supports LLaVA (image_token_index/image_token_id) and Qwen2.5-VL
        (image_token_id + video_token_id).
        """
        if input_ids is None or input_ids.numel() == 0:
            return None
        arch = self.base_model.config.architectures[0]
        cfg = self.base_model.config
        if arch in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        ):
            tid = getattr(cfg, "image_token_index", None)
            if tid is None:
                tid = getattr(cfg, "image_token_id", None)
            if tid is None:
                return None
            return (input_ids[0] == tid).to(torch.bool)
        if arch == "Qwen2_5_VLForConditionalGeneration":
            img_id = getattr(cfg, "image_token_id", None)
            vid_id = getattr(cfg, "video_token_id", None)
            mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
            if img_id is not None:
                mask = mask | (input_ids[0] == img_id)
            if vid_id is not None:
                mask = mask | (input_ids[0] == vid_id)
            return mask
        return None

    def _resolve_image_token_id(self) -> Optional[int]:
        """Primary image-token id (used to mark visual placeholders in input_ids)."""
        arch = self.base_model.config.architectures[0]
        cfg = self.base_model.config
        if arch in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        ):
            tid = getattr(cfg, "image_token_index", None)
            if tid is None:
                tid = getattr(cfg, "image_token_id", None)
            return tid
        if arch == "Qwen2_5_VLForConditionalGeneration":
            return getattr(cfg, "image_token_id", None)
        return None

    # ----------------------------------------------------------------- override

    @torch.no_grad()
    def specgenerate(self, input_ids, *args, **kwargs):
        """Run EAGLE's specgenerate with SAGE injected at initialize_tree.

        Mechanism: the parent's specgenerate calls `initialize_tree(...)` and
        `update_inference_inputs(...)` via its module-level namespace. We
        transiently replace those names with closures that wrap SAGE behavior
        and (optionally) notify the TextImportanceProbe.
        """
        eagle_spec_module = sys.modules["method.eagle.spec_model"]
        # Safety: ensure the symbols exist on the parent module.
        if not hasattr(eagle_spec_module, "initialize_tree"):
            from method.eagle.utils import initialize_tree as _eagle_initialize_tree
            eagle_spec_module.initialize_tree = _eagle_initialize_tree
        if not hasattr(eagle_spec_module, "update_inference_inputs"):
            from method.eagle.utils import (
                update_inference_inputs as _eagle_update_inference_inputs,
            )
            eagle_spec_module.update_inference_inputs = _eagle_update_inference_inputs

        original_initialize_tree = eagle_spec_module.initialize_tree
        original_update_inference_inputs = eagle_spec_module.update_inference_inputs
        sage_self = self
        verify_counter = [0]

        # Reset probe state once per specgenerate (= once per turn).
        if sage_self.text_importance_probe is not None:
            sage_self.text_importance_probe.reset()

        def wrapped_initialize_tree(
            inp_ids,
            model,
            past_kv,
            logits_proc,
            inputs_embeds=None,
            embed_weights=None,
            image_mask=None,
            **kw,
        ):
            # SAGE branch: with or without visual_processor.
            if sage_self.visual_processor is None:
                result = original_initialize_tree(
                    inp_ids,
                    model,
                    past_kv,
                    logits_proc,
                    inputs_embeds=inputs_embeds,
                    embed_weights=embed_weights,
                    image_mask=image_mask,
                    **kw,
                )
            else:
                visual_mask_1d = sage_self._build_visual_mask_1d(inp_ids)
                pre_proj = sage_self.vision_hook.pop()
                result = sage_initialize_tree(
                    inp_ids,
                    model,
                    past_kv,
                    logits_proc,
                    inputs_embeds=inputs_embeds,
                    embed_weights=embed_weights,
                    image_mask=image_mask,
                    visual_processor=sage_self.visual_processor,
                    visual_mask_1d=visual_mask_1d,
                    pre_projector_features=pre_proj,
                    target_position_ids=None,
                    debug=sage_self.sage_debug,
                    **kw,
                )

            # TI-Probe: capture H_v + H_t_initial after prefill.
            probe = sage_self.text_importance_probe
            if probe is not None:
                img_id = sage_self._resolve_image_token_id()
                if img_id is not None:
                    probe.on_prefill(
                        input_ids=inp_ids,
                        image_token_id=int(img_id),
                        sample_id=sage_self._sage_sample_id,
                        topk=int(sage_self.text_importance_topk),
                    )
            return result

        def wrapped_update_inference_inputs(*u_args, **u_kwargs):
            # Pull args we need BEFORE calling original (positions are stable).
            # update_inference_inputs signature:
            #   (input_ids, candidates, best_candidate, accept_length,
            #    retrieve_indices, logits_processor, new_token,
            #    past_key_values_data_list, current_length_data, model,
            #    hidden_state_new, sample_p, cache_base_len=None)
            best_candidate = u_args[2] if len(u_args) > 2 else u_kwargs.get("best_candidate")
            accept_length = u_args[3] if len(u_args) > 3 else u_kwargs.get("accept_length")
            retrieve_indices = u_args[4] if len(u_args) > 4 else u_kwargs.get("retrieve_indices")

            result = original_update_inference_inputs(*u_args, **u_kwargs)

            probe = sage_self.text_importance_probe
            if probe is not None and retrieve_indices is not None:
                try:
                    al = (
                        int(accept_length.item())
                        if isinstance(accept_length, torch.Tensor)
                        else int(accept_length)
                    )
                    bc = (
                        int(best_candidate.item())
                        if isinstance(best_candidate, torch.Tensor)
                        else int(best_candidate)
                    )
                    select_indices = retrieve_indices[bc, : al + 1]
                    verify_counter[0] += 1
                    probe.on_verify_accepted(
                        select_indices=select_indices,
                        sample_id=sage_self._sage_sample_id,
                        topk=int(sage_self.text_importance_topk),
                        verify_step=verify_counter[0],
                    )
                except Exception as e:  # robust against shape edge cases
                    print(f"[TI-Probe] WARN verify-step notify failed: {e}")
            return result

        try:
            eagle_spec_module.initialize_tree = wrapped_initialize_tree
            eagle_spec_module.update_inference_inputs = wrapped_update_inference_inputs
            return super().specgenerate(input_ids, *args, **kwargs)
        finally:
            eagle_spec_module.initialize_tree = original_initialize_tree
            eagle_spec_module.update_inference_inputs = original_update_inference_inputs

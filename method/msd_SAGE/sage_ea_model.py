"""SageEaModel — MSD's EaModel with SAGE visual-token compression on the draft.

Subclasses `method.msd.ea_model.EaModel`. Only the HF-LLaVA branch of
`msdgenerate` is overridden. Other architectures (Qwen2-VL, native LLaVA) fall
back to upstream MSD untouched.

Wiring:
    model = SageEaModel.from_pretrained(...)
    model.setup_sage(visual_processor, image_token_id, debug=False)
    model.msdgenerate(...)   # SAGE-aware path
"""
from __future__ import annotations

from typing import Optional

import torch

from method.msd.ea_model import (
    EaModel,
    _collect_stop_token_ids,
    _truncate_at_first_stop_token,
)
from method.msd.utils import (
    prepare_logits_processor,
    reset_tree_mode,
    tree_decoding_hf,
    evaluate_posterior,
)

from method.eagle_SAGE.visual_hooks import VisionTowerHook
from .sage_utils import (
    sage_initialize_tree_hf,
    sage_update_inference_inputs_hf,
)


class SageEaModel(EaModel):
    """EaModel with SAGE visual-token compression for the draft head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SAGE state — populated by setup_sage(). Defaults are no-op (acts as MSD).
        self._sage_visual_processor = None
        self._sage_vision_hook: Optional[VisionTowerHook] = None
        self._sage_image_token_id: Optional[int] = None
        self._sage_debug: bool = False
        self._sage_sample_id: Optional[str] = None  # set per-turn by the eval script

    # ------------------------------------------------------------------ wiring

    def setup_sage(
        self,
        visual_processor,
        image_token_id: int,
        debug: bool = False,
    ) -> None:
        """Install the vision-tower hook and store the visual processor.

        The hook captures pre-projector ViT features whenever
        `base_model.vision_tower` is called (typically inside
        `get_inputs_embeds`). The compressor stage in `visual_processor`
        consumes them at prefill time.

        Also expands `ea_layer.embed_tokens` to match the base model's
        lm_head vocab size if smaller. SAGE's compressed visual context
        causes the draft to occasionally predict special tokens
        (e.g. image_token_index=32000) which are valid for the base
        lm_head (vocab=32064 on LLaVA-1.5) but OOB for the upstream MSD
        draft embed table (vocab=32000). The OOB CUDA assert surfaces
        asynchronously at the next op (apply_rotary_pos_emb's
        cos[position_ids] indexing). Vanilla MSD never trips this on
        testmini because its full-visual draft barely ever samples a
        token in [32000, 32064) — SAGE's sparser visual context does.
        """
        self._sage_visual_processor = visual_processor
        self._sage_image_token_id = int(image_token_id)
        self._sage_debug = bool(debug)
        if self._sage_vision_hook is None:
            self._sage_vision_hook = VisionTowerHook(self.base_model)

        self._maybe_expand_draft_embed_tokens()

    def _maybe_expand_draft_embed_tokens(self) -> None:
        """Expand draft embed_tokens to base lm_head vocab size if smaller."""
        try:
            base_vocab = int(self._get_lm_head().out_features)
        except Exception:
            return
        embed = self.ea_layer.embed_tokens
        draft_vocab = int(embed.num_embeddings)
        if draft_vocab >= base_vocab:
            return
        old_weight = embed.weight.data
        new_embed = torch.nn.Embedding(
            num_embeddings=base_vocab,
            embedding_dim=embed.embedding_dim,
            padding_idx=embed.padding_idx,
        ).to(device=old_weight.device, dtype=old_weight.dtype)
        # Copy original rows; new rows default to whatever Embedding init gives
        # (Normal(0, 1)) — overwrite them with the mean of existing rows so the
        # draft outputs reasonable embeddings instead of huge random vectors
        # for tokens it was never trained on.
        with torch.no_grad():
            new_embed.weight[:draft_vocab] = old_weight
            mean_row = old_weight.mean(dim=0)
            new_embed.weight[draft_vocab:] = mean_row
        self.ea_layer.embed_tokens = new_embed
        self.ea_layer.vocab_size = base_vocab
        if self._sage_debug:
            print(
                f"[SAGE-MSD] expanded draft embed_tokens: "
                f"{draft_vocab} -> {base_vocab} (filled new rows with row-mean)",
                flush=True,
            )

    def remove_sage_hook(self) -> None:
        if self._sage_vision_hook is not None:
            self._sage_vision_hook.remove()
            self._sage_vision_hook = None

    # ------------------------------------------------------------ msdgenerate

    @torch.no_grad()
    def msdgenerate(
        self,
        input_ids,
        inputs_embeds=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
    ):
        """SAGE-aware msdgenerate. Only the HF-LLaVA branch is replaced; other
        backbones (or SAGE not configured) fall back to upstream MSD.
        """
        sage_active = (
            self._is_hf_llava
            and self._sage_visual_processor is not None
        )
        if not sage_active:
            return super().msdgenerate(
                input_ids,
                inputs_embeds=inputs_embeds,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                log=log,
            )

        # ---- HF-LLaVA SAGE path (mirrors EaModel.msdgenerate's HF branch) ----
        max_length = max_length - self.ea_layer.total_tokens - 10
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        input_len = input_ids.shape[1]
        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        reset_tree_mode(self)

        # Reset SAGE state per generation so a previous turn's mask doesn't leak.
        self.ea_layer._sage_keep_mask = None
        self.ea_layer._sage_original_prefix_len = None

        from transformers import DynamicCache
        past_key_values = DynamicCache()

        result = sage_initialize_tree_hf(
            input_ids,
            self,
            past_key_values,
            logits_processor,
            inputs_embeds,
            visual_processor=self._sage_visual_processor,
            image_token_id=int(self._sage_image_token_id),
            sample_id=self._sage_sample_id,
            debug=self._sage_debug,
        )
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            logits,
            hidden_state,
            sample_token,
            past_key_values,
            ea_input_ids,
            ea_inputs_embeds,
        ) = result
        new_token = 0

        for idx in range(max_length):
            draft_tokens = draft_tokens.to(input_ids.device)

            # Tree decoding on the TARGET (full-prefix KV cache, untouched by SAGE).
            logits, hidden_state_new, outputs = tree_decoding_hf(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                tree_mask=tree_mask,
            )
            past_key_values = outputs.past_key_values

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            self.acclen += accept_length
            self.accnum += 1

            result = sage_update_inference_inputs_hf(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values,
                self,
                hidden_state_new,
                sample_p,
                ea_input_ids,
                ea_inputs_embeds,
            )
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                hidden_state,
                sample_token,
                past_key_values,
                ea_input_ids,
                ea_inputs_embeds,
            ) = result

            input_ids, new_token, hit_stop = _truncate_at_first_stop_token(
                input_ids, input_len, stop_token_ids
            )
            if hit_stop:
                break
            if new_token >= max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        if not log:
            return input_ids
        return input_ids, new_token, idx


import torch
import torch.nn as nn
import time
import numpy as np
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoConfig

from method.vispec.utils import *
from method.vispec.kv_cache import initialize_past_key_values

PLD_NGRAM = 4
PLD_N_PRED = 10
TAIL_REPEAT_NGRAM_MIN = 1
TAIL_REPEAT_NGRAM_MAX = 32
TAIL_REPEAT_MIN_REPEATS = 12
TAIL_REPEAT_MIN_TOKENS = 64


def _has_repetitive_tail(
    generated_ids: torch.Tensor,
    ngram_min: int = 1,
    ngram_max: int = 32,
    min_repeats: int = 12,
    min_generated_tokens: int = 64,
) -> bool:
    """Return True when the generated tail is repeated n-gram loops."""
    if generated_ids is None:
        return False
    if generated_ids.dim() != 1:
        generated_ids = generated_ids.view(-1)

    total = int(generated_ids.numel())
    if total < min_generated_tokens or total < max(1, ngram_min) * max(2, min_repeats):
        return False

    ngram_min = max(1, int(ngram_min))
    ngram_max = min(int(ngram_max), total // max(2, int(min_repeats)))
    if ngram_max < ngram_min:
        return False

    for n in range(ngram_min, ngram_max + 1):
        unit = generated_ids[-n:]
        repeats = 0
        pos = total
        while pos - n >= 0 and torch.equal(generated_ids[pos - n: pos], unit):
            repeats += 1
            if repeats >= min_repeats:
                return True
            pos -= n
    return False


class SpecModel(nn.Module):

    def __init__(self, base_model, tokenizer=None, max_new_tokens=512):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        Type="LLaMA",
        base_model_path=None,
        spec_model_path=None,
        total_token=30,
        depth=3,
        top_k=8,
        threshold=1.0,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(base_model_path)
        arch = config.architectures[0]
        if arch == "Qwen2_5_VLForConditionalGeneration":
            from method.vispec.modeling_qwen2_5_vl_kv import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path, **kwargs
            )
        elif arch == "LlavaForConditionalGeneration":
            from method.llava_adapter import CustomLlavaForConditionalGeneration
            model = CustomLlavaForConditionalGeneration.from_pretrained(
                base_model_path, **kwargs
            )
        elif arch == "LlavaNextForConditionalGeneration":
            from method.llava_adapter import CustomLlavaNextForConditionalGeneration
            model = CustomLlavaNextForConditionalGeneration.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Model type {arch} is not supported. Please use a supported model type."
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

        # spec_layer_state_dict = None
        model = cls(
            model,
            tokenizer = tokenizer,
            max_new_tokens=512,
        )

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        inputs_embeds=None,
        output_real_hidden=False,
        **kwargs,
    ):
        if (
            inputs_embeds is not None
            and self.base_model.config.architectures[0]
            == "LlavaNextForConditionalGeneration"
        ):
            input_ids = None
            kwargs = {}

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                output_hidden_states=True,
                **kwargs,
            )
            if output_orig:
                orig = outputs.logits
            hidden_states = outputs.hidden_states[-1]

        if output_real_hidden:
            return None, orig, hidden_states, outputs.hidden_states
        if output_orig:
            return None, orig, hidden_states
        else:
            return None, hidden_states

    @torch.no_grad()
    def specgenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
        inputs_embeds=None,
        return_acceptance_len=False,
        return_decode_time=False,
        **kwargs,
    ):
        # print("=" * 60)
        # print("specgenerate() CALLED")
        # print(f"  input_ids.shape={input_ids.shape}, temperature={temperature}")
        # print(f"  max_new_tokens={max_new_tokens}, max_length={max_length}")
        # print(f"  pld_ngram={pld_ngram}, pld_n_pred={pld_n_pred}")
        # print(f"  log={log}, return_acceptance_len={return_acceptance_len}, return_decode_time={return_decode_time}")
        # print(f"  kwargs keys: {list(kwargs.keys())}")
        # print(f"  inputs_embeds is None: {inputs_embeds is None}")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Keep PLD and anti-loop parameters fixed in spec_model.
        pld_ngram = PLD_NGRAM
        pld_n_pred = PLD_N_PRED
        tail_repeat_ngram_min = TAIL_REPEAT_NGRAM_MIN
        tail_repeat_ngram_max = TAIL_REPEAT_NGRAM_MAX
        tail_repeat_min_repeats = TAIL_REPEAT_MIN_REPEATS
        tail_repeat_min_tokens = TAIL_REPEAT_MIN_TOKENS
        kwargs.pop("pld_ngram", None)
        kwargs.pop("pld_n_pred", None)
        kwargs.pop("tail_repeat_ngram_min", None)
        kwargs.pop("tail_repeat_ngram_max", None)
        kwargs.pop("tail_repeat_min_repeats", None)
        kwargs.pop("tail_repeat_min_tokens", None)
        arch = self.base_model.config.architectures[0]
        is_llava = arch in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        )

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            # print("KV cache EXISTS on self, reusing and resetting length to 0")
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
            # print(f"  current_length_data after zero: {current_length_data.tolist()}")
        else:
            # print("KV cache NOT on self, initializing new KV cache")
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
                # print("  Initialized KV cache from base_model")
            except:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
                # print("  Initialized KV cache from base_model.language_model (fallback)")
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
            # print(f"  num_layers={len(past_key_values)}, current_length_data shape={current_length_data.shape}")

        image_token_id = None
        prefill_kwargs = {}
        if arch == "Qwen2_5_VLForConditionalGeneration":
            image_token_id = self.base_model.config.image_token_id
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")
            # print(f"  Qwen2.5-VL mode: image_token_id={image_token_id}")
            # print(f"  pixel_values is None: {pixel_values is None}")
            # print(f"  image_grid_thw: {image_grid_thw}")

            if inputs_embeds is None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.base_model.visual.dtype)
                    image_embeds = self.base_model.visual(
                        pixel_values, grid_thw=image_grid_thw
                    )
                    n_image_tokens = (
                        (input_ids == image_token_id)
                        .sum()
                        .item()
                    )
                    n_image_features = image_embeds.shape[0]
                    # print(f"  n_image_tokens={n_image_tokens}, n_image_features={n_image_features}")
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        image_mask, image_embeds
                    )
                    # print("  Image embeds merged into inputs_embeds")
                else:
                    # print("  No pixel_values, skipping image embedding")
                    pass
        elif is_llava:
            pixel_values = kwargs.get("pixel_values")
            image_sizes = kwargs.get("image_sizes")
            if pixel_values is not None:
                prefill_kwargs["pixel_values"] = pixel_values
                if arch == "LlavaNextForConditionalGeneration" and image_sizes is not None:
                    prefill_kwargs["image_sizes"] = image_sizes

        # Prompt lookup decoding (PLD)

        input_len = input_ids.shape[1]
        # print(f"  input_len (prompt length) = {input_len}")

        # Initialize acceptance length tracking if needed
        if return_acceptance_len:
            acceptance_len = []

        # print("Running prefill (init forward pass)...")
        init_out = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **prefill_kwargs,
        )
        init_logits = init_out.logits[:, -1, :]

        inputs_embeds = None
        kwargs = {}

        if temperature > 0:
            probs = torch.softmax(init_logits / temperature, dim=-1)
            init_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            init_token = torch.argmax(init_logits, dim=-1)

        # print(f"  init_token = {init_token.item()} (decoded: {repr(self.tokenizer.decode([init_token.item()]))})")

        input_ids = torch.cat((input_ids, init_token.unsqueeze(-1).to(input_ids.device)), dim=1)

        if hasattr(self, "current_length_data") and (not is_llava):
            # Only the prompt has been forwarded, init_token hasn't
            current_length_data.fill_(input_len)
            # print(f"  [REDUNDANT] current_length_data set to input_len={input_len}")

        if hasattr(self, "current_length_data") and (not is_llava):
            current_length_data.fill_(input_ids.shape[1] - 1)
            # print(f"  current_length_data set to {input_ids.shape[1] - 1} (input_ids.shape[1]-1)")

        # print(f"Entering main decode loop (max_length={max_length})...")
        idx = 0  # default in case loop doesn't execute
        total_draft_attempts = 0
        total_draft_found = 0
        total_accepted = 0
        total_fallback = 0

        for idx in range(max_length):
            curr_len = input_ids.shape[1]
            generated_so_far = curr_len - input_len

            # Check max_new_tokens
            if generated_so_far >= max_new_tokens:
                # print(f"  [ITER {idx}] BREAK: max_new_tokens reached ({generated_so_far} >= {max_new_tokens})")
                break
            
            # Check EOS
            if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                # print(f"  [ITER {idx}] BREAK: EOS token found in generated tokens")
                break

            if _has_repetitive_tail(
                input_ids[0, input_len:],
                ngram_min=tail_repeat_ngram_min,
                ngram_max=tail_repeat_ngram_max,
                min_repeats=tail_repeat_min_repeats,
                min_generated_tokens=tail_repeat_min_tokens,
            ):
                break
            
            ids = input_ids[0]
            # kv_len_before = current_length_data[0].item() if hasattr(self, "current_length_data") else -1
            
            # ===== N-gram lookup phase =====
            total_draft_attempts += 1
            draft_reason = ""
            
            if ids.shape[0] < pld_ngram + 1:
                draft_tokens = torch.tensor([], device=input_ids.device).long()
                draft_reason = f"seq too short ({ids.shape[0]} < {pld_ngram + 1})"
            else:
                query = ids[-pld_ngram:]
                # print(f"  [ITER {idx}] QUERY n-gram (last {pld_ngram} tokens): {query.tolist()} "
                                # f"(decoded: {repr(self.tokenizer.decode(query.tolist()))})")
                if image_token_id is not None and (query == image_token_id).any():
                    draft_tokens = torch.tensor([], device=input_ids.device).long()
                    draft_reason = "query contains image_token_id"
                else:
                    search_ids = ids[:-pld_ngram]
                    if search_ids.shape[0] < pld_ngram:
                        draft_tokens = torch.tensor([], device=input_ids.device).long()
                        draft_reason = f"search_ids too short ({search_ids.shape[0]} < {pld_ngram})"
                    else:
                        windows = search_ids.unfold(0, pld_ngram, 1)
                        matches = (windows == query).all(dim=-1)
                        match_indices = matches.nonzero(as_tuple=True)[0]
                        if match_indices.numel() > 0:
                            draft_tokens = torch.tensor([], device=input_ids.device).long()
                            n_matches_total = match_indices.numel()
                            n_skipped_image = 0
                            chosen_match_idx = None
                            for match_idx in torch.flip(match_indices, dims=[0]):
                                start = match_idx + pld_ngram
                                end = start + pld_n_pred
                                valid_end = min(end, ids.shape[0])
                                candidate = ids[start: valid_end]

                                if image_token_id is not None and (candidate == image_token_id).any():
                                    n_skipped_image += 1
                                    continue
                                else:
                                    draft_tokens = candidate
                                    # chosen_match_idx = match_idx.item()
                                    # Print matched historical context
                                    # hist_start = max(0, match_idx.item())
                                    # hist_ngram = ids[hist_start : hist_start + pld_ngram]
                                    # hist_after = ids[hist_start + pld_ngram : valid_end]
                                    # print(f"    MATCH DETAIL: history pos {hist_start}")
                                    # print(f"      history n-gram : {hist_ngram.tolist()} "
                                                    # f"(decoded: {repr(self.tokenizer.decode(hist_ngram.tolist()))})")
                                    # print(f"      history follow : {hist_after.tolist()} "
                                                    # f"(decoded: {repr(self.tokenizer.decode(hist_after.tolist()))})")
                                    # print(f"      current query  : {query.tolist()} "
                                                    # f"(decoded: {repr(self.tokenizer.decode(query.tolist()))})")
                                    # print(f"      draft (copied) : {draft_tokens.tolist()} "
                                                    # f"(decoded: {repr(self.tokenizer.decode(draft_tokens.tolist()))})")
                                    break
                            if draft_tokens.numel() == 0:
                                draft_reason = f"all {n_matches_total} matches contain image tokens (skipped {n_skipped_image})"
                            else:
                                draft_reason = f"found match at idx {chosen_match_idx} (from {n_matches_total} total, {n_skipped_image} skipped)"
                        else:
                            draft_tokens = torch.tensor([], device=input_ids.device).long()
                            draft_reason = "no n-gram match found"

            last_tok = input_ids[:, -1:]

            # ===== No draft: fallback to normal decoding =====
            if draft_tokens.numel() == 0:
                total_fallback += 1
                # print(f"  [ITER {idx}] FALLBACK (no draft): {draft_reason} | "
                                # f"kv_len={kv_len_before}, input_ids_len={curr_len}, "
                                # f"generated={generated_so_far}")
                out = self.base_model(
                    last_tok,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                next_logits = out.logits[:, -1, :]
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_id = torch.argmax(next_logits, dim=-1)
                input_ids = torch.cat((input_ids, next_id.unsqueeze(0)), dim=-1)
                if hasattr(self, "current_length_data") and (not is_llava):
                    current_length_data.fill_(input_ids.shape[1] - 1)
                    # print(f"    -> next_id={next_id.item()} (decoded: {repr(self.tokenizer.decode([next_id.item()]))}), "
                                    # f"new kv_len={current_length_data[0].item()}, new input_ids_len={input_ids.shape[1]}")
                if return_acceptance_len:
                    acceptance_len.append(0)
                continue

            # ===== Draft found: speculative verification =====
            total_draft_found += 1
            D = draft_tokens.numel()
            # print(f"  [ITER {idx}] DRAFT FOUND: D={D}, {draft_reason} | "
                            # f"kv_len={kv_len_before}, input_ids_len={curr_len}, "
                            # f"generated={generated_so_far}")
            # print(f"    draft_tokens = {draft_tokens.tolist()} "
                            # f"(decoded: {repr(self.tokenizer.decode(draft_tokens.tolist()))})")

            draft_tokens = draft_tokens.to(dtype=ids.dtype, device=ids.device).view(1, -1)

            # print(f"    Forward pass 1: last_tok={last_tok[0].tolist()}")
            out_last = self.base_model(
                last_tok,
                use_cache=True,
                past_key_values=past_key_values,
            )
            # kv_after_last = current_length_data[0].item() if hasattr(self, "current_length_data") else -1
            # print(f"    kv_len after last_tok forward: {kv_after_last}")

            # print(f"    Forward pass 2: draft_tokens shape={draft_tokens.shape}")
            out = self.base_model(
                draft_tokens,
                use_cache=True,
                past_key_values=past_key_values,
            )
            # kv_after_draft = current_length_data[0].item() if hasattr(self, "current_length_data") else -1
            # print(f"    kv_len after draft forward: {kv_after_draft}")

            logits = out.logits

            D = draft_tokens.shape[1]
            
            # Verification: 
            last_pred = torch.argmax(out_last.logits[:, -1:, :], dim=-1)
            draft_pred_ids = torch.argmax(logits, dim=-1)

            # Compute per-token likelihood of draft tokens (disabled for performance)
            # Position 0: P(draft[0] | context) from out_last
            # last_probs = torch.softmax(out_last.logits[:, -1, :], dim=-1)
            # p_draft0 = last_probs[0, draft_tokens[0, 0]].item()
            # Positions 1..D-1: P(draft[i] | context, draft[0..i-1]) from out
            # draft_probs = torch.softmax(logits, dim=-1)  # [1, D, vocab]
            # per_token_info = []
            # per_token_info.append({
            #     "pos": 0,
            #     "token_id": draft_tokens[0, 0].item(),
            #     "token_str": repr(self.tokenizer.decode([draft_tokens[0, 0].item()])),
            #     "p_draft": f"{p_draft0:.4f}",
            #     "model_top1_id": last_pred[0, 0].item(),
            #     "model_top1_str": repr(self.tokenizer.decode([last_pred[0, 0].item()])),
            #     "match": draft_tokens[0, 0].item() == last_pred[0, 0].item(),
            # })
            # for di in range(D):
            #     if di < D - 1:
            #         next_tok = draft_tokens[0, di + 1].item()
            #     else:
            #         next_tok = draft_pred_ids[0, di].item()  # bonus token
            #     p_tok = draft_probs[0, di, next_tok].item()
            #     model_top1 = draft_pred_ids[0, di].item()
            #     is_draft = di < D - 1
            #     per_token_info.append({
            #         "pos": di + 1,
            #         "token_id": next_tok,
            #         "token_str": repr(self.tokenizer.decode([next_tok])),
            #         "p": f"{p_tok:.4f}",
            #         "model_top1_id": model_top1,
            #         "model_top1_str": repr(self.tokenizer.decode([model_top1])),
            #         "match": next_tok == model_top1 if is_draft else "(bonus)",
            #         "type": "draft" if is_draft else "bonus",
            #     })
            # print(f"    --- Per-token draft likelihood ---")
            # print(f"    pos 0: draft_tok={per_token_info[0]['token_id']} {per_token_info[0]['token_str']}  "
                            # f"P={per_token_info[0]['p_draft']}  "
                            # f"model_top1={per_token_info[0]['model_top1_id']} {per_token_info[0]['model_top1_str']}  "
                            # f"match={per_token_info[0]['match']}")
            # for info in per_token_info[1:]:
                # print(f"    pos {info['pos']}: {info['type']}_tok={info['token_id']} {info['token_str']}  "
                                # f"P={info['p']}  "
                                # f"model_top1={info['model_top1_id']} {info['model_top1_str']}  "
                                # f"match={info['match']}")
            # print(f"    ---------------------------------")

            # print(f"    last_pred (model prediction for last_tok) = {last_pred[0].tolist()} "
                            # f"(decoded: {repr(self.tokenizer.decode(last_pred[0].tolist()))})")
            # print(f"    draft_tokens[0]  = {draft_tokens[0, 0].item()} "
                            # f"(decoded: {repr(self.tokenizer.decode([draft_tokens[0, 0].item()]))})")
            # print(f"    draft_pred_ids   = {draft_pred_ids[0].tolist()} "
                            # f"(decoded: {repr(self.tokenizer.decode(draft_pred_ids[0].tolist()))})")
            
            if last_pred[:, 0] != draft_tokens[:, 0]:
                accept_len = 0
                # print(f"    REJECT: last_pred ({last_pred[0, 0].item()}) != draft[0] ({draft_tokens[0, 0].item()})")
            else:
                if D == 1:
                    accept_len = 1
                    # print(f"    ACCEPT all (D=1)")
                else:
                    same = (draft_pred_ids[:, :-1] == draft_tokens[:, 1:])
                    mismatch = (~same).squeeze(0).nonzero(as_tuple=True)[0]
                    if mismatch.numel() > 0:
                        accept_len = int(mismatch[0].item()) + 1
                        # print(f"    PARTIAL ACCEPT: first mismatch at position {mismatch[0].item()}, accept_len={accept_len}/{D}")
                    else:
                        accept_len = D
                        # print(f"    ACCEPT ALL: accept_len={accept_len}/{D}")

            total_accepted += accept_len

            if return_acceptance_len:
                acceptance_len.append(accept_len)

            if accept_len == 0:
                tokens_to_add = last_pred
                # print(f"    tokens_to_add (reject, use last_pred): {tokens_to_add[0].tolist()}")
            elif accept_len < D:
                tokens_to_add = torch.cat([
                    draft_tokens[:, :accept_len], 
                    draft_pred_ids[:, accept_len - 1 : accept_len]
                ], dim=1)
                # print(f"    tokens_to_add (partial accept {accept_len}/{D} + correction): {tokens_to_add[0].tolist()}")
            else:
                tokens_to_add = torch.cat([
                    draft_tokens, 
                    draft_pred_ids[:, -1:]
                ], dim=1)
                # print(f"    tokens_to_add (full accept {D}/{D} + bonus): {tokens_to_add[0].tolist()}")

            # Cap tokens_to_add to not exceed max_new_tokens
            remaining = max_new_tokens - (input_ids.shape[1] - input_len)
            if tokens_to_add.shape[1] > remaining:
                # print(f"    CAPPING tokens_to_add from {tokens_to_add.shape[1]} to {remaining} (max_new_tokens={max_new_tokens})")
                tokens_to_add = tokens_to_add[:, :remaining]

            input_ids = torch.cat([input_ids, tokens_to_add], dim=1)

            # Update KV cache length to match actual accepted tokens
            if hasattr(self, "current_length_data") and (not is_llava):
                current_length_data.fill_(input_ids.shape[1] - 1)
                # print(f"    KV cache updated: kv_len={current_length_data[0].item()}, "
                                # f"input_ids_len={input_ids.shape[1]}, "
                                # f"tokens_added={tokens_to_add.shape[1]}, "
                                # f"kv_was={kv_after_draft} -> now={current_length_data[0].item()} "
                                # f"(trimmed {kv_after_draft - current_length_data[0].item()} rejected entries)")

        # ===== Summary =====
        new_token = input_ids.shape[1] - input_len
        # print(f"GENERATION COMPLETE:")
        # print(f"  total iterations: {idx + 1 if idx > 0 or new_token > 0 else 0}")
        # print(f"  new_tokens: {new_token}")
        # print(f"  draft attempts: {total_draft_attempts}, drafts found: {total_draft_found}, fallbacks: {total_fallback}")
        # print(f"  total accepted draft tokens: {total_accepted}")
        if total_draft_found > 0:
            # print(f"  avg acceptance per draft: {total_accepted / total_draft_found:.2f}")
            pass
        # print(f"  final input_ids length: {input_ids.shape[1]}")
        # final_kv = current_length_data[0].item() if hasattr(self, "current_length_data") else -1
        # print(f"  final kv_len: {final_kv}")
        # print(f"  SANITY CHECK: input_ids_len - 1 == kv_len? {input_ids.shape[1] - 1 == final_kv}")
        # if return_acceptance_len:
            # print(f"  acceptance_len list: {acceptance_len}")
        
        # Decode and log generated text (disabled for performance)
        # generated_ids = input_ids[0, input_len:].tolist()
        # generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print(f"  generated text (first 200 chars): {repr(generated_text[:200])}")

        # Check for EOS in generated tokens (disabled for performance)
        # if self.tokenizer.eos_token_id is not None:
        #     eos_positions = [i for i, tid in enumerate(generated_ids) if tid == self.tokenizer.eos_token_id]
        #     if eos_positions:
        #         # print(f"  EOS found at positions: {eos_positions} (in generated tokens)")
        #         if eos_positions[0] < len(generated_ids) - 1:
        #             pass  # WARNING: tokens exist AFTER first EOS
        #             # print(f"  WARNING: tokens exist AFTER first EOS! "
        #                               # f"({len(generated_ids) - eos_positions[0] - 1} extra tokens)")
        # print("=" * 60)

        outputs = (input_ids,)

        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_len,)
        if return_decode_time:
            outputs += (0.0,)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def forward(self, input_ids, **kwargs):
        return self.specgenerate(input_ids, **kwargs)

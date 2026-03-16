
import torch
import torch.nn as nn
# import time
# import numpy as np
# from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoConfig

from method.vispec.utils import *
from method.vispec.kv_cache import initialize_past_key_values

MATRIX_TOP_K = 8
DRAFT_LEN = 10
TAIL_REPEAT_NGRAM_MIN = 1     # 1-gram and up (same as PLD: catches any cycle size)
TAIL_REPEAT_NGRAM_MAX = 64    # cover long phrase cycles (~15-token phrases)
TAIL_REPEAT_MIN_REPEATS = 5   # fire after 2 consecutive repeats
TAIL_REPEAT_MIN_TOKENS = 50   # start checking after 20 generated tokens


def _has_repetitive_tail(
    generated_ids: torch.Tensor,
    ngram_min: int = 1,
    ngram_max: int = 64,
    min_repeats: int = 2,
    min_generated_tokens: int = 20,
) -> bool:
    """Return True when the generated tail is repeated n-gram loops (consecutive)."""
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


def _has_recurring_ngram(
    generated_ids: torch.Tensor,
    window: int = 250,
    ngram_min: int = 20,
    ngram_max: int = 50,
    min_occurrences: int = 6,
    min_generated_tokens: int = 200,
):
    if generated_ids is None:
        return False
    if generated_ids.dim() != 1:
        generated_ids = generated_ids.view(-1)
    total = int(generated_ids.numel())
    if total < min_generated_tokens:
        return False
    tail = generated_ids[-window:].tolist()
    L = len(tail)
    for n in range(ngram_min, min(ngram_max + 1, L // min_occurrences + 1)):
        unit = tail[-n:]
        count = 0
        for start in range(L - n + 1):
            if tail[start: start + n] == unit:
                count += 1
                if count >= min_occurrences:
                    return True
    return False


class SpecModel(nn.Module):

    def __init__(self, base_model, tokenizer=None, max_new_tokens=512):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        

    def get_tokenizer(self):
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
            from method.recycling.modeling_qwen2_5_vl_kv import Qwen2_5_VLForConditionalGeneration
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
        """Token Recycling speculative decoding.

        Core idea: maintain an adjacency matrix M[vocab_size, top_k] that stores
        top-k next-token predictions for each token. Use M to generate chain drafts
        by following top-1 predictions. Verify drafts in a single forward pass.
        Update M from verification logits.

        This is a chain-draft simplification of the full tree-based Token Recycling
        (which uses a static tree structure for parallel branch exploration).
        The chain approach follows only the top-1 path from M, verified sequentially.

        Reference: Token Recycling (https://arxiv.org/abs/2408.05233)
        """
        # print("=" * 60)
        # print("[RECYCLING] Token Recycling Speculative Decoding")
        # print("=" * 60)
        # print(f"[RECYCLING] Config: matrix_top_k={matrix_top_k}, draft_len={draft_len}")
        # print(f"[RECYCLING] input_ids.shape={input_ids.shape}, temperature={temperature}")
        # print(f"[RECYCLING] max_new_tokens={max_new_tokens}, max_length={max_length}")
        # print(f"[RECYCLING] log={log}, return_acceptance_len={return_acceptance_len}, return_decode_time={return_decode_time}")
        # print(f"[RECYCLING] inputs_embeds is None: {inputs_embeds is None}")
        # print(f"[RECYCLING] kwargs keys: {list(kwargs.keys())}")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Allow CLI overrides of Recycling parameters; fall back to module-level constants.
        matrix_top_k = kwargs.pop("matrix_top_k", MATRIX_TOP_K)
        draft_len = kwargs.pop("draft_len", DRAFT_LEN)
        tail_repeat_ngram_min = kwargs.pop("tail_repeat_ngram_min", TAIL_REPEAT_NGRAM_MIN)
        tail_repeat_ngram_max = kwargs.pop("tail_repeat_ngram_max", TAIL_REPEAT_NGRAM_MAX)
        tail_repeat_min_repeats = kwargs.pop("tail_repeat_min_repeats", TAIL_REPEAT_MIN_REPEATS)
        tail_repeat_min_tokens = kwargs.pop("tail_repeat_min_tokens", TAIL_REPEAT_MIN_TOKENS)
        arch = self.base_model.config.architectures[0]
        is_llava = arch in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        )

        # ===== Initialize adjacency matrix =====
        vocab_size = self.base_model.config.vocab_size
        adjacency_matrix = torch.zeros(
            vocab_size, matrix_top_k, dtype=torch.long, device=input_ids.device
        )
        # print(f"[RECYCLING] Adjacency matrix initialized: shape=[{vocab_size}, {matrix_top_k}], "
        #       f"memory={vocab_size * matrix_top_k * 8 / 1024 / 1024:.1f} MB")

        # ===== Initialize KV cache =====
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
            # print("[RECYCLING] KV cache reused, reset length to 0")
        else:
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
                # print("[RECYCLING] KV cache initialized from base_model")
            except:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
                # print("[RECYCLING] KV cache initialized from base_model.language_model (fallback)")
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
            # print(f"[RECYCLING] num_layers={len(past_key_values)}, current_length_data shape={current_length_data.shape}")

        # ===== Handle Qwen2.5-VL image embeddings =====
        image_token_id = None
        prefill_kwargs = {}
        if arch == "Qwen2_5_VLForConditionalGeneration":
            image_token_id = self.base_model.config.image_token_id
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")
            # print(f"[RECYCLING] Qwen2.5-VL mode: image_token_id={image_token_id}")
            # print(f"[RECYCLING] pixel_values is None: {pixel_values is None}")
            # print(f"[RECYCLING] image_grid_thw: {image_grid_thw}")

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
                    # print(f"[RECYCLING] n_image_tokens={n_image_tokens}, n_image_features={n_image_features}")
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
                    # print("[RECYCLING] Image embeds merged into inputs_embeds")
        elif is_llava:
            pixel_values = kwargs.get("pixel_values")
            image_sizes = kwargs.get("image_sizes")
            if pixel_values is not None:
                prefill_kwargs["pixel_values"] = pixel_values
                if arch == "LlavaNextForConditionalGeneration" and image_sizes is not None:
                    prefill_kwargs["image_sizes"] = image_sizes

        input_len = input_ids.shape[1]
        # print(f"[RECYCLING] input_len (prompt length) = {input_len}")

        if return_acceptance_len:
            acceptance_len_list = []

        # ===== Prefill =====
        # print("[RECYCLING] Running prefill (init forward pass)...")
        init_out = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **prefill_kwargs,
        )
        init_logits = init_out.logits

        # Initialize adjacency matrix from prompt logits
        # For each token in prompt, store top-k predictions
        # If same token appears multiple times, last occurrence wins
        prompt_token_ids = input_ids[0]  # [seq_len]
        topk_indices = init_logits[0].topk(matrix_top_k, dim=-1).indices  # [seq_len, top_k]
        adjacency_matrix[prompt_token_ids] = topk_indices
        # n_unique_prompt = prompt_token_ids.unique().numel()
        # n_nonzero_M = (adjacency_matrix.sum(dim=1) != 0).sum().item()
        # print(f"[RECYCLING] M initialized from prefill: {n_unique_prompt} unique prompt tokens, "
        #       f"{n_nonzero_M} M entries non-zero")

        inputs_embeds = None
        kwargs = {}

        if temperature > 0:
            probs = torch.softmax(init_logits[:, -1, :] / temperature, dim=-1)
            init_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            init_token = torch.argmax(init_logits[:, -1, :], dim=-1)

        # print(f"[RECYCLING] init_token = {init_token.item()} "
        #       f"(decoded: {repr(self.tokenizer.decode([init_token.item()]))})")

        input_ids = torch.cat((input_ids, init_token.unsqueeze(-1).to(input_ids.device)), dim=1)

        if hasattr(self, "current_length_data") and (not is_llava):
            current_length_data.fill_(input_ids.shape[1] - 1)
            # print(f"[RECYCLING] current_length_data set to {current_length_data[0].item()}")

        # ===== Main decode loop =====
        total_draft_attempts = 0
        total_draft_found = 0
        total_accepted = 0
        total_fallback = 0
        # Draft stop reason counters (for debug)
        _debug_cycle = 0
        _debug_img = 0
        _debug_uninit = 0
        _break_reason = "max_iter"

        for idx in range(max_length):
            curr_len = input_ids.shape[1]
            generated_so_far = curr_len - input_len

            # Check max_new_tokens
            if generated_so_far >= max_new_tokens:
                _break_reason = "max_new_tokens"
                break
            
            # Check EOS
            if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                _break_reason = "eos"
                break

            if _has_repetitive_tail(
                input_ids[0, input_len:],
                ngram_min=tail_repeat_ngram_min,
                ngram_max=tail_repeat_ngram_max,
                min_repeats=tail_repeat_min_repeats,
                min_generated_tokens=tail_repeat_min_tokens,
            ):
                _break_reason = "anti_rep"
                break

            # _has_recurring_ngram disabled: too many false positives on complex answers
            # (math formulas, multi-choice options). draft_len=3 limits cycle growth instead.
            
            # kv_len_before = current_length_data[0].item() if hasattr(self, "current_length_data") else -1
            total_draft_attempts += 1

            # ===== Generate chain draft from adjacency matrix =====
            last_token_id = input_ids[0, -1].item()
            draft_tokens = []
            current = last_token_id
            # draft_chain_info = [f"{current}({repr(self.tokenizer.decode([current]))})"]

            # Build a tail of recently-generated tokens for n-gram cycle check.
            # Use draft_len * 5 window to catch up to 20-gram phrase cycles.
            _recent_tail = input_ids[0, max(input_len, input_ids.shape[1] - draft_len * 5):].tolist()

            for step in range(draft_len):
                candidates = adjacency_matrix[current]
                next_tok = candidates[0].item()  # top-1 prediction
                if next_tok == 0:  # uninitialized entry
                    _debug_uninit += 1
                    break
                # Skip image token
                if image_token_id is not None and next_tok == image_token_id:
                    _debug_img += 1
                    break
                # N-gram cycle detection (1..10-gram):
                # Check if appending next_tok would create an immediate repeat in the
                # combined [recent_context + draft + next_tok] tail.
                # Range up to 10 covers phrase-level loops like "beachgoers enjoying a day"
                # (~9 tokens) that are common in LLaVA greedy repetition.
                _tentative = _recent_tail + draft_tokens + [next_tok]
                _tlen = len(_tentative)
                _ngram_cycle = False
                for _n in range(1, 11):
                    if _tlen >= 2 * _n and _tentative[-_n:] == _tentative[-2 * _n:-_n]:
                        _ngram_cycle = True
                        break
                if _ngram_cycle:
                    _debug_cycle += 1
                    break
                draft_tokens.append(next_tok)
                # draft_chain_info.append(f"{next_tok}({repr(self.tokenizer.decode([next_tok]))})")
                current = next_tok

            last_tok = input_ids[:, -1:]

            # ===== No draft: fallback to normal decoding =====
            if len(draft_tokens) == 0:
                total_fallback += 1
                reason = "M uninitialized" if adjacency_matrix[last_token_id][0].item() == 0 else "image_token hit"
                # print(f"[RECYCLING] [ITER {idx}] FALLBACK: {reason}, "
                #       f"last_token={last_token_id}({repr(self.tokenizer.decode([last_token_id]))}), "
                #       f"kv_len={kv_len_before}, generated={generated_so_far}")
                out = self.base_model(
                    last_tok,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                next_logits = out.logits[:, -1, :]

                # Update M for last_token
                adjacency_matrix[last_token_id] = next_logits.topk(matrix_top_k, dim=-1).indices.squeeze(0)

                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_id = torch.argmax(next_logits, dim=-1)
                # Cycle guard: if the chosen token continues a 2..20-gram repeat,
                # suppress it and pick the next-best token (up to 20 attempts).
                _fb_ctx = input_ids[0, max(input_len, input_ids.shape[1] - 50):].tolist()
                _fb_logits = next_logits.clone()
                for _attempt in range(20):
                    _fb_cand = next_id.item()
                    _fb_seq = _fb_ctx + [_fb_cand]
                    _fb_cyc = any(
                        len(_fb_seq) >= 2 * _n and _fb_seq[-_n:] == _fb_seq[-2 * _n:-_n]
                        for _n in range(2, 21)
                    )
                    if not _fb_cyc:
                        break
                    _fb_logits[0, _fb_cand] = float('-inf')
                    next_id = torch.argmax(_fb_logits, dim=-1)
                input_ids = torch.cat((input_ids, next_id.unsqueeze(0)), dim=-1)
                if hasattr(self, "current_length_data") and (not is_llava):
                    current_length_data.fill_(input_ids.shape[1] - 1)
                # print(f"[RECYCLING]   -> next_id={next_id.item()} "
                #       f"({repr(self.tokenizer.decode([next_id.item()]))}), "
                #       f"M[{last_token_id}] updated, "
                #       f"new kv_len={current_length_data[0].item()}")
                if return_acceptance_len:
                    acceptance_len_list.append(0)
                continue

            # ===== Draft found: speculative verification =====
            total_draft_found += 1
            draft = torch.tensor([draft_tokens], device=input_ids.device, dtype=input_ids.dtype)
            D = draft.shape[1]

            # print(f"[RECYCLING] [ITER {idx}] DRAFT: D={D}, "
            #       f"kv_len={kv_len_before}, generated={generated_so_far}")
            # print(f"[RECYCLING]   chain: {' -> '.join(draft_chain_info)}")
            # print(f"[RECYCLING]   draft_tokens = {draft_tokens} "
            #       f"(decoded: {repr(self.tokenizer.decode(draft_tokens))})")

            # Single forward pass: [last_tok, draft_0, ..., draft_D-1]
            chunk = torch.cat([last_tok, draft], dim=1)  # [1, 1+D]
            out = self.base_model(
                chunk,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits  # [1, 1+D, vocab]
            # kv_after_forward = current_length_data[0].item() if hasattr(self, "current_length_data") else -1
            # print(f"[RECYCLING]   forward pass done: chunk_len={chunk.shape[1]}, "
            #       f"kv_len after forward={kv_after_forward}")

            # Update adjacency matrix from ALL forwarded positions
            chunk_ids = chunk[0]  # [1+D]
            topk_vals = logits[0].topk(matrix_top_k, dim=-1).indices  # [1+D, top_k]
            adjacency_matrix[chunk_ids] = topk_vals
            n_updated = chunk_ids.unique().numel()
            # print(f"[RECYCLING]   M updated: {n_updated} unique tokens")

            # Verification: compare model predictions with draft tokens
            pred_ids = torch.argmax(logits, dim=-1)  # [1, 1+D]
            # pred_ids[0, 0] = model's top-1 prediction after last_tok -> should match draft[0]
            # pred_ids[0, i] = model's top-1 prediction after draft[i-1] -> should match draft[i]
            # pred_ids[0, D] = bonus token (prediction after draft[D-1])

            same = (pred_ids[:, :D] == draft)  # [1, D]
            mismatch = (~same).squeeze(0).nonzero(as_tuple=True)[0]
            if mismatch.numel() > 0:
                accept_len = int(mismatch[0].item())
            else:
                accept_len = D

            total_accepted += accept_len

            # Per-token verification log
            # print(f"[RECYCLING]   --- Per-token verification (accept_len={accept_len}/{D}) ---")
            # for i in range(min(D, 20)):  # cap at 20 to avoid too much output
            #     draft_tok = draft[0, i].item()
            #     pred_tok = pred_ids[0, i].item()
            #     match_sym = "MATCH" if draft_tok == pred_tok else "MISS"
            #     if i < accept_len:
            #         status = "ACCEPTED"
            #     elif i == accept_len:
            #         status = "REJECTED (first mismatch)"
            #     else:
            #         status = "SKIPPED"
                # print(f"[RECYCLING]     pos {i}: draft={draft_tok}({repr(self.tokenizer.decode([draft_tok]))}) "
                    #   f"vs pred={pred_tok}({repr(self.tokenizer.decode([pred_tok]))}) "
                    #   f"[{match_sym}] {status}")
            # if D > 20:
            #     print(f"[RECYCLING]     ... ({D - 20} more positions omitted)")
            bonus_tok = pred_ids[0, D].item()
            # print(f"[RECYCLING]     bonus token: {bonus_tok}({repr(self.tokenizer.decode([bonus_tok]))})")

            if return_acceptance_len:
                acceptance_len_list.append(accept_len)

            # Build tokens to add
            if accept_len == 0:
                tokens_to_add = pred_ids[:, :1]  # correction token (model's prediction)
            elif accept_len < D:
                tokens_to_add = torch.cat([
                    draft[:, :accept_len], 
                    pred_ids[:, accept_len:accept_len + 1]
                ], dim=1)
            else:  # accept_len == D (all accepted)
                tokens_to_add = torch.cat([
                    draft, 
                    pred_ids[:, D:D + 1]
                ], dim=1)

            # Post-acceptance cycle guard: scan tokens_to_add for 2..20-gram cycles.
            # When the base model confirms a repetitive draft (e.g. at temp=0 greedy),
            # truncate at the first cyclic position and replace with the next-best token.
            # Window=50 to detect long phrase cycles (~10-15 tokens common in LLaVA).
            _pa_ctx = input_ids[0, max(input_len, input_ids.shape[1] - 50):].tolist()
            _pa_safe = []
            _pa_rescue_tok = None
            for _pa_i, _pa_tok in enumerate(tokens_to_add[0].tolist()):
                _pa_seq = _pa_ctx + _pa_safe + [_pa_tok]
                _pa_cyc = any(
                    len(_pa_seq) >= 2 * _n and _pa_seq[-_n:] == _pa_seq[-2 * _n:-_n]
                    for _n in range(2, 21)
                )
                if _pa_cyc:
                    # Use logits at this position to find a non-cyclic escape token.
                    _pa_esc_logits = logits[0, _pa_i:_pa_i + 1, :].clone()
                    for _pa_attempt in range(20):
                        _pa_rescue_cand = torch.argmax(_pa_esc_logits, dim=-1).item()
                        _pa_seq2 = _pa_ctx + _pa_safe + [_pa_rescue_cand]
                        _pa_still_cyc = any(
                            len(_pa_seq2) >= 2 * _n2 and _pa_seq2[-_n2:] == _pa_seq2[-2 * _n2:-_n2]
                            for _n2 in range(2, 21)
                        )
                        if not _pa_still_cyc:
                            _pa_rescue_tok = _pa_rescue_cand
                            break
                        _pa_esc_logits[0, _pa_rescue_cand] = float('-inf')
                    else:
                        _pa_rescue_tok = _pa_rescue_cand  # use last attempt
                    break
                _pa_safe.append(_pa_tok)
            if _pa_rescue_tok is not None:
                _pa_safe.append(_pa_rescue_tok)
                tokens_to_add = torch.tensor(
                    [_pa_safe], device=input_ids.device, dtype=input_ids.dtype
                )

            # Cap tokens_to_add to not exceed max_new_tokens
            remaining = max_new_tokens - (input_ids.shape[1] - input_len)
            if tokens_to_add.shape[1] > remaining:
                # print(f"[RECYCLING]   CAPPING tokens_to_add from {tokens_to_add.shape[1]} to {remaining} "
                #       f"(max_new_tokens={max_new_tokens})")
                tokens_to_add = tokens_to_add[:, :remaining]

            input_ids = torch.cat([input_ids, tokens_to_add], dim=1)

            # Update KV cache length
            # After forward: KV has kv_before + 1 + D entries
            # We want: kv_before + 1 + accept_len (keep last_tok + accepted drafts)
            # Stale entries beyond that are ignored (current_length_data controls visibility)
            if hasattr(self, "current_length_data") and (not is_llava):
                current_length_data.fill_(input_ids.shape[1] - 1)

            # print(f"[RECYCLING]   tokens_added={tokens_to_add[0].tolist()} "
            #       f"(decoded: {repr(self.tokenizer.decode(tokens_to_add[0].tolist()))})")
            # print(f"[RECYCLING]   accept_len={accept_len}/{D}, "
            #       f"new kv_len={current_length_data[0].item()}, "
            #       f"total_generated={input_ids.shape[1] - input_len}")

        # ===== Summary =====
        new_token = input_ids.shape[1] - input_len
        _last_toks = self.tokenizer.decode(input_ids[0, -min(20, new_token):].tolist(), skip_special_tokens=False)
        # print(
        #     f"[REC] new_tok={new_token:4d} | stop={_break_reason:15s} | "
        #     f"iters={idx+1:4d} fallback={total_fallback:4d} "
        #     f"cycle_cut={_debug_cycle:4d} uninit_cut={_debug_uninit:4d} img_cut={_debug_img:3d} | "
        #     f"last20={repr(_last_toks)}"
        # )
        # print(f"[RECYCLING]   total iterations: {idx + 1 if idx > 0 or new_token > 0 else 0}")
        # print(f"[RECYCLING]   new_tokens: {new_token}")
        # print(f"[RECYCLING]   draft attempts: {total_draft_attempts}")
        # print(f"[RECYCLING]   drafts found: {total_draft_found}, fallbacks: {total_fallback}")
        # print(f"[RECYCLING]   total accepted draft tokens: {total_accepted}")
        # if total_draft_found > 0:
        #     print(f"[RECYCLING]   avg acceptance per draft: {total_accepted / total_draft_found:.2f}")
        # M coverage stats
        # n_nonzero_final = (adjacency_matrix.sum(dim=1) != 0).sum().item()
        # print(f"[RECYCLING]   M coverage: {n_nonzero_final}/{vocab_size} tokens "
        #       f"({100 * n_nonzero_final / vocab_size:.2f}%)")
        # final_kv = current_length_data[0].item() if hasattr(self, "current_length_data") else -1
        # print(f"[RECYCLING]   final kv_len: {final_kv}, "
        #       f"final input_ids_len: {input_ids.shape[1]}")
        # print(f"[RECYCLING]   SANITY CHECK: input_ids_len - 1 == kv_len? "
        #       f"{input_ids.shape[1] - 1 == final_kv}")

        # generated_ids = input_ids[0, input_len:].tolist()
        # generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print(f"[RECYCLING]   generated text (first 200 chars): {repr(generated_text[:200])}")

        # if self.tokenizer.eos_token_id is not None:
        #     eos_positions = [i for i, tid in enumerate(generated_ids) if tid == self.tokenizer.eos_token_id]
            # if eos_positions:
            #     print(f"[RECYCLING]   EOS found at positions: {eos_positions}")
            #     if eos_positions[0] < len(generated_ids) - 1:
            #         print(f"[RECYCLING]   WARNING: tokens exist AFTER first EOS! "
            #               f"({len(generated_ids) - eos_positions[0] - 1} extra tokens)")
        # if return_acceptance_len:
        #     print(f"[RECYCLING]   acceptance_len list: {acceptance_len_list}")
        # print(f"[RECYCLING] {'=' * 50}")

        outputs = (input_ids,)

        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_len_list,)
        if return_decode_time:
            outputs += (0.0,)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def forward(self, input_ids, **kwargs):
        return self.specgenerate(input_ids, **kwargs)

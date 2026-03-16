import torch

from method.msd.ea_model import (
    EaModel as _BaseEaModel,
    _collect_stop_token_ids,
    _normalize_token_ids,
    _truncate_at_first_stop_token,
)
from method.msd.kv_cache import initialize_past_key_values
from method.msd.utils import (
    evaluate_posterior,
    initialize_tree,
    prepare_logits_processor,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
)


def _has_repetitive_tail(
    generated_ids: torch.Tensor,
    ngram_min: int = 1,
    ngram_max: int = 64,
    min_repeats: int = 2,
    min_generated_tokens: int = 20,
) -> bool:
    """Return True when the generated tail contains consecutive repeated n-gram loops."""
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


class EaModel(_BaseEaModel):
    def _build_visual_key_mask(self, input_ids):
        if input_ids is None or input_ids.numel() == 0:
            return None

        if self._is_qwen2vl:
            image_token_id = getattr(self.base_model.config, "image_token_id", None)
            video_token_id = getattr(self.base_model.config, "video_token_id", None)
            mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
            if image_token_id is not None:
                mask = mask | (input_ids[0] == image_token_id)
            if video_token_id is not None:
                mask = mask | (input_ids[0] == video_token_id)
            return mask

        if self._is_hf_llava:
            image_token_id = getattr(
                self.base_model.config,
                "image_token_index",
                getattr(self.base_model.config, "image_token_id", None),
            )
            if image_token_id is None:
                return None
            return (input_ids[0] == image_token_id).to(torch.bool)

        return None

    def _visual_attention_ratio(self, attentions, visual_key_mask, query_index=-1):
        if attentions is None or visual_key_mask is None:
            return 0.0
        if isinstance(attentions, (tuple, list)):
            if len(attentions) == 0:
                return 0.0
            attn = attentions[-1]
        else:
            attn = attentions
        if attn is None or attn.numel() == 0:
            return 0.0

        q_len = attn.shape[2]
        if query_index < 0:
            query_index = q_len + query_index
        query_index = max(0, min(query_index, q_len - 1))
        query_attn = attn[:, :, query_index : query_index + 1, :]

        key_len = query_attn.shape[-1]
        if visual_key_mask.shape[-1] < key_len:
            pad = torch.zeros(
                key_len - visual_key_mask.shape[-1],
                dtype=visual_key_mask.dtype,
                device=visual_key_mask.device,
            )
            key_mask = torch.cat([visual_key_mask, pad], dim=0)
        else:
            key_mask = visual_key_mask[:key_len]
        key_mask = key_mask.to(query_attn.device, query_attn.dtype).view(1, 1, 1, -1)

        ratio = (query_attn * key_mask).sum(dim=-1).mean()
        return float(ratio.item())

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
        skip_when_visual=True,
        visual_attn_threshold=0.35,
        no_op_on_visual_hit=False,
        visual_probe_interval=1,
        is_llama3=False,
    ):
        # Keep HF-LLaVA path unchanged for now; vskip is applied on MSD non-HF path.
        if self._is_hf_llava:
            return super().msdgenerate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                log=log,
            )

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
        if is_llama3:
            stop_token_ids.update(
                _normalize_token_ids(
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                )
            )
        reset_tree_mode(self)

        visual_key_mask = self._build_visual_key_mask(input_ids)
        if visual_key_mask is None:
            visual_key_mask = torch.zeros(
                input_ids.shape[1], dtype=torch.bool, device=input_ids.device
            )
        else:
            visual_key_mask = visual_key_mask.to(input_ids.device)
        use_visual_gate = skip_when_visual and bool(visual_key_mask.any().item())
        visual_attn_threshold = float(visual_attn_threshold)
        visual_probe_interval = max(int(visual_probe_interval), 1)

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            _,
            hidden_state,
            sample_token,
        ) = initialize_tree(
            input_ids, self, past_key_values, logits_processor, inputs_embeds
        )
        new_token = 0
        idx = 0

        for idx in range(max_length):
            self._get_language_model().tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            do_probe = use_visual_gate and (idx % visual_probe_interval == 0)
            logits, hidden_state_new, tree_outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                output_attentions=do_probe,
            )
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accept_length = int(accept_length)
            self.acclen += accept_length
            self.accnum += 1
            prev_retrieve_indices = retrieve_indices

            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                hidden_state,
                sample_token,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
            )

            if (
                do_probe
                and new_token < max_new_tokens
                and input_ids.shape[1] < max_length
            ):
                query_index = int(
                    prev_retrieve_indices[best_candidate, accept_length].item()
                )
                visual_ratio = self._visual_attention_ratio(
                    getattr(tree_outputs, "attentions", None),
                    visual_key_mask,
                    query_index=query_index,
                )
                if visual_ratio >= visual_attn_threshold:
                    if no_op_on_visual_hit:
                        pass
                    else:
                        if logits_processor is not None:
                            forced_token = torch.multinomial(sample_p, 1).view(1, 1)
                        else:
                            forced_token = torch.argmax(sample_p).view(1, 1)
                        forced_token = forced_token.to(input_ids.device)

                        # Continue with one target-only token when visual attention is high.
                        reset_tree_mode(self)
                        force_out = self.base_model(
                            input_ids=forced_token,
                            use_cache=True,
                            past_key_values=past_key_values,
                            output_attentions=False,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        input_ids = torch.cat((input_ids, forced_token), dim=1)
                        new_token += 1
                        visual_key_mask = torch.cat(
                            [
                                visual_key_mask,
                                torch.zeros(
                                    1,
                                    dtype=torch.bool,
                                    device=visual_key_mask.device,
                                ),
                            ],
                            dim=0,
                        )
                        current_length_data.fill_(input_ids.shape[1])

                        forced_logits = force_out.logits[:, -1, :]
                        if logits_processor is not None:
                            forced_logits = logits_processor(None, forced_logits)
                            forced_probs = torch.softmax(forced_logits, dim=-1)
                            seed_token = torch.multinomial(forced_probs, 1)
                        else:
                            seed_token = torch.argmax(
                                forced_logits, dim=-1, keepdim=True
                            )
                        seed_token = seed_token.to(input_ids.device)
                        seed_input_ids = torch.cat((input_ids, seed_token), dim=1)
                        forced_hidden = force_out.hidden_states[-1]
                        (
                            draft_tokens,
                            retrieve_indices,
                            tree_mask,
                            tree_position_ids,
                        ) = self.ea_layer.topK_genrate(
                            forced_hidden,
                            seed_input_ids,
                            self._get_lm_head(),
                            logits_processor,
                        )
                        sample_token = seed_token

            input_ids, new_token, hit_stop = _truncate_at_first_stop_token(
                input_ids, input_len, stop_token_ids
            )
            if hit_stop:
                break
            if new_token >= max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
            if _has_repetitive_tail(input_ids[0, input_len:]):
                break

        if not log:
            return input_ids
        return input_ids, new_token, idx

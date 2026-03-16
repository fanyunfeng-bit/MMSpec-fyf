import time

import torch

from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import (
    SpecModel as _BaseSpecModel,
    _collect_stop_token_ids,
    _has_stop_token,
    _normalize_token_ids,
)
from method.vispec.utils import (
    evaluate_posterior,
    initialize_tree,
    prepare_logits_processor,
    reset_tree_mode,
    tree_decoding,
    update_inference_inputs,
)


class SpecModel(_BaseSpecModel):
    def _sample_token(self, logits, temperature):
        if temperature > 1e-5:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def _build_visual_key_mask(self, input_ids):
        if input_ids is None or input_ids.numel() == 0:
            return None

        arch = self.base_model.config.architectures[0]
        if arch in ("LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration"):
            image_token_id = getattr(
                self.base_model.config,
                "image_token_index",
                getattr(self.base_model.config, "image_token_id", None),
            )
            if image_token_id is None:
                return None
            return (input_ids[0] == image_token_id).to(torch.bool)

        if arch == "Qwen2_5_VLForConditionalGeneration":
            image_token_id = getattr(self.base_model.config, "image_token_id", None)
            video_token_id = getattr(self.base_model.config, "video_token_id", None)
            mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
            if image_token_id is not None:
                mask = mask | (input_ids[0] == image_token_id)
            if video_token_id is not None:
                mask = mask | (input_ids[0] == video_token_id)
            return mask

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

    def _collect_prefill_kwargs(self, **kwargs):
        arch = self.base_model.config.architectures[0]
        if arch in ("LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration"):
            allowed = (
                "pixel_values",
                "image_sizes",
                "vision_feature_layer",
                "vision_feature_select_strategy",
            )
        elif arch == "Qwen2_5_VLForConditionalGeneration":
            allowed = (
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "second_per_grid_ts",
            )
        else:
            allowed = tuple()

        prefill_kwargs = {}
        for key in allowed:
            value = kwargs.get(key)
            if value is not None:
                prefill_kwargs[key] = value
        return prefill_kwargs

    @torch.no_grad()
    def _target_only_generate(
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
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You must specify at least one of input_ids or inputs_embeds"
            )
        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = None

        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        if is_llama3:
            stop_token_ids.update(
                _normalize_token_ids(
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                )
            )

        input_ids = input_ids.clone()
        input_len = input_ids.shape[1]
        acceptance_len = [] if return_acceptance_len else None

        decode_start = None
        if return_decode_time and torch.cuda.is_available():
            torch.cuda.synchronize()
            decode_start = time.time()

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
            except Exception:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        if max_new_tokens <= 0 or input_ids.shape[1] >= max_length:
            outputs = (input_ids,)
            if log:
                outputs += (0, 0)
            if return_acceptance_len:
                outputs += (acceptance_len if acceptance_len is not None else [])
            if return_decode_time:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_time = 0.0 if decode_start is None else (time.time() - decode_start)
                outputs += (decode_time,)
            return outputs[0] if len(outputs) == 1 else outputs

        prefill_kwargs = self._collect_prefill_kwargs(**kwargs)
        prefill_out = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            **prefill_kwargs,
        )

        next_id = self._sample_token(prefill_out.logits[:, -1, :], temperature)
        input_ids = torch.cat((input_ids, next_id.unsqueeze(0).to(input_ids.device)), dim=1)
        generated = 1

        is_llava = self.base_model.config.architectures[0] in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        )
        if not is_llava:
            current_length_data.fill_(input_ids.shape[1] - 1)
        if acceptance_len is not None:
            acceptance_len.append(0)

        idx = 0
        while generated < max_new_tokens and input_ids.shape[1] < max_length:
            if _has_stop_token(input_ids[0, input_len:], stop_token_ids):
                break

            idx += 1
            out = self.base_model(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
            next_id = self._sample_token(out.logits[:, -1, :], temperature)
            input_ids = torch.cat((input_ids, next_id.unsqueeze(0).to(input_ids.device)), dim=1)
            generated += 1
            if not is_llava:
                current_length_data.fill_(input_ids.shape[1] - 1)
            if acceptance_len is not None:
                acceptance_len.append(0)

        new_token = max(int(input_ids.shape[1] - input_len), 0)
        idx = max(min(idx, new_token - 1), 0) if new_token > 0 else 0

        outputs = (input_ids,)
        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_len,)
        if return_decode_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_time = 0.0 if decode_start is None else (time.time() - decode_start)
            outputs += (decode_time,)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

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
        skip_when_visual=True,
        visual_attn_threshold=0.35,
        no_op_on_visual_hit=False,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        if is_llama3:
            stop_token_ids.update(
                _normalize_token_ids(
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                )
            )
        arch = self.base_model.config.architectures[0]
        max_length = max_length - self.spec_layer.total_tokens - 10
        visual_attn_threshold = float(visual_attn_threshold)

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.spec_layer.reset_kv()

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
            except Exception:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        cache_max_len = past_key_values_data[0].shape[-2] if past_key_values_data else None

        embed_weights = None
        special_image_mask = None
        if arch == "LlavaNextForConditionalGeneration":
            vision_feature_layer = kwargs.get("vision_feature_layer")
            vision_feature_select_strategy = kwargs.get(
                "vision_feature_select_strategy"
            )
            pixel_values = kwargs.get("pixel_values")
            image_sizes = kwargs.get("image_sizes")

            vision_feature_layer = (
                vision_feature_layer
                if vision_feature_layer is not None
                else self.base_model.config.vision_feature_layer
            )
            vision_feature_select_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else self.base_model.config.vision_feature_select_strategy
            )

            if pixel_values is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

            if pixel_values is not None and pixel_values.size(0) > 0:
                image_features = self.base_model.get_image_features(
                    pixel_values,
                    image_sizes,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )
                image_features, _ = self.base_model.pack_image_features(
                    image_features,
                    image_sizes,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_newline=self.base_model.image_newline,
                )
                special_image_mask = (
                    input_ids == self.base_model.config.image_token_index
                ).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )
                if inputs_embeds[special_image_mask].numel() != image_features.numel():
                    n_image_tokens = (
                        input_ids == self.base_model.config.image_token_index
                    ).sum()
                    n_image_features = image_features.shape[0]
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_features = image_features.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_features
                )
                special_image_mask = special_image_mask[..., 0]
        elif arch == "LlavaForConditionalGeneration":
            vision_feature_layer = kwargs.get("vision_feature_layer")
            vision_feature_select_strategy = kwargs.get(
                "vision_feature_select_strategy"
            )
            pixel_values = kwargs.get("pixel_values")
            image_sizes = kwargs.get("image_sizes")

            vision_feature_layer = (
                vision_feature_layer
                if vision_feature_layer is not None
                else self.base_model.config.vision_feature_layer
            )
            vision_feature_select_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else self.base_model.config.vision_feature_select_strategy
            )

            if pixel_values is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

            if pixel_values is not None and pixel_values.size(0) > 0:
                image_features = self.base_model.get_image_features(
                    pixel_values=pixel_values,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_sizes=image_sizes,
                )
                if isinstance(image_features, (list, tuple)):
                    image_features = torch.cat(image_features, dim=0)

                image_token_id = getattr(
                    self.base_model.config,
                    "image_token_id",
                    getattr(self.base_model.config, "image_token_index", None),
                )
                if image_token_id is None:
                    raise ValueError("Cannot find image token id/index in Llava config")

                special_image_mask = input_ids == image_token_id
                n_image_tokens = int(special_image_mask.sum().item())
                expanded_image_mask = special_image_mask.unsqueeze(-1).expand_as(
                    inputs_embeds
                )
                expanded_image_mask = expanded_image_mask.to(inputs_embeds.device)

                if inputs_embeds[expanded_image_mask].numel() != image_features.numel():
                    if image_features.dim() == 2:
                        n_image_features = int(image_features.shape[0])
                    else:
                        n_image_features = int(
                            image_features.shape[0] * image_features.shape[1]
                        )
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                image_features = image_features.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    expanded_image_mask, image_features
                )
        elif arch == "Qwen2_5_VLForConditionalGeneration":
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")
            pixel_values_videos = kwargs.get("pixel_values_videos")
            video_grid_thw = kwargs.get("video_grid_thw")

            if inputs_embeds is None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.base_model.visual.dtype)
                    image_embeds = self.base_model.visual(
                        pixel_values, grid_thw=image_grid_thw
                    )
                    n_image_tokens = (
                        (input_ids == self.base_model.config.image_token_id)
                        .sum()
                        .item()
                    )
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == self.base_model.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        image_mask, image_embeds
                    )
                    special_image_mask = mask

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(
                        self.base_model.visual.dtype
                    )
                    video_embeds = self.base_model.visual(
                        pixel_values_videos, grid_thw=video_grid_thw
                    )
                    n_video_tokens = (
                        (input_ids == self.base_model.config.video_token_id)
                        .sum()
                        .item()
                    )
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )

                    mask = input_ids == self.base_model.config.video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    video_mask = mask_expanded.to(inputs_embeds.device)

                    video_embeds = video_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        video_mask, video_embeds
                    )
                    if special_image_mask is None:
                        special_image_mask = mask
                    else:
                        special_image_mask = special_image_mask | mask

        input_len = input_ids.shape[1]
        visual_key_mask = self._build_visual_key_mask(input_ids)
        if visual_key_mask is None:
            visual_key_mask = torch.zeros(
                input_len, dtype=torch.bool, device=input_ids.device
            )
        else:
            visual_key_mask = visual_key_mask.to(input_ids.device)
        use_visual_gate = skip_when_visual and bool(visual_key_mask.any().item())
        reset_tree_mode(self)
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            _,
            hidden_state,
            sample_token,
        ) = initialize_tree(
            input_ids,
            self,
            past_key_values,
            logits_processor,
            inputs_embeds,
            embed_weights,
            image_mask=special_image_mask,
            **kwargs,
        )
        new_token = 0

        if return_acceptance_len:
            acceptance_len = []
        if return_decode_time:
            torch.cuda.synchronize()
            start_time = time.time()

        for idx in range(max_length):
            kv_cursor = input_ids.shape[1]
            if current_length_data is not None and current_length_data.numel() > 0:
                kv_cursor = max(kv_cursor, int(current_length_data.max().item()))
            if (
                cache_max_len is not None
                and kv_cursor + draft_tokens.shape[1] > cache_max_len
            ):
                break

            if not hasattr(self.base_model, "language_model"):
                self.base_model.model.tree_mask = tree_mask
            else:
                self.base_model.language_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_state_new, tree_outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                current_length_data=current_length_data,
                output_attentions=use_visual_gate,
            )
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accept_length = int(accept_length)
            remaining_tokens = max_new_tokens - new_token
            if remaining_tokens <= 0:
                break
            accept_length = min(accept_length, remaining_tokens - 1)
            if cache_max_len is not None:
                cache_remaining = cache_max_len - kv_cursor
                if cache_remaining <= 0:
                    break
                accept_length = min(accept_length, cache_remaining - 1)
            if accept_length < 0:
                break
            if return_acceptance_len:
                acceptance_len.append(int(accept_length))
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
                cache_base_len=kv_cursor,
            )

            if (
                use_visual_gate
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
                        kv_force = input_ids.shape[1]
                        if current_length_data is not None and current_length_data.numel() > 0:
                            kv_force = max(kv_force, int(current_length_data.max().item()))
                        can_force_decode = cache_max_len is None or kv_force < cache_max_len
                        if can_force_decode:
                            if logits_processor is not None:
                                forced_token = torch.multinomial(sample_p, 1).view(1, 1)
                            else:
                                forced_token = torch.argmax(sample_p).view(1, 1)
                            forced_token = forced_token.to(input_ids.device)

                            # Continue with one target token when visual attention is high.
                            reset_tree_mode(self)
                            force_out = self.base_model(
                                input_ids=forced_token,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                output_attentions=False,
                                output_hidden_states=True,
                            )
                            input_ids = torch.cat((input_ids, forced_token), dim=1)
                            new_token += 1
                            visual_key_mask = torch.cat(
                                [
                                    visual_key_mask,
                                    torch.zeros(
                                        1, dtype=torch.bool, device=visual_key_mask.device
                                    ),
                                ],
                                dim=0,
                            )
                            if arch not in (
                                "LlavaForConditionalGeneration",
                                "LlavaNextForConditionalGeneration",
                            ):
                                current_length_data.fill_(input_ids.shape[1])
                            if return_acceptance_len:
                                acceptance_len.append(0)

                            forced_logits = force_out.logits[:, -1, :]
                            if logits_processor is not None:
                                forced_logits = logits_processor(None, forced_logits)
                                forced_probs = torch.softmax(forced_logits, dim=-1)
                                seed_token = torch.multinomial(forced_probs, 1)
                            else:
                                seed_token = torch.argmax(
                                    forced_logits, dim=-1, keepdim=True
                                )

                            seed_input_ids = torch.cat(
                                (input_ids, seed_token.to(input_ids.device)), dim=1
                            )
                            forced_hidden = force_out.hidden_states[-1]
                            try:
                                (
                                    draft_tokens,
                                    retrieve_indices,
                                    tree_mask,
                                    tree_position_ids,
                                ) = self.spec_layer.topK_genrate(
                                    forced_hidden,
                                    seed_input_ids,
                                    self.base_model.lm_head,
                                    logits_processor,
                                )
                            except Exception:
                                (
                                    draft_tokens,
                                    retrieve_indices,
                                    tree_mask,
                                    tree_position_ids,
                                ) = self.spec_layer.topK_genrate(
                                    forced_hidden,
                                    seed_input_ids,
                                    self.base_model.language_model.lm_head,
                                    logits_processor,
                                )
                            sample_token = seed_token

            if _has_stop_token(input_ids[0, input_len:], stop_token_ids):
                break
            if new_token >= max_new_tokens:
                break

        outputs = (input_ids,)
        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_len,)
        if return_decode_time:
            torch.cuda.synchronize()
            end_time = time.time()
            outputs += (end_time - start_time,)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

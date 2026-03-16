import copy
import json
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from .cnets_medusa import Model
from .configs import EConfig
from .kv_cache import initialize_past_key_values
from .utils import *

# ── Medusa runtime debug logging ──────────────────────────────────────
_MEDUSA_DEBUG = os.environ.get("MEDUSA_DEBUG", "0") == "1"
_MEDUSA_DEBUG_MAX_ITERS = int(os.environ.get("MEDUSA_DEBUG_MAX_ITERS", "999999"))

def _dlog(*args, **kwargs):
    """Print only when MEDUSA_DEBUG=1."""
    if _MEDUSA_DEBUG:
        print("[MEDUSA]", *args, **kwargs, flush=True)


def _get_model_class(arch_type):
    if arch_type == "LlamaForCausalLM":
        from .modeling_llama_kv import LlamaForCausalLM
        return LlamaForCausalLM
    elif arch_type == "Qwen2ForCausalLM":
        from .modeling_qwen2_kv import LlamaForCausalLM
        return LlamaForCausalLM
    elif arch_type == "MixtralForCausalLM":
        from .modeling_mixtral_kv import MixtralForCausalLM
        return MixtralForCausalLM
    elif arch_type == "LlavaNextForConditionalGeneration":
        from method.llava_adapter import CustomLlavaNextForConditionalGeneration
        return CustomLlavaNextForConditionalGeneration
    elif arch_type == "LlavaForConditionalGeneration":
        from method.llava_adapter import CustomLlavaForConditionalGeneration
        return CustomLlavaForConditionalGeneration
    elif arch_type == "Qwen2_5_VLForConditionalGeneration":
        from .modeling_qwen2_5_vl_kv import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        raise NotImplementedError(f"Model type {arch_type} is not supported.")


def _normalize_token_ids(token_ids):
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        raw_ids = token_ids
    else:
        raw_ids = [token_ids]
    normalized = []
    for token_id in raw_ids:
        if token_id is None:
            continue
        if isinstance(token_id, torch.Tensor):
            if token_id.numel() == 1:
                token_id = token_id.item()
            else:
                normalized.extend(int(x) for x in token_id.view(-1).tolist())
                continue
        try:
            normalized.append(int(token_id))
        except (TypeError, ValueError):
            continue
    return normalized


def _collect_stop_token_ids(tokenizer, model):
    stop_ids = set(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        stop_ids.update(
            _normalize_token_ids(getattr(generation_config, "eos_token_id", None))
        )
    config = getattr(model, "config", None)
    if config is not None:
        stop_ids.update(_normalize_token_ids(getattr(config, "eos_token_id", None)))
    stop_ids.update(_normalize_token_ids(getattr(tokenizer, "eod_id", None)))
    return stop_ids


def _has_stop_token(generated_ids, stop_ids):
    if not stop_ids:
        return False
    if isinstance(generated_ids, torch.Tensor):
        generated_set = set(int(x) for x in generated_ids.view(-1).tolist())
    else:
        generated_set = set(int(x) for x in generated_ids)
    return bool(generated_set & stop_ids)


class SpecModel(nn.Module):

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
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        if hasattr(base_model, "language_model"):
            base_model = base_model.language_model

        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )
        config = EConfig.from_pretrained(spec_model_path)
        with open(spec_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        self.spec_layer = Model(
            config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold,
        )

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.spec_layer.diff_device = True
            if not low_memory:
                self.spec_layer.headweight = base_model.lm_head.weight.clone().to(
                    device
                )
            else:
                self.spec_layer.layer_device = device

        else:
            self.spec_layer.diff_device = False
        if spec_layer_state_dict is not None:
            # Shape validation (防呆校验): every checkpoint key that exists in
            # the model must have exactly the same shape — no silent reshape.
            model_sd = self.spec_layer.state_dict()
            for k, v in spec_layer_state_dict.items():
                if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape):
                    raise RuntimeError(
                        f"[medusa] Shape mismatch for '{k}': "
                        f"checkpoint={tuple(v.shape)}, model={tuple(model_sd[k].shape)}. "
                        f"Please check that the spec config matches the checkpoint."
                    )
            missing_keys, unexpected_keys = self.spec_layer.load_state_dict(
                spec_layer_state_dict, strict=False
            )
            if len(missing_keys) > 0:
                print(f"missing_keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"unexpected_keys: {unexpected_keys}")
        self.spec_layer.to(self.base_model.dtype).to(device)
        self.spec_layer.init_tree()
        # print(f"[DEBUG] spec_layer device: {device}, dtype: {self.base_model.dtype}")

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
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        model_class = _get_model_class(Type)
        base_model = model_class.from_pretrained(base_model_path, **kwargs)

        # ── Resolve config.json ─────────────────────────────────────────
        configpath = os.path.join(spec_model_path, "config.json")
        if not os.path.exists(configpath):
            # Try HF hub – top-level first, then state_19/ subfolder
            for hf_path in ("config.json", "state_19/config.json"):
                try:
                    configpath = hf_hub_download(spec_model_path, hf_path)
                    break
                except Exception:
                    continue
            else:
                raise FileNotFoundError(
                    f"Cannot find config.json for medusa checkpoint in {spec_model_path}"
                )

        # ── Resolve weight file(s) ──────────────────────────────────────
        spec_layer_state_dict = None

        # 1) Try local paths
        for fname in ("pytorch_model.bin", "pytorch_model_fsdp.bin",
                       "model.safetensors"):
            local_path = os.path.join(spec_model_path, fname)
            if os.path.exists(local_path):
                if fname.endswith(".bin"):
                    spec_layer_state_dict = torch.load(
                        local_path, map_location=base_model.device
                    )
                else:
                    from safetensors.torch import load_file
                    spec_layer_state_dict = load_file(local_path)
                break

        # 2) Try HF hub (top-level, then state_19/ subfolder)
        if spec_layer_state_dict is None:
            from safetensors.torch import load_file as st_load_file
            # Candidate weight files in priority order
            hf_candidates = [
                "pytorch_model.bin",
                "model.safetensors",
                "state_19/pytorch_model.bin",
                "state_19/model.safetensors",
            ]
            for hf_fname in hf_candidates:
                try:
                    dl_path = hf_hub_download(spec_model_path, hf_fname)
                    if hf_fname.endswith(".bin"):
                        spec_layer_state_dict = torch.load(
                            dl_path, map_location=base_model.device
                        )
                    else:
                        spec_layer_state_dict = st_load_file(dl_path)
                    # Check for multi-shard safetensors (model_1.safetensors, …)
                    if hf_fname.endswith(".safetensors"):
                        prefix = hf_fname.rsplit("/", 1)
                        prefix = prefix[0] + "/" if len(prefix) > 1 else ""
                        for shard_idx in range(1, 100):
                            shard_name = f"{prefix}model_{shard_idx}.safetensors"
                            try:
                                shard_path = hf_hub_download(spec_model_path, shard_name)
                                shard_sd = st_load_file(shard_path)
                                spec_layer_state_dict.update(shard_sd)
                                print(f"[medusa] Merged shard: {shard_name}")
                            except Exception:
                                break
                    break
                except Exception:
                    continue
            else:
                raise FileNotFoundError(
                    f"Cannot find weight file for medusa checkpoint in {spec_model_path}"
                )

        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            spec_layer_state_dict,
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(
                    0, model.config.vocab_size - 200, (1, length)
                ).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.spec_layer.total_tokens = total_token - 1

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
        _dlog(f"  [forward] input_ids={input_ids.shape if input_ids is not None else None}, "
              f"inputs_embeds={inputs_embeds.shape if inputs_embeds is not None else None}, "
              f"position_ids={position_ids.shape if position_ids is not None else None}, "
              f"output_orig={output_orig}, "
              f"kv_len={past_key_values[0][0].current_length.item() if past_key_values else 'None'}")
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
        max_length = max_length - self.spec_layer.total_tokens - 10

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
            except:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        embed_weights = None
        special_image_mask = None
        if (
            self.base_model.config.architectures[0]
            == "LlavaNextForConditionalGeneration"
        ):
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

                image_features, feature_lens = self.base_model.pack_image_features(
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

        elif (
            self.base_model.config.architectures[0]
            == "Qwen2_5_VLForConditionalGeneration"
        ):
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")

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

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        _dlog(f"===== specgenerate START =====")
        _dlog(f"  input_ids.shape={input_ids.shape}, input_len={input_len}")
        _dlog(f"  temperature={temperature}, max_new_tokens={max_new_tokens}, max_length={max_length}")
        _dlog(f"  logits_processor={'None (greedy)' if logits_processor is None else 'set'}")
        _dlog(f"  total_tokens={self.spec_layer.total_tokens}, depth={self.spec_layer.depth}, top_k={self.spec_layer.top_k}")
        _dlog(f"  kv_len_before_init={past_key_values[0][0].current_length.item()}")

        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            logits,
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

        _dlog(f"  [after initialize_tree]")
        _dlog(f"    draft_tokens.shape={draft_tokens.shape}, values={draft_tokens[0,:10].tolist()}...")
        _dlog(f"    retrieve_indices.shape={retrieve_indices.shape}")
        _dlog(f"    tree_mask.shape={tree_mask.shape}")
        _dlog(f"    tree_position_ids={tree_position_ids.tolist()}")
        _dlog(f"    sample_token={sample_token.tolist()}")
        _dlog(f"    kv_len_after_init={past_key_values[0][0].current_length.item()}")

        if return_acceptance_len:
            acceptance_len = []
        if return_decode_time:
            torch.cuda.synchronize()
            start_time = time.time()

        for idx in range(max_length):
            _should_log = _MEDUSA_DEBUG and idx < _MEDUSA_DEBUG_MAX_ITERS

            if _should_log:
                _dlog(f"\n  ── iter {idx} ── new_token={new_token}, input_ids.shape={input_ids.shape}")
                _dlog(f"    kv_len={past_key_values[0][0].current_length.item()}")
                _dlog(f"    draft_tokens={draft_tokens[0].tolist()}")
                _dlog(f"    tree_position_ids={tree_position_ids.tolist()}")
                _dlog(f"    tree_mask.shape={tree_mask.shape}, sum_per_row={tree_mask[0,0].sum(dim=-1).int().tolist()}")

            if not hasattr(self.base_model, "language_model"):
                self.base_model.model.tree_mask = tree_mask
            else:
                self.base_model.language_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            if _should_log:
                _dlog(f"    [after tree_decoding] logits.shape={logits.shape}")
                _dlog(f"    kv_len_after_decode={past_key_values[0][0].current_length.item()}")
                # Show argmax of logits for each candidate path
                argmax_tokens = torch.argmax(logits[:, :-1], dim=-1)
                _dlog(f"    logits argmax (per candidate, per pos): shape={argmax_tokens.shape}")
                for ci in range(min(argmax_tokens.shape[0], 5)):
                    _dlog(f"      cand[{ci}] argmax={argmax_tokens[ci].tolist()}")

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            if _should_log:
                _dlog(f"    candidates.shape={candidates.shape}")
                for ci in range(min(candidates.shape[0], 5)):
                    _dlog(f"      cand[{ci}]={candidates[ci].tolist()}")

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            if _should_log:
                _dlog(f"    best_candidate={best_candidate.item()}, accept_length={accept_length}")
                accepted_tokens = candidates[best_candidate, :accept_length+1].tolist()
                _dlog(f"    accepted_tokens={accepted_tokens}")
                if logits_processor is None:
                    sample_token_val = torch.argmax(sample_p).item()
                    _dlog(f"    sample_p argmax (next token)={sample_token_val}")
                    _dlog(f"    sample_p top5_vals={torch.topk(sample_p, 5).values.tolist()}")
                    _dlog(f"    sample_p top5_ids={torch.topk(sample_p, 5).indices.tolist()}")

            remaining_tokens = max_new_tokens - new_token
            if remaining_tokens <= 0:
                break
            accept_length = min(accept_length, remaining_tokens - 1)
            if return_acceptance_len:
                acceptance_len.append(int(accept_length))
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

            if _should_log:
                _dlog(f"    [after update] new_token={new_token}, input_ids.shape={input_ids.shape}")
                _dlog(f"    kv_len_after_update={past_key_values[0][0].current_length.item()}")
                _dlog(f"    last {min(10, new_token)} generated tokens={input_ids[0, max(input_len, input_ids.shape[1]-10):].tolist()}")

            if _has_stop_token(input_ids[0, input_len:], stop_token_ids):
                if _should_log:
                    _dlog(f"    STOP: stop token found")
                break
            if new_token >= max_new_tokens:
                if _should_log:
                    _dlog(f"    STOP: max_new_tokens reached")
                break

        _dlog(f"\n===== specgenerate END =====")
        _dlog(f"  total iters={idx+1 if 'idx' in dir() else 0}, new_token={new_token}")
        _dlog(f"  generated_ids={input_ids[0, input_len:].tolist()}")
        try:
            _dlog(f"  generated_text={self.tokenizer.decode(input_ids[0, input_len:], skip_special_tokens=True)}")
        except:
            pass

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
        else:
            return outputs

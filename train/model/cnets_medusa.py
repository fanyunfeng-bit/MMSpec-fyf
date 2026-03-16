import copy
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN

try:
    from train.model.configs import EConfig
    from train.model.utils_c import *
    from train.model.choices import *
except:
    from train.model.configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor


class ResBlock(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class Model(nn.Module):

    def __init__(
        self,
        config,
        load_emb=False,
        path=None,
        bias=True,
        total_tokens=30,
        depth=3,
        top_k=8,
        threshold=1.0,
    ):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        if load_emb:
            import json

            from safetensors import safe_open
            from transformers import AutoModelForImageTextToText

            try:
                try:
                    with open(
                        os.path.join(path, "model.safetensors.index.json"), "r"
                    ) as f:
                        index_json = json.loads(f.read())
                        emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                    with safe_open(
                        os.path.join(path, emb_path), framework="pt", device="cpu"
                    ) as f:
                        tensor_slice = f.get_slice("model.embed_tokens.weight")
                        vocab_size, hidden_dim = tensor_slice.get_shape()
                        tensor = tensor_slice[:, :hidden_dim].float()
                except:
                    with open(
                        os.path.join(path, "pytorch_model.bin.index.json"), "r"
                    ) as f:
                        index_json = json.loads(f.read())
                        emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                    weights = torch.load(os.path.join(path, emb_path))
                    tensor = weights["model.embed_tokens.weight"].float()
            except:
                m = AutoModelForImageTextToText.from_pretrained(
                    path, torch_dtype="auto"
                )
                try:
                    tensor = m.language_model.model.embed_tokens.weight.float()
                except:
                    tensor = m.model.embed_tokens.weight.float()
                del m

            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)

        medusa_num_heads = 3
        medusa_num_layers = 1
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                )
                for _ in range(medusa_num_heads)
            ]
        )

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.register_buffer(
            "tree_mask_init",
            torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None],
            persistent=False,
        )
        self.register_buffer(
            "position_ids",
            torch.zeros(
                self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long
            ),
            persistent=False,
        )

    def reset(self):
        self.tree_mask = None

    def forward(
        self,
        hidden_states,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        std=None,
        image_mask=None,
    ):
        hidden_states = hidden_states[0]
        medusa_logits = []
        for i in range(self.medusa):
            mhidden_states = self.medusa_head[i](hidden_states)
            medusa_logits.append(mhidden_states)

        medusa_logits = torch.stack(medusa_logits, dim=0)

        if use_cache:
            return medusa_logits, None

        return medusa_logits

    def reset_kv(self):
        self.stable_kv = None

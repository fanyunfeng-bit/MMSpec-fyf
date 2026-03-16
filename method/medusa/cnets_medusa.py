# coding=utf-8
import copy
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

# ── Medusa runtime debug logging ──────────────────────────────────────
_MEDUSA_DEBUG = os.environ.get("MEDUSA_DEBUG", "0") == "1"

def _dlog(*args, **kwargs):
    if _MEDUSA_DEBUG:
        print("[MEDUSA]", *args, **kwargs, flush=True)
from transformers.activations import ACT2FN

try:
    from .choices import *
    from .configs import EConfig
    from .utils_c import *
except:
    from choices import *
    from configs import EConfig
    from utils import prepare_logits_processor
    from utils_c import *


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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_states,
        input_ids,
        head,
        logits_processor,
        inputs_embeds=None,
        embed_weights=None,
        image_mask=None,
    ):
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        _dlog(f"  [topK_genrate] hidden_states.shape={hidden_states.shape}, "
              f"input_ids.shape={input_ids.shape}, total_tokens={total_tokens}, depth={depth}, top_k={top_k}")
        _dlog(f"    sample_token={sample_token.tolist()}")
        _dlog(f"    medusa_heads={self.medusa}, medusa_num_layers={self.medusa_num_layers}")

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        self.reset()

        out_hidden, past_key_values = self(
            hidden_states,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        out_hidden = out_hidden[:, -1, :]

        _dlog(f"    out_hidden.shape={out_hidden.shape} (num_heads x hidden_dim)")

        last_hidden = out_hidden[0].unsqueeze(0)

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        _dlog(f"    head0 topk_index={topk_index[0].tolist()}, topk_logp={topk_p[0].tolist()}")

        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        for i in range(out_hidden.shape[0] - 1):
            cur_hidden = out_hidden[i + 1].expand(1, top_k, -1)

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = head(cur_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            _dlog(f"    head{i+1} topk_index (per parent, top-{top_k}):")
            for pi in range(min(top_k, 3)):
                _dlog(f"      parent[{pi}]: tokens={topk_index[pi].tolist()}, logp={[f'{x:.3f}' for x in topk_p[pi].tolist()]}")

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            _dlog(f"    head{i+1} beam topk_cs_index={topk_cs_index.tolist()}, selected_tokens={input_ids[0].tolist()}")

            ss_token.append(topk_index)
            scores_list.append(cu_scores)

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        _dlog(f"    draft_tokens (with sample)={draft_tokens.tolist()}")
        _dlog(f"    top_scores_index={top_scores_index.tolist()}")

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(
            top_scores_index, draft_parents - 1, right=False
        )
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        _dlog(f"    mask_index_list (parent pointers)={mask_index_list}")

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        _dlog(f"    tree_position_ids={tree_position_ids.tolist()}")
        _dlog(f"    tree_mask sum_per_row={tree_mask.sum(dim=-1).int().tolist()}")

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        _dlog(f"    max_depth={max_depth.item()}, leaf_num={leaf_num}, noleaf_num={noleaf_num}")

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

        _dlog(f"    retrieve_indices.shape={retrieve_indices.shape}")
        for ri in range(min(retrieve_indices.shape[0], 5)):
            _dlog(f"      path[{ri}]={retrieve_indices[ri].tolist()}")

        del (
            mask_index,
            mask_index_list,
            noleaf_index,
            noleaf_num,
            leaf_num,
            max_depth,
            rid,
        )
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


if __name__ == "__main__":
    config = EConfig.from_pretrained("config.json")
    model = Model(config, load_emb=False)
    print(model)

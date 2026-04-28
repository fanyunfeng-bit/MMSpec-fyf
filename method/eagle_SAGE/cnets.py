"""SageModel: EAGLE draft model with SAGE-aware position_ids propagation.

Subclass of `method.eagle.cnets.Model`. The only override is `topK_genrate`,
which now accepts an optional `position_ids` parameter that is propagated into
the FIRST draft forward (the prefill). The tree-expansion loop is unchanged:
new tree tokens are always post-visual, their position_ids come from
`len_posi + self.position_ids`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401  (kept for compatibility)

from method.eagle.cnets import Model as EagleModel


def _slice_pids(position_ids, start: int):
    """Slice position_ids for the un-cached tail. Handles 1D [B, L] and 3D [3, B, L]."""
    if position_ids is None:
        return None
    if position_ids.dim() == 2:
        return position_ids[:, start:]
    if position_ids.dim() == 3:
        return position_ids[:, :, start:]
    return position_ids


class SageModel(EagleModel):
    """Draft model with position_ids-aware first forward for SAGE repositioning."""

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
        position_ids=None,
    ):
        """Identical to EagleModel.topK_genrate except the FIRST self(...) call
        forwards the caller-provided `position_ids`. Tree-expansion code path is
        unchanged.
        """
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.clone()
            assert inputs_embeds.shape[-2] < input_ids.shape[-1]
            if embed_weights is not None:
                assert embed_weights.dim() == 3
                assert embed_weights.shape[-1] == 1
                inputs_embeds[
                    : embed_weights.shape[0], : embed_weights.shape[1]
                ] *= embed_weights
            inputs_embeds.to(input_ids.device)
            new_embeds = self.embed_tokens(input_ids[:, inputs_embeds.shape[-2] :])
            inputs_embeds = torch.cat((inputs_embeds[:, 1:, :], new_embeds), dim=-2)

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            if len(self.stable_kv[0]) == 3:
                kv_len = int(self.stable_kv[0][2])
            else:
                kv_len = int(self.stable_kv[0][0].shape[2])
            kv_len = max(0, min(kv_len, input_ids.shape[1]))
            uncached_input_ids = input_ids[:, kv_len:]
            if uncached_input_ids.shape[1] == 0:
                uncached_input_ids = input_ids[:, -hidden_states.shape[1] :]
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=uncached_input_ids,
                past_key_values=self.stable_kv,
                use_cache=True,
                position_ids=_slice_pids(position_ids, kv_len),
            )
        else:
            if inputs_embeds is not None:
                input_ids = None
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                position_ids=position_ids,
            )
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids_loop = len_posi + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids_loop,
                use_cache=True,
            )
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat(
                (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3
            )

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(
            top_scores_index, draft_parents - 1, right=False
        )
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth_i = position_ids_list[i]
                for j in reversed(range(depth_i + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for idx_l in range(len(lst)):
                    sort_keys.append(lst[idx_l] if lst[idx_l] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
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

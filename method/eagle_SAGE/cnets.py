"""SageModel: EAGLE draft model with SAGE-aware position_ids propagation.

Subclass of `method.eagle.cnets.Model`. Two extensions to `topK_genrate`:

1. Accepts an optional `position_ids` argument that is propagated into the
   FIRST draft forward (the prefill). Required for prefill-time visual
   repositioning so each kept visual token rotates by its true position.

2. When `_sage_keep_mask` and `_sage_original_prefix_len` are set on the
   instance (by `sage_initialize_tree` after VisualCompressor runs), AND the
   incoming `input_ids` carries the original (uncompressed) prefix layout
   (i.e. shape[1] > _sage_original_prefix_len), this method compresses the
   prefix portion through the keep mask before the standard EAGLE flow runs.
   This is required on subsequent calls (from update_inference_inputs), which
   re-pass the FULL original input_ids while the draft's stable_kv is sized to
   the COMPRESSED prefix.

Tree-expansion loop is unchanged in shape; the only edit is that the per-loop
`position_ids` offset is `max(prefix position_ids) + 1` instead of
`input_ids.shape[1]`. They are equal in the no-compression case (positions are
contiguous arange) and differ when compression is active.
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
        """EAGLE topK_genrate with optional first-call position_ids and SAGE
        prefix compression. See module docstring."""
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        # --- SAGE prefix compression on subsequent calls -----------------
        # update_inference_inputs re-passes the FULL ORIGINAL prefix; the
        # draft's stable_kv is sized to the COMPRESSED prefix. Slice down to
        # match. On the first call (from sage_initialize_tree), input_ids was
        # already compressed by the pipeline so this branch is skipped.
        keep_mask = getattr(self, "_sage_keep_mask", None)
        original_prefix_len = getattr(self, "_sage_original_prefix_len", None)
        if (
            keep_mask is not None
            and original_prefix_len is not None
            and input_ids.shape[1] >= original_prefix_len + 1
            and keep_mask.shape[0] == original_prefix_len
        ):
            keep_mask = keep_mask.to(input_ids.device)
            prefix = input_ids[:, :original_prefix_len]
            tail = input_ids[:, original_prefix_len:]
            compressed_prefix = prefix[:, keep_mask]
            input_ids = torch.cat([compressed_prefix, tail], dim=1)

            # SAGE convention (matches sage_initialize_tree): position_ids
            # length = post-shift input_ids length = pre-shift length - 1
            # (i.e. excludes the LAST pre-shift element, which is the sample).
            kept_pos = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).long()
            tail_pos = torch.arange(
                original_prefix_len,
                original_prefix_len + tail.shape[1] - 1,
                device=input_ids.device,
                dtype=torch.long,
            )
            position_ids = torch.cat([kept_pos, tail_pos], dim=0)[None, :]

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

        # Tree-expansion next-position offset: continue from the LAST position
        # used in the prefix, not from input_ids.shape[1]. Under compression
        # the latter is smaller than the true last position; without this fix
        # new tree tokens would land in the middle of the original positions.
        if position_ids is not None:
            if position_ids.dim() == 2:
                next_pos_offset = int(position_ids[0, -1].item()) + 1
            elif position_ids.dim() == 3:
                next_pos_offset = int(position_ids[..., -1].max().item()) + 1
            else:
                next_pos_offset = len_posi
        else:
            next_pos_offset = len_posi

        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids_loop = next_pos_offset + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids_loop,
                use_cache=True,
            )
            next_pos_offset += 1

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

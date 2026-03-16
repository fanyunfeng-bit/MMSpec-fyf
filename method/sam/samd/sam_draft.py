import torch
from typing import List, Dict, Optional, Literal, Any
from dataclasses import field
from enum import Enum
from collections import namedtuple
from .dyn_sam import DynSAM
from .static_sam import NullStaticSAM


class SamdConfig:
    n_predicts: int = 40
    max_predicts: int = 70
    len_threshold: int = 5
    len_bias: int = 5

    cache_type: Literal["dynamic", "static"] = field(
        default="static"
    )
    use_last_hidden_states: bool = field(default=False)

    tree_method: Literal["token_recycle", "eagle", "eagle2"] = field(
        default=None
    )
    tree_model_path: Optional[str] = field(default=None)
    tree_path: Optional[str] = field(default=None)
    tree: Optional[List[List[int]]] = field(default=None)
    tree_config: Optional[Dict[str, Any]] = field(default=None)

class CandidateType(str, Enum):
    sequence = "sequence"
    tree = "tree"

Candidates = namedtuple('Candidates', ['type', 'tokens', 'candidate_tokens', 'buffers_kwargs'])

TOPK = 8

class DraftModel(torch.nn.Module):
    
    def __init__(self,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.config = SamdConfig()
        self.sam_dyn = DynSAM(self.config.n_predicts, device=device)
        self.sam_static = NullStaticSAM(self.config.n_predicts, device=device)
        
        self.sam_dyn.n_predicts = self.config.n_predicts
        self.sam_static.n_predicts = self.config.n_predicts
        self.len_bias = self.config.len_bias
        self.len_threshold = self.config.len_threshold
        
    def reset(self):
        self.sam_dyn.reset()
        self.sam_static.reset()

    def lookup(self, start_token: int):
        index_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        # index_static, match_static = self.sam_static.lookup(start_token)
        seq, buffers_kwargs = self.sam_dyn.gen_dyn_draft(index_dyn, match_dyn, start_token)
        draft = torch.tensor(seq[1:], device=self.device).long()
        return (CandidateType.sequence, draft, buffers_kwargs)
        # if match_dyn >= match_static:
        #     seq, buffers_kwargs = self.sam_dyn.gen_dyn_draft(index_dyn, match_dyn, start_token)
        #     return (CandidateType.sequence, seq, buffers_kwargs)
        # else:
        #     tree, buffers_kwargs = self.sam_static.gen_dyn_draft(index_static, match_static, start_token)
        #     return (CandidateType.tree, tree, buffers_kwargs)

    def update(self,
        tokens: Optional[torch.Tensor] = None,
    ):
        tokens_list = tokens.tolist()
        self.sam_dyn.add_tokens(tokens_list)
        self.sam_static.transfer_tokens(tokens_list)
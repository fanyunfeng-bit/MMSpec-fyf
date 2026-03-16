# MSD (Multimodal Speculative Decoding) Module
# Adapted from EAGLE model for LLaVA integration

from .ea_model import EaModel
from .configs import EConfig
from .kv_cache import initialize_past_key_values, KVCache
from .utils import (
    prepare_logits_processor,
    generate_tree_buffers,
    initialize_tree,
    reset_tree_mode,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    temp_cache,
)
from .choices import mc_sim_7b_63

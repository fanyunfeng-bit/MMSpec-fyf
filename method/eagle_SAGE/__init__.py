"""SAGE: Sink-Aware Generative Enhancement for speculative decoding on VLMs.

Built on top of EAGLE-1. This package adds a pluggable VisualProcessor that
runs between the target's prefill output and the draft's first forward:
  1. Detects visual sink tokens by L2 norm at the vision tower's pre-projector output.
  2. Repositions visual sink tokens (with their positional embeddings) to the front
     of the visual sub-sequence.

Future pluggable stages (compression, draft-skip gate) can be registered via
VisualProcessor.register_stage().
"""

from .spec_model import SageSpecModel as SpecModel
from .visual_processor import VisualContext, VisualProcessor
from .sink_detector import L2NormSinkDetector
from .repositioner import SinkFirstRepositioner
from .visual_hooks import VisionTowerHook
from .text_importance_probe import TextImportanceProbe
from .visual_compressor import VisualCompressor

__all__ = [
    "SpecModel",
    "SageSpecModel",
    "VisualContext",
    "VisualProcessor",
    "L2NormSinkDetector",
    "SinkFirstRepositioner",
    "VisionTowerHook",
    "TextImportanceProbe",
    "VisualCompressor",
]

SageSpecModel = SpecModel

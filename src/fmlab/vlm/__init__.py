"""Visual-language-model experiments with a dependency-light offline mode.

The public API intentionally keeps model loading lazy: importing :mod:`fmlab.vlm`
never allocates GPU memory or imports ``transformers``.
"""

from .inference import Qwen3VLConfig, Qwen3VLRunner, resolve_qwen_vl_path
from .local_report import render_local_inference_report
from .manifest import prepare_training_manifests
from .schema import GenerationResult, GroundingCase, MediaInput, QAItem

__all__ = [
    "GenerationResult",
    "GroundingCase",
    "MediaInput",
    "QAItem",
    "Qwen3VLConfig",
    "Qwen3VLRunner",
    "prepare_training_manifests",
    "render_local_inference_report",
    "resolve_qwen_vl_path",
]

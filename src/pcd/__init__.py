"""pcd: parallel constrained decoding for open LLMs, on MLX or PyTorch.

    from pcd import Engine

    engine = Engine.load("qwen2.5-1.5b")            # alias, repo id or local path
    out = engine.extract(ticket_text, {
        "priority": {"type": "enum", "choices": ["P0", "P1", "P2"], "description": "Urgency"},
        "needs_refund": {"type": "boolean", "description": "Whether to refund"},
    })
    out.values        # {"priority": "P0", "needs_refund": True}
    out.fields        # per-field probability, candidate mass, method, alternatives
    out.stats         # timings, measured forward passes
"""

from .backends import available_backends, load_backend
from .compiler import TokenizationError
from .decoding import DecodeOptions
from .engine import Engine
from .models import MODELS
from .prompt import PromptFormat
from .result import Extraction, FieldResult, Stats
from .schema import Field, Schema, SchemaError

__version__ = "0.1.0"

__all__ = [
    "DecodeOptions",
    "Engine",
    "Extraction",
    "Field",
    "FieldResult",
    "MODELS",
    "PromptFormat",
    "Schema",
    "SchemaError",
    "Stats",
    "TokenizationError",
    "available_backends",
    "load_backend",
]

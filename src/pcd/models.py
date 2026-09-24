"""Curated aliases for open models that fit on an Apple Silicon laptop.

Any Hugging Face repo id works directly (``Engine.load("org/model")``); the aliases only
save you from looking up the right MLX / PyTorch repositories. Every repo id below was
checked to exist on the Hub. ``tested`` lists the backends on which that exact checkpoint
was run end to end by ``tests/test_real_models.py`` (see README for the full list).

Memory: 4-bit MLX weights need about 0.6 GB per billion parameters; bf16 PyTorch
weights about 2 GB per billion. A 16 GB Mac runs everything up to ~9B in 4-bit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    params: str
    mlx: str | None
    torch: str | None
    notes: str = ""
    tested: tuple[str, ...] = ()


_SPECS = [
    # Qwen 2.5: dense, strong at following label vocabularies. The original project's model.
    ModelSpec("qwen2.5-0.5b", "0.5B", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "Qwen/Qwen2.5-0.5B-Instruct", tested=("torch",)),
    ModelSpec("qwen2.5-1.5b", "1.5B", "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "Qwen/Qwen2.5-1.5B-Instruct", tested=("mlx",)),
    ModelSpec("qwen2.5-3b", "3B", "mlx-community/Qwen2.5-3B-Instruct-4bit", "Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("qwen2.5-7b", "7B", "mlx-community/Qwen2.5-7B-Instruct-4bit", "Qwen/Qwen2.5-7B-Instruct"),
    # Qwen 3: thinking is switched off through the chat template (enable_thinking=False).
    ModelSpec("qwen3-0.6b", "0.6B", "mlx-community/Qwen3-0.6B-4bit", "Qwen/Qwen3-0.6B", tested=("mlx",)),
    ModelSpec("qwen3-1.7b", "1.7B", "mlx-community/Qwen3-1.7B-4bit", "Qwen/Qwen3-1.7B"),
    ModelSpec("qwen3-4b", "4B", "mlx-community/Qwen3-4B-Instruct-2507-4bit", "Qwen/Qwen3-4B-Instruct-2507"),
    ModelSpec("qwen3-8b", "8B", "mlx-community/Qwen3-8B-4bit", "Qwen/Qwen3-8B"),
    # Qwen 3.5: hybrid linear-attention (Gated DeltaNet) + attention layers.
    ModelSpec("qwen3.5-0.8b", "0.8B", "mlx-community/Qwen3.5-0.8B-4bit", "Qwen/Qwen3.5-0.8B", "hybrid recurrent cache", ("mlx",)),
    ModelSpec("qwen3.5-2b", "2B", "mlx-community/Qwen3.5-2B-4bit", "Qwen/Qwen3.5-2B", "hybrid recurrent cache"),
    ModelSpec("qwen3.5-4b", "4B", "mlx-community/Qwen3.5-4B-4bit", "Qwen/Qwen3.5-4B", "hybrid recurrent cache"),
    ModelSpec("qwen3.5-9b", "9B", "mlx-community/Qwen3.5-9B-4bit", "Qwen/Qwen3.5-9B", "hybrid recurrent cache"),
    # Llama: the official PyTorch repos are gated (accept the license, then `hf auth login`).
    ModelSpec("llama3.2-1b", "1B", "mlx-community/Llama-3.2-1B-Instruct-4bit", "meta-llama/Llama-3.2-1B-Instruct", "torch repo gated", ("mlx",)),
    ModelSpec("llama3.2-3b", "3B", "mlx-community/Llama-3.2-3B-Instruct-4bit", "meta-llama/Llama-3.2-3B-Instruct", "torch repo gated"),
    ModelSpec("llama3.1-8b", "8B", "mlx-community/Llama-3.1-8B-Instruct-4bit", "meta-llama/Llama-3.1-8B-Instruct", "torch repo gated"),
    # Gemma: sliding-window attention; Gemma 2 soft-caps logits (the fast path turns itself off).
    ModelSpec("gemma2-2b", "2B", "mlx-community/gemma-2-2b-it-4bit", "google/gemma-2-2b-it", "no system role; torch repo gated"),
    ModelSpec("gemma3-1b", "1B", "mlx-community/gemma-3-1b-it-4bit", "google/gemma-3-1b-it", "torch repo gated", ("mlx",)),
    ModelSpec("gemma3-4b", "4B", "mlx-community/gemma-3-4b-it-4bit", "google/gemma-3-4b-it", "multimodal checkpoint; torch repo gated"),
    ModelSpec("gemma4-e2b", "E2B", "mlx-community/gemma-4-e2b-it-4bit", "google/gemma-4-e2b-it"),
    ModelSpec("gemma4-e4b", "E4B", "mlx-community/gemma-4-e4b-it-4bit", "google/gemma-4-e4b-it"),
    # Microsoft Phi.
    ModelSpec("phi3.5-mini", "3.8B", "mlx-community/Phi-3.5-mini-instruct-4bit", "microsoft/Phi-3.5-mini-instruct"),
    ModelSpec("phi4-mini", "3.8B", "mlx-community/Phi-4-mini-instruct-4bit", "microsoft/Phi-4-mini-instruct"),
    # Mistral.
    ModelSpec("mistral-7b", "7B", "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mistralai/Mistral-7B-Instruct-v0.3", "no system role"),
    ModelSpec("ministral-8b", "8B", "mlx-community/Ministral-8B-Instruct-2410-4bit", "mistralai/Ministral-8B-Instruct-2410"),
    # Small open models.
    ModelSpec("smollm2-360m", "360M", "mlx-community/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", tested=("torch",)),
    ModelSpec("smollm2-1.7b", "1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "MLX converts bf16 weights on load"),
    ModelSpec("smollm3-3b", "3B", "mlx-community/SmolLM3-3B-4bit", "HuggingFaceTB/SmolLM3-3B"),
    ModelSpec("lfm2.5-1.2b", "1.2B", "mlx-community/LFM2.5-1.2B-Instruct-4bit", "LiquidAI/LFM2.5-1.2B-Instruct", "hybrid conv + attention cache", ("mlx",)),
    ModelSpec("granite3.3-2b", "2B", "mlx-community/granite-3.3-2b-instruct-4bit", "ibm-granite/granite-3.3-2b-instruct"),
    # Needs ~16 GB for weights alone: a 32 GB Mac or a GPU box.
    ModelSpec("gpt-oss-20b", "21B MoE", "mlx-community/gpt-oss-20b-MXFP4-Q8", "openai/gpt-oss-20b", "harmony format; needs 24 GB+"),
]

MODELS: dict[str, ModelSpec] = {s.alias: s for s in _SPECS}


def resolve(model: str, backend: str) -> str:
    """Map an alias to the repo id for ``backend``; anything else is returned unchanged."""
    spec = MODELS.get(model.lower())
    if spec is None:
        return model
    repo = spec.mlx if backend == "mlx" else spec.torch
    if repo is None:
        raise ValueError(f"model alias {model!r} has no {backend} checkpoint")
    return repo

"""Offline test fixtures: a byte-level chat tokenizer, a tiny random checkpoint, and an
oracle backend whose probabilities can be computed by brute force."""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import numpy as np

from pcd.backends.base import Backend, Prefix, Query
from pcd.tokenizer import HFTokenizer

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# Same format, but rejects system messages like Gemma 2's template.
NO_SYSTEM = (
    "{% for m in messages %}{% if m['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
    "{% endif %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def byte_tokenizer(chat_template: str | None = CHATML):
    """A byte-level BPE tokenizer without merges: every byte is one token."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    tok.chat_template = chat_template
    return tok


TINY_ARCHS = ("llama", "qwen3", "gemma2", "gemma3_text")


def save_tiny(path, tokenizer, arch: str = "llama", seed: int = 0) -> str:
    """A 2-layer random model in float32, saved with the tokenizer (loadable by both backends).

    * gemma2 soft-caps its logits, so the "decision positions only" fast path must turn off.
    * gemma3_text uses an 8-token sliding window on every other layer, much shorter than
      the prompts, so rotating / sliding KV caches are exercised.
    """
    import json
    from pathlib import Path

    import torch
    import transformers as tf

    torch.manual_seed(seed)
    common = dict(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, rms_norm_eps=1e-6,
    )
    extra_cfg = {}
    if arch == "llama":
        model = tf.LlamaForCausalLM(tf.LlamaConfig(**common, rope_theta=10000.0, tie_word_embeddings=False))
    elif arch == "qwen3":
        model = tf.Qwen3ForCausalLM(tf.Qwen3Config(**common, head_dim=16, rope_theta=1_000_000.0, tie_word_embeddings=True))
        extra_cfg = {"rope_theta": 1_000_000.0}  # transformers v5 nests it in rope_parameters; mlx-lm reads it here
    elif arch == "gemma2":
        model = tf.Gemma2ForCausalLM(tf.Gemma2Config(
            **common, head_dim=16, query_pre_attn_scalar=16, sliding_window=4096,
            final_logit_softcapping=5.0, attn_logit_softcapping=50.0, rope_theta=10000.0,
        ))
    elif arch == "gemma3_text":
        model = tf.Gemma3ForCausalLM(tf.Gemma3TextConfig(
            **common, head_dim=16, query_pre_attn_scalar=16, sliding_window=8,
            layer_types=["sliding_attention", "full_attention"], rope_theta=1_000_000.0,
            rope_local_base_freq=10000.0,
        ))
        extra_cfg = {"sliding_window_pattern": 2}  # the key mlx-lm reads
    else:
        raise ValueError(arch)
    with torch.no_grad():  # larger weights than the default init, so outputs are far from uniform
        for p in model.parameters():
            if p.dim() == 2:
                p.normal_(0.0, 0.2)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra_cfg:
        cfg_path = Path(path) / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg.update(extra_cfg)
        cfg_path.write_text(json.dumps(cfg, indent=2))
    return str(path)


def save_tiny_llama(path, tokenizer, seed: int = 0) -> str:
    return save_tiny(path, tokenizer, "llama", seed)


class OracleBackend(Backend):
    """A deterministic fake LM: next-token logits are a hash of the last two tokens,
    plus optional hand-written rules. Everything is computed on the fly from the full
    context, so the engine's results can be checked against brute force."""

    name = "oracle"

    def __init__(self, tokenizer, vocab_size: int | None = None, seed: int = 0,
                 rules: Sequence[Callable[[tuple[int, ...], np.ndarray], None]] = (), scale: float = 3.0, **kw):
        super().__init__(tokenizer if isinstance(tokenizer, HFTokenizer) else HFTokenizer(tokenizer), **kw)
        self.V = vocab_size or self.tokenizer.vocab_size
        self.seed = seed
        self.rules = list(rules)
        self.scale = scale

    # -- the "model" ----------------------------------------------------------
    def logits(self, ctx: tuple[int, ...]) -> np.ndarray:
        key = hashlib.sha256(f"{self.seed}:{ctx[-2:]}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        z = rng.normal(0.0, self.scale, self.V)
        for rule in self.rules:
            rule(ctx, z)
        return z

    def logprobs(self, ctx: tuple[int, ...]) -> np.ndarray:
        z = self.logits(ctx)
        m = z.max()
        return z - (m + np.log(np.exp(z - m).sum()))

    def seq_logprob(self, ctx: tuple[int, ...], seq: Sequence[int]) -> float:
        total, c = 0.0, tuple(ctx)
        for t in seq:
            total += float(self.logprobs(c)[t])
            c = c + (t,)
        return total

    # -- backend contract -------------------------------------------------------
    def prefill(self, tokens, parent=None):
        toks = (parent.tokens if parent is not None else ()) + tuple(int(t) for t in tokens)
        if tokens:
            self.forward_passes += 1
        return Prefix(tokens=toks, state=None, nbytes=len(toks))

    def score(self, prefix, rows, queries):
        self.validate(rows, queries)
        self.forward_passes += len(self.plan_chunks(prefix, rows, queries))
        out = []
        for q in queries:
            lp = self.logprobs(prefix.tokens + tuple(rows[q.row][: q.position + 1]))
            out.append(lp[list(q.tokens)])
        return out

    def greedy(self, prefix, inputs, max_new_tokens, stop_ids, should_stop=None):
        ctx, out = prefix.tokens + tuple(inputs), []
        while len(out) < max_new_tokens:
            self.forward_passes += 1
            t = int(np.argmax(self.logits(ctx)))
            if t in stop_ids:
                break
            out.append(t)
            ctx = ctx + (t,)
            if should_stop and should_stop(out):
                break
        return out

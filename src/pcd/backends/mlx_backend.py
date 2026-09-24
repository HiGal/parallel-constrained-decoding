"""MLX backend (Apple Silicon), built on mlx-lm.

Works with any architecture mlx-lm can load. Cache copies use mlx-lm's generic
``state`` / ``meta_state`` / ``from_state`` protocol, so plain, sliding-window
(``RotatingKVCache``), quantized and SSM (``ArraysCache``) caches are all handled
the same way.

Fast path: instead of projecting every position of every row onto the vocabulary,
run the transformer body, gather only the queried positions, and project those
(notebook 05 §3b). Some architectures post-process logits (e.g. Gemma 2 soft-capping),
so the fast path is enabled only after a probe shows it reproduces ``model(...)``.
Model-specific overrides can be registered with ``register_head``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from ..tokenizer import HFTokenizer
from .base import Backend, Prefix, Query, gather_layout, pad_rows, verify_close

# model_type -> fn(model) -> (body, head) or None. See CONTRIBUTING.md.
_HEAD_REGISTRY: dict[str, Callable[[Any], tuple[Callable, Callable] | None]] = {}


def register_head(model_type: str, fn: Callable[[Any], tuple[Callable, Callable] | None]) -> None:
    """Teach the MLX backend how to split a model into (body -> hidden states, head -> logits)."""
    _HEAD_REGISTRY[model_type] = fn


def _default_split(model: Any) -> tuple[Callable, Callable] | None:
    body = getattr(model, "model", None)
    if body is None or not callable(body):
        return None
    args = getattr(model, "args", None)
    tied = getattr(model, "tie_word_embeddings", None)
    if tied is None:
        tied = getattr(args, "tie_word_embeddings", None)
    lm_head = getattr(model, "lm_head", None)
    embed = getattr(body, "embed_tokens", None)
    if lm_head is not None and not tied:
        head = lm_head
    elif embed is not None and hasattr(embed, "as_linear"):
        head = embed.as_linear
    elif lm_head is not None:
        head = lm_head
    else:
        return None
    return (lambda x, cache: body(x, cache=cache)), head


def _softcapped_split(model: Any) -> tuple[Callable, Callable] | None:
    """Gemma 2 soft-caps its logits: apply the cap after projecting the gathered positions."""
    cap = getattr(model, "final_logit_softcapping", None)
    split = _default_split(model)
    if not cap or split is None:
        return split
    body, project = split
    return body, (lambda h: mx.tanh(project(h) / cap) * cap)


register_head("gemma2", _softcapped_split)


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model: Any, tokenizer: Any, *, fast_path: bool | None = None, **kwargs: Any):
        tok = tokenizer if isinstance(tokenizer, HFTokenizer) else HFTokenizer(tokenizer)
        super().__init__(tok, **kwargs)
        from mlx_lm.models.cache import make_prompt_cache

        self.model = model
        self._make_cache = lambda: make_prompt_cache(model)
        self._split = None
        if fast_path is not False:
            self._split = self._probe_fast_path()
            if fast_path and self._split is None:
                raise RuntimeError("fast_path=True, but the body/head split does not reproduce the model's logits")
        self.forward_passes = 0

    @classmethod
    def load(cls, model_id: str, **kwargs: Any) -> "MLXBackend":
        from mlx_lm import load

        model, tokenizer = load(model_id)
        return cls(model, tokenizer, model_id=model_id, **kwargs)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "fast_path": self._split is not None,
            "model_type": getattr(getattr(self.model, "args", None), "model_type", type(self.model).__name__),
        }

    # ------------------------------------------------------------------ contract

    def prefill(self, tokens: Sequence[int], parent: Prefix | None = None) -> Prefix:
        tokens = [int(t) for t in tokens]
        if not tokens:
            if parent is None:
                raise ValueError("cannot prefill an empty sequence without a parent")
            return parent
        cache = self._clone(parent.state, 1) if parent is not None else self._make_cache()
        for i in range(0, len(tokens), self.prefill_step):
            # Logits are never evaluated, so MLX skips the vocabulary projection.
            self.model(mx.array(tokens[i : i + self.prefill_step])[None], cache=cache)
            self.forward_passes += 1
            mx.eval([c.state for c in cache])
        all_tokens = (parent.tokens if parent is not None else ()) + tuple(tokens)
        return Prefix(tokens=all_tokens, state=cache, nbytes=_cache_nbytes(cache))

    def score(self, prefix: Prefix, rows: Sequence[Sequence[int]], queries: Sequence[Query]) -> list[np.ndarray]:
        self.validate(rows, queries)
        out: list[np.ndarray | None] = [None] * len(queries)
        pad = self.tokenizer.pad_id
        for chunk in self.plan_chunks(prefix, rows, queries):
            batch = mx.array(pad_rows([rows[r] for r in chunk], pad))
            cache = self._clone(prefix.state, len(chunk))
            qids, ridx, pidx, toks = gather_layout(chunk, queries)
            ridx_a, pidx_a = mx.array(ridx), mx.array(pidx)
            if self._split is not None:
                body, head = self._split
                hidden = body(batch, cache)
                logits = head(hidden[ridx_a, pidx_a])
            else:
                logits = self.model(batch, cache=cache)[ridx_a, pidx_a]
            self.forward_passes += 1
            logits = logits.astype(mx.float32)
            lse = mx.logsumexp(logits, axis=-1, keepdims=True)
            lp = mx.take_along_axis(logits, mx.array(toks), axis=1) - lse
            lp = np.array(lp, dtype=np.float64)
            for i, q in enumerate(qids):
                out[q] = lp[i, : len(queries[q].tokens)]
        return out  # type: ignore[return-value]

    def greedy(self, prefix, inputs, max_new_tokens, stop_ids, should_stop=None) -> list[int]:
        cache = self._clone(prefix.state, 1)
        y = mx.array([int(t) for t in inputs])[None]
        generated: list[int] = []

        def step(y: mx.array) -> mx.array:
            self.forward_passes += 1
            if self._split is not None:
                body, head = self._split
                logits = head(body(y, cache)[:, -1])
            else:
                logits = self.model(y, cache=cache)[:, -1]
            return mx.argmax(logits, axis=-1)

        # Pipelined like mlx-lm's generate_step: queue step t+1 before reading token t.
        nxt = step(y)
        mx.async_eval(nxt)
        while len(generated) < max_new_tokens:
            following = step(nxt[None]) if len(generated) + 1 < max_new_tokens else None
            if following is not None:
                mx.async_eval(following)
            tok = int(nxt.item())
            if tok in stop_ids:
                break
            generated.append(tok)
            if should_stop is not None and should_stop(generated):
                break
            if following is None:
                break
            nxt = following
        return generated

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _clone(cache: list, n: int) -> list:
        """Independent copy of a per-layer cache with its batch axis repeated ``n`` times."""

        def rep(a: Any) -> Any:
            if a is None:
                return None
            # mx.array(a) makes a new array object, so writes to the copy never reach `a`.
            return mx.repeat(a, n, axis=0) if n > 1 else mx.array(a)

        return [type(c).from_state(tree_map(rep, c.state), c.meta_state) for c in cache]

    def _probe_fast_path(self):
        model_type = getattr(getattr(self.model, "args", None), "model_type", "")
        make = _HEAD_REGISTRY.get(model_type, _default_split)
        try:
            split = make(self.model)
            if split is None:
                return None
            body, head = split
            # Same shapes as the model's own forward, so both sides use the same kernels:
            # the check is about what the head computes, not about rounding.
            x = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
            ref = self.model(x, cache=self._make_cache())[0, -1]
            got = head(body(x, self._make_cache()))[0, -1]
            mx.eval(ref, got)
            ok = verify_close(np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32)))
            return split if ok else None
        except Exception:
            return None


def _cache_nbytes(cache: list) -> int:
    total = 0
    for c in cache:
        for _, a in tree_flatten(c.state):
            if a is not None:
                total += a.nbytes
    return total

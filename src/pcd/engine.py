"""The engine: prompt -> cached prefill -> batched constrained decoding -> typed JSON.

Per request:

1. Compile the schema for this tokenizer (cached per schema and prompt format).
2. Tokenize ``head + context + tail``. If the head's tokens are a prefix of the result,
   reuse the head's KV state (computed once per schema) and prefill only the rest.
   This was the largest single saving in notebook 05 §3a.
3. For each wave: decide all its fields together (``decoding.py``). Between waves,
   write the answers so far into the prompt as JSON lines and prefill them, so later
   fields condition on them. Waves come from ``depends_on``; by default the first field
   is also decided on its own first (``DecodeOptions.anchor_first``).
4. Assemble the output in schema order with real booleans, and report what was
   measured: timings per phase, forward passes, rows scored, candidate mass.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Iterable

from .backends.base import Backend, Prefix
from .compiler import CompiledSchema, compile_schema, encode_after
from .decoding import DecodeOptions, decode_fields
from .prompt import FAMILY_DEFAULTS, PromptFormat
from .result import Extraction, Stats
from .schema import Schema

_LOW_MASS = 0.1


class Engine:
    def __init__(
        self,
        backend: Backend,
        prompt_format: PromptFormat | None = None,
        decode: DecodeOptions | None = None,
        *,
        head_cache_size: int = 4,
    ):
        self.backend = backend
        if prompt_format is None:
            model_type = backend.describe().get("model_type", "")
            prompt_format = PromptFormat(**FAMILY_DEFAULTS.get(model_type, {}))
        self.prompt_format = prompt_format
        self.decode = decode or DecodeOptions()
        self._compiled: OrderedDict[tuple, CompiledSchema] = OrderedDict()
        self._heads: OrderedDict[tuple[int, ...], Prefix] = OrderedDict()
        self._head_cache_size = head_cache_size
        self._lock = threading.RLock()

    @classmethod
    def load(
        cls,
        model: str,
        backend: str = "auto",
        *,
        prompt_format: PromptFormat | None = None,
        decode: DecodeOptions | None = None,
        **backend_kwargs: Any,
    ) -> "Engine":
        """Load a model by repo id, local path or alias (see ``pcd.models``)."""
        from .backends import load_backend

        return cls(load_backend(model, backend, **backend_kwargs), prompt_format, decode)

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    # ------------------------------------------------------------------ caching

    def compile(self, schema: Any) -> CompiledSchema:
        schema = Schema.coerce(schema)
        key = (schema.fingerprint, self.prompt_format.key())
        with self._lock:
            cs = self._compiled.get(key)
            if cs is None:
                cs = compile_schema(schema, self.tokenizer, self.prompt_format)
                self._compiled[key] = cs
                while len(self._compiled) > 32:
                    self._compiled.popitem(last=False)
            else:
                self._compiled.move_to_end(key)
            return cs

    def warmup(self, schema: Any, context: str = "Warm-up input.") -> None:
        """Compile the schema, cache its prompt head, and run the kernels once."""
        self.extract(context, schema)

    def clear_cache(self) -> None:
        with self._lock:
            self._compiled.clear()
            self._heads.clear()

    def _head_prefix(self, cs: CompiledSchema) -> tuple[Prefix, float]:
        """KV state of the static head; returns (prefix, ms spent computing it now)."""
        cached = self._heads.get(cs.head_ids)
        if cached is not None:
            self._heads.move_to_end(cs.head_ids)
            return cached, 0.0
        t0 = time.perf_counter()
        prefix = self.backend.prefill(cs.head_ids)
        self._heads[cs.head_ids] = prefix
        while len(self._heads) > self._head_cache_size:
            self._heads.popitem(last=False)
        return prefix, (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------------ inference

    def extract(self, context: str, schema: Any, **decode_overrides: Any) -> Extraction:
        """Decide every field of ``schema`` for ``context``.

        ``decode_overrides`` replace fields of the engine's ``DecodeOptions`` for this call,
        e.g. ``strategy="exact"`` or ``temperature=0.5``.
        """
        opts = replace(self.decode, **decode_overrides) if decode_overrides else self.decode
        with self._lock:
            return self._extract(context, schema, opts)

    def extract_many(self, contexts: Iterable[str], schema: Any, **decode_overrides: Any) -> list[Extraction]:
        return [self.extract(c, schema, **decode_overrides) for c in contexts]

    def _extract(self, context: str, schema: Any, opts: DecodeOptions) -> Extraction:
        t_start = time.perf_counter()
        passes_before = self.backend.forward_passes
        stats, warnings = Stats(), []
        tok = self.tokenizer

        cs = self.compile(schema)
        text = cs.prompt.render(context.strip())
        ids = tok.encode(text)
        stats.prompt_tokens = len(ids)
        t = time.perf_counter()
        stats.tokenize_ms = (t - t_start) * 1e3

        n_head = len(cs.head_ids)
        if n_head < len(ids) and tuple(ids[:n_head]) == cs.head_ids:
            head, head_ms = self._head_prefix(cs)
            stats.head_prefill_ms = head_ms
            stats.cached_tokens = n_head if head_ms == 0.0 else 0
            t = time.perf_counter()
            prefix = self.backend.prefill(ids[n_head:], parent=head)
        else:
            warnings.append("the prompt head tokenizes differently with this context; prefix cache skipped")
            prefix = self.backend.prefill(ids)
        if tuple(ids[-len(cs.tail_ids):]) != cs.tail_ids:
            warnings.append("the prompt tail tokenizes differently with this context")
        stats.prefill_ms = (time.perf_counter() - t) * 1e3

        results: dict[str, Any] = {}
        written = cs.prompt.tail  # text the current prefix ends with
        waves = _plan_waves(cs.schema.waves, opts.anchor_first)
        for w, names in enumerate(waves):
            if w > 0:
                t = time.perf_counter()
                new_lines = cs.answer_text({n: results[n].value for n in waves[w - 1]})
                ext = encode_after(tok, written, new_lines, "answers of the previous wave")
                prefix = self.backend.prefill(ext, parent=prefix)
                written += new_lines
                stats.prefill_ms += (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            res, rounds, rows = decode_fields(self.backend, prefix, [cs.fields[n] for n in names], opts, wave=w)
            stats.decode_ms += (time.perf_counter() - t) * 1e3
            stats.decode_rounds += rounds
            stats.rows_scored += rows
            results.update(res)

        low = [n for n, r in results.items() if r.candidate_mass is not None and r.candidate_mass < _LOW_MASS]
        if low:
            warnings.append(
                f"{len(low)} field(s) with candidate mass < {_LOW_MASS}: the model wanted to write something "
                f"outside the allowed labels ({', '.join(low[:5])}{', ...' if len(low) > 5 else ''})"
            )

        names = cs.schema.names
        stats.waves = len(waves)
        stats.forward_passes = self.backend.forward_passes - passes_before
        stats.total_ms = (time.perf_counter() - t_start) * 1e3
        return Extraction(
            values={n: results[n].value for n in names},
            fields={n: results[n] for n in names},
            stats=stats,
            backend={**self.backend.describe(), "prompt_mode": cs.prompt.mode},
            warnings=warnings,
        )


def _plan_waves(waves: list[list[str]], anchor_first: bool) -> list[list[str]]:
    """Dependency waves, optionally with the first field decided on its own before the rest."""
    if not anchor_first or len(waves[0]) < 2:
        return waves
    first, rest = waves[0][0], waves[0][1:]
    return [[first]] + ([rest] if rest else []) + waves[1:]

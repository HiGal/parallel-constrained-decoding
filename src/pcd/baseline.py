"""Autoregressive baseline: let the model write the whole JSON object token by token.

It uses the same prompt (and the same cached head) as the parallel engine, so a
comparison isolates the decoding method. Output is parsed and checked against the
schema; nothing is assumed valid.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .engine import Engine


@dataclass
class BaselineResult:
    text: str
    values: dict[str, Any] | None
    valid_json: bool
    schema_match: bool
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    new_tokens: int = 0
    total_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    forward_passes: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / (self.decode_ms / 1e3) if self.decode_ms > 0 else 0.0


def generate_json(engine: Engine, context: str, schema: Any, max_new_tokens: int | None = None) -> BaselineResult:
    backend, tok = engine.backend, engine.tokenizer
    with engine._lock:
        t0 = time.perf_counter()
        passes_before = backend.forward_passes
        cs = engine.compile(schema)
        ids = tok.encode(cs.prompt.render(context.strip()))
        if max_new_tokens is None:
            per_field = [len(cf.prefix) + max(len(v.seq) for v in cf.variants) for cf in cs.fields.values()]
            max_new_tokens = int(sum(per_field) * 1.5) + 16

        n_head = len(cs.head_ids)
        if n_head < len(ids) - 1 and tuple(ids[:n_head]) == cs.head_ids:
            head, _ = engine._head_prefix(cs)
            t1 = time.perf_counter()
            prefix = backend.prefill(ids[n_head:-1], parent=head)
        else:
            t1 = time.perf_counter()
            prefix = backend.prefill(ids[:-1])
        t2 = time.perf_counter()

        def closed(generated: list[int]) -> bool:
            if "}" not in tok.decode(generated[-1:]):
                return False
            return _parse(cs.prompt.opener + tok.decode(generated)) is not None

        out = backend.greedy(prefix, ids[-1:], max_new_tokens, tok.eos_ids, closed)
        t3 = time.perf_counter()

    text = cs.prompt.opener + tok.decode(out)
    values = _parse(text)
    res = BaselineResult(
        text=text,
        values=values,
        valid_json=values is not None,
        schema_match=False,
        new_tokens=len(out),
        total_ms=(t3 - t0) * 1e3,
        prefill_ms=(t2 - t1) * 1e3,
        decode_ms=(t3 - t2) * 1e3,
        forward_passes=backend.forward_passes - passes_before,
    )
    if isinstance(values, dict):
        res.missing, res.extra, res.invalid = cs.schema.validate_values(values)
        res.schema_match = not (res.missing or res.extra or res.invalid)
    return res


def lenient_values(schema: Any, values: Any) -> dict[str, Any]:
    """Map what an autoregressive decoder wrote onto labels where the intent is clear
    (``"true"`` -> ``True``, surrounding whitespace). Used for agreement, not validation."""
    from .schema import BOOLEAN, Schema

    schema = Schema.coerce(schema)
    out: dict[str, Any] = {}
    if not isinstance(values, dict):
        return out
    for name, v in values.items():
        if name not in schema:
            continue
        f = schema[name]
        if f.type == BOOLEAN and isinstance(v, str) and v.strip().lower() in ("true", "false"):
            v = v.strip().lower() == "true"
        elif isinstance(v, str):
            v = v.strip()
        out[name] = v
    return out


def _parse(text: str) -> Any:
    end = text.rfind("}")
    if end < 0:
        return None
    body = text[: end + 1]
    for candidate in (body, re.sub(r",\s*}", "}", body)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None

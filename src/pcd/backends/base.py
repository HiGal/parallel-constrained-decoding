"""The backend contract.

A backend wraps one model and exposes two primitives. Everything else (first-token
decisions, exact scoring, trie walks, dependency waves) is built on top of them in a
backend-independent way.

``prefill(tokens, parent=None) -> Prefix``
    Run ``tokens`` through the model on top of ``parent``'s KV state (or from scratch)
    and return a new, immutable prefix. Must not modify ``parent``.

``score(prefix, rows, queries) -> list[np.ndarray]``
    Treat every row as a continuation of ``prefix`` (all rows share the prefix's KV
    state, broadcast to the batch). For each query ``(row, position, token_ids)`` return
    the full-vocabulary log-softmax, at that position of that row, of the requested
    token ids. ``position`` indexes into the row; the distribution at position ``p``
    predicts the token at ``p + 1``. Must not modify ``prefix``.

Rows may have different lengths. Backends right-pad them, which is safe: under a
causal mask no position ever attends to anything after it (notebook 02 §3). Rows are
never *continued* from a padded batch; any continuation is a new row scored against
the untouched prefix, so pad tokens cannot leak into later steps (notebook 04 §4).

Optional: ``greedy(prefix, inputs, max_new_tokens, stop_ids, should_stop)`` for the
autoregressive baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..tokenizer import Tokenizer


@dataclass(frozen=True)
class Query:
    row: int
    position: int
    tokens: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class Prefix:
    """An immutable KV state for a token sequence."""

    tokens: tuple[int, ...]
    #: Backend-specific cache object. Never mutate it.
    state: Any = field(repr=False)
    #: Bytes held by one copy of the state; used to size batches.
    nbytes: int = 0

    def __len__(self) -> int:
        return len(self.tokens)


class Backend(ABC):
    """Base class with shared batching logic. See the module docstring for the contract."""

    name: str = "abstract"

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        model_id: str = "",
        max_rows: int = 64,
        kv_budget_bytes: int = 2 << 30,
        prefill_step: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.max_rows = max_rows
        self.kv_budget_bytes = kv_budget_bytes
        self.prefill_step = prefill_step
        #: Number of model forward calls made so far (prefill chunks, scoring chunks, decode steps).
        self.forward_passes = 0

    # ------------------------------------------------------------------ contract

    @abstractmethod
    def prefill(self, tokens: Sequence[int], parent: Prefix | None = None) -> Prefix: ...

    @abstractmethod
    def score(self, prefix: Prefix, rows: Sequence[Sequence[int]], queries: Sequence[Query]) -> list[np.ndarray]: ...

    def greedy(
        self,
        prefix: Prefix,
        inputs: Sequence[int],
        max_new_tokens: int,
        stop_ids: frozenset[int],
        should_stop: Callable[[list[int]], bool] | None = None,
    ) -> list[int]:
        """Greedy decoding after ``prefix`` + ``inputs``. Optional; used by the baseline."""
        raise NotImplementedError(f"{type(self).__name__} does not implement greedy decoding")

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_id}

    # ------------------------------------------------------------------ shared helpers

    def plan_chunks(self, prefix: Prefix, rows: Sequence[Sequence[int]], queries: Sequence[Query]) -> list[list[int]]:
        """Group the rows that have queries into forward passes.

        Rows are sorted by length so similar lengths share a pass (less padding). A pass
        holds at most ``max_rows`` rows and at most ``kv_budget_bytes`` of broadcast KV.
        """
        needed = sorted({q.row for q in queries}, key=lambda r: (len(rows[r]), r))
        if not needed:
            return []
        per_row = max(prefix.nbytes, 1)
        cap = max(1, min(self.max_rows, self.kv_budget_bytes // per_row))
        return [needed[i : i + cap] for i in range(0, len(needed), cap)]

    @staticmethod
    def validate(rows: Sequence[Sequence[int]], queries: Sequence[Query]) -> None:
        for q in queries:
            if not 0 <= q.row < len(rows):
                raise IndexError(f"query refers to row {q.row}, but there are {len(rows)} rows")
            if not 0 <= q.position < len(rows[q.row]):
                raise IndexError(f"query position {q.position} outside row {q.row} of length {len(rows[q.row])}")
            if not q.tokens:
                raise ValueError("query without token ids")


def pad_rows(rows: Sequence[Sequence[int]], pad_id: int) -> list[list[int]]:
    width = max(len(r) for r in rows)
    return [list(r) + [pad_id] * (width - len(r)) for r in rows]


def gather_layout(chunk: list[int], queries: Sequence[Query]) -> tuple[list[int], list[int], list[int], list[list[int]]]:
    """Flatten the queries of a chunk: (query ids, local row index, position, padded token ids)."""
    local = {r: i for i, r in enumerate(chunk)}
    qids = [i for i, q in enumerate(queries) if q.row in local]
    width = max(len(queries[i].tokens) for i in qids)
    toks = [list(queries[i].tokens) + [queries[i].tokens[0]] * (width - len(queries[i].tokens)) for i in qids]
    return qids, [local[queries[i].row] for i in qids], [queries[i].position for i in qids], toks


def verify_close(a: np.ndarray, b: np.ndarray, rtol: float = 5e-3) -> bool:
    """Whether two logit vectors agree (used to verify optional fast paths)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return False
    scale = 1.0 + float(np.max(np.abs(a)))
    return float(np.max(np.abs(a - b))) <= rtol * scale and int(np.argmax(a)) == int(np.argmax(b))

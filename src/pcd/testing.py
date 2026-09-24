"""Conformance checks for backends.

Run ``check_backend(backend)`` against any backend (built-in or your own) to verify the
properties the engine depends on. Each check computes the same next-token distributions
in two different ways and compares them, so it works with any model, including randomly
initialised ones.

Distributions are compared by total-variation distance over the whole vocabulary
(half the L1 distance between the two probability vectors). Rounding noise in bf16 or
4-bit models gives distances around 1e-3 to 1e-2; a broken cache, a pad token leaking
into a row, or a wrong position gives distances of 0.3 or more.

    >>> from pcd.backends import load_backend
    >>> from pcd.testing import check_backend
    >>> for c in check_backend(load_backend("qwen2.5-0.5b")): print(c)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends.base import Backend, Query

_TEXT = (
    "Parallel constrained decoding reads one distribution per field from a shared prefix. "
    "The quick brown fox jumps over the lazy dog, then files a support ticket about it."
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def check_backend(backend: Backend, tol: float = 0.1) -> list[Check]:
    """Verify prefix caching, batching, chunking, immutability and normalization.

    ``tol`` bounds the total-variation distance between distributions that should be
    identical. The default absorbs bf16 / 4-bit noise; use ~1e-4 for float32 models.
    """
    ids = backend.tokenizer.encode(_TEXT)
    if len(ids) < 36:
        ids = (ids * (36 // max(len(ids), 1) + 1))[:48]
    vocab = tuple(range(backend.tokenizer.vocab_size))
    a, b = ids[:12], ids[12:20]
    rows = [tuple(ids[20:23]), tuple(ids[23:28]), tuple(ids[28:29])]
    queries = [Query(i, len(r) - 1, vocab) for i, r in enumerate(rows)]
    checks: list[Check] = []

    def dist(x: list[np.ndarray], y: list[np.ndarray]) -> tuple[float, bool]:
        """Largest total-variation distance, and whether every argmax agrees."""
        tv = max(0.5 * float(np.abs(np.exp(p) - np.exp(q)).sum()) for p, q in zip(x, y))
        same = all(int(np.argmax(p)) == int(np.argmax(q)) for p, q in zip(x, y))
        return tv, same

    def check(name: str, x, y, extra_ok: bool = True) -> None:
        tv, same = dist(x, y)
        checks.append(Check(name, tv <= tol and extra_ok, f"max TV distance = {tv:.1e}, argmax {'same' if same else 'differs'}"))

    try:
        # 1. Extending a cached prefix equals prefilling the whole sequence at once.
        full = backend.prefill(a + b)
        parent = backend.prefill(a)
        ext = backend.prefill(b, parent=parent)
        check("prefix extension is exact", backend.score(full, rows, queries), backend.score(ext, rows, queries))

        # 2. Extending a prefix leaves the parent untouched.
        before = backend.score(parent, rows, queries)
        backend.prefill(ids[30:34], parent=parent)
        check("prefill does not modify its parent", before, backend.score(parent, rows, queries), len(parent) == len(a))

        # 3. A right-padded batch equals each row scored alone.
        batched = backend.score(full, rows, queries)
        single = [backend.score(full, [r], [Query(0, len(r) - 1, vocab)])[0] for r in rows]
        check("batched rows match single rows", batched, single)

        # 4. Splitting a batch into several passes changes nothing.
        saved = backend.max_rows
        try:
            backend.max_rows = 1
            chunked = backend.score(full, rows, queries)
        finally:
            backend.max_rows = saved
        check("chunking is invisible", batched, chunked)

        # 5. Scoring does not modify the prefix (repeating gives the same answer).
        check("score does not modify the prefix", batched, backend.score(full, rows, queries))

        # 6. A query at position p of a long row equals the last position of the row cut at p.
        long_row = tuple(ids[20:28])
        mid = backend.score(full, [long_row], [Query(0, p, vocab) for p in range(len(long_row))])
        cut = [backend.score(full, [long_row[: p + 1]], [Query(0, p, vocab)])[0] for p in range(len(long_row))]
        check("positions index into the row", mid, cut)

        # 7. Log-probs are normalized over the vocabulary.
        total = float(np.exp(batched[0]).sum())
        checks.append(Check("log-probs are normalized", 0.98 <= total <= 1.0 + 1e-3, f"sum p over tokenizer vocab = {total:.4f}"))
    except Exception as e:  # report instead of crashing, so `pcd check` shows what broke
        checks.append(Check("backend runs", False, f"{type(e).__name__}: {e}"))
    return checks

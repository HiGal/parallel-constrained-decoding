"""Backend-independent decoding of compiled fields.

Every field is a token trie over its *variants* (label x surface form, see
``compiler.py``). Three ways to read a decision from the model:

``marginal`` (default for up to ``exact_max`` labels)
    Score every variant up to the token that *identifies* its label, and sum the
    variants of each label. For labels whose first tokens differ this is plain
    first-token scoring (notebook 03); for colliding labels such as ``1_HOUR`` /
    ``12_HOURS`` it follows each branch just deep enough to tell them apart
    (notebook 04). All the distributions it needs come from a handful of rows in the
    same batched pass, e.g. one row ``prefix + ' "'`` covers a boolean's bare
    ``true``/``false`` and its quoted ``"true"``/``"false"`` forms.

``exact``
    Score every variant's full token sequence, including its closing ``",\\n``, and sum
    per label (notebook 04 §6). Lets tokens after the identifying one (for example the
    name after a numeric code) weigh in.

``trie`` / ``hybrid`` (default above ``exact_max`` labels)
    Greedy walk: at each branch restrict the next-token distribution to the tokens that
    keep a variant alive, renormalize, follow the best. Forced tokens cost no model
    call. With ``hybrid_exact_below`` the walk switches to exact scoring once few labels
    survive, so the words after a code can decide (notebook 04 §7).

All pending fields advance together: every round gathers the rows and queries of all
fields and sends them to the backend in one call. Marginal and exact fields finish in a
single round; trie fields need one round per branch level.

Probabilities are renormalized over the allowed labels: they describe how the model
splits its belief among the options, not how much it believes any of them. The field's
``candidate_mass`` is the unconstrained probability that the model's own continuation
reaches a token identifying some allowed label (for trie fields: that its next token
starts one). Low mass means the model wanted to write something else (notebook 03 §4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .backends.base import Backend, Prefix, Query
from .compiler import CompiledField
from .result import FieldResult

Strategy = Literal["auto", "exact", "greedy"]


@dataclass(frozen=True)
class DecodeOptions:
    """How fields are decided."""

    #: "auto": marginal scoring up to ``exact_max`` labels, hybrid above.
    #: "exact": full-string scoring up to ``exact_max`` labels, hybrid above.
    #: "greedy": greedy trie walk only (fewest rows, can need several rounds).
    strategy: Strategy = "auto"
    exact_max: int = 32
    hybrid_exact_below: int = 10
    #: Sharpens (<1) or flattens (>1) the reported probabilities. Never changes the winner.
    temperature: float = 1.0
    #: How many alternatives to report per field.
    top_k: int = 5
    #: Decide the schema's first field before the others and write its answer into the
    #: prompt. A key placed straight after ``{`` reads as "the first key", and small models
    #: answer it worse when it is not the first key of the catalog (Qwen3-0.6B: 1% -> 95%
    #: on the right label with any line in front). Costs one extra round (~15% latency).
    anchor_first: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in ("auto", "exact", "greedy"):
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


def field_plan(cf: CompiledField, opts: DecodeOptions) -> tuple[str, int]:
    """(mode, exact_below): mode is "marginal", "exact" or "trie"."""
    k = len(cf.labels)
    if opts.strategy == "greedy":
        return "trie", 0
    if k > opts.exact_max:
        return "trie", opts.hybrid_exact_below
    return ("exact" if opts.strategy == "exact" else "marginal"), 0


class FieldDecoder:
    """State machine that decides one field over one or more rounds."""

    def __init__(self, cf: CompiledField, opts: DecodeOptions):
        self.cf = cf
        self.opts = opts
        self.mode, self.exact_below = field_plan(cf, opts)
        self.alive = list(range(len(cf.variants)))
        self.path: list[int] = []  # tokens chosen after cf.prefix (trie mode)
        self.prob = 1.0
        self.mass: float | None = None
        self.top: dict[int, float] = {}
        self.branches = 0
        self.scored = ""  # "", "marginal" or "exact": how the final step was scored
        self.deep = False  # marginal scoring needed nodes below the decision position
        self.done = len(cf.labels) == 1
        self._pending: tuple[str, Any] | None = None

    # ---- round protocol -------------------------------------------------------

    def request(self, rows: list[tuple[int, ...]], queries: list[Query]) -> None:
        if self.mode == "trie":
            self._skip_forced()
            if self.done:
                return
            if len(self._alive_labels()) <= self.exact_below:
                self._request_tree(rows, queries, full=True)
            else:
                self._request_branch(rows, queries)
        else:
            self._request_tree(rows, queries, full=self.mode == "exact")

    def consume(self, logps: Sequence[np.ndarray]) -> None:
        kind, payload = self._pending
        self._pending = None
        if kind == "branch":
            self._consume_branch(logps, *payload)
        else:
            self._consume_tree(logps, *payload)

    # ---- tree scoring (marginal / exact) ---------------------------------------

    def _last_depth(self, vi: int, full: bool) -> int:
        v = self.cf.variants[vi]
        return len(v.seq) - 1 if full else max(v.ident, len(self.path))

    def _request_tree(self, rows, queries, full: bool) -> None:
        d0 = len(self.path)
        base = self.cf.prefix + tuple(self.path)
        children: dict[tuple[int, ...], set[int]] = {}
        for vi in self.alive:
            s = self.cf.variants[vi].seq
            for j in range(d0, self._last_depth(vi, full) + 1):
                children.setdefault(s[d0:j], set()).add(s[j])
        # One row per maximal node path; every node is read from a row that contains it.
        paths: list[tuple[int, ...]] = []
        for node in sorted(children, key=len, reverse=True):
            if not any(p[: len(node)] == node for p in paths):
                paths.append(node)
        row_of = {}
        for p in paths:
            row_of[p] = len(rows)
            rows.append(base + p)
        nodes = {}
        for node, kids in children.items():
            p = next(p for p in paths if p[: len(node)] == node)
            toks = tuple(sorted(kids))
            nodes[node] = (len(queries), toks)
            queries.append(Query(row_of[p], len(base) - 1 + len(node), toks))
        self.deep = self.deep or any(len(n) > 0 for n in children)
        self._pending = ("tree", (nodes, full, d0))

    def _consume_tree(self, logps, nodes, full: bool, d0: int) -> None:
        lp = {node: dict(zip(toks, np.asarray(logps[qi], dtype=np.float64))) for node, (qi, toks) in nodes.items()}
        by_label: dict[int, list[float]] = {}
        ident_mass = 0.0
        for vi in self.alive:
            v = self.cf.variants[vi]
            steps = [lp[v.seq[d0:j]][v.seq[j]] for j in range(d0, self._last_depth(vi, full) + 1)]
            by_label.setdefault(v.label, []).append(float(sum(steps)))
            ident_mass += math.exp(sum(steps[: max(v.ident, d0) - d0 + 1]))
        if d0 == 0:
            self.mass = ident_mass
        labels = list(by_label)
        scores = np.array([_logsumexp(by_label[li]) for li in labels])
        p = _softmax(scores / self.opts.temperature)
        before = self.prob
        j = int(np.argmax(p))
        self.prob *= float(p[j])
        self.top = {li: before * float(pi) for li, pi in zip(labels, p)}
        self.alive = [vi for vi in self.alive if self.cf.variants[vi].label == labels[j]]
        self.scored = "exact" if full else "marginal"
        self.done = True

    # ---- greedy trie walk -----------------------------------------------------

    def _alive_labels(self) -> set[int]:
        return {self.cf.variants[vi].label for vi in self.alive}

    def _skip_forced(self) -> None:
        """Append tokens every surviving variant agrees on; they need no model call."""
        while True:
            if len(self._alive_labels()) == 1:
                self.done = True
                return
            if len(self._alive_labels()) <= self.exact_below:
                return
            d = len(self.path)
            nxt = {self.cf.variants[vi].seq[d] for vi in self.alive}
            if len(nxt) != 1:
                return
            self.path.append(nxt.pop())

    def _request_branch(self, rows, queries) -> None:
        d = len(self.path)
        options = tuple(sorted({self.cf.variants[vi].seq[d] for vi in self.alive}))
        base = self.cf.prefix + tuple(self.path)
        rows.append(base)
        queries.append(Query(len(rows) - 1, len(base) - 1, options))
        self._pending = ("branch", (len(queries) - 1, options))

    def _consume_branch(self, logps, qi: int, options: tuple[int, ...]) -> None:
        lp = np.asarray(logps[qi], dtype=np.float64)
        d = len(self.path)
        if d == 0:
            self.mass = float(np.exp(lp).sum())
        p = _softmax(lp / self.opts.temperature)
        j = int(np.argmax(p))
        groups = {t: [vi for vi in self.alive if self.cf.variants[vi].seq[d] == t] for t in options}
        before = self.prob
        self.prob *= float(p[j])
        self.branches += 1
        # Alternatives whose probability is known: branches that lead to a single label.
        self.top = {}
        for t, pi in zip(options, p):
            labs = {self.cf.variants[vi].label for vi in groups[t]}
            if len(labs) == 1:
                li = labs.pop()
                self.top[li] = self.top.get(li, 0.0) + before * float(pi)
        self.alive = groups[options[j]]
        self.path.append(options[j])
        if len(self._alive_labels()) == 1:
            self.done = True

    # ---- result ---------------------------------------------------------------

    def result(self, wave: int = 0) -> FieldResult:
        labels = self.cf.labels
        winner = self.cf.variants[self.alive[0]].label
        if len(labels) == 1:
            method, prob, mass = "fixed", 1.0, 1.0
        else:
            prob, mass = self.prob, self.mass
            if self.scored == "exact":
                method = "hybrid" if self.branches else "exact"
            elif self.scored == "marginal":
                method = "marginal" if self.deep else "first_token"
            else:
                method = "trie"
        top = sorted(self.top.items(), key=lambda kv: -kv[1])[: self.opts.top_k]
        return FieldResult(
            name=self.cf.name,
            value=labels[winner],
            probability=prob,
            candidate_mass=mass,
            method=method,
            top=[(labels[li], p) for li, p in top] or [(labels[winner], prob)],
            wave=wave,
        )


def decode_fields(
    backend: Backend, prefix: Prefix, fields: Sequence[CompiledField], opts: DecodeOptions, wave: int = 0
) -> tuple[dict[str, FieldResult], int, int]:
    """Decide ``fields`` against ``prefix``. Returns (results, rounds, rows scored)."""
    decoders = [FieldDecoder(cf, opts) for cf in fields]
    rounds = rows_scored = 0
    while True:
        rows: list[tuple[int, ...]] = []
        queries: list[Query] = []
        active = []
        for d in decoders:
            if not d.done:
                d.request(rows, queries)
                if not d.done:
                    active.append(d)
        if not active:
            break
        logps = backend.score(prefix, rows, queries)
        rounds += 1
        rows_scored += len(rows)
        for d in active:
            d.consume(logps)
    return {d.cf.name: d.result(wave) for d in decoders}, rounds, rows_scored


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    if not math.isfinite(m):
        return np.full_like(x, 1.0 / len(x))
    e = np.exp(x - m)
    return e / e.sum()


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))

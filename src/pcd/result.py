"""Result types. Everything reported here is measured, nothing is hard-coded."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FieldResult:
    name: str
    value: Any
    #: Probability of ``value`` renormalized over the allowed labels (for trie fields: the
    #: probability of the chosen path under the constraint). Not a calibrated confidence.
    probability: float
    #: Unconstrained probability that the model's next token at the decision position
    #: starts an allowed label. Low values mean the model wanted to write something else.
    candidate_mass: float | None
    #: "first_token", "exact", "trie", "hybrid" or "fixed" (single-choice field).
    method: str
    #: Best alternatives with their probabilities (only labels whose probability is known).
    top: list[tuple[Any, float]] = field(default_factory=list)
    #: Dependency wave the field was decided in (0 = first).
    wave: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["top"] = [{"label": lab, "probability": p} for lab, p in self.top]
        return d


@dataclass
class Stats:
    total_ms: float = 0.0
    tokenize_ms: float = 0.0
    #: Prefill of the static head, only when it was not cached yet (0 when reused).
    head_prefill_ms: float = 0.0
    #: Prefill of the per-request context (on top of the cached head).
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    prompt_tokens: int = 0
    #: Prompt tokens whose KV state came from the cache.
    cached_tokens: int = 0
    #: Model forward calls actually made for this request (counted by the backend).
    forward_passes: int = 0
    #: Sequential scoring rounds (each is one batched call, possibly several passes).
    decode_rounds: int = 0
    rows_scored: int = 0
    waves: int = 1


@dataclass
class Extraction:
    #: The output object: every key present, in schema order, with real booleans.
    values: dict[str, Any]
    fields: dict[str, FieldResult]
    stats: Stats
    backend: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def low_confidence(self, min_probability: float = 0.5, min_mass: float = 0.2) -> list[str]:
        """Fields worth a second look: an unsure pick, or a model that wanted another answer."""
        out = []
        for name, f in self.fields.items():
            if f.probability < min_probability or (f.candidate_mass is not None and f.candidate_mass < min_mass):
                out.append(name)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "stats": asdict(self.stats),
            "backend": self.backend,
            "warnings": self.warnings,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.values, ensure_ascii=False, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Extraction({len(self.values)} fields, {self.stats.total_ms:.0f} ms, "
            f"{self.stats.forward_passes} forward passes)"
        )

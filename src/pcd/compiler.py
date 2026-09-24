"""Tokenizer-aware schema compilation.

For every field, label and *surface form* of that label we tokenize the complete JSON
line the model would write, in context, i.e. right after the prompt's tail:

    tail + '  "risk_tier": "HIGH",\\n'

and strip the tail's tokens. This gives each form's exact token sequence as the model
saw it in training. Two details matter a lot in practice:

* Tokenizing pieces separately produces splits the model never saw (the original
  engine's ``": "`` + ``"true"``; notebook 03 §3).
* The line must include its newline, where the tokenizer puts it: Qwen merges ``",\\n``
  into one token and gives the bare ``",`` almost no probability (~7 nats of noise on
  every full-string score), while GPT-2-style tokenizers glue the newline to the next
  line's indentation. ``compile_schema`` picks the layout that tokenizes cleanly.

The field's row is the longest common *token* prefix of all its sequences; its last
token is the decision position. What follows is one *variant* per (label, form): a
token trie that the decoders walk. Each variant records the depth at which its label is
*identified* (no other label shares the prefix up to that token).

Compilation runs once per (schema, tokenizer, prompt format) and is cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompt import PromptFormat, RenderedPrompt, build_prompt
from .schema import Field, Schema
from .tokenizer import Tokenizer


class TokenizationError(RuntimeError):
    """The tokenizer merges tokens across a boundary the engine relies on."""


@dataclass(frozen=True)
class Variant:
    #: Index into the field's labels.
    label: int
    #: Tokens after the field prefix: value, closing punctuation, newline.
    seq: tuple[int, ...]
    #: Index in ``seq`` of the first token that only this variant's label can produce.
    ident: int


@dataclass(frozen=True)
class CompiledField:
    field: Field
    #: Tokens appended to the shared prompt for this field. The last one is the decision position.
    prefix: tuple[int, ...]
    #: One entry per (label, surface form), canonical form first within each label.
    variants: tuple[Variant, ...]

    @property
    def name(self) -> str:
        return self.field.name

    @property
    def labels(self) -> tuple:
        return self.field.labels

    @property
    def has_collision(self) -> bool:
        """Two labels' canonical forms start with the same token."""
        firsts = [v.seq[0] for v in self.canonical_variants()]
        return len(set(firsts)) < len(firsts)

    def canonical_variants(self) -> list[Variant]:
        seen, out = set(), []
        for v in self.variants:
            if v.label not in seen:
                seen.add(v.label)
                out.append(v)
        return out


@dataclass(frozen=True)
class CompiledSchema:
    schema: Schema
    fmt: PromptFormat
    prompt: RenderedPrompt
    head_ids: tuple[int, ...]
    tail_ids: tuple[int, ...]
    fields: dict[str, CompiledField]

    def answer_text(self, answers: dict[str, Any]) -> str:
        """JSON lines for already-decided fields, in schema order (written between waves)."""
        return "".join(
            self.fmt.field_line(self.schema[n], answers[n], layout=self.prompt.layout)
            for n in self.schema.names if n in answers
        )


def compile_schema(schema: Schema, tokenizer: Tokenizer, fmt: PromptFormat) -> CompiledSchema:
    """Compile for the first newline layout whose token boundaries are clean."""
    errors = []
    for layout in fmt.layouts():
        try:
            return _compile(schema, tokenizer, fmt, layout)
        except TokenizationError as e:
            errors.append(f"{layout!r} layout: {e}")
    raise TokenizationError("no JSON layout tokenizes cleanly for this tokenizer:\n  " + "\n  ".join(errors))


def _compile(schema: Schema, tokenizer: Tokenizer, fmt: PromptFormat, layout: str) -> CompiledSchema:
    prompt = build_prompt(schema, tokenizer, fmt, layout)
    # A second anchor that ends like a prompt extended with earlier answers (wave > 0).
    dummy = Field("_", "enum", choices=("x",))
    wave_anchor = prompt.tail + fmt.field_line(dummy, "x", layout=layout)
    fields = {f.name: _compile_field(f, tokenizer, fmt, layout, prompt.tail, wave_anchor) for f in schema}
    return CompiledSchema(
        schema=schema,
        fmt=fmt,
        prompt=prompt,
        head_ids=tuple(tokenizer.encode(prompt.head)),
        tail_ids=tuple(tokenizer.encode(prompt.tail)),
        fields=fields,
    )


def encode_after(tokenizer: Tokenizer, anchor: str, text: str, what: str = "text") -> tuple[int, ...]:
    """Tokens of ``text`` as they appear right after ``anchor``.

    Raises TokenizationError if the anchor's own tokens change when ``text`` follows,
    i.e. if the tokenizer merges across the boundary.
    """
    a = tokenizer.encode(anchor)
    full = tokenizer.encode(anchor + text)
    if full[: len(a)] != a:
        raise TokenizationError(
            f"tokenization of {what} merges with the text before it "
            f"(anchor ends {anchor[-12:]!r}, text starts {text[:24]!r}). "
            "Try a different PromptFormat.indent, or see CONTRIBUTING.md."
        )
    return tuple(full[len(a):])


def _compile_field(
    f: Field, tok: Tokenizer, fmt: PromptFormat, layout: str, anchor: str, wave_anchor: str
) -> CompiledField:
    what = f"field {f.name!r}"
    entries: list[tuple[int, tuple[int, ...]]] = []
    for li, lab in enumerate(f.labels):
        for form in fmt.value_forms(f, lab):
            entries.append((li, encode_after(tok, anchor, fmt.field_line(f, lab, form, layout), what)))

    # The same line must tokenize identically after earlier answers (wave > 0).
    if encode_after(tok, wave_anchor, fmt.field_line(f, f.labels[0], layout=layout), what) != entries[0][1]:
        raise TokenizationError(f"{what} tokenizes differently after a previous answer line")

    seqs = [s for _, s in entries]
    if len(entries) == 1:
        return CompiledField(field=f, prefix=seqs[0], variants=(Variant(0, (), 0),))

    n = _common_prefix_len(seqs)
    if n == 0:
        raise TokenizationError(f"{what}: its lines share no leading token (not even the key)")
    rest = [s[n:] for s in seqs]
    ordered = sorted(rest)
    for a, b in zip(ordered, ordered[1:]):
        if not a or b[: len(a)] == a:
            raise TokenizationError(f"{what}: one value's tokens are a prefix of another's")

    variants = []
    for (li, _), s in zip(entries, rest):
        ident = next(
            j for j in range(len(s))
            if all(l2 == li for (l2, _), s2 in zip(entries, rest) if s2[: j + 1] == s[: j + 1])
        )
        variants.append(Variant(li, s, ident))
    return CompiledField(field=f, prefix=seqs[0][:n], variants=tuple(variants))


def _common_prefix_len(seqs: list[tuple[int, ...]]) -> int:
    n = min(len(s) for s in seqs)
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs):
            return i
    return n

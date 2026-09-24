"""Prompt construction.

The shared prompt has three parts:

    head     static: system turn with the instructions and the field catalog,
             and the start of the user turn. Identical for every request with the
             same schema, so its KV cache is computed once and reused.
    context  the per-request input.
    tail     static: the end of the user turn, the assistant header, and the JSON
             opener (``{\\n`` or ``{``). Every field row continues from here.

The parts come from the model's own chat template, rendered once with a marker in
place of the context. Templates that reject a system turn (Gemma 2, some Mistral
versions) get the system text merged into the user turn; tokenizers without a chat
template (base models) get a plain-text format.

The catalog lists every allowed option. Without the list, most of a small model's
probability falls outside the candidates (notebook 03 §4) and numeric codes become
guesswork (notebook 04 §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schema import BOOLEAN, Field, Schema, render_label
from .tokenizer import Tokenizer

DEFAULT_INSTRUCTIONS = (
    "You are a precise classification engine. Read the input and fill in a JSON object "
    "with exactly the keys listed below. Every value must be copied exactly from that "
    "key's allowed options."
)

_MARKER = "[[[PCD_CONTEXT_SLOT]]]"


@dataclass(frozen=True)
class PromptFormat:
    """Knobs for the shared prompt. The defaults work for instruction-tuned chat models."""

    instructions: str = DEFAULT_INSTRUCTIONS
    #: Text placed in the user turn before the context ("" to omit).
    input_label: str = "Input:"
    #: True lists every option; an int lists at most that many per field; False lists none.
    list_choices: bool | int = True
    #: None: use the chat template when the tokenizer has one. False: always plain text.
    use_chat_template: bool | None = None
    #: None: auto-detect whether the template accepts a system turn. False: merge into user turn.
    system_role: bool | None = None
    #: Extra variables for the chat template. ``enable_thinking=False`` turns off
    #: Qwen3-style reasoning blocks; templates that do not use it ignore it.
    template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    #: Text the assistant turn starts with before the JSON, for formats that need it
    #: (e.g. gpt-oss: ``<|channel|>final<|message|>``). See ``FAMILY_DEFAULTS``.
    assistant_prefix: str = ""
    #: Indentation of each key line in the JSON the model "is writing".
    indent: str = "  "
    #: Also accept booleans written as strings (``"true"``) and add their probability to the
    #: label. Small models often quote booleans; without this, a model that puts 80% on
    #: ``"true"`` can be read as answering ``false`` (see README, "Surface forms").
    quoted_booleans: bool = True
    #: Where each JSON line's newline goes. "after": ``{\\n`` + ``  "k": v,\\n`` (GPT-4-style
    #: and SentencePiece tokenizers glue the newline to the punctuation before it). "before":
    #: ``{`` + ``\\n  "k": v,`` (GPT-2-style tokenizers glue it to the indentation after it).
    #: "auto" picks the first layout whose token boundaries are clean for the tokenizer.
    newline: str = "auto"

    def key(self) -> tuple:
        """Hashable identity, used to cache compiled schemas."""
        return (
            self.instructions, self.input_label, self.list_choices, self.use_chat_template,
            self.system_role, tuple(sorted(self.template_kwargs.items())), self.assistant_prefix,
            self.indent, self.quoted_booleans, self.newline,
        )

    def layouts(self) -> list[str]:
        if self.newline == "auto":
            return ["after", "before"]
        if self.newline not in ("after", "before"):
            raise ValueError(f"newline must be 'auto', 'after' or 'before', not {self.newline!r}")
        return [self.newline]

    # ------------------------------------------------------------------ JSON pieces

    def value_forms(self, f: Field, label: Any) -> list[str]:
        """Ways the model may write ``label``; the first is the canonical one used in output."""
        canonical = render_label(label)
        if f.type == BOOLEAN and self.quoted_booleans:
            return [canonical, f'"{canonical}"']
        return [canonical]

    def field_line(self, f: Field, label: Any, form: str | None = None, layout: str = "after") -> str:
        """One line of the JSON object, including its newline (see ``newline``).

        The newline matters: Qwen writes ``",\\n`` as one token and gives the bare ``",``
        almost no probability, which would add ~7 nats of noise to every full-string score.
        """
        value = render_label(label) if form is None else form
        line = f"{self.indent}{render_label(f.name)}: {value},"
        return line + "\n" if layout == "after" else "\n" + line

    def catalog_line(self, f: Field) -> str:
        desc = f.description.rstrip(" .")
        head = f"- {render_label(f.name)}" + (f": {desc}." if desc else ":")
        if self.list_choices is False:
            return head
        if f.type == BOOLEAN:
            return f"{head} Allowed: true | false"
        choices = [render_label(c) for c in f.choices]
        limit = len(choices) if self.list_choices is True else int(self.list_choices)
        shown = " | ".join(choices[:limit])
        if len(choices) > limit:
            shown += f" | ... ({len(choices)} options in total)"
        return f"{head} Allowed: {shown}"

    def system_text(self, schema: Schema) -> str:
        lines = "\n".join(self.catalog_line(f) for f in schema)
        return f"{self.instructions}\n\nKeys:\n{lines}"


@dataclass(frozen=True)
class RenderedPrompt:
    head: str
    tail: str
    #: "chat" (system turn), "chat-merged" (system text inside the user turn) or "plain".
    mode: str
    #: Newline layout of the JSON lines ("after" or "before", see ``PromptFormat.newline``).
    layout: str = "after"

    @property
    def opener(self) -> str:
        return JSON_OPENERS[self.layout]

    def render(self, context: str) -> str:
        return self.head + context + self.tail


JSON_OPENERS = {"after": "{\n", "before": "{"}

#: PromptFormat overrides applied automatically by model type (``config.model_type``).
FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    # Harmony format: the answer goes in the "final" channel, after the analysis channel.
    "gpt_oss": {"assistant_prefix": "<|channel|>final<|message|>"},
}


def build_prompt(schema: Schema, tokenizer: Tokenizer, fmt: PromptFormat, layout: str = "after") -> RenderedPrompt:
    system = fmt.system_text(schema)
    user = (f"{fmt.input_label}\n" if fmt.input_label else "") + _MARKER
    kwargs = dict(fmt.template_kwargs)

    rendered, mode = None, "plain"
    if fmt.use_chat_template is not False:
        if fmt.system_role is not False:
            rendered = _try_render(
                tokenizer, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                kwargs, must_contain=system,
            )
            mode = "chat"
        if rendered is None:
            rendered = _try_render(
                tokenizer, [{"role": "user", "content": f"{system}\n\n{user}"}], kwargs,
                must_contain=system,
            )
            mode = "chat-merged"
        if rendered is None and fmt.use_chat_template:
            raise ValueError("the tokenizer's chat template could not be rendered")
        if rendered is not None and tokenizer.bos_text and not rendered.startswith(tokenizer.bos_text):
            rendered = tokenizer.bos_text + rendered
    if rendered is None:
        mode = "plain"
        rendered = f"{tokenizer.bos_text}{system}\n\n{user}\n\nJSON:\n"

    if rendered.count(_MARKER) != 1:
        raise ValueError("the chat template altered or duplicated the context slot")
    head, tail = rendered.split(_MARKER)
    return RenderedPrompt(
        head=head, tail=tail + fmt.assistant_prefix + JSON_OPENERS[layout], mode=mode, layout=layout
    )


def _try_render(tokenizer: Tokenizer, messages: list[dict], kwargs: dict, must_contain: str) -> str | None:
    try:
        out = tokenizer.render_chat(messages, **kwargs)
    except Exception:  # templates raise on unsupported roles
        return None
    if out is None or _MARKER not in out or must_contain not in out:
        return None
    return out

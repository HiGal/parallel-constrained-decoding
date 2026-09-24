"""The tokenizer interface the engine relies on, and an adapter for Hugging Face tokenizers.

The engine never calls a tokenizer library directly. It needs exactly:

* ``encode(text)``: token ids *without* adding special tokens (the chat template
  already contains them as text);
* ``decode(ids)``;
* ``render_chat(messages, **kwargs)``: the chat template rendered to text with the
  generation prompt appended, or ``None`` when the tokenizer has no template;
* ``bos_text``: text to put in front of a non-chat prompt, if the model expects BOS;
* ``eos_ids``, ``pad_id``, ``vocab_size``.

Anything that provides these can drive the engine (see CONTRIBUTING.md).
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def render_chat(self, messages: list[dict], **template_kwargs: Any) -> str | None: ...

    @property
    def bos_text(self) -> str: ...

    @property
    def eos_ids(self) -> frozenset[int]: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...


# End-of-turn markers used by popular chat formats. They are stop tokens for the
# autoregressive baseline even when a tokenizer does not list them as EOS.
_END_OF_TURN = (
    "<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
    "</s>", "<|end_of_text|>", "<|return|>",
)


class HFTokenizer:
    """Adapter for ``transformers`` tokenizers and mlx-lm's ``TokenizerWrapper``."""

    def __init__(self, tokenizer: Any):
        # mlx-lm wraps the HF tokenizer in `_tokenizer`; unwrap it so both backends behave the
        # same. (HF fast tokenizers also have `_tokenizer`, the Rust object: leave those alone.)
        inner = getattr(tokenizer, "_tokenizer", None)
        self.raw = inner if hasattr(inner, "apply_chat_template") else tokenizer
        extra_eos = getattr(tokenizer, "eos_token_ids", None) or ()
        self._eos = self._collect_eos(extra_eos)
        self._bos_text = self._detect_bos()

    def encode(self, text: str) -> list[int]:
        return list(self.raw.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self.raw.decode(list(ids), skip_special_tokens=False)

    def render_chat(self, messages: list[dict], **template_kwargs: Any) -> str | None:
        if not getattr(self.raw, "chat_template", None):
            return None
        out = self.raw.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        if not isinstance(out, str):  # some versions return a list for batched input
            out = out[0]
        return out

    @property
    def bos_text(self) -> str:
        return self._bos_text

    @property
    def eos_ids(self) -> frozenset[int]:
        return self._eos

    @property
    def pad_id(self) -> int:
        for tid in (getattr(self.raw, "pad_token_id", None), getattr(self.raw, "eos_token_id", None)):
            if isinstance(tid, int):
                return tid
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.raw)

    # ------------------------------------------------------------------ helpers

    def _collect_eos(self, extra: Any) -> frozenset[int]:
        ids: set[int] = set()
        eos = getattr(self.raw, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple, set)):
            ids.update(int(e) for e in eos)
        if isinstance(extra, int):  # transformers v5 exposes a single id here, mlx-lm a set
            extra = (extra,)
        ids.update(int(e) for e in extra if e is not None)
        vocab: Mapping[str, int] = self.raw.get_vocab()
        for tok in _END_OF_TURN:
            if tok in vocab:
                ids.add(vocab[tok])
        return frozenset(ids)

    def _detect_bos(self) -> str:
        """Return the BOS text if encoding with special tokens would normally prepend it."""
        bos = getattr(self.raw, "bos_token", None)
        bos_id = getattr(self.raw, "bos_token_id", None)
        if not bos or bos_id is None:
            return ""
        try:
            with_special = self.raw.encode("a", add_special_tokens=True)
        except Exception:
            return ""
        return bos if with_special and with_special[0] == bos_id else ""

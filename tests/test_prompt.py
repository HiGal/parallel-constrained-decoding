import pytest
from helpers import NO_SYSTEM, byte_tokenizer

from pcd import PromptFormat, Schema
from pcd.prompt import build_prompt
from pcd.tokenizer import HFTokenizer

SCHEMA = Schema.from_dict({
    "urgent": {"type": "boolean", "description": "Whether it is urgent."},
    "tier": {"type": "enum", "choices": ["TIER_1", "TIER_2"], "description": "Support tier"},
    "code": {"type": "enum", "choices": [f"C{i}" for i in range(12)]},
})


def test_chat_prompt_splits_around_context(byte_tok):
    tok = HFTokenizer(byte_tok)
    fmt = PromptFormat()
    p = build_prompt(SCHEMA, tok, fmt)
    assert p.mode == "chat"
    assert p.head.startswith("<|im_start|>system\n") and p.head.endswith("<|im_start|>user\nInput:\n")
    assert p.tail == "<|im_end|>\n<|im_start|>assistant\n{\n"
    # Rendering the real messages gives exactly head + context + tail.
    ctx = "Server is down!"
    real = byte_tok.apply_chat_template(
        [{"role": "system", "content": fmt.system_text(SCHEMA)}, {"role": "user", "content": "Input:\n" + ctx}],
        tokenize=False, add_generation_prompt=True,
    )
    assert p.render(ctx) == real + "{\n"


def test_catalog_lists_every_choice():
    fmt = PromptFormat()
    text = fmt.system_text(SCHEMA)
    assert '- "urgent": Whether it is urgent. Allowed: true | false' in text
    assert '- "tier": Support tier. Allowed: "TIER_1" | "TIER_2"' in text
    short = PromptFormat(list_choices=3).system_text(SCHEMA)
    assert '"C0" | "C1" | "C2" | ... (12 options in total)' in short
    assert "Allowed" not in PromptFormat(list_choices=False).system_text(SCHEMA)


def test_system_role_fallback_and_plain_mode():
    tok = HFTokenizer(byte_tokenizer(NO_SYSTEM))
    p = build_prompt(SCHEMA, tok, PromptFormat())
    assert p.mode == "chat-merged"
    assert "system" not in p.head and PromptFormat().instructions in p.head

    plain = build_prompt(SCHEMA, HFTokenizer(byte_tokenizer(None)), PromptFormat())
    assert plain.mode == "plain" and plain.tail.endswith("JSON:\n{\n")

    with pytest.raises(ValueError):
        build_prompt(SCHEMA, HFTokenizer(byte_tokenizer(None)), PromptFormat(use_chat_template=True))


def test_assistant_prefix_and_field_lines(byte_tok):
    tok = HFTokenizer(byte_tok)
    p = build_prompt(SCHEMA, tok, PromptFormat(assistant_prefix="<|channel|>final<|message|>"))
    assert p.tail.endswith("assistant\n<|channel|>final<|message|>{\n")
    fmt = PromptFormat()
    assert fmt.field_line(SCHEMA["urgent"], True) == '  "urgent": true,\n'
    assert fmt.field_line(SCHEMA["urgent"], True, layout="before") == '\n  "urgent": true,'
    assert fmt.layouts() == ["after", "before"] and PromptFormat(newline="before").layouts() == ["before"]
    before = build_prompt(SCHEMA, tok, fmt, layout="before")
    assert before.tail.endswith("assistant\n{") and before.opener == "{"
    assert fmt.value_forms(SCHEMA["urgent"], False) == ["false", '"false"']
    assert fmt.value_forms(SCHEMA["tier"], "TIER_1") == ['"TIER_1"']
    assert PromptFormat(quoted_booleans=False).value_forms(SCHEMA["urgent"], True) == ["true"]

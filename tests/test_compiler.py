import pytest

from pcd import PromptFormat, Schema, TokenizationError
from pcd.compiler import compile_schema, encode_after
from pcd.tokenizer import HFTokenizer


def _text(tok, ids):
    return tok.decode(list(ids))


def test_byte_level_compilation(byte_tok):
    tok = HFTokenizer(byte_tok)
    schema = Schema.from_dict({
        "urgent": {"type": "boolean"},
        "tier": {"type": "enum", "choices": ["TIER_1", "TIER_2", "SENIOR"]},
        "only": {"type": "enum", "choices": ["X"]},
    })
    cs = compile_schema(schema, tok, PromptFormat())
    urgent = cs.fields["urgent"]
    # The row stops right before the first character where the lines differ.
    assert _text(tok, urgent.prefix) == '  "urgent": '
    got = {(v.label, _text(tok, v.seq), v.ident) for v in urgent.variants}
    assert got == {(0, "true,\n", 0), (1, "false,\n", 0), (0, '"true",\n', 1), (1, '"false",\n', 1)}

    tier = cs.fields["tier"]
    assert _text(tok, tier.prefix) == '  "tier": "'
    assert [(_text(tok, v.seq), v.ident) for v in tier.variants] == [('TIER_1",\n', 5), ('TIER_2",\n', 5), ('SENIOR",\n', 0)]
    assert tier.has_collision and not urgent.has_collision

    only = cs.fields["only"]
    assert len(only.variants) == 1 and _text(tok, only.prefix) == '  "only": "X",\n'


def test_prefix_labels_and_escaping(byte_tok):
    tok = HFTokenizer(byte_tok)
    schema = Schema.from_dict({
        "risk": {"type": "enum", "choices": ["HIGH", "HIGH_RISK", 'say "hi"', "naïve"]},
    })
    cf = compile_schema(schema, tok, PromptFormat()).fields["risk"]
    texts = [_text(tok, cf.prefix + v.seq) for v in cf.variants]
    assert texts == ['  "risk": "HIGH",\n', '  "risk": "HIGH_RISK",\n', '  "risk": "say \\"hi\\"",\n', '  "risk": "naïve",\n']
    # HIGH is identified by its closing quote, HIGH_RISK by the underscore.
    ident = {cf.labels[v.label]: _text(tok, v.seq[: v.ident + 1]) for v in cf.variants}
    assert ident["HIGH"] == 'HIGH"' and ident["HIGH_RISK"] == "HIGH_"


def test_answer_text_for_waves(byte_tok):
    tok = HFTokenizer(byte_tok)
    schema = Schema.from_dict({"a": {"type": "boolean"}, "b": {"type": "enum", "choices": ["X", "Y"], "depends_on": ["a"]}})
    cs = compile_schema(schema, tok, PromptFormat())
    assert cs.answer_text({"a": True}) == '  "a": true,\n'
    assert cs.answer_text({"b": "Y", "a": False}) == '  "a": false,\n  "b": "Y",\n'


def test_encode_after_detects_merges():
    class MergingTokenizer(HFTokenizer):
        """Pretends "x\\n" + "  " merge into a different token sequence."""

        def __init__(self):
            pass

        def encode(self, text):
            ids = [ord(c) for c in text]
            if text.endswith("\n  y"):
                ids[-4:-2] = [999]
            return ids

    with pytest.raises(TokenizationError, match="merges"):
        encode_after(MergingTokenizer(), "x\n", "  y")


# --------------------------------------------------------------------------- real tokenizers

TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/gemma-3-1b-it-4bit",
    "google/gemma-4-e2b-it",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "ibm-granite/granite-3.3-2b-instruct",
    "openai/gpt-oss-20b",
]


@pytest.mark.network
@pytest.mark.parametrize("repo", TOKENIZERS)
def test_real_tokenizers_compile_every_preset(repo, presets):
    from transformers import AutoTokenizer

    from pcd.prompt import FAMILY_DEFAULTS

    try:
        raw = AutoTokenizer.from_pretrained(repo)
    except Exception as e:  # offline, gated, or removed from the Hub
        pytest.skip(f"cannot load {repo}: {e}")
    tok = HFTokenizer(raw)
    fmt = PromptFormat(**FAMILY_DEFAULTS.get("gpt_oss" if "gpt-oss" in repo else "", {}))
    gpt2_style = any(k in repo for k in ("SmolLM2", "granite"))
    assert compile_schema(Schema.coerce(presets["fintech_fraud"]), tok, fmt).prompt.layout == (
        "before" if gpt2_style else "after"
    )
    for name, p in presets.items():
        cs = compile_schema(Schema.coerce(p), tok, fmt)
        full = tok.encode(cs.prompt.render(p["context"].strip()))
        assert tuple(full[: len(cs.head_ids)]) == cs.head_ids, f"{name}: head is not a token prefix"
        assert tuple(full[-len(cs.tail_ids):]) == cs.tail_ids, f"{name}: tail is not a token suffix"
        for cf in cs.fields.values():
            # Every (label, form) round-trips to the exact JSON line after the tail.
            layout = cs.prompt.layout
            for v in cf.variants:
                label = cf.labels[v.label]
                text = tok.decode(list(cs.tail_ids + cf.prefix + v.seq))
                forms = fmt.value_forms(cf.field, label)
                assert any(text.endswith(fmt.field_line(cf.field, label, form, layout).lstrip()) for form in forms), (
                    repo, cf.name, label, text[-60:]
                )
            if cf.field.type == "boolean" and layout == "after":
                # The natural form: the space belongs to the value token (' true'), not the key.
                assert tok.decode(list(cf.prefix[-1:])).endswith(":"), (repo, tok.decode(list(cf.prefix[-2:])))

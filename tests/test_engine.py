"""End-to-end engine behaviour on the oracle backend and on tiny real models."""

import json

import numpy as np
import pytest
from helpers import OracleBackend

from pcd import DecodeOptions, Engine, PromptFormat, Schema
from pcd.baseline import generate_json, lenient_values
from pcd.tokenizer import HFTokenizer


def oracle_engine(byte_tok, rules=(), **kw):
    return Engine(OracleBackend(HFTokenizer(byte_tok), rules=rules), **kw)


def test_output_is_typed_complete_and_ordered(byte_tok, presets):
    engine = oracle_engine(byte_tok)
    for name, p in presets.items():
        out = engine.extract(p["context"], p["schema"])
        schema = Schema.coerce(p)
        assert list(out.values) == schema.names
        assert schema.validate_values(out.values) == ([], [], [])
        for f in out.fields.values():
            assert 0.0 < f.probability <= 1.0
            assert f.candidate_mass is not None and 0.0 <= f.candidate_mass <= 1.0 + 1e-9
        json.loads(out.to_json())
        json.dumps(out.to_dict(), default=str)


def test_head_cache_is_reused(byte_tok, presets):
    engine = oracle_engine(byte_tok, decode=DecodeOptions(anchor_first=False))
    p = presets["support_triage"]
    first = engine.extract(p["context"], p["schema"])
    second = engine.extract("A different ticket about billing.", p["schema"])
    assert first.stats.cached_tokens == 0 and first.stats.head_prefill_ms > 0
    assert second.stats.cached_tokens > 0 and second.stats.head_prefill_ms == 0
    # head prefill + context prefill + one scoring pass (the oracle counts one pass per 64 rows)
    assert first.stats.forward_passes == 3 and second.stats.forward_passes == 2
    assert first.stats.decode_rounds == 1


def test_measured_passes_grow_with_trie_levels(byte_tok, presets):
    p = presets["high_cardinality_255"]
    engine = oracle_engine(byte_tok)
    out = engine.extract(p["context"], p["schema"])
    assert out.fields["customs_category"].method in ("hybrid", "trie")
    assert out.stats.decode_rounds >= 2
    assert out.stats.forward_passes >= 2 + out.stats.decode_rounds


def test_dependent_fields_see_earlier_answers(byte_tok):
    """`copy` depends on `source`; the oracle makes `copy` repeat whatever `source` said."""
    tok = HFTokenizer(byte_tok)
    src_line = {lab: tuple(tok.encode(f'  "source": "{lab}",\n')) for lab in ("RED", "BLUE")}
    key = tuple(tok.encode('  "copy": "'))
    first = {lab: tok.encode(lab)[0] for lab in ("RED", "BLUE")}

    def rule(ctx, z):
        if ctx[-len(key):] == key:
            for lab, line in src_line.items():
                if any(ctx[i : i + len(line)] == line for i in range(len(ctx) - len(line))):
                    z[first[lab]] += 50.0

    def prefer(label):
        tok_id = first[label]
        skey = tuple(tok.encode('  "source": "'))

        def r(ctx, z):
            if ctx[-len(skey):] == skey:
                z[tok_id] += 50.0
        return r

    schema = {
        "source": {"type": "enum", "choices": ["RED", "BLUE"]},
        "copy": {"type": "enum", "choices": ["RED", "BLUE"], "depends_on": ["source"]},
    }
    for label in ("RED", "BLUE"):
        engine = oracle_engine(byte_tok, rules=[rule, prefer(label)])
        out = engine.extract("anything", schema)
        assert out.values == {"source": label, "copy": label}
        assert out.fields["copy"].wave == 1 and out.stats.waves == 2
        assert out.fields["copy"].probability > 0.99


def test_first_field_anchors_the_rest(byte_tok):
    """With anchoring, every later field is scored after the first field's answer line."""
    tok = HFTokenizer(byte_tok)
    line = tuple(tok.encode('  "a": "RED",\n'))
    key_a, key_b = tuple(tok.encode('  "a": "')), tuple(tok.encode('  "b": "'))
    red, blue = tok.encode("R")[0], tok.encode("B")[0]

    def rule(ctx, z):
        if ctx[-len(key_a):] == key_a:
            z[red] += 50.0
        elif ctx[-len(key_b):] == key_b:
            seen = any(ctx[i : i + len(line)] == line for i in range(len(ctx) - len(line)))
            z[red if seen else blue] += 50.0

    schema = {"a": {"type": "enum", "choices": ["RED", "BLUE"]}, "b": {"type": "enum", "choices": ["RED", "BLUE"]}}
    engine = oracle_engine(byte_tok, rules=[rule])
    anchored = engine.extract("x", schema)
    assert anchored.values == {"a": "RED", "b": "RED"}
    assert anchored.stats.waves == 2 and anchored.fields["b"].wave == 1 and anchored.stats.decode_rounds == 2
    isolated = engine.extract("x", schema, anchor_first=False)
    assert isolated.values == {"a": "RED", "b": "BLUE"}
    assert isolated.stats.waves == 1 and isolated.stats.decode_rounds == 1


def test_decode_overrides_and_strategies(byte_tok, presets):
    engine = oracle_engine(byte_tok, decode=DecodeOptions(strategy="exact"))
    p = presets["fintech_fraud"]
    exact = engine.extract(p["context"], p["schema"])
    assert {f.method for f in exact.fields.values()} <= {"exact"}
    greedy = engine.extract(p["context"], p["schema"], strategy="greedy")
    assert {f.method for f in greedy.fields.values()} == {"trie"}
    auto = engine.extract(p["context"], p["schema"], strategy="auto")
    assert {f.method for f in auto.fields.values()} <= {"first_token", "marginal"}


def test_low_mass_is_reported(byte_tok):
    tok = HFTokenizer(byte_tok)
    key = tuple(tok.encode('  "x": "'))
    other = tok.encode("z")[0]

    def wants_something_else(ctx, z):
        if ctx[-len(key):] == key:
            z[other] += 20.0  # the model would write "z..." rather than any allowed label

    engine = oracle_engine(byte_tok, rules=[wants_something_else])
    out = engine.extract("text", {"x": {"type": "enum", "choices": ["ALPHA", "BETA", "GAMMA"]}})
    assert out.fields["x"].candidate_mass < 0.01
    assert any("candidate mass" in w for w in out.warnings)
    assert "x" in out.low_confidence()


def test_schema_inputs_are_interchangeable(byte_tok):
    engine = oracle_engine(byte_tok)
    as_dict = {"ok": {"type": "boolean"}, "lvl": {"type": "enum", "choices": ["LOW", "HIGH"]}}
    as_json_schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "lvl": {"enum": ["LOW", "HIGH"]}}}
    a = engine.extract("ctx", as_dict)
    b = engine.extract("ctx", as_json_schema)
    c = engine.extract("ctx", Schema.from_dict(as_dict))
    assert a.values == b.values == c.values


def test_prompt_format_family_defaults(byte_tok):
    class GptOss(OracleBackend):
        def describe(self):
            return {**super().describe(), "model_type": "gpt_oss"}

    engine = Engine(GptOss(HFTokenizer(byte_tok)))
    assert engine.prompt_format.assistant_prefix == "<|channel|>final<|message|>"
    custom = Engine(GptOss(HFTokenizer(byte_tok)), prompt_format=PromptFormat())
    assert custom.prompt_format.assistant_prefix == ""


def test_baseline_parses_and_validates(byte_tok):
    tok = HFTokenizer(byte_tok)
    script = tok.encode('  "ok": "true",\n  "lvl": "MID"\n}')

    class Scripted(OracleBackend):
        def greedy(self, prefix, inputs, max_new_tokens, stop_ids, should_stop=None):
            out = []
            for t in script[:max_new_tokens]:
                out.append(t)
                self.forward_passes += 1
                if should_stop and should_stop(out):
                    break
            return out

    engine = Engine(Scripted(tok))
    schema = {"ok": {"type": "boolean"}, "lvl": {"type": "enum", "choices": ["LOW", "HIGH"]}}
    res = generate_json(engine, "ctx", schema)
    assert res.valid_json and not res.schema_match
    assert res.invalid == ["ok='true'", "lvl='MID'"]
    assert lenient_values(schema, res.values) == {"ok": True, "lvl": "MID"}
    assert res.new_tokens == len(script) and res.forward_passes >= len(script)


def test_tiny_model_end_to_end(tiny_backend, presets):
    engine = Engine(tiny_backend)
    for name, p in presets.items():
        out = engine.extract(p["context"], p["schema"])
        assert Schema.coerce(p).validate_values(out.values) == ([], [], [])
        assert out.stats.forward_passes >= 2
        again = engine.extract(p["context"], p["schema"])
        assert again.values == out.values
        for k in out.fields:
            assert again.fields[k].probability == pytest.approx(out.fields[k].probability, abs=1e-4)
    res = generate_json(engine, presets["fintech_fraud"]["context"], presets["fintech_fraud"]["schema"], max_new_tokens=20)
    assert res.new_tokens <= 20 and res.forward_passes >= 1


def test_temperature_leaves_values_unchanged_on_real_model(tiny_backend, presets):
    engine = Engine(tiny_backend)
    p = presets["code_security"]
    a = engine.extract(p["context"], p["schema"])
    b = engine.extract(p["context"], p["schema"], temperature=0.3)
    assert a.values == b.values
    assert np.mean([b.fields[k].probability for k in b.fields]) >= np.mean([a.fields[k].probability for k in a.fields])

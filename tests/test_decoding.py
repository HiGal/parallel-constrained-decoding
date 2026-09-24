"""Decoders checked against brute force on an oracle LM (see helpers.OracleBackend)."""

import math

import numpy as np
import pytest
from helpers import OracleBackend

from pcd import PromptFormat, Schema
from pcd.compiler import compile_schema
from pcd.decoding import DecodeOptions, decode_fields
from pcd.tokenizer import HFTokenizer

SCHEMA = {
    "urgent": {"type": "boolean"},
    "tier": {"type": "enum", "choices": ["TIER_1", "TIER_2", "SENIOR", "DIRECTOR"]},
    "sla": {"type": "enum", "choices": ["1_HOUR", "4_HOURS", "12_HOURS", "24_HOURS"]},
    "risk": {"type": "enum", "choices": ["HIGH", "HIGH_RISK", "LOW"]},
    "mood": {"type": "enum", "choices": ["happy", "sad", "angry"]},
}


def setup(byte_tok, seed=0, rules=(), schema=SCHEMA, fmt=None):
    tok = HFTokenizer(byte_tok)
    backend = OracleBackend(tok, seed=seed, rules=rules)
    cs = compile_schema(Schema.from_dict(schema), tok, fmt or PromptFormat())
    prefix = backend.prefill(tok.encode(cs.prompt.render("The context.")))
    return backend, cs, prefix


def brute(backend, prefix, cf, full: bool, temperature=1.0):
    """Label distribution and candidate mass computed directly from the oracle."""
    ctx = prefix.tokens + cf.prefix
    scores, mass = {}, 0.0
    for v in cf.variants:
        last = len(v.seq) - 1 if full else v.ident
        lp = backend.seq_logprob(ctx, v.seq[: last + 1])
        scores.setdefault(v.label, []).append(lp)
        mass += math.exp(backend.seq_logprob(ctx, v.seq[: v.ident + 1]))
    labels = sorted(scores)
    s = np.array([np.logaddexp.reduce(scores[li]) for li in labels]) / temperature
    p = np.exp(s - s.max())
    p /= p.sum()
    return {cf.labels[li]: float(pi) for li, pi in zip(labels, p)}, mass


def brute_greedy(backend, prefix, cf):
    ctx, alive, path, prob = prefix.tokens + cf.prefix, list(cf.variants), [], 1.0
    while len({v.label for v in alive}) > 1:
        d = len(path)
        options = sorted({v.seq[d] for v in alive})
        if len(options) > 1:
            lp = backend.logprobs(ctx + tuple(path))[options]
            p = np.exp(lp - lp.max()) / np.exp(lp - lp.max()).sum()
            j = int(np.argmax(p))
            prob *= p[j]
        else:
            j = 0
        alive = [v for v in alive if v.seq[d] == options[j]]
        path.append(options[j])
    return cf.labels[alive[0].label], prob


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("strategy", ["auto", "exact"])
def test_marginal_and_exact_match_brute_force(byte_tok, seed, strategy):
    backend, cs, prefix = setup(byte_tok, seed)
    res, rounds, _ = decode_fields(backend, prefix, list(cs.fields.values()), DecodeOptions(strategy=strategy))
    assert rounds == 1
    for name, cf in cs.fields.items():
        dist, mass = brute(backend, prefix, cf, full=strategy == "exact")
        r = res[name]
        assert r.value == max(dist, key=dist.get)
        assert r.probability == pytest.approx(dist[r.value], rel=1e-9)
        assert r.candidate_mass == pytest.approx(mass, rel=1e-9)
        for label, p in r.top:
            assert p == pytest.approx(dist[label], rel=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_greedy_trie_matches_brute_force(byte_tok, seed):
    backend, cs, prefix = setup(byte_tok, seed)
    res, rounds, _ = decode_fields(backend, prefix, list(cs.fields.values()), DecodeOptions(strategy="greedy"))
    for name, cf in cs.fields.items():
        value, prob = brute_greedy(backend, prefix, cf)
        assert res[name].value == value
        assert res[name].probability == pytest.approx(prob, rel=1e-9)
        assert res[name].method == "trie"


def test_hybrid_switches_to_exact(byte_tok):
    schema = {"code": {"type": "enum", "choices": [f"CAT_{i:03d}_{n}" for i, n in enumerate(
        ["Animals", "Meat", "Fish", "Dairy", "Plants", "Fruit", "Coffee", "Cereal", "Oil", "Sugar",
         "Cocoa", "Wine", "Salt", "Ores", "Fuel", "Glass", "Steel", "Tools"])]}}
    backend, cs, prefix = setup(byte_tok, 3, schema=schema)
    cf = cs.fields["code"]
    opts = DecodeOptions(exact_max=4, hybrid_exact_below=10)  # 18 labels -> 10 or 8 after one branch
    res, rounds, _ = decode_fields(backend, prefix, [cf], opts)
    r = res["code"]
    assert r.method == "hybrid" and rounds >= 2
    # Reproduce: greedy branches while more than 10 labels survive, then exact scoring.
    ctx, alive, path, prob = prefix.tokens + cf.prefix, list(cf.variants), [], 1.0
    while len({v.label for v in alive}) > 10:
        d = len(path)
        options = sorted({v.seq[d] for v in alive})
        if len(options) > 1:
            lp = backend.logprobs(ctx + tuple(path))[options]
            p = np.exp(lp - np.logaddexp.reduce(lp))
            j = int(np.argmax(p))
            prob *= p[j]
        else:
            j = 0
        alive = [v for v in alive if v.seq[d] == options[j]]
        path.append(options[j])
    full = {v.label: backend.seq_logprob(ctx + tuple(path), v.seq[len(path):]) for v in alive}
    labels = list(full)
    s = np.array([full[l] for l in labels])
    p = np.exp(s - np.logaddexp.reduce(s))
    assert r.value == cf.labels[labels[int(np.argmax(p))]]
    assert r.probability == pytest.approx(prob * p.max(), rel=1e-9)


def test_quoted_booleans_are_marginalized(byte_tok):
    """The situation found on Qwen2.5-1.5B: the model mostly wants to quote the boolean.

    Bare tokens slightly prefer false, but after an opening quote the model is sure of
    true. Summing the two surface forms recovers the model's actual answer."""

    tok = HFTokenizer(byte_tok)
    quote, t, f = (tok.encode(c)[0] for c in '"tf')
    key = tuple(tok.encode('"urgent": '))

    def rule(ctx, z):
        if ctx[-len(key):] == key:  # at the decision position
            z[:] = -20
            z[quote], z[f], z[t] = 5.0, 3.2, 3.0
        elif ctx[-len(key) - 1:] == key + (quote,):  # after an opening quote
            z[:] = -20
            z[t], z[f] = 5.0, 2.0

    schema = {"urgent": {"type": "boolean"}}
    backend, cs, prefix = setup(byte_tok, rules=[rule], schema=schema)
    res, _, _ = decode_fields(backend, prefix, list(cs.fields.values()), DecodeOptions())
    assert res["urgent"].value is True and res["urgent"].method == "marginal"
    assert res["urgent"].candidate_mass > 0.99

    backend, cs, prefix = setup(byte_tok, rules=[rule], schema=schema, fmt=PromptFormat(quoted_booleans=False))
    res, _, _ = decode_fields(backend, prefix, list(cs.fields.values()), DecodeOptions())
    assert res["urgent"].value is False and res["urgent"].candidate_mass < 0.25


def test_temperature_changes_confidence_not_winner(byte_tok):
    backend, cs, prefix = setup(byte_tok, 1)
    fields = list(cs.fields.values())
    base, _, _ = decode_fields(backend, prefix, fields, DecodeOptions())
    sharp, _, _ = decode_fields(backend, prefix, fields, DecodeOptions(temperature=0.5))
    flat, _, _ = decode_fields(backend, prefix, fields, DecodeOptions(temperature=3.0))
    for name in base:
        assert base[name].value == sharp[name].value == flat[name].value
        assert sharp[name].probability >= base[name].probability - 1e-12 >= flat[name].probability - 2e-12


def test_fields_are_independent_of_batching(byte_tok):
    backend, cs, prefix = setup(byte_tok, 2)
    fields = list(cs.fields.values())
    together, _, _ = decode_fields(backend, prefix, fields, DecodeOptions())
    for cf in fields:
        alone, _, _ = decode_fields(backend, prefix, [cf], DecodeOptions())
        assert alone[cf.name] == together[cf.name]


def test_options_validation():
    with pytest.raises(ValueError):
        DecodeOptions(strategy="beam")
    with pytest.raises(ValueError):
        DecodeOptions(temperature=0)

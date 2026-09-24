"""Real models end to end (``uv run pytest -m slow -s``). Downloads ~4.5 GB on first run.

Each model must pass the backend conformance checks (cache broadcast, prefix extension,
batching) and answer an unambiguous ticket at least as well as the same model writing
the JSON token by token. The ticket covers booleans, a field with colliding labels
(1_HOUR / 12_HOURS / 1_WEEK all start with "1") and a dependent field.
"""

import gc
import json

import pytest

from pcd import Engine, Schema
from pcd.backends import available_backends
from pcd.baseline import generate_json, lenient_values
from pcd.testing import check_backend

pytestmark = pytest.mark.slow

TICKET = (
    "Subject: Charged twice for my March invoice!!\n\n"
    "Hi, I was billed twice for my March invoice ($49 each). I am really annoyed, this is the "
    "third time I write about it. Please refund the duplicate charge within 12 hours or I will "
    "cancel. I'm on the Premium plan.\n-- Dana"
)
SCHEMA = {
    "sentiment": {"type": "enum", "choices": ["POSITIVE", "NEUTRAL", "NEGATIVE"], "description": "Customer mood"},
    "topic": {"type": "enum", "choices": ["BILLING", "TECHNICAL_ISSUE", "FEATURE_REQUEST", "ACCOUNT_ACCESS"],
              "description": "What the ticket is about"},
    "wants_refund": {"type": "boolean", "description": "Whether the customer asks for money back"},
    "is_spam": {"type": "boolean", "description": "Whether the message is spam"},
    "deadline": {"type": "enum", "choices": ["1_HOUR", "12_HOURS", "1_WEEK", "NONE"],
                 "description": "Deadline the customer gives"},
    "plan": {"type": "enum", "choices": ["FREE", "PREMIUM", "ENTERPRISE"], "description": "Customer's plan"},
    "route_to": {"type": "enum", "choices": ["BILLING_TEAM", "ENGINEERING", "SALES"],
                 "description": "Team that should handle it", "depends_on": ["topic"]},
}
EXPECTED = {
    "sentiment": "NEGATIVE", "topic": "BILLING", "wants_refund": True, "is_spam": False,
    "deadline": "12_HOURS", "plan": "PREMIUM", "route_to": "BILLING_TEAM",
}

# (backend, model, check accuracy). SmolLM2-360M is too small to classify this ticket (its
# autoregressive JSON does not even parse); it is here for the GPT-2-style "before" layout.
CASES = [
    # One Hugging Face checkpoint, loaded by both backends (mlx-lm reads bf16 safetensors).
    ("mlx", "Qwen/Qwen2.5-0.5B-Instruct", True),
    ("torch", "Qwen/Qwen2.5-0.5B-Instruct", True),
    ("mlx", "HuggingFaceTB/SmolLM2-360M-Instruct", False),
    ("torch", "HuggingFaceTB/SmolLM2-360M-Instruct", False),
    # 4-bit MLX checkpoints of other families.
    ("mlx", "qwen2.5-1.5b", True),
    ("mlx", "qwen3-0.6b", True),
    ("mlx", "qwen3.5-0.8b", True),  # hybrid linear-attention cache
    ("mlx", "llama3.2-1b", True),
    ("mlx", "gemma3-1b", True),  # sliding-window cache
    ("mlx", "lfm2.5-1.2b", True),  # hybrid convolution cache
]

RESULTS: dict = {}


@pytest.mark.parametrize("backend,model,accuracy", CASES, ids=[f"{b}:{m}" for b, m, _ in CASES])
def test_real_model(backend, model, accuracy, presets):
    if backend not in available_backends():
        pytest.skip(f"{backend} not installed")
    engine = Engine.load(model, backend)
    try:
        info = engine.backend.describe()
        checks = check_backend(engine.backend)
        assert all(c.passed for c in checks), [str(c) for c in checks if not c.passed]

        out = engine.extract(TICKET, SCHEMA)
        schema = Schema.coerce(SCHEMA)
        assert schema.validate_values(out.values) == ([], [], [])
        assert out.fields["route_to"].wave > out.fields["topic"].wave
        wrong = {k: out.values[k] for k, v in EXPECTED.items() if out.values[k] != v}
        ar = lenient_values(SCHEMA, generate_json(engine, TICKET, SCHEMA).values)
        ar_wrong = {k: ar.get(k) for k, v in EXPECTED.items() if ar.get(k) != v}

        preset_ms = {}
        for name, p in presets.items():
            engine.extract(p["context"], p["schema"])  # cache the head
            r = engine.extract(p["context"], p["schema"])
            assert Schema.coerce(p).validate_values(r.values) == ([], [], [])
            assert r.stats.cached_tokens > 0
            preset_ms[name] = round(r.stats.total_ms)

        RESULTS[f"{backend}:{model}"] = {
            "fast_path": info.get("fast_path"), "layout": engine.compile(SCHEMA).prompt.layout,
            "prompt_mode": out.backend.get("prompt_mode"), "wrong": wrong, "ar_wrong": ar_wrong,
            "mean_mass": round(sum(f.candidate_mass for f in out.fields.values()) / len(out.fields), 3),
            "ms": preset_ms,
        }
        print(f"\n{backend}:{model} {json.dumps(RESULTS[f'{backend}:{model}'], default=str)}")
        # Faithfulness: at most one miss beyond the model's own autoregressive answer. Fields
        # are decided without seeing each other, and small models sometimes answer a yes/no
        # field differently in isolation (LFM2.5-1.2B: is_spam); see README, Limitations.
        if accuracy:
            assert len(wrong) <= len(ar_wrong) + 1, (wrong, ar_wrong)
    finally:
        del engine
        gc.collect()
        if backend == "mlx":
            import mlx.core as mx

            mx.clear_cache()


def test_same_checkpoint_same_answers_on_both_backends(presets):
    """Qwen2.5-0.5B in bf16 on MLX and on PyTorch: the decisions should (almost) all match."""
    if not {"mlx", "torch"} <= set(available_backends()):
        pytest.skip("needs both backends")
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    engines = {b: Engine.load(model, b) for b in ("mlx", "torch")}
    same = total = 0
    for p in list(presets.values()) + [{"context": TICKET, "schema": SCHEMA}]:
        a = engines["mlx"].extract(p["context"], p["schema"]).values
        b = engines["torch"].extract(p["context"], p["schema"]).values
        same += sum(a[k] == b[k] for k in a)
        total += len(a)
    print(f"\nMLX vs PyTorch, same bf16 checkpoint: {same}/{total} decisions identical")
    assert same / total >= 0.9

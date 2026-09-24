"""Minimal end-to-end example.

    uv run python examples/quickstart.py                      # MLX on Apple Silicon, else PyTorch
    uv run python examples/quickstart.py qwen3-4b torch       # any alias / repo id, any backend
"""

import sys

from pcd import Engine

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-1.5b"
backend = sys.argv[2] if len(sys.argv) > 2 else "auto"

engine = Engine.load(model, backend)

schema = {
    "sentiment": {"type": "enum", "choices": ["POSITIVE", "NEUTRAL", "NEGATIVE"], "description": "Customer mood"},
    "topic": {"type": "enum", "choices": ["BILLING", "TECHNICAL_ISSUE", "ACCOUNT_ACCESS"], "description": "Main topic"},
    "wants_refund": {"type": "boolean", "description": "Whether the customer asks for money back"},
    "deadline": {"type": "enum", "choices": ["1_HOUR", "12_HOURS", "1_WEEK", "NONE"], "description": "Deadline given"},
    # Decided after `topic`, with the topic's answer visible in the prompt.
    "route_to": {"type": "enum", "choices": ["BILLING_TEAM", "ENGINEERING", "SUPPORT"],
                 "description": "Team that should handle it", "depends_on": ["topic"]},
}

ticket = (
    "I was billed twice for my March invoice. Please refund the duplicate charge "
    "within 12 hours or I'm cancelling. Really disappointed."
)

out = engine.extract(ticket, schema)
print(out.values)
for name, f in out.fields.items():
    print(f"  {name:13} {str(f.value):14} p={f.probability:.2f}  mass={f.candidate_mass:.2f}  {f.method}")
s = out.stats
print(f"{s.total_ms:.0f} ms, {s.forward_passes} forward passes, {s.prompt_tokens} prompt tokens")

# The second request with the same schema reuses the cached prompt head.
again = engine.extract("The app logs me out every five minutes since the update.", schema)
print(again.values, f"{again.stats.total_ms:.0f} ms ({again.stats.cached_tokens} cached prompt tokens)")
if again.low_confidence():
    print("worth a second look:", again.low_confidence())

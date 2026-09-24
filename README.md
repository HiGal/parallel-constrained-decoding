# pcd: parallel constrained decoding for open LLMs

`pcd` fills in classification and extraction schemas (booleans and closed enums) with
any open LLM, on **MLX** (Apple Silicon) or **PyTorch** (CUDA, Apple `mps`, CPU).

Instead of letting the model write a JSON object token by token, `pcd` prefills the
prompt once, broadcasts its KV cache to one row per field, and reads every decision from
a single batched forward pass. The output always has every key, and every value is an
allowed label. Each field also comes with a probability and a diagnostic that tells you
when the model wanted to answer something else.

```python
from pcd import Engine

engine = Engine.load("qwen2.5-1.5b")        # alias, Hugging Face repo id, or local path
out = engine.extract(ticket_text, {
    "sentiment":    {"type": "enum", "choices": ["POSITIVE", "NEUTRAL", "NEGATIVE"]},
    "wants_refund": {"type": "boolean", "description": "Whether the customer asks for money back"},
    "deadline":     {"type": "enum", "choices": ["1_HOUR", "12_HOURS", "1_WEEK", "NONE"]},
})
out.values   # {'sentiment': 'NEGATIVE', 'wants_refund': True, 'deadline': '12_HOURS'}
out.fields   # per field: probability, candidate_mass, method, alternatives
out.stats    # per-phase timings, measured forward passes, cached prompt tokens
```

## Install

With [uv](https://docs.astral.sh/uv/), from this directory:

```bash
uv sync --extra mlx                 # Apple Silicon
uv sync --extra torch               # NVIDIA / CPU (also runs on Apple mps)
uv sync --extra mlx --extra torch   # both, e.g. to compare them
```

To use it from another project: `uv add --editable /path/to/pcd --extra mlx`.

## Command line

```bash
uv run pcd models                                          # list model aliases
uv run pcd check qwen3-4b --backend mlx                    # does this model work? (see CONTRIBUTING.md)
uv run pcd run qwen2.5-1.5b examples/presets/support_triage.json
uv run pcd run llama3.2-3b --schema schema.json --context ticket.txt --json
uv run pcd bench qwen2.5-1.5b examples/presets/*.json      # vs autoregressive JSON generation
```

## Models

Any decoder model that mlx-lm or `transformers` can load should work; `pcd check` tells
you in a minute. The aliases in `src/pcd/models.py` map short names to MLX (4-bit) and
PyTorch checkpoints that fit a 16 GB Mac:

| Family | Aliases | Notes |
|---|---|---|
| Qwen 2.5 | `qwen2.5-0.5b` `-1.5b` `-3b` `-7b` | |
| Qwen 3 | `qwen3-0.6b` `-1.7b` `-4b` `-8b` | thinking switched off through the chat template |
| Qwen 3.5 | `qwen3.5-0.8b` `-2b` `-4b` `-9b` | hybrid linear-attention cache |
| Llama 3.x | `llama3.2-1b` `-3b`, `llama3.1-8b` | PyTorch repos are gated |
| Gemma | `gemma2-2b`, `gemma3-1b` `-4b`, `gemma4-e2b` `-e4b` | sliding windows; Gemma 2 soft-caps logits |
| Phi | `phi3.5-mini`, `phi4-mini` | |
| Mistral | `mistral-7b`, `ministral-8b` | |
| Small models | `smollm2-360m` `-1.7b`, `smollm3-3b`, `lfm2.5-1.2b`, `granite3.3-2b` | LFM2 has a hybrid conv cache |
| gpt-oss | `gpt-oss-20b` | harmony format handled; needs 24 GB+ |

What was actually run for this README:

| Checkpoint | Backend | Cache type exercised | Ticket: pcd errors / same model's own AR errors |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` (bf16) | MLX **and** PyTorch (mps) | KV | 1 / 1 (MLX), 1 / 2 (PyTorch) |
| `HuggingFaceTB/SmolLM2-360M-Instruct` (bf16) | MLX **and** PyTorch (mps) | KV, "before" newline layout | 2 / 7 (its AR JSON does not parse) |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | MLX | KV | 2 / 2 |
| `mlx-community/Qwen3-0.6B-4bit` | MLX | KV | 0 / 0 |
| `mlx-community/Qwen3.5-0.8B-4bit` | MLX | hybrid linear attention (`ArraysCache`) | 1 / 1 |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | MLX | KV | 1 / 1 |
| `mlx-community/gemma-3-1b-it-4bit` | MLX | sliding window (`RotatingKVCache`) | 1 / 1 |
| `mlx-community/LFM2.5-1.2B-Instruct-4bit` | MLX | hybrid convolution (`ArraysCache`) | 2 / 1 |

Every run passed the backend conformance checks. On the test ticket (7 fields), `pcd`
made exactly the same mistakes as the model writing the JSON itself in 6 of the 8 runs
checked for accuracy. It was better once (Qwen2.5-0.5B on PyTorch) and worse once
(LFM2.5 answered `is_spam` differently when asked in isolation). SmolLM2-360M is only
checked for conformance: its own JSON does not parse, while `pcd` still gets 5 of 7. The same bf16 checkpoint gave identical
decisions on MLX and PyTorch for 93 of 95 fields across all presets.

Tokenizers and chat templates of all 15 families above compile every preset
(`uv run pytest -m network`).

## How it works

1. **Compile the schema for the tokenizer** (once, cached). Every allowed value of
   every field is tokenized as the complete JSON line the model would write
   (`  "risk": "HIGH",\n`) right after the prompt. Each field's row is the longest common
   token prefix of its lines, so the decision lands on the first token where labels
   differ. The rest forms a token trie over *(label, surface form)* pairs.
2. **Build the prompt from the model's chat template**: a system turn listing every
   key with its allowed values, the input in the user turn, and an assistant turn that
   opens the JSON object. The static head is prefilled once per schema and reused.
3. **Prefill** only the per-request part on top of the cached head.
4. **Decide all fields at once.** Rows from all fields go into one batched pass on
   copies of the prompt's KV cache (chunked by a memory budget). For each field the
   default scorer reads the probability of reaching the token that identifies each
   label, summed over its surface forms. Colliding labels (`1_HOUR` / `12_HOURS`) are
   followed just deep enough to tell apart, in the same pass. Fields with more than 32
   labels use a greedy trie walk and switch to exact scoring once ≤10 labels survive.
5. **Waves.** By default the schema's first field is decided on its own, its answer is
   written into the prompt, and then all other fields are decided together after it.
   Fields with `depends_on` wait for a later wave in the same way.
6. **Assemble** the object in schema order with real booleans, and report what was
   measured.

## Reading the results

| Field attribute | Meaning |
|---|---|
| `value` | The chosen label. Always one of the allowed ones. |
| `probability` | How the model splits its belief **among the allowed labels** (renormalized). Not a calibrated confidence. |
| `candidate_mass` | Unconstrained probability that the model's own continuation reaches a token identifying *some* allowed label. Low means the model wanted to write something else, so treat the answer with suspicion. |
| `method` | `first_token`, `marginal`, `exact`, `trie`, `hybrid`, or `fixed` (one-choice field). |
| `top` | Alternatives with their probabilities. |

`out.low_confidence()` lists fields with low probability or low mass, and
`out.warnings` explains anything unusual. `out.stats.forward_passes` is counted by the
backend, not assumed.

## Options

```python
from pcd import DecodeOptions, Engine, PromptFormat

engine = Engine.load(
    "qwen3-4b", "mlx",
    decode=DecodeOptions(
        strategy="auto",        # "exact": score full label strings; "greedy": fewest rows
        exact_max=32,           # above this many labels, use the trie walk
        hybrid_exact_below=10,  # ...and switch to exact scoring when this few remain
        temperature=1.0,        # reshapes probabilities, never changes the winner
        anchor_first=True,      # decide the first field before the others (see below)
    ),
    prompt_format=PromptFormat(
        instructions="You label customer tickets. ...",
        list_choices=True,      # list every allowed value in the prompt (strongly recommended)
        quoted_booleans=True,   # score "true"/"false" as forms of true/false
    ),
    max_rows=64,                # rows per forward pass
    kv_budget_bytes=2 << 30,    # memory for broadcast KV copies per pass
)
out = engine.extract(text, schema, strategy="exact")   # per-call overrides
```

Schemas can be the preset-style dict shown above, a JSON Schema whose properties are
`boolean` / `enum` / `const`, a Pydantic model with `bool`, `Literal` and `Enum` fields,
or a `pcd.Schema`.

## Benchmarks

Apple M1 Pro (16 GB), macOS 15.1, mlx 0.32.2, torch 2.14 (mps). Each run uses the
defaults, a warm head cache, and the median of 3. "AR" is the same model writing the
JSON token by token from the same prompt (`pcd bench`). "Agreement" counts fields where
the answer equals AR's; it is a proxy for faithfulness, not accuracy.

**MLX, `qwen2.5-1.5b` (4-bit), against the original `core/engine_mlx.py`:**

| Preset | Fields | Old engine | **pcd** | AR (tokens) | Agreement with AR: old → **pcd** | AR output schema-valid? |
|---|---|---|---|---|---|---|
| code_security | 28 | 920 ms | **646 ms** | 2,910 ms (323) | 21 → **26** | no (14 invalid values) |
| fintech_fraud | 28 | 938 ms | **699 ms** | 2,815 ms (304) | 19 → **26** | no (missing key, 17 invalid) |
| support_triage | 28 | 957 ms | **650 ms** | 2,436 ms (271) | 6 → **13** | no (missing key) |
| high_cardinality_255 | 4 | 297 ms | 873 ms | 657 ms (47) | 2 → 2 | no |

On the 28-field presets `pcd` is 3.7–4.3× faster than AR and ~30% faster than the old
engine. The cached prompt head saves more time than listing every allowed value and the
anchor round cost. The 255-choice preset is the exception: its 4,400-token catalog makes
every batched row carry a large KV copy, and the trie needs several rounds.
The old engine was fast there only because it decided the field by a fallback rule
(`CAT_005_Live_Trees_&_Plants`, reported at 0.75). `pcd` answers
`CAT_098_Satellite_Telemetry_Transceivers` at p=0.20, with the correct
`CAT_095_Aerospace_Titanium_Fasteners_&_Bolts` second, and AR writes `CAT_099`, which is
not a valid label. With labels that put the code before the meaning, no decoder recovers
the right answer at this model size (see notebook 04 §7).

**PyTorch on mps, `Qwen/Qwen2.5-0.5B-Instruct` (bf16):**

| Preset | Fields | **pcd** | AR (tokens) | Speed-up | Agreement with AR |
|---|---|---|---|---|---|
| code_security | 28 | 489 ms | 5,973 ms (251) | 12.2× | 16/28 |
| fintech_fraud | 28 | 498 ms | 6,648 ms (288) | 13.4× | 23/28 |
| support_triage | 28 | 479 ms | 5,999 ms (251) | 12.5× | 23/28 |
| high_cardinality_255 | 4 | 754 ms | 1,380 ms (50) | 1.8× | 4/4 |

A typical 28-field request costs 4 measured forward passes when the head is cached:
the context prefill, the anchor field, its answer line, and one batched pass for the
remaining 27 fields.

## Tests

```bash
uv run pytest                 # offline, ~10 s
uv run pytest -m network      # real tokenizers (downloads ~200 MB of tokenizer files)
uv run pytest -m slow -s      # real models end to end (~4.5 GB of checkpoints)
```

The offline suite needs no downloads. It checks:
* every decoding strategy against brute-force enumeration on an oracle LM;
* both backends against the conformance suite (`pcd.testing.check_backend`) on tiny
  random Llama, Qwen3, Gemma 2 and Gemma 3 checkpoints;
* **MLX and PyTorch against each other on the same weights** (log-probs agree to 1e-5,
  decisions identical), including sliding-window and soft-capped models.

## Limitations

* Only booleans and closed enums. Free text, numbers and nested objects need normal
  generation.
* Probabilities are renormalized over the allowed labels, not calibrated. Use
  `candidate_mass` and a labelled sample to decide thresholds.
* Fields in the same wave are decided independently. Asked in isolation, small models
  sometimes answer yes/no questions differently from when they write a whole object
  (notebook 05; LFM2.5 on `is_spam` above). Use `depends_on` for fields that must agree,
  and measure accuracy on real labelled data before trusting any decoder.
* Numeric-code labels such as `CAT_185_Aircraft_Parts` make the model commit to digits
  before it reaches the meaning. If you control the labels, put the meaning first.
* Conformance tolerances: 4-bit hybrid models such as LFM2.5 differ by a total-variation
  distance of ~0.06 between one-shot and two-step prefill *in mlx-lm itself*, which is
  why `check_backend` defaults to `tol=0.1`. A broken cache shows up as 0.3 or more.
* Every batched row holds a copy of the prompt's KV cache. With long prompts and big
  models, lower `kv_budget_bytes` to trade speed for memory.

## Layout

```
src/pcd/
  schema.py       fields, labels, dependency waves; dict / JSON Schema / Pydantic input
  prompt.py       chat-template rendering, head/tail split, catalog, surface forms
  compiler.py     in-context tokenization, decision positions, label tries
  decoding.py     marginal / exact / trie / hybrid decoders (backend-independent)
  engine.py       caching, waves, stats
  baseline.py     autoregressive JSON baseline for comparisons
  backends/       base contract, MLX backend, PyTorch backend
  testing.py      conformance checks for any backend
  models.py       model aliases
  cli.py          `pcd` command
tests/            offline, network and slow suites
examples/         quickstart and the original presets
CONTRIBUTING.md   adding models, adapters and backends
```

The presets in `examples/presets/` come from the original project (Apache 2.0).

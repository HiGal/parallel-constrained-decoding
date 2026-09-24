# Contributing: new models, adapters and backends

`pcd` is built so that supporting a new model is usually zero work, sometimes a
one-line setting, and occasionally a small adapter. This guide goes from the cheapest
fix to the most involved one.

## How the pieces fit

```
Schema ──► PromptFormat ──► compiler ──► decoding ──► Backend (MLX / PyTorch / yours)
fields      chat template     token trie    marginal /     prefill(tokens, parent)
labels      head/tail split   per field     exact / trie   score(prefix, rows, queries)
```

* `schema.py`: fields and labels. No model knowledge.
* `prompt.py`: renders the model's chat template around a context slot, then splits it
  into a cached **head**, the **context** and a static **tail** ending in `{`.
* `compiler.py`: tokenizes every JSON line *in context*, finds each field's decision
  position (longest common token prefix) and builds the label trie. It checks that token
  boundaries are clean and raises `TokenizationError` if they are not.
* `decoding.py`: backend-independent. Turns the trie into rows and queries, sends them as
  one batched call per round, and turns log-probabilities into decisions.
* `backends/`: the only code that touches a framework. Two primitives, nothing else.

## Step 1: does the model already work?

```bash
uv run pcd check <repo-or-alias> --backend mlx     # or --backend torch
```

This runs the backend conformance checks (prefix caching, broadcast, batching,
chunking, immutability, normalization), compiles a schema with colliding labels, and
does one end-to-end extraction. If everything says `PASS`, the model is supported: add an
alias to `src/pcd/models.py` (optional) and you are done.

## Step 2: prompt-level quirks (no code)

Most model-specific behaviour lives in the chat template, and `PromptFormat` has a knob
for each problem we have met:

| Symptom | Setting |
|---|---|
| Template raises on a system message | nothing: detected automatically, system text is merged into the user turn |
| Model "thinks" before answering (Qwen3, SmolLM3 hybrids) | `template_kwargs={"enable_thinking": False}` (the default) |
| Assistant turn needs a header before the content (gpt-oss "harmony") | `assistant_prefix="<\|channel\|>final<\|message\|>"` |
| Base model without a chat template | nothing: a plain-text format is used automatically |
| Very long label lists | `list_choices=50` lists at most 50 per field (quality drops, prefill shrinks) |
| Model keeps writing booleans as strings | nothing: `"true"`/`"false"` are scored as forms of `True`/`False` by default |

To make a setting the default for a whole model family, add it to
`FAMILY_DEFAULTS` in `src/pcd/prompt.py`, keyed by `config.model_type`:

```python
FAMILY_DEFAULTS = {
    "gpt_oss": {"assistant_prefix": "<|channel|>final<|message|>"},
    "my_family": {"template_kwargs": {"reasoning": "off"}},
}
```

## Step 3: tokenizer quirks

The compiler must know where tokens start and end around the JSON lines. It tries two
newline layouts and keeps the first that tokenizes cleanly:

* `after`: prompt ends with `{\n`, lines are `  "key": value,\n` (GPT-4-style BPE like Qwen,
  Llama 3, Phi-4, and SentencePiece models like Gemma, Mistral, Phi-3.5);
* `before`: prompt ends with `{`, lines are `\n  "key": value,` (GPT-2-style BPE like
  SmolLM2 and Granite, which glue the newline to the following spaces).

If neither works you get a `TokenizationError` naming the boundary that merges. Try
`PromptFormat(indent="\t")` or `indent=" "` first. If the tokenizer is not a Hugging Face
tokenizer at all, implement the small `Tokenizer` protocol (`src/pcd/tokenizer.py`):

```python
class MyTokenizer:
    def encode(self, text: str) -> list[int]: ...        # no BOS/EOS added
    def decode(self, ids) -> str: ...
    def render_chat(self, messages, **kwargs) -> str | None: ...  # None = no chat template
    bos_text: str          # text to prepend to plain prompts ("" if none)
    eos_ids: frozenset[int]
    pad_id: int
    vocab_size: int
```

## Step 4: model-level adapters (MLX / PyTorch)

Two things can be model-specific inside the built-in backends.

**Caches.** Every KV row is a copy of the prompt's cache.
* MLX clones caches through mlx-lm's `state` / `meta_state` / `from_state` protocol,
  which covers `KVCache`, `RotatingKVCache` (sliding windows), `QuantizedKVCache` and
  `ArraysCache` (SSM / linear-attention layers). A new cache class works if it
  implements that protocol with the batch on axis 0.
* PyTorch uses `copy.deepcopy(cache)` plus `cache.batch_repeat_interleave(n)`, which all
  `transformers` caches implement.

**The fast path.** To avoid projecting every position onto the vocabulary, the backends
split the model into a *body* (hidden states) and a *head* (logits), and project only the
positions they read. The split is verified against a full forward pass when the backend
is created and switched off if it differs. Correctness never depends on it. To enable it
for an architecture whose head does extra work, register the split. This is the
built-in adapter for Gemma 2's logit soft-capping (`src/pcd/backends/mlx_backend.py`):

```python
import mlx.core as mx
from pcd.backends.mlx_backend import register_head

def gemma2_split(model):
    cap = model.final_logit_softcapping
    body = lambda x, cache: model.model(x, cache=cache)
    head = lambda h: mx.tanh(model.model.embed_tokens.as_linear(h) / cap) * cap
    return body, head

register_head("gemma2", gemma2_split)   # verified on load like the default split
```

## Step 5: a new backend (llama.cpp, vLLM, ONNX, Core ML, ...)

Subclass `Backend` and implement two methods. The base class gives you chunk planning
(`plan_chunks`), padding and validation helpers.

```python
import numpy as np
from pcd.backends.base import Backend, Prefix, Query, gather_layout, pad_rows

class MyBackend(Backend):
    name = "mine"

    def prefill(self, tokens, parent=None) -> Prefix:
        # Run `tokens` on top of parent's KV state. Never modify `parent`.
        state = my_copy(parent.state) if parent else my_new_cache()
        my_forward(tokens, state)
        self.forward_passes += 1
        return Prefix(tokens=(parent.tokens if parent else ()) + tuple(tokens),
                      state=state, nbytes=my_state_size(state))

    def score(self, prefix, rows, queries) -> list[np.ndarray]:
        # Every row continues `prefix`. For each Query(row, position, tokens) return the
        # log-softmax over the full vocabulary at that position, for those token ids.
        self.validate(rows, queries)
        out = [None] * len(queries)
        for chunk in self.plan_chunks(prefix, rows, queries):
            batch = pad_rows([rows[r] for r in chunk], self.tokenizer.pad_id)  # right padding is safe
            cache = my_broadcast(prefix.state, len(chunk))                     # copies, never the original
            logits = my_forward_batch(batch, cache)                            # (n, L, V)
            self.forward_passes += 1
            qids, ridx, pidx, toks = gather_layout(chunk, queries)
            for i, q in enumerate(qids):
                row_logits = logits[ridx[i], pidx[i]]
                lp = row_logits - np.logaddexp.reduce(row_logits)
                out[q] = lp[list(queries[q].tokens)]
        return out

    # Optional, only for the autoregressive baseline (`pcd bench`):
    def greedy(self, prefix, inputs, max_new_tokens, stop_ids, should_stop=None) -> list[int]: ...
```

Then prove it with the conformance suite. Every check compares two ways of computing
the same numbers, so a random model works too:

```python
from pcd.testing import check_backend
for c in check_backend(MyBackend(...)):
    print(c)          # all must PASS
```

and run the engine tests against it (`tests/test_engine.py` shows how to plug a backend
into `Engine`).

## Adding a model alias

Add a `ModelSpec` to `src/pcd/models.py` with the MLX and PyTorch repo ids (check they
exist), then add the model to `CASES` in `tests/test_real_models.py` and run:

```bash
uv run pytest -m slow -s -k "<alias>"
```

## Tests

```bash
uv run pytest                 # offline: schema, prompt, compiler, decoders vs brute force,
                              # both backends on tiny random models, MLX <-> PyTorch parity
uv run pytest -m network      # 15 real tokenizers compile every preset (downloads tokenizers)
uv run pytest -m slow -s      # real models end to end on both backends (~4.5 GB download)
```

Keep new code backend-independent unless it has to touch a framework, and add a
brute-force or cross-backend test for any change to decoding.

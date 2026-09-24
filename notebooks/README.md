# pcd from the inside

Six notebooks that explain how `pcd` works, written as engineering blog posts. Each post makes
its case by running the library on a real model; the saved outputs are part of the post, and
every number in the prose was printed by a cell above it.

| # | Post | What it covers |
|---|---|---|
| 01 | [Stop generating the JSON you already know](01_why_parallel_decoding.ipynb) | The autoregressive baseline, where its tokens go, the core idea, `pcd` end to end, and how the code is organized |
| 02 | [What the model reads](02_schemas_and_prompts.ipynb) | Schemas and waves, the catalog of allowed values, rendering through the model's own chat template, template quirks, and why the catalog is not optional |
| 03 | [Tokens are not characters](03_compiling_for_the_tokenizer.ipynb) | In-context tokenization, the boundary check, where the newline goes, decision positions, label tries, and ten tokenizers |
| 04 | [Reading decisions off the logits](04_decoding.ipynb) | Rows and queries, first-token decisions, quoted booleans, colliding labels, exact scoring, the 255-label trie, and a brute-force proof |
| 05 | [Two primitives, every cache](05_backends.ipynb) | The backend contract, copying KV caches, batch shapes, memory budgets, the fast path, four architectures, and MLX against PyTorch |
| 06 | [Waves, caches and the bill](06_engine_and_benchmarks.ipynb) | A request phase by phase, the head cache, waves and anchoring, the benchmark against token-by-token JSON, scaling, and when `pcd` loses |

Read them in order: each post builds on the previous ones and links back instead of repeating them.
Post 01 stands on its own if you only want the idea and the numbers.

## Running them

The outputs were produced on an Apple M1 Pro with 16 GB (each notebook prints its machine and
library versions). Timings will differ on other hardware; the decisions, probabilities and token
counts should not.

```bash
cd pcd
uv sync --extra mlx --extra torch --group notebooks
uv run --with jupyter jupyter lab notebooks/      # or open them in your IDE with the project's .venv kernel
```

- **Run one notebook at a time.** Each loads its models one after another and unloads them in
  between; two notebooks at once can run a 16 GB machine out of memory. Post 05 needs the most:
  about 7 GB at its peak in our runs, during the float32 PyTorch comparison.
- **Models are loaded offline.** `nbutils.py` sets `HF_HUB_OFFLINE=1`, so download the checkpoints
  first, or set `HF_HUB_OFFLINE=0` before starting Jupyter to let them download on first use:

| Used by | Models (MLX, 4-bit unless noted) | Tokenizers only |
|---|---|---|
| all posts | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | |
| 02, 06 | `mlx-community/Qwen3-0.6B-4bit` | |
| 05 | `mlx-community/gemma-3-1b-it-4bit`, `mlx-community/LFM2.5-1.2B-Instruct-4bit`, `mlx-community/Qwen3.5-0.8B-4bit`; `Qwen/Qwen2.5-0.5B-Instruct` (bf16, on MLX and PyTorch) | |
| 02, 03 | | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3.5-0.8B`, `HuggingFaceTB/SmolLM2-360M-Instruct`, `HuggingFaceTB/SmolLM3-3B`, `LiquidAI/LFM2.5-1.2B-Instruct`, `google/gemma-4-e2b-it`, and the tokenizers of `mlx-community/Llama-3.2-1B-Instruct-4bit`, `gemma-3-1b-it-4bit`, `gemma-2-2b-it-4bit` |

Once the models are cached, 01 to 04 each run top to bottom in about a minute or less, and 05 and 06 in about three.

`nbutils.py` is the only shared code: model loading and unloading, presets, and display and chart
helpers. It does not reimplement anything in `pcd`; the notebooks call the library directly.

The investigation that led to `pcd`, a teardown of the original engine in `Qwen-2.5-1B-RLCD/`,
lives in [`../../Qwen-2.5-1B-RLCD/notebooks/`](../../Qwen-2.5-1B-RLCD/notebooks/). This series does
not depend on it.

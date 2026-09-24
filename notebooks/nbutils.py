"""Glue for the notebook series: paths, presets, model loading, display helpers, chart style.

Nothing here reimplements `pcd`. The notebooks call the library directly; this module only
keeps their setup cells short.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # every model and tokenizer in the series is cached locally
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message="IProgress not found")

import logging

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import gc
import inspect
import json
import platform
import statistics
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pcd"
PRESETS = ROOT / "examples" / "presets"
PRESET_NAMES = ["support_triage", "fintech_fraud", "code_security", "high_cardinality_255"]

#: The model most posts use: small, fast, and the original project's model family.
MODEL = "qwen2.5-1.5b"


def load_preset(name: str) -> dict:
    """A preset from ``examples/presets``: ``{"id", "title", "context", "schema", ...}``."""
    return json.loads((PRESETS / f"{name}.json").read_text())


# ---------------------------------------------------------------------- models

_BACKENDS: dict[tuple[str, str], Any] = {}


def load(model: str = MODEL, backend: str = "mlx"):
    """Load a backend once per kernel. Wrap it in as many ``Engine``s as you like:
    ``Engine(load(), prompt_format=..., decode=...)`` reuses the weights."""
    from pcd import load_backend

    key = (model, backend)
    if key not in _BACKENDS:
        _BACKENDS[key] = load_backend(model, backend)
    return _BACKENDS[key]


def unload(model: str | None = None) -> None:
    """Drop cached backends (all, or one model) and give the memory back."""
    for key in [k for k in _BACKENDS if model is None or k[0] == model]:
        del _BACKENDS[key]
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def timed(fn: Callable[[], Any], repeats: int = 3, warmup: int = 1) -> tuple[float, Any]:
    """(median milliseconds, last result) of ``fn()``."""
    for _ in range(warmup):
        fn()
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), out


def environment() -> None:
    """Print the machine and library versions the saved outputs came from."""
    chip = ""
    if platform.system() == "Darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
            chip += f", {mem:.0f} GB"
        except Exception:
            pass
    versions = []
    for pkg in ("pcd", "mlx", "mlx-lm", "torch", "transformers"):
        try:
            versions.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    os_name = f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else f"{platform.system()} {platform.release()}"
    print(f"{chip or platform.machine()} | {os_name} | "
          f"Python {platform.python_version()} | " + ", ".join(versions))


# ---------------------------------------------------------------------- display

def md(text: str) -> None:
    display(Markdown(text))


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Render rows as a Markdown table."""

    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    md("\n".join(out))


def show_source(target: Any, start: str | None = None, end: str | None = None, max_lines: int = 60) -> None:
    """Print library source with line numbers.

    ``target`` is a function, class or method, or a path relative to ``src/pcd``
    (``"decoding.py"``). With ``start``/``end``, print from the first line containing
    ``start`` through the first later line containing ``end``.
    """
    if isinstance(target, (str, Path)):
        path = SRC / target
        lines = path.read_text().splitlines()
        first = 0
    else:
        target = inspect.unwrap(target)
        path = Path(inspect.getsourcefile(target))
        lines, first = inspect.getsourcelines(target)
        lines = [l.rstrip("\n") for l in lines]
        first -= 1
    i = 0 if start is None else next(k for k, l in enumerate(lines) if start in l)
    j = i + max_lines if end is None else next(k for k in range(i + 1, len(lines)) if end in lines[k]) + 1
    j = min(j, len(lines))
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        rel = path
    print(f"# {rel}")
    width = len(str(first + j))
    for k in range(i, j):
        print(f"{first + k + 1:>{width}}  {lines[k]}")
    if j < len(lines) and end is None and max_lines < len(lines) - i:
        print(f"{'':>{width}}  ...")


def vis(s: str) -> str:
    """Make whitespace visible: space -> '·', newline -> '⏎', tab -> '→'."""
    return s.replace(" ", "·").replace("\n", "⏎").replace("\t", "→")


def pieces(tok: Any, ids: Sequence[int]) -> list[str]:
    """Decode every token id on its own (``tok`` is a pcd tokenizer or a HF tokenizer)."""
    return [tok.decode([int(i)]) for i in ids]


def show_tokens(tok: Any, ids_or_text: Any, label: str = "") -> list[str]:
    """Print tokens separated by '│', with whitespace made visible. Returns the pieces."""
    ids = tok.encode(ids_or_text) if isinstance(ids_or_text, str) else list(ids_or_text)
    ps = pieces(tok, ids)
    print((f"{label:<14}" if label else "") + "│" + "│".join(vis(p) for p in ps) + f"│   ({len(ps)} tokens)")
    return ps


def clip(text: str, head: int = 400, tail: int = 200) -> str:
    """Shorten long text for display, keeping both ends."""
    if len(text) <= head + tail + 20:
        return text
    return f"{text[:head]}\n   [... {len(text) - head - tail} characters ...]\n{text[-tail:]}"


# ---------------------------------------------------------------------- charts

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#3d3c38"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


style()

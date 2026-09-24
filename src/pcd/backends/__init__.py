"""Backends: MLX (Apple Silicon) and PyTorch (CUDA / MPS / CPU).

``load_backend("auto")`` picks MLX on Apple Silicon when ``mlx-lm`` is installed and
PyTorch otherwise. Third-party backends only need to subclass ``Backend``; see
CONTRIBUTING.md and ``pcd.testing.check_backend``.
"""

from __future__ import annotations

import importlib.util
import platform
from typing import Any

from .base import Backend, Prefix, Query

__all__ = ["Backend", "Prefix", "Query", "available_backends", "load_backend", "resolve_backend_name"]


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def available_backends() -> list[str]:
    out = []
    if platform.system() == "Darwin" and platform.machine() == "arm64" and _has("mlx") and _has("mlx_lm"):
        out.append("mlx")
    if _has("torch") and _has("transformers"):
        out.append("torch")
    return out


def resolve_backend_name(name: str = "auto") -> str:
    name = name.lower()
    avail = available_backends()
    if name == "auto":
        if not avail:
            raise RuntimeError("no backend installed: `uv pip install 'pcd[mlx]'` or `'pcd[torch]'`")
        return avail[0]
    if name not in ("mlx", "torch"):
        raise ValueError(f"unknown backend {name!r}; use 'mlx', 'torch' or 'auto'")
    if name not in avail:
        extra = "mlx" if name == "mlx" else "torch"
        raise RuntimeError(f"backend {name!r} is not available here; install `pcd[{extra}]`")
    return name


def load_backend(model: str, backend: str = "auto", **kwargs: Any) -> Backend:
    """Load ``model`` (a repo id, local path or alias from ``pcd.models``) on a backend."""
    from ..models import resolve

    name = resolve_backend_name(backend)
    repo = resolve(model, name)
    if name == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend.load(repo, **kwargs)
    from .torch_backend import TorchBackend

    return TorchBackend.load(repo, **kwargs)

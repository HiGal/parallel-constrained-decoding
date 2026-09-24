"""PyTorch backend, built on Hugging Face ``transformers``.

Runs on CUDA, Apple ``mps`` or CPU, and works with any ``AutoModelForCausalLM`` whose
cache supports ``batch_repeat_interleave`` (all ``transformers`` caches do).

Fast path: run the decoder body, gather the queried positions, and project only those
through the output embedding. It is enabled only after a probe shows it reproduces the
full forward pass. Otherwise the backend asks the model for the logits of the queried
positions only (``logits_to_keep``), which keeps any model-specific post-processing
such as Gemma's logit soft-capping.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Sequence

import numpy as np
import torch

from ..tokenizer import HFTokenizer
from .base import Backend, Prefix, Query, gather_layout, pad_rows, verify_close


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.startswith("mps"):
        try:
            torch.ones(2, dtype=torch.bfloat16, device="mps").sum().item()
            return torch.bfloat16
        except Exception:
            return torch.float16
    return torch.float32


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, model: Any, tokenizer: Any, *, fast_path: bool | None = None, **kwargs: Any):
        tok = tokenizer if isinstance(tokenizer, HFTokenizer) else HFTokenizer(tokenizer)
        super().__init__(tok, **kwargs)
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self._keep_arg = "logits_to_keep" if "logits_to_keep" in inspect.signature(model.forward).parameters else None
        self._split = None
        if fast_path is not False:
            self._split = self._probe_fast_path()
            if fast_path and self._split is None:
                raise RuntimeError("fast_path=True, but the body/head split does not reproduce the model's logits")
        self.forward_passes = 0

    @classmethod
    def load(cls, model_id: str, *, device: str | None = None, dtype: Any = None, **kwargs: Any) -> "TorchBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or default_device()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        dtype = dtype or default_dtype(device)
        from transformers import AutoConfig

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        extra: dict[str, Any] = {}
        config = AutoConfig.from_pretrained(model_id)
        if getattr(config, "attn_logit_softcapping", None):
            # SDPA kernels skip attention soft-capping (Gemma 2); eager attention applies it.
            extra["attn_implementation"] = "eager"
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **extra)
        except TypeError:  # transformers < 4.56 only knows torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **extra)
        return cls(model.to(device), tokenizer, model_id=model_id, **kwargs)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
            "fast_path": self._split is not None,
            "model_type": getattr(self.model.config, "model_type", type(self.model).__name__),
        }

    # ------------------------------------------------------------------ contract

    @torch.inference_mode()
    def prefill(self, tokens: Sequence[int], parent: Prefix | None = None) -> Prefix:
        tokens = [int(t) for t in tokens]
        if not tokens:
            if parent is None:
                raise ValueError("cannot prefill an empty sequence without a parent")
            return parent
        cache = copy.deepcopy(parent.state) if parent is not None else None
        for i in range(0, len(tokens), self.prefill_step):
            ids = torch.tensor([tokens[i : i + self.prefill_step]], device=self.device)
            kwargs = {self._keep_arg: 1} if self._keep_arg else {}
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, **kwargs)
            cache = out.past_key_values
            self.forward_passes += 1
        all_tokens = (parent.tokens if parent is not None else ()) + tuple(tokens)
        return Prefix(tokens=all_tokens, state=cache, nbytes=_cache_nbytes(cache))

    @torch.inference_mode()
    def score(self, prefix: Prefix, rows: Sequence[Sequence[int]], queries: Sequence[Query]) -> list[np.ndarray]:
        self.validate(rows, queries)
        out: list[np.ndarray | None] = [None] * len(queries)
        pad = self.tokenizer.pad_id
        for chunk in self.plan_chunks(prefix, rows, queries):
            batch = torch.tensor(pad_rows([rows[r] for r in chunk], pad), device=self.device)
            cache = self._broadcast(prefix.state, len(chunk))
            qids, ridx, pidx, toks = gather_layout(chunk, queries)
            r_t = torch.tensor(ridx, device=self.device)
            if self._split is not None:
                body, head = self._split
                hidden = body(input_ids=batch, past_key_values=cache, use_cache=True).last_hidden_state
                logits = head(hidden[r_t, torch.tensor(pidx, device=self.device)])
            elif self._keep_arg:
                keep = sorted(set(pidx))
                col = {p: j for j, p in enumerate(keep)}
                full = self.model(
                    input_ids=batch, past_key_values=cache, use_cache=True,
                    **{self._keep_arg: torch.tensor(keep, device=self.device)},
                ).logits
                logits = full[r_t, torch.tensor([col[p] for p in pidx], device=self.device)]
            else:
                full = self.model(input_ids=batch, past_key_values=cache, use_cache=True).logits
                logits = full[r_t, torch.tensor(pidx, device=self.device)]
            self.forward_passes += 1
            logits = logits.float()
            lp = torch.gather(logits, 1, torch.tensor(toks, device=self.device)) - torch.logsumexp(
                logits, dim=-1, keepdim=True
            )
            lp = lp.cpu().double().numpy()
            for i, q in enumerate(qids):
                out[q] = lp[i, : len(queries[q].tokens)]
        return out  # type: ignore[return-value]

    @torch.inference_mode()
    def greedy(self, prefix, inputs, max_new_tokens, stop_ids, should_stop=None) -> list[int]:
        cache = copy.deepcopy(prefix.state)
        ids = torch.tensor([[int(t) for t in inputs]], device=self.device)
        kwargs = {self._keep_arg: 1} if self._keep_arg else {}
        generated: list[int] = []
        while len(generated) < max_new_tokens:
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, **kwargs)
            self.forward_passes += 1
            cache = out.past_key_values
            tok = int(out.logits[0, -1].argmax().item())
            if tok in stop_ids:
                break
            generated.append(tok)
            if should_stop is not None and should_stop(generated):
                break
            ids = torch.tensor([[tok]], device=self.device)
        return generated

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _broadcast(cache: Any, n: int) -> Any:
        c = copy.deepcopy(cache)
        if n > 1:
            c.batch_repeat_interleave(n)
        return c

    @torch.inference_mode()
    def _probe_fast_path(self):
        try:
            body = self.model.get_decoder()
            head = self.model.get_output_embeddings()
            if body is None or head is None:
                return None
            x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=self.device)
            # Same shapes as the model's own forward (see the MLX backend).
            ref = self.model(input_ids=x).logits[0, -1]
            got = head(body(input_ids=x).last_hidden_state)[0, -1]
            ok = verify_close(ref.float().cpu().numpy(), got.float().cpu().numpy())
            return (body, head) if ok else None
        except Exception:
            return None


def _cache_nbytes(cache: Any) -> int:
    total = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:  # transformers >= 4.56
        for layer in layers:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total
    for name in ("key_cache", "value_cache"):  # older DynamicCache
        for t in getattr(cache, name, []) or []:
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
    return total

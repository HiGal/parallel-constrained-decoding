"""Backend conformance on tiny random models, and MLX <-> PyTorch parity on shared weights."""

import numpy as np
import pytest
from helpers import TINY_ARCHS, save_tiny

from pcd import Engine
from pcd.backends import available_backends
from pcd.backends.base import Query
from pcd.testing import check_backend

needs_both = pytest.mark.skipif(
    not {"mlx", "torch"} <= set(available_backends()), reason="needs both MLX and PyTorch"
)


@pytest.fixture(scope="module")
def arch_pairs(tmp_path_factory, byte_tok):
    """For every tiny architecture: (mlx backend, torch backend) on the same float32 weights."""
    if not {"mlx", "torch"} <= set(available_backends()):
        pytest.skip("needs both MLX and PyTorch")
    import mlx_lm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pcd.backends.mlx_backend import MLXBackend
    from pcd.backends.torch_backend import TorchBackend

    out = {}
    for arch in TINY_ARCHS:
        d = save_tiny(tmp_path_factory.mktemp(arch), byte_tok, arch)
        model, tok = mlx_lm.load(d)
        kw = {"attn_implementation": "eager"} if arch == "gemma2" else {}
        tmodel = AutoModelForCausalLM.from_pretrained(d, dtype="float32", **kw)
        out[arch] = (MLXBackend(model, tok, model_id=arch), TorchBackend(tmodel, AutoTokenizer.from_pretrained(d), model_id=arch))
    return out


def test_conformance(tiny_backend):
    failed = [str(c) for c in check_backend(tiny_backend, tol=1e-4) if not c.passed]
    assert not failed, failed


@needs_both
@pytest.mark.parametrize("arch", TINY_ARCHS)
def test_every_architecture_conforms_on_both_backends(arch_pairs, arch):
    for backend in arch_pairs[arch]:
        failed = [str(c) for c in check_backend(backend, tol=1e-4) if not c.passed]
        assert not failed, (backend.name, failed)


@needs_both
@pytest.mark.parametrize("arch", TINY_ARCHS)
def test_mlx_and_torch_agree(arch_pairs, arch, byte_tok):
    mlx_b, torch_b = arch_pairs[arch]
    # gemma2 soft-caps its logits: the generic split fails verification and PyTorch falls back
    # to logits_to_keep; MLX has a registered soft-capping head that passes verification.
    assert mlx_b.describe()["fast_path"] is True
    assert torch_b.describe()["fast_path"] == (arch != "gemma2")
    ids = byte_tok.encode("A prompt long enough to overflow an eight-token sliding window.", add_special_tokens=False)
    rows = [tuple(ids[30:41]), tuple(ids[41:44]), tuple(ids[44:45])]
    vocab = tuple(range(len(byte_tok)))
    queries = [Query(r, p, vocab) for r, row in enumerate(rows) for p in range(len(row))]
    a = mlx_b.score(mlx_b.prefill(ids[:30]), rows, queries)
    b = torch_b.score(torch_b.prefill(ids[:30]), rows, queries)
    assert max(float(np.max(np.abs(x - y))) for x, y in zip(a, b)) < 1e-4


@needs_both
def test_engine_decisions_identical_across_backends(arch_pairs, presets):
    for arch in ("llama", "gemma3_text"):
        mlx_b, torch_b = arch_pairs[arch]
        for name in ("fintech_fraud", "support_triage"):
            p = presets[name]
            a = Engine(mlx_b).extract(p["context"], p["schema"])
            b = Engine(torch_b).extract(p["context"], p["schema"])
            assert a.values == b.values, (arch, name)
            for k in a.fields:
                assert a.fields[k].probability == pytest.approx(b.fields[k].probability, abs=1e-3)


def test_unverified_fast_path_is_disabled(tmp_path, byte_tok):
    """A head that does not reproduce the model's logits must be rejected on load."""
    if "mlx" not in available_backends():
        pytest.skip("needs MLX")
    import mlx_lm

    from pcd.backends import mlx_backend

    model, tok = mlx_lm.load(save_tiny(tmp_path, byte_tok, "gemma2"))
    saved = mlx_backend._HEAD_REGISTRY.pop("gemma2")
    try:
        assert mlx_backend.MLXBackend(model, tok).describe()["fast_path"] is False
        with pytest.raises(RuntimeError):
            mlx_backend.MLXBackend(model, tok, fast_path=True)
    finally:
        mlx_backend.register_head("gemma2", saved)


def test_greedy_matches_step_by_step_argmax(tiny_backend, byte_tok):
    ids = byte_tok.encode("Greedy decoding check: ", add_special_tokens=False)
    prefix = tiny_backend.prefill(ids[:-1])
    out = tiny_backend.greedy(prefix, ids[-1:], 12, frozenset())
    vocab = tuple(range(len(byte_tok)))
    row = tuple(ids[-1:])
    for t in out:
        lp = tiny_backend.score(prefix, [row], [Query(0, len(row) - 1, vocab)])[0]
        assert int(np.argmax(lp)) == t
        row += (t,)
    assert len(out) == 12


def test_chunking_by_memory_budget_counts_passes(tiny_backend, byte_tok):
    ids = byte_tok.encode("Some shared prefix for the rows.", add_special_tokens=False)
    prefix = tiny_backend.prefill(ids)
    rows = [tuple(ids[i : i + 3]) for i in range(5)]
    queries = [Query(i, 2, (1, 2, 3)) for i in range(5)]
    saved = tiny_backend.kv_budget_bytes
    try:
        tiny_backend.kv_budget_bytes = prefix.nbytes * 2  # room for two broadcast copies per pass
        before = tiny_backend.forward_passes
        small = tiny_backend.score(prefix, rows, queries)
        assert tiny_backend.forward_passes - before == 3
    finally:
        tiny_backend.kv_budget_bytes = saved
    big = tiny_backend.score(prefix, rows, queries)
    assert max(float(np.max(np.abs(x - y))) for x, y in zip(small, big)) < 1e-4


def test_invalid_queries_are_rejected(tiny_backend):
    prefix = tiny_backend.prefill([1, 2, 3])
    with pytest.raises(IndexError):
        tiny_backend.score(prefix, [(4, 5)], [Query(0, 2, (1,))])
    with pytest.raises(IndexError):
        tiny_backend.score(prefix, [(4, 5)], [Query(1, 0, (1,))])
    with pytest.raises(ValueError):
        tiny_backend.prefill([])

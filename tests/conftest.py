import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import byte_tokenizer, save_tiny_llama  # noqa: E402

from pcd.backends import available_backends  # noqa: E402

PRESETS = Path(__file__).parents[1] / "examples" / "presets"


@pytest.fixture(scope="session")
def presets() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(PRESETS.glob("*.json"))}


@pytest.fixture(scope="session")
def byte_tok():
    return byte_tokenizer()


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory, byte_tok):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return save_tiny_llama(tmp_path_factory.mktemp("tiny_llama"), byte_tok)


def _make_backend(name: str, tiny_dir: str):
    if name not in available_backends():
        pytest.skip(f"{name} backend not installed")
    if name == "mlx":
        import mlx_lm

        from pcd.backends.mlx_backend import MLXBackend

        model, tok = mlx_lm.load(tiny_dir)
        return MLXBackend(model, tok, model_id="tiny-llama")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pcd.backends.torch_backend import TorchBackend

    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype="float32")
    return TorchBackend(model, AutoTokenizer.from_pretrained(tiny_dir), model_id="tiny-llama")


@pytest.fixture(scope="session", params=["mlx", "torch"])
def tiny_backend(request, tiny_dir):
    return _make_backend(request.param, tiny_dir)


@pytest.fixture(scope="session")
def mlx_tiny(tiny_dir):
    return _make_backend("mlx", tiny_dir)


@pytest.fixture(scope="session")
def torch_tiny(tiny_dir):
    return _make_backend("torch", tiny_dir)

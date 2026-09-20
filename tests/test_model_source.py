import json

import httpx

from kv_cache_capacity.model import ModelConfig, load_model_config


CONFIG = {
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "hidden_size": 3584,
}


def test_load_local_config(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CONFIG), encoding="utf-8")
    model = load_model_config(path)
    assert model.num_hidden_layers == 28
    assert model.num_key_value_heads == 4


def test_load_huggingface_config(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return CONFIG

    def fake_get(*args, **kwargs):
        assert args[0].endswith("/org/model/resolve/main/config.json")
        assert kwargs["follow_redirects"] is True
        return Response()

    monkeypatch.setattr(httpx, "get", fake_get)
    model = ModelConfig.from_huggingface("org/model")
    assert model.hidden_size == 3584

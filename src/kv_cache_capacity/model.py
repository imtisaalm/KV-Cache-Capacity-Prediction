from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class ModelConfig:
    """Normalized transformer fields required for KV-cache accounting."""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int | None = None
    sliding_window: int | None = None
    full_attention_layers: int | None = None

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    def token_layer_units(self, context_length: int) -> int:
        """Return cached token-layer positions for one sequence."""
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.sliding_window is None:
            return self.num_hidden_layers * context_length

        full = self.full_attention_layers or 0
        if not 0 <= full <= self.num_hidden_layers:
            raise ValueError("full_attention_layers must be within the layer count")
        windowed = self.num_hidden_layers - full
        return full * context_length + windowed * min(context_length, self.sliding_window)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ModelConfig":
        attention_heads = int(raw["num_attention_heads"])
        full_layers = raw.get("full_attention_layers")
        return cls(
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=attention_heads,
            num_key_value_heads=int(raw.get("num_key_value_heads", attention_heads)),
            hidden_size=int(raw["hidden_size"]),
            head_dim=int(raw["head_dim"]) if raw.get("head_dim") is not None else None,
            sliding_window=int(raw["sliding_window"]) if raw.get("sliding_window") is not None else None,
            full_attention_layers=int(full_layers) if full_layers is not None else None,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        *,
        token: str | None = None,
        timeout: float = 15.0,
    ) -> "ModelConfig":
        """Load config.json for a model hosted on Hugging Face."""
        if not model_id or "/" not in model_id:
            raise ValueError("model_id must use the namespace/model form")
        headers: dict[str, str] = {}
        auth_token = token or os.getenv("HF_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        response = httpx.get(
            url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Hugging Face config response must be a JSON object")
        return cls.from_mapping(payload)


def load_model_config(source: str | Path) -> ModelConfig:
    """Load a local config.json path or a Hugging Face model ID."""
    path = Path(source).expanduser()
    if path.is_file():
        return ModelConfig.from_json(path)
    return ModelConfig.from_huggingface(str(source))

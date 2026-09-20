from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int | None = None
    sliding_window: int | None = None

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ModelConfig":
        attention_heads = int(raw["num_attention_heads"])
        return cls(
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=attention_heads,
            num_key_value_heads=int(raw.get("num_key_value_heads", attention_heads)),
            hidden_size=int(raw["hidden_size"]),
            head_dim=int(raw["head_dim"]) if raw.get("head_dim") is not None else None,
            sliding_window=int(raw["sliding_window"]) if raw.get("sliding_window") is not None else None,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

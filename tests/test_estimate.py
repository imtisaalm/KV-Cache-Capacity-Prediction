import pytest

from kv_cache_capacity.estimate import estimate_kv_cache
from kv_cache_capacity.model import ModelConfig


def llama_like() -> ModelConfig:
    return ModelConfig(
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
    )


def test_gqa_uses_kv_heads_not_query_heads() -> None:
    result = estimate_kv_cache(llama_like(), context_length=4096, kv_dtype="bf16")
    expected = 2 * 32 * 4096 * 8 * 128 * 2
    assert result.total_bytes_per_rank == expected


def test_tensor_parallel_shards_cache_per_rank() -> None:
    single = estimate_kv_cache(llama_like(), context_length=2048, concurrency=4)
    tp4 = estimate_kv_cache(
        llama_like(), context_length=2048, concurrency=4, tensor_parallel_size=4
    )
    assert tp4.total_bytes_per_rank == single.total_bytes_per_rank / 4


def test_sliding_window_caps_windowed_layers() -> None:
    model = ModelConfig(
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_size=5120,
        sliding_window=4096,
        full_attention_layers=8,
    )
    result = estimate_kv_cache(model, context_length=16384)
    assert result.token_layer_units_per_sequence == 8 * 16384 + 32 * 4096


def test_invalid_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported KV dtype"):
        estimate_kv_cache(llama_like(), context_length=1024, kv_dtype="int4")

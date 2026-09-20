from __future__ import annotations

from dataclasses import dataclass

from .model import ModelConfig


DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
}


@dataclass(frozen=True)
class CacheEstimate:
    context_length: int
    concurrency: int
    kv_dtype: str
    tensor_parallel_size: int
    bytes_per_element: float
    bytes_per_token_per_layer: float
    token_layer_units_per_sequence: int
    bytes_per_sequence_per_rank: float
    total_bytes_per_rank: float

    @property
    def total_gib_per_rank(self) -> float:
        return self.total_bytes_per_rank / (1024**3)

    @property
    def per_sequence_gib_per_rank(self) -> float:
        return self.bytes_per_sequence_per_rank / (1024**3)


def estimate_kv_cache(
    model: ModelConfig,
    *,
    context_length: int,
    concurrency: int = 1,
    kv_dtype: str = "bf16",
    tensor_parallel_size: int = 1,
) -> CacheEstimate:
    """Estimate the KV tensor footprint per tensor-parallel rank.

    This is the mathematical tensor footprint. Serving engines may reserve more
    memory because of block allocation, alignment, CUDA graphs, fragmentation,
    model weights, activations, and other runtime state.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")

    normalized_dtype = kv_dtype.lower()
    try:
        element_bytes = DTYPE_BYTES[normalized_dtype]
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"unsupported KV dtype {kv_dtype!r}; expected one of: {supported}") from exc

    kv_width = model.num_key_value_heads * model.effective_head_dim
    # Key and value tensors are both retained for each cached token.
    bytes_per_token_per_layer = 2 * kv_width * element_bytes
    token_layer_units = model.token_layer_units(context_length)
    per_sequence = bytes_per_token_per_layer * token_layer_units / tensor_parallel_size
    total = per_sequence * concurrency

    return CacheEstimate(
        context_length=context_length,
        concurrency=concurrency,
        kv_dtype=normalized_dtype,
        tensor_parallel_size=tensor_parallel_size,
        bytes_per_element=element_bytes,
        bytes_per_token_per_layer=bytes_per_token_per_layer,
        token_layer_units_per_sequence=token_layer_units,
        bytes_per_sequence_per_rank=per_sequence,
        total_bytes_per_rank=total,
    )

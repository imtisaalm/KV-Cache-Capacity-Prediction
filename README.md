# KV Cache Capacity Prediction

A Python package for estimating transformer KV-cache tensor memory and inspecting live KV-cache utilization from a vLLM server.

The estimator operates on normalized model configuration fields rather than a hard-coded model table. The monitor reads vLLM's Prometheus-compatible `/metrics` endpoint.

## Scope

For a decoder-only attention layer, the KV tensor footprint is computed from

```text
2 × num_kv_heads × head_dim × bytes_per_element × cached_token_positions
```

where the factor of two accounts for key and value tensors. The implementation supports:

- multi-head, grouped-query, and multi-query attention through `num_key_value_heads`
- FP32, FP16, BF16, and FP8 cache element widths
- tensor-parallel cache sharding
- sliding-window layers
- hybrid full-attention/sliding-window layer counts
- concurrent sequences

The result is a tensor-footprint estimate. It does not include model weights, activations, CUDA graphs, allocator fragmentation, block-alignment overhead, or engine-specific reservation policy.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Estimate

The input file uses the same core fields commonly present in a Hugging Face `config.json`.

```bash
kv-cache estimate examples/qwen2_5_7b.json \
  --context 32768 \
  --concurrency 8 \
  --kv-dtype bf16 \
  --tensor-parallel 1
```

Example output:

```text
layers: 28
attention heads: 28
KV heads: 4
head dimension: 128
context: 32768
concurrency: 8
KV dtype: bf16
tensor parallel size: 1
per sequence / rank: 1.750 GiB
total / rank: 14.000 GiB
```

For a hybrid attention model, add the following fields when applicable:

```json
{
  "sliding_window": 4096,
  "full_attention_layers": 8
}
```

## Monitor vLLM

```bash
kv-cache monitor http://127.0.0.1:8000
```

One sample:

```bash
kv-cache monitor http://127.0.0.1:8000 --once
```

The monitor currently reads:

- `vllm:kv_cache_usage_perc`
- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- prefix-cache hit/query counters
- preemption count

Example line:

```text
kv=62.5%  running=7  waiting=3  prefix_hit=75.0%  preemptions=4
```

## Validation

Unit tests cover GQA accounting, tensor-parallel sharding, sliding-window layer accounting, dtype validation, Prometheus parsing, and derived prefix-cache hit rate.

```bash
pytest -q
```

## Notes on runtime capacity

Serving engines typically allocate KV cache in blocks and may reserve a fixed cache pool at startup. Runtime allocation can therefore differ from the mathematical tensor footprint calculated here. The live monitor is intentionally separate from the estimator so predicted tensor memory and observed engine utilization are not presented as the same quantity.

## References

- vLLM production metrics: https://docs.vllm.ai/en/latest/usage/metrics/
- vLLM metrics design: https://docs.vllm.ai/en/latest/design/metrics/

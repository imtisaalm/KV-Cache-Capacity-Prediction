from __future__ import annotations

import time

import typer

from .estimate import estimate_kv_cache
from .model import load_model_config
from .monitor import VLLMSnapshot, fetch_vllm_snapshot

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _gib(value: float) -> str:
    return f"{value / (1024**3):.3f} GiB"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _snapshot_line(snapshot: VLLMSnapshot) -> str:
    running = "n/a" if snapshot.running_requests is None else f"{snapshot.running_requests:.0f}"
    waiting = "n/a" if snapshot.waiting_requests is None else f"{snapshot.waiting_requests:.0f}"
    preemptions = "n/a" if snapshot.preemptions is None else f"{snapshot.preemptions:.0f}"
    return (
        f"kv={_pct(snapshot.kv_cache_usage)}  running={running}  waiting={waiting}  "
        f"prefix_hit={_pct(snapshot.prefix_cache_hit_rate)}  preemptions={preemptions}"
    )


@app.command()
def estimate(
    source: str = typer.Argument(
        ...,
        help="Local config.json path or Hugging Face model ID.",
    ),
    context: int = typer.Option(..., min=1),
    concurrency: int = typer.Option(1, min=1),
    kv_dtype: str = typer.Option("bf16", "--kv-dtype"),
    tensor_parallel: int = typer.Option(1, "--tensor-parallel", min=1),
) -> None:
    """Estimate KV tensor memory from a local config or hosted model config."""
    model = load_model_config(source)
    result = estimate_kv_cache(
        model,
        context_length=context,
        concurrency=concurrency,
        kv_dtype=kv_dtype,
        tensor_parallel_size=tensor_parallel,
    )
    typer.echo(f"layers: {model.num_hidden_layers}")
    typer.echo(f"attention heads: {model.num_attention_heads}")
    typer.echo(f"KV heads: {model.num_key_value_heads}")
    typer.echo(f"head dimension: {model.effective_head_dim}")
    typer.echo(f"context: {context}")
    typer.echo(f"concurrency: {concurrency}")
    typer.echo(f"KV dtype: {result.kv_dtype}")
    typer.echo(f"tensor parallel size: {tensor_parallel}")
    typer.echo(f"per sequence / rank: {_gib(result.bytes_per_sequence_per_rank)}")
    typer.echo(f"total / rank: {_gib(result.total_bytes_per_rank)}")


@app.command()
def monitor(
    url: str = typer.Argument("http://127.0.0.1:8000"),
    interval: float = typer.Option(1.0, min=0.1),
    once: bool = typer.Option(False, help="Fetch one sample and exit."),
) -> None:
    """Read cache and scheduler metrics from a vLLM /metrics endpoint."""
    while True:
        snapshot = fetch_vllm_snapshot(url)
        typer.echo(_snapshot_line(snapshot))
        if once:
            return
        time.sleep(interval)


if __name__ == "__main__":
    app()

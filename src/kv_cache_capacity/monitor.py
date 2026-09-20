from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import httpx


_SAMPLE = re.compile(
    r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+|NaN|Inf|-Inf)$'
)


@dataclass(frozen=True)
class VLLMSnapshot:
    kv_cache_usage: float | None
    running_requests: float | None
    waiting_requests: float | None
    prefix_cache_hits: float | None
    prefix_cache_queries: float | None
    preemptions: float | None

    @property
    def prefix_cache_hit_rate(self) -> float | None:
        if self.prefix_cache_hits is None or not self.prefix_cache_queries:
            return None
        return self.prefix_cache_hits / self.prefix_cache_queries


def parse_prometheus(text: str) -> dict[str, list[float]]:
    """Parse numeric samples from Prometheus text exposition.

    Labels are intentionally collapsed because the vLLM gauges consumed here are
    engine-level values. Multiple samples with the same name remain available as
    a list and are summed by metric_sum.
    """
    result: dict[str, list[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        value = float(match.group("value"))
        if math.isnan(value):
            continue
        result.setdefault(match.group("name"), []).append(value)
    return result


def metric_sum(metrics: dict[str, list[float]], names: Iterable[str]) -> float | None:
    for name in names:
        values = metrics.get(name)
        if values is not None:
            return sum(values)
    return None


def snapshot_from_metrics(text: str) -> VLLMSnapshot:
    metrics = parse_prometheus(text)
    return VLLMSnapshot(
        kv_cache_usage=metric_sum(metrics, ["vllm:kv_cache_usage_perc"]),
        running_requests=metric_sum(metrics, ["vllm:num_requests_running"]),
        waiting_requests=metric_sum(metrics, ["vllm:num_requests_waiting"]),
        prefix_cache_hits=metric_sum(
            metrics,
            [
                "vllm:prefix_cache_hits",
                "vllm:prefix_cache_hits_total",
                "vllm:gpu_prefix_cache_hits",
                "vllm:gpu_prefix_cache_hits_total",
            ],
        ),
        prefix_cache_queries=metric_sum(
            metrics,
            [
                "vllm:prefix_cache_queries",
                "vllm:prefix_cache_queries_total",
                "vllm:gpu_prefix_cache_queries",
                "vllm:gpu_prefix_cache_queries_total",
            ],
        ),
        preemptions=metric_sum(
            metrics, ["vllm:num_preemptions", "vllm:num_preemptions_total"]
        ),
    )


def fetch_vllm_snapshot(base_url: str, timeout: float = 5.0) -> VLLMSnapshot:
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    response = httpx.get(metrics_url, timeout=timeout)
    response.raise_for_status()
    return snapshot_from_metrics(response.text)

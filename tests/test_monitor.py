from kv_cache_capacity.monitor import parse_prometheus, snapshot_from_metrics


SAMPLE = """
# HELP vllm:kv_cache_usage_perc KV-cache usage.
vllm:kv_cache_usage_perc 0.625
vllm:num_requests_running{model_name="test"} 7
vllm:num_requests_waiting{model_name="test"} 3
vllm:prefix_cache_hits 900
vllm:prefix_cache_queries 1200
vllm:num_preemptions 4
"""


def test_parser_skips_comments_and_retains_values() -> None:
    parsed = parse_prometheus(SAMPLE)
    assert parsed["vllm:kv_cache_usage_perc"] == [0.625]
    assert parsed["vllm:num_requests_running"] == [7.0]


def test_snapshot_derives_prefix_hit_rate() -> None:
    snapshot = snapshot_from_metrics(SAMPLE)
    assert snapshot.kv_cache_usage == 0.625
    assert snapshot.running_requests == 7
    assert snapshot.waiting_requests == 3
    assert snapshot.prefix_cache_hit_rate == 0.75
    assert snapshot.preemptions == 4

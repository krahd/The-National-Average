from __future__ import annotations

import math

from tna.video.neurips_metrics import count_below, effective_count, normalised_entropy, stats_for, top_share


def test_uniform_distribution_has_maximum_normalised_entropy():
    weights = {str(i): 1.0 for i in range(8)}
    assert math.isclose(normalised_entropy(weights), 1.0, abs_tol=1e-12)
    assert math.isclose(effective_count(weights), 8.0, abs_tol=1e-12)
    assert math.isclose(top_share(weights), 1.0 / 8.0, abs_tol=1e-12)


def test_concentrated_distribution_reduces_effective_count():
    uniform = {"a": 0.5, "b": 0.5}
    concentrated = {"a": 0.99, "b": 0.01}
    assert normalised_entropy(concentrated) < normalised_entropy(uniform)
    assert effective_count(concentrated) < effective_count(uniform)
    assert top_share(concentrated) > top_share(uniform)


def test_threshold_count_uses_normalised_shares():
    weights = {"a": 900.0, "b": 90.0, "c": 10.0}
    assert count_below(weights, 0.02) == 1
    assert count_below(weights, 0.10) == 2


def test_stats_are_scale_invariant():
    a = stats_for("x", {"a": 1.0, "b": 2.0, "c": 3.0})
    b = stats_for("x", {"a": 100.0, "b": 200.0, "c": 300.0})
    assert math.isclose(a.normalised_entropy, b.normalised_entropy)
    assert math.isclose(a.effective_count, b.effective_count)
    assert math.isclose(a.top_share, b.top_share)

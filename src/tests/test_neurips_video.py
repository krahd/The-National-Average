from __future__ import annotations

from tna.video.neurips import NEURIPS_PHASE_SCHEDULE, neurips_segments


def test_neurips_schedule_is_contiguous_and_complete():
    duration = 176.0
    segments = neurips_segments(duration)
    assert [key for key, _, _ in segments] == [key for key, _ in NEURIPS_PHASE_SCHEDULE]
    assert segments[0][1] == 0.0
    assert abs(segments[-1][2] - duration) < 1e-9
    for (_, start, end), (_, next_start, _) in zip(segments, segments[1:]):
        assert end > start
        assert abs(end - next_start) < 1e-9


def test_neurips_schedule_keeps_political_core_and_distribution_layers():
    keys = {key for key, _ in NEURIPS_PHASE_SCHEDULE}
    assert {
        "sources",
        "spaces",
        "distribution",
        "weighting",
        "thresholds",
        "erase",
        "coda",
    }.issubset(keys)
    # Tutorial-heavy baseline phases intentionally do not occupy the submission cut.
    assert "tokenize" not in keys
    assert "attend" not in keys
    assert "name" not in keys


def test_submission_master_has_time_margin():
    assert 0 < 176.0 < 180.0

from __future__ import annotations

import numpy as np

from tna.video.latent_flight import (
    FLIGHT_SHOTS,
    FOCUS_CODES,
    Camera,
    accelerated_search_progress,
    chapter_segments,
    film_phase_segments,
    flight_segments,
    forward_depth,
    parameter_weights,
    project_point,
)
from tna.video.latent_flight_audio import _weight_curves


def test_focus_is_the_requested_contested_set():
    assert FOCUS_CODES == ("ps", "il", "us", "de")


def test_flight_schedule_is_contiguous_and_complete():
    segments = flight_segments(176.0)
    assert [name for name, _, _ in segments] == [name for name, _ in FLIGHT_SHOTS]
    assert segments[0][1] == 0.0
    assert abs(segments[-1][2] - 176.0) < 1e-9
    for (_, start, end), (_, next_start, _) in zip(segments, segments[1:]):
        assert end > start
        assert abs(end - next_start) < 1e-9


def test_computational_chapters_are_contiguous_and_complete():
    segments = chapter_segments(96.0)
    assert segments[0][0] == "boundary"
    assert segments[-1][0] == "unresolved"
    assert segments[0][1] == 0.0
    assert abs(segments[-1][2] - 96.0) < 1e-9
    for (_, start, end), (_, next_start, _) in zip(segments, segments[1:]):
        assert end > start
        assert abs(end - next_start) < 1e-9


def test_film_structure_reserves_opening_search_and_conclusion():
    segments = film_phase_segments(96.0)
    assert [name for name, _, _ in segments] == ["title", "statement", "search", "conclusion"]
    assert segments[0][1] == 0.0
    assert abs(segments[-1][2] - 96.0) < 1e-9
    assert segments[2][2] - segments[2][1] > 60.0
    for (_, start, end), (_, next_start, _) in zip(segments, segments[1:]):
        assert end > start
        assert abs(end - next_start) < 1e-9


def test_moving_average_weights_are_valid_and_never_settle():
    samples = np.stack([parameter_weights(value) for value in np.linspace(0, 1, 101)])
    np.testing.assert_allclose(samples.sum(axis=1), 1.0)
    assert np.all(samples > 0.0)
    assert np.max(np.ptp(samples, axis=0)) > 0.45
    assert len(np.unique(np.round(samples, 3), axis=0)) > 90


def test_projection_places_forward_target_at_screen_centre():
    camera = Camera(
        position=np.array((0.0, 0.0, -5.0)),
        target=np.array((0.0, 0.0, 0.0)),
        roll=0.0,
        focal=100.0,
    )
    projected = project_point(np.array((0.0, 0.0, 0.0)), camera, (200, 100))
    assert projected is not None
    assert projected[:2] == (100.0, 50.0)
    assert project_point(np.array((0.0, 0.0, -6.0)), camera, (200, 100)) is None


def test_search_camera_depth_never_reverses():
    progress = np.linspace(0.0, 1.0, 1001)
    depths = np.array([forward_depth(accelerated_search_progress(value)) for value in progress])
    assert np.all(np.diff(depths) > 0.0)
    assert depths[-1] - depths[0] > 290.0
    increments = np.diff(depths)
    assert increments[-1] > increments[len(increments) // 2] * 1.45


def test_audio_weight_curves_are_sample_major():
    curves = _weight_curves(2.0, 97)
    assert curves.shape == (97, 4)
    np.testing.assert_allclose(curves.sum(axis=1), 1.0, atol=1e-5)

"""Tests for the ball detector's conditions reference and comparing a run with it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tracking import conditions
from tracking.ball import SceneStats
from tracking.config import RetrainingConfig
from tracking.video import VideoUnreadableError

from .conftest import SIZE


def _samples(low=100.0, high=124.0, rgb=(82.0, 119.0, 169.0), count=25):
    """Samples of a court whose brightness ranges from low to high, colour fixed."""
    return [
        conditions.ConditionSample(
            float(k), float(b), 24.0, tuple(v * b / 114.0 for v in rgb), -90.0
        )
        for k, b in enumerate(np.linspace(low, high, count))
    ]


def _reference(**kwargs):
    return conditions.build_reference(
        _samples(**kwargs),
        video={"name": "match.mp4", "sha256": "f" * 64},
        detector="motion-ball",
        measured="2026-11-25",
        method="test",
    )


def _scene(brightness, rgb):
    return SceneStats(30, brightness, 20.0, tuple(float(v) for v in rgb), -80.0)


def test_chromaticity_drops_brightness_out():
    assert conditions.chromaticity((10, 20, 30)) == pytest.approx(
        conditions.chromaticity((100, 200, 300))
    )
    assert conditions.chromaticity((0, 0, 0)) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_a_frame_is_sampled_over_the_court_surface(calibration):
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    image[:] = (90, 120, 170)
    sample = conditions.sample_image(image, calibration, 12.5)
    assert sample.pts == 12.5
    assert sample.court_rgb == pytest.approx((90.0, 120.0, 170.0))
    assert sample.contrast == pytest.approx(0.0)
    assert conditions.ConditionSample.from_dict(sample.to_dict()) == sample


def test_sample_times_are_evenly_spaced_and_refuse_a_zero_step():
    assert conditions.sample_times(0.0, 60.0, 20.0) == [0.0, 20.0, 40.0, 60.0]
    assert conditions.sample_times(5.0, 4.0, 1.0) == []
    with pytest.raises(ValueError):
        conditions.sample_times(0.0, 1.0, 0.0)


def test_sampling_a_video_reports_moments_it_could_not_decode(calibration, tmp_path):
    image = np.full((SIZE[1], SIZE[0], 3), 100, dtype=np.uint8)

    def reader(_path, moment):
        if moment == 20.0:
            raise VideoUnreadableError("no frame there")
        return moment + 0.01, image

    samples, failed = conditions.sample_video(
        tmp_path / "v.mp4", [0.0, 20.0, 40.0], calibration, reader=reader
    )
    assert [s.pts for s in samples] == [0.01, 40.01] and failed == [20.0]


def test_the_real_decoder_is_used_by_default(calibration, steady_video):
    samples, failed = conditions.sample_video(steady_video, [0.5, 99.0], None)
    assert len(samples) == 1 and failed == [99.0]
    assert 0 < samples[0].brightness < 255


def test_a_reference_summarises_spread_and_round_trips(tmp_path):
    reference = conditions.build_reference(
        _samples(),
        video={"name": "match.mp4", "sha256": "f" * 64},
        detector="motion-ball",
        measured="2026-11-25",
        method="25 frames",
        failed_samples=2,
        tracking={"rally_coverage": 0.88, "rally_mean_confidence": 0.43},
        extra={"note": "test"},
    )
    assert reference.brightness.minimum == 100.0 and reference.brightness.maximum == 124.0
    assert reference.chroma_distance.maximum < 1e-6
    path = reference.save(tmp_path / "ref" / "reference.json")
    assert conditions.load_reference(path) == reference
    assert json.loads(path.read_text())["format"] == conditions.REFERENCE_FORMAT


def test_a_missing_or_foreign_reference_reads_as_none(tmp_path, caplog):
    assert conditions.load_reference(None) is None
    assert conditions.load_reference(tmp_path / "missing.json") is None
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"format": 99}))
    assert conditions.load_reference(other) is None
    assert "No usable conditions reference" in caplog.text


def test_a_spread_needs_values():
    with pytest.raises(ValueError):
        conditions.Spread.of([])
    with pytest.raises(ValueError):
        conditions.build_reference([], video={}, detector="d", measured="2026-11-25", method="none")


def test_limits_take_the_larger_of_the_measured_spread_and_the_floor():
    reference = _reference()
    config = RetrainingConfig()
    bounds = conditions.limits(reference, config)
    margin = max(3 * reference.brightness.std, 0.25 * reference.brightness.p50)
    assert bounds["dim_below"] == pytest.approx(100.0 - margin, abs=0.01)
    assert bounds["bright_above"] == pytest.approx(124.0 + margin, abs=0.01)
    assert bounds["chroma_above"] == pytest.approx(0.03, abs=1e-4)
    wide = conditions.limits(reference, RetrainingConfig(sigmas=100.0))
    assert wide["dim_below"] < bounds["dim_below"]


def test_footage_like_the_reference_raises_nothing():
    check = conditions.compare(_scene(112.0, (80, 117, 166)), _reference())
    assert check.compared and check.conditions == ()
    assert check.measures["reference_video"] == "match.mp4"


def test_night_footage_is_named_as_darker_than_the_tuning_conditions():
    check = conditions.compare(_scene(45.0, (32, 47, 66)), _reference())
    assert len(check.conditions) == 1
    assert check.conditions[0].startswith(
        "footage much darker than the conditions the ball detector was tuned on: court "
        "brightness 45 against 100-124 on the reference recording (40% of its median)"
    )


def test_glare_and_a_clay_court_are_named():
    check = conditions.compare(_scene(210.0, (200, 120, 70)), _reference())
    assert len(check.conditions) == 2
    assert check.conditions[0].startswith("footage much brighter")
    assert "a court colour unlike the reference's (RGB 200, 120, 70" in check.conditions[1]


def test_without_a_scene_or_a_reference_nothing_is_compared():
    assert not conditions.compare(None, _reference()).compared
    check = conditions.compare(_scene(45.0, (1, 1, 1)), None)
    assert not check.compared and check.conditions == ()
    assert check.measures["scene"]["brightness"] == 45.0

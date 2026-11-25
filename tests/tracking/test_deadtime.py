"""Tests for the dead-time pre-pass: hearing where the play is."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from swingvision_import import footage
from tracking import deadtime
from tracking.config import DeadTimeConfig

RATE = deadtime.SAMPLE_RATE


def _clicks(seconds, at, rng_seed=0):
    """Quiet noise with a sharp click at each moment in `at`."""
    rng = np.random.default_rng(rng_seed)
    samples = rng.normal(0, 30, int(seconds * RATE))
    for moment in at:
        start = int(moment * RATE)
        samples[start : start + 40] += 8000 * np.sin(np.arange(40))
    return np.clip(samples, -32768, 32767).astype(np.int16)


def test_the_curve_is_the_same_however_the_audio_is_split():
    audio = _clicks(6.0, [1.0, 1.5, 4.2])
    whole = deadtime.OnsetCurve()
    whole.feed(audio)
    split = deadtime.OnsetCurve()
    for block in np.array_split(audio, 7):
        split.feed(block)
    assert np.allclose(whole.result(), split.result())
    assert len(whole.result()) == 600


def test_the_curve_scaled_is_footages_own_onset_envelope():
    audio = _clicks(6.0, [1.0, 1.5, 4.2])
    curve = deadtime.OnsetCurve()
    curve.feed(audio)
    raw = curve.result()
    scaled = np.minimum(raw / np.percentile(raw, 99), 3.0)
    assert np.allclose(scaled, footage.onset_envelope(audio, RATE), atol=1e-4)


def test_too_little_audio_makes_no_curve():
    curve = deadtime.OnsetCurve()
    curve.feed(np.zeros(10, dtype=np.int16))
    assert len(curve.result()) == 0


def test_strikes_are_the_sharp_peaks():
    curve = deadtime.OnsetCurve()
    curve.feed(_clicks(6.0, [1.0, 1.5, 4.2]))
    # Strength is relative to the recording's 99th percentile, so the loudest
    # 1% of plain noise reaches 1.0; a click is far above it.
    loose = deadtime.strikes(curve.result(), DeadTimeConfig())
    times = deadtime.strikes(curve.result(), DeadTimeConfig(min_strength=2.0))
    assert len(loose) > 3 and np.allclose(times, [1.0, 1.5, 4.2], atol=0.02)
    assert len(deadtime.strikes(np.zeros(0), DeadTimeConfig())) == 0
    assert len(deadtime.strikes(np.zeros(100), DeadTimeConfig())) == 0


def _curve(strike_times, length):
    """A synthetic curve with strikes exactly where asked."""
    curve = np.full(int(length * 100), 0.1, dtype=np.float32)
    curve[::50] = 1.0
    for moment in strike_times:
        curve[int(round(moment * 100))] = 5.0
    return curve


CONFIG = DeadTimeConfig(enabled=True, min_strength=2.0)


def test_quiet_stretches_between_rallies_are_planned_dead():
    plan = deadtime.plan_dead_time(_curve([30, 32, 34, 90, 93], 150.0), 0.0, 150.0, CONFIG)
    assert plan.active == ((25.0, 39.0), (85.0, 98.0))
    assert plan.dead == ((0.0, 25.0), (39.0, 85.0), (98.0, 150.0))
    assert plan.strikes == 5
    assert plan.skipped_seconds == pytest.approx(123.0)
    assert plan.skipped_share == pytest.approx(123.0 / 150.0)
    assert not plan.is_active(10.0) and plan.is_active(30.0)
    assert deadtime.outside_active(plan, [10.0, 30.0, 120.0]) == [10.0, 120.0]
    data = plan.to_dict()
    assert data["dead"][0] == [0.0, 25.0] and data["settings"]["min_strength"] == 2.0
    assert plan.note().startswith(
        "Dead-time pre-pass: 123s of the 150s tracked (82%) in 3 stretch(es)"
    )


def test_a_lone_sound_is_not_play_and_short_quiet_gaps_are_tracked():
    plan = deadtime.plan_dead_time(_curve([5, 60, 62, 72, 74], 100.0), 0.0, 100.0, CONFIG)
    # 5s is one strike (not a burst); 62->72 is too short a gap to skip.
    assert plan.active == ((55.0, 79.0),)
    assert plan.dead == ((0.0, 55.0), (79.0, 100.0))


def test_play_near_the_edges_of_the_run_is_kept_whole():
    plan = deadtime.plan_dead_time(_curve([2, 4, 60, 62, 95, 97], 100.0), 0.0, 100.0, CONFIG)
    assert plan.active == ((0.0, 9.0), (55.0, 67.0), (90.0, 100.0))
    assert plan.dead == ((9.0, 55.0), (67.0, 90.0))


def test_play_wholly_outside_the_run_is_ignored_and_a_short_run_is_tracked():
    plan = deadtime.plan_dead_time(_curve([2, 4, 60, 62], 100.0), 40.0, 100.0, CONFIG)
    assert plan.active == ((55.0, 67.0),) and plan.dead == ((40.0, 55.0), (67.0, 100.0))
    short = deadtime.plan_dead_time(_curve([2, 4], 100.0), 40.0, 45.0, CONFIG)
    assert short.active == ((40.0, 45.0),) and short.dead == ()


def test_no_audio_or_no_play_at_all_means_no_plan(caplog):
    assert deadtime.plan_dead_time(None, 0.0, 10.0) is None
    assert deadtime.plan_dead_time(np.zeros(0), 0.0, 10.0) is None
    assert deadtime.plan_dead_time(_curve([], 100.0), 0.0, 100.0, CONFIG) is None
    assert "heard no play at all" in caplog.text


def test_a_recordings_audio_is_read_in_blocks(tmp_path):
    path = tmp_path / "clicks.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(_clicks(4.0, [1.0, 2.5]).tobytes())
    curve = deadtime.read_onset_curve(path, block_seconds=0.7)
    assert len(curve) == 400
    found = deadtime.strikes(curve, DeadTimeConfig(min_strength=2.0))
    assert np.allclose(found, [1.0, 2.5], atol=0.02)


def test_no_audio_track_or_no_ffmpeg_reads_as_none(steady_video, tmp_path, caplog):
    assert deadtime.read_onset_curve(steady_video) is None
    assert "No usable audio" in caplog.text
    assert deadtime.read_onset_curve(steady_video, ffmpeg=str(tmp_path / "no-ffmpeg")) is None
    assert "Could not start ffmpeg" in caplog.text

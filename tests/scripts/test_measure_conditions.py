"""Tests for the measure_conditions CLI. No video is decoded here."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import measure_conditions as mc
from tests.tracking.conftest import exact_calibration, make_camera
from tracking import conditions


@pytest.fixture
def calibration():
    """An exact calibration of the standard test camera."""
    return exact_calibration(make_camera())


def _fake_sampler(calls):
    def sample_video(video, times, calibration):
        calls.append((video, list(times), calibration))
        samples = [
            conditions.ConditionSample(t, 110.0 + t / 100, 22.0, (80.0, 118.0, 170.0), -90.0)
            for t in times
            if t != 40.0
        ]
        return samples, [40.0] if 40.0 in times else []

    return sample_video


def test_samples_are_appended_and_built_into_a_reference(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(mc.conditions, "sample_video", _fake_sampler(calls))
    samples = tmp_path / "samples.jsonl"
    video = tmp_path / "match.mp4"
    assert mc.main(["sample", str(video), str(samples), "--end", "60", "--every", "20"]) == 0
    assert mc.main(["sample", str(video), str(samples), "--start", "60", "--end", "100"]) == 0
    assert calls[0][1] == [0.0, 20.0, 40.0, 60.0] and calls[0][2] is None
    assert "3 sample(s) written" in capsys.readouterr().out
    read, failed = mc.read_samples(samples)
    assert [s.pts for s in read] == [0.0, 20.0, 60.0, 90.0] and failed == 1

    monkeypatch.setattr(
        mc.video_stage,
        "probe",
        lambda path: SimpleNamespace(duration=100.0, width=1920, height=1080),
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"rally_coverage": 0.88, "rally_mean_confidence": 0.43}))
    out = tmp_path / "reference.json"
    code = mc.main(
        [
            "build",
            str(samples),
            str(video),
            "--detector",
            "motion-ball",
            "--sha256",
            "b" * 64,
            "--tracking-baseline",
            str(baseline),
            "--measured",
            "2026-11-25",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    reference = conditions.load_reference(out)
    assert reference.samples == 4 and reference.failed_samples == 1
    assert reference.video == {
        "name": "match.mp4",
        "sha256": "b" * 64,
        "duration": 100.0,
        "width": 1920,
        "height": 1080,
    }
    assert reference.tracking["rally_coverage"] == 0.88
    assert reference.method.startswith("4 single frames, one seek each, from 0s to 90s")


def test_a_calibration_file_or_automatic_calibration_is_used(tmp_path, monkeypatch, calibration):
    calls = []
    monkeypatch.setattr(mc.conditions, "sample_video", _fake_sampler(calls))
    saved = tmp_path / "calibration.json"
    saved.write_text(json.dumps(calibration.to_dict()))
    samples = tmp_path / "s.jsonl"
    args = ["sample", "v.mp4", str(samples), "--end", "0"]
    assert mc.main([*args, "--calibration", str(saved)]) == 0
    assert calls[-1][2].image_size == calibration.image_size
    results = iter(
        [SimpleNamespace(ok=True, calibration=calibration), SimpleNamespace(ok=False, reason="no")]
    )
    monkeypatch.setattr(mc, "calibrate_video", lambda video, at: next(results))
    assert mc.main([*args, "--calibrate-at", "5"]) == 0 and calls[-1][2] is calibration
    assert mc.main([*args, "--calibrate-at", "5"]) == 2


def test_an_empty_samples_file_cannot_build_a_reference(tmp_path, monkeypatch, capsys):
    samples = tmp_path / "s.jsonl"
    samples.write_text("\n")
    monkeypatch.setattr(
        mc.video_stage, "probe", lambda path: SimpleNamespace(duration=1.0, width=1, height=1)
    )
    code = mc.main(["build", str(samples), "v.mp4", "--detector", "d", "--sha256", "c" * 64])
    assert code == 2 and "at least one sample" in capsys.readouterr().err

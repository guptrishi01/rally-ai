"""Tests for the tracker's configuration objects."""

import dataclasses

import pytest

import tracking
from tracking.config import CalibrationConfig, CourtConfig, TrackerConfig, VideoConfig


def test_tracker_config_nests_one_config_per_stage():
    config = TrackerConfig()
    assert isinstance(config.video, VideoConfig)
    assert isinstance(config.court, CourtConfig)
    assert isinstance(config.calibration, CalibrationConfig)


def test_configs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TrackerConfig().video = VideoConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        VideoConfig().reject_vfr = True


def test_a_stage_can_be_overridden_without_touching_the_others():
    config = TrackerConfig(video=VideoConfig(reject_vfr=True))
    assert config.video.reject_vfr
    assert config.court == CourtConfig()


def test_irregular_timing_is_accepted_by_default():
    assert VideoConfig().reject_vfr is False


def test_the_package_exports_its_configs_and_provenance():
    assert tracking.TrackerConfig is TrackerConfig
    assert tracking.Provenance.MEASURED.value == "measured"


def test_the_phase_three_stages_have_their_own_configs():
    config = TrackerConfig()
    assert isinstance(config.models, tracking.ModelConfig)
    assert isinstance(config.players, tracking.PlayerConfig)
    assert isinstance(config.ball, tracking.BallConfig)
    assert config.models.weights_dir is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.ball.max_gap_seconds = 1.0


def test_the_phase_four_stages_have_their_own_configs():
    config = TrackerConfig()
    assert isinstance(config.events, tracking.EventConfig)
    assert isinstance(config.strokes, tracking.StrokeConfig)
    assert config.events.court == "singles"
    # Right-handed until intake asks: an assumption, stated in the config itself.
    assert dict(config.strokes.handedness) == {"A": "right", "B": "right"}
    assert config.models.pose_model == "yolo11n-pose"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.events.call_sigmas = 2.0


def test_the_phase_five_stages_have_their_own_configs():
    config = TrackerConfig()
    assert isinstance(config.segment, tracking.SegmentConfig)
    assert isinstance(config.assemble, tracking.AssembleConfig)
    # Thresholds measured on the real sheet, stated in the config itself.
    assert config.segment.rally_gap_seconds == 5.0
    assert config.segment.match_start is None
    assert config.assemble.unknown_stroke == "Unknown"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.segment.serve_max_x_m = 3.0

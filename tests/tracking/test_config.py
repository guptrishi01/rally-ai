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

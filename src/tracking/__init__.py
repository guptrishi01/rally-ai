"""The independent tracker: per-shot tracking data from a fixed-camera video.

Built in phases (see docs). So far: reading a recording with real frame
timestamps (`video`), finding the court and calibrating the image<->court
homography (`court`, `calibrate`), provenance tags (`provenance`),
following the two players (`players`), finding and linking the ball
(`ball`), lazy, version-pinned model loading (`models`), reading hits,
bounces, net contacts and shot results off the ball's track (`events`),
naming each hit's stroke (`strokes`), splitting the hits into serve
attempts numbered as SwingVision numbers them (`segment`), and emitting
them as the `RawShotRow`s reconstruction consumes, with a provenance
trace per row (`assemble`), and walking a whole recording in checkpointed
chunks into a staged match (`stitch`, `pipeline`). Phase 8 adds
measuring it: scoring tracker output against a reference export or the
reviewer's confirmations (`evaluate`), the conditions the ball detector
was tuned on (`conditions`), an audio pre-pass that skips stretches with
no play (`deadtime`), and exporting labelled frames for training the
learned ball model (`dataset`).

Importing this package loads no machine-learning library; see `models`.
"""

from .config import (
    AssembleConfig,
    BallConfig,
    CalibrationConfig,
    CourtConfig,
    DeadTimeConfig,
    EventConfig,
    ModelConfig,
    PipelineConfig,
    PlayerConfig,
    RetrainingConfig,
    SegmentConfig,
    StrokeConfig,
    TrackerConfig,
    VideoConfig,
)
from .provenance import Provenance

__all__ = [
    "AssembleConfig",
    "BallConfig",
    "CalibrationConfig",
    "CourtConfig",
    "DeadTimeConfig",
    "EventConfig",
    "ModelConfig",
    "PipelineConfig",
    "PlayerConfig",
    "Provenance",
    "RetrainingConfig",
    "SegmentConfig",
    "StrokeConfig",
    "TrackerConfig",
    "VideoConfig",
]

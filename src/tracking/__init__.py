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
trace per row (`assemble`).

Importing this package loads no machine-learning library; see `models`.
"""

from .config import (
    AssembleConfig,
    BallConfig,
    CalibrationConfig,
    CourtConfig,
    EventConfig,
    ModelConfig,
    PlayerConfig,
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
    "EventConfig",
    "ModelConfig",
    "PlayerConfig",
    "Provenance",
    "SegmentConfig",
    "StrokeConfig",
    "TrackerConfig",
    "VideoConfig",
]

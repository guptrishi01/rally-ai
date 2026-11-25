"""The independent tracker: per-shot tracking data from a fixed-camera video.

Built in phases (see docs). So far: reading a recording with real frame
timestamps (`video`), finding the court and calibrating the image<->court
homography (`court`, `calibrate`), provenance tags (`provenance`),
following the two players (`players`), finding and linking the ball
(`ball`), lazy, version-pinned model loading (`models`), reading hits,
bounces, net contacts and shot results off the ball's track (`events`),
and naming each hit's stroke (`strokes`).

Importing this package loads no machine-learning library; see `models`.
"""

from .config import (
    BallConfig,
    CalibrationConfig,
    CourtConfig,
    EventConfig,
    ModelConfig,
    PlayerConfig,
    StrokeConfig,
    TrackerConfig,
    VideoConfig,
)
from .provenance import Provenance

__all__ = [
    "BallConfig",
    "CalibrationConfig",
    "CourtConfig",
    "EventConfig",
    "ModelConfig",
    "PlayerConfig",
    "Provenance",
    "StrokeConfig",
    "TrackerConfig",
    "VideoConfig",
]

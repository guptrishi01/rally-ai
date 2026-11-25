"""The independent tracker: per-shot tracking data from a fixed-camera video.

Built in phases (see docs). So far: reading a recording with real frame
timestamps (`video`), finding the court and calibrating the image<->court
homography (`court`, `calibrate`), provenance tags (`provenance`),
following the two players (`players`), finding and linking the ball
(`ball`), and lazy, version-pinned model loading (`models`).

Importing this package loads no machine-learning library; see `models`.
"""

from .config import (
    BallConfig,
    CalibrationConfig,
    CourtConfig,
    ModelConfig,
    PlayerConfig,
    TrackerConfig,
    VideoConfig,
)
from .provenance import Provenance

__all__ = [
    "BallConfig",
    "CalibrationConfig",
    "CourtConfig",
    "ModelConfig",
    "PlayerConfig",
    "Provenance",
    "TrackerConfig",
    "VideoConfig",
]

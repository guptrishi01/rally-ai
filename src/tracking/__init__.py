"""The independent tracker: per-shot tracking data from a fixed-camera video.

Built in phases (see docs). So far: reading a recording with real frame
timestamps (`video`), finding the court and calibrating the image<->court
homography (`court`, `calibrate`), and provenance tags (`provenance`).
"""

from .config import CalibrationConfig, CourtConfig, TrackerConfig, VideoConfig
from .provenance import Provenance

__all__ = ["CalibrationConfig", "CourtConfig", "Provenance", "TrackerConfig", "VideoConfig"]

"""Configuration for the independent tracker.

One frozen `TrackerConfig` holds a nested, frozen config per pipeline stage
(`video`, `court`, `calibration` so far). Later phases add their own stage
config as a new field with a default, so nothing already built has to change
when they do.

Every pixel threshold is expressed as a **fraction of the image** (its width
or its diagonal), never as a raw pixel count, so the same config works on a
1920x1080 recording and on a 640x360 test fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VideoConfig:
    """Settings for decoding a recording and indexing its timestamps.

    Attributes:
        reject_vfr: When True, `video.load_video` refuses a recording whose
            frame timestamps are irregular (`VariableFrameRateError`).
            Off by default: the frame iterator always carries each frame's
            real presentation timestamp, so irregular timing is handled, not
            assumed away - and the one real recording this was built
            against has 610 dropped-frame gaps in 342,928 frames.
        vfr_jitter: How far a frame-to-frame gap may sit from the median
            gap, as a fraction of it, before that gap counts as irregular.
            0.15 absorbs time-base rounding (a 1/600 time base shows 60fps
            as gaps of 9, 10 and 11 ticks) without hiding a dropped frame.
        checksum_chunk_bytes: Read size used when hashing the file.
        decode_timeout_seconds: How long to wait for ffmpeg to exit after
            its output has been consumed, before it is killed.
    """

    reject_vfr: bool = False
    vfr_jitter: float = 0.15
    checksum_chunk_bytes: int = 4 * 1024 * 1024
    decode_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class CourtConfig:
    """Settings for finding the court's painted lines in a frame.

    Attributes:
        line_reach: Half-width, as a fraction of image width, of the
            "is this pixel brighter than both its neighbours" test that
            picks out painted lines. Must exceed the thickest line's
            half-thickness. A third of it is tried as well, for thin far
            lines crowded by another bright band.
        line_contrast: How much brighter (of 255) a line pixel must be than
            the pixels `line_reach` away on both sides.
        line_min_brightness: Luminance floor (of 255) for a line pixel.
        line_max_saturation: Largest max-minus-min colour channel spread
            (of 255) a line pixel may have; court lines are near-white.
        hough_threshold: Votes a Hough segment needs, as a fraction of image
            width.
        hough_min_length: Shortest segment kept, as a fraction of image
            width.
        hough_max_gap: Largest gap bridged inside one segment, as a fraction
            of image width.
        merge_angle_deg: Segments closer in angle than this may be the same
            line.
        merge_distance: ...and closer than this, as a fraction of the image
            diagonal, are merged into one line.
        transverse_max_angle_deg: A line within this many degrees of
            horizontal is a baseline/service-line candidate. Sidelines seen
            from a low camera are shallow (about 20 degrees on the real
            recording), so this cannot be generous.
        vanishing_tolerance: How close, as a fraction of the image diagonal,
            a line must pass to the sidelines' vanishing point to count as
            a sideline/centre-line candidate.
        max_transverse: Most baseline/service-line candidates searched.
        max_longitudinal: Most sideline/centre-line candidates searched.
        min_focal: Shortest focal length, as a fraction of image width, a
            candidate fit may imply (see `court.plausible_camera`).
        max_focal: Longest focal length it may imply.
        max_axis_skew: Largest cosine of the angle between the court's two
            axes, as the implied camera sees them, before a candidate fit
            is discarded. 0 is a perfect right angle.
        max_axis_stretch: Largest ratio between the implied lengths of the
            court's two axes before a candidate fit is discarded.
        hit_tolerance: How close, as a fraction of the image diagonal, a
            projected template line must be to a line pixel to count as
            supported during the search.
        miss_penalty: Weight of an unsupported stretch of projected template
            line relative to a supported one. This is what stops a service
            line being accepted as a baseline: the lines that assignment
            predicts elsewhere are not there.
        line_weight_cap: In-frame length, as a fraction of image width, at
            which a template line counts fully toward a candidate's score.
            Longer lines count no more, so the score counts lines found
            rather than pixels.
        samples_per_line: Points sampled along each template line when
            scoring a candidate fit.
        min_support: Smallest fraction of the in-frame template that must
            lie on line pixels for a detection to be reported at all.
        max_runner_up: Largest score a *different* fit may reach, as a
            fraction of the best fit's, before the detection is called
            ambiguous and abandoned.
        distinct_fit: How far apart, as a fraction of the image diagonal,
            two fits must put some in-frame court keypoint for them to
            count as different.
        refit_band: Half-width, as a fraction of the image diagonal, of the
            strip of line pixels gathered around each projected template
            line when it is refitted.
        refit_iterations: Rounds of refit-lines-then-refit-homography.
        min_line_length: Shortest stretch of supported line, as a fraction
            of image width, accepted as a found template line.
        min_line_coverage: Smallest fraction of a template line's in-frame
            length that must be supported for it to count as found.
        paint_width_m: Width of a painted line, in metres, used to predict
            how thick it should look.
        thickness_ratio: A found line may be this many times thicker than
            predicted...
        thickness_slack_px: ...plus this many pixels, before it is rejected
            as not a painted ground line. This is what refuses the net tape
            when it sits where a far line should be.
        net_zone_pad: How far, as a fraction of the image diagonal, the
            predicted net-tape zone is grown before its pixels are barred
            from being read as painted line (see `court.net_tape_zone`).
        keypoint_margin: How far outside the frame, as a fraction of its
            width/height, a line intersection may fall and still be used.
        templates: Court templates tried, in order.
    """

    line_reach: float = 0.006
    line_contrast: float = 22.0
    line_min_brightness: float = 110.0
    line_max_saturation: float = 90.0
    hough_threshold: float = 0.03
    hough_min_length: float = 0.045
    hough_max_gap: float = 0.012
    merge_angle_deg: float = 2.0
    merge_distance: float = 0.004
    transverse_max_angle_deg: float = 12.0
    vanishing_tolerance: float = 0.02
    max_transverse: int = 9
    max_longitudinal: int = 9
    min_focal: float = 0.3
    max_focal: float = 6.0
    max_axis_skew: float = 0.2
    max_axis_stretch: float = 1.33
    hit_tolerance: float = 0.003
    miss_penalty: float = 0.6
    line_weight_cap: float = 0.15
    samples_per_line: int = 16
    min_support: float = 0.5
    max_runner_up: float = 0.95
    distinct_fit: float = 0.03
    refit_band: float = 0.004
    refit_iterations: int = 3
    min_line_length: float = 0.03
    min_line_coverage: float = 0.35
    paint_width_m: float = 0.10
    thickness_ratio: float = 3.0
    thickness_slack_px: float = 2.5
    net_zone_pad: float = 0.001
    keypoint_margin: float = 0.35
    templates: tuple[str, ...] = ("doubles", "singles")


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings for fitting and judging the image<->court homography.

    Attributes:
        ransac_threshold: RANSAC inlier distance, as a fraction of the image
            diagonal.
        min_points: Fewest correspondences a fit may be made from.
        min_inlier_fraction: Smallest share of the supplied correspondences
            that must agree with the fit.
        max_residual_px: Largest median reprojection residual accepted, as a
            fraction of the image diagonal.
        max_residual_m: Largest median reprojection residual accepted on the
            court plane, in metres.
        min_spread: Smallest ratio of the correspondences' minor to major
            extent (on either side) before they are called collinear.
        max_far_depth_m_per_px: Coarsest depth resolution at the far
            baseline, in metres of court per image pixel, still accepted.
            Past this the far end is a smear and the fit abstains.
        warn_far_depth_m_per_px: Depth resolution at the far baseline above
            which the fit is kept but carries a "raise the camera" warning.
        min_near_far_ratio: The near baseline must look at least this many
            times as long as the far one; otherwise the two ends were very
            likely labelled the wrong way round.
        frames_for_median: Frames blended (by per-pixel median, which
            removes the players) when calibrating from a video.
        median_span_seconds: Stretch of video those frames are spread over.
    """

    ransac_threshold: float = 0.004
    min_points: int = 4
    min_inlier_fraction: float = 0.6
    max_residual_px: float = 0.004
    max_residual_m: float = 0.25
    min_spread: float = 0.02
    max_far_depth_m_per_px: float = 1.0
    warn_far_depth_m_per_px: float = 0.15
    min_near_far_ratio: float = 1.0
    frames_for_median: int = 9
    median_span_seconds: float = 8.0


@dataclass(frozen=True)
class TrackerConfig:
    """Everything the tracker is configured by, one field per stage.

    Attributes:
        video: Decode and timestamp-index settings.
        court: Court-line detection settings.
        calibration: Homography fit and validation settings.
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

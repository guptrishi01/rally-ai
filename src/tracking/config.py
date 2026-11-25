"""Configuration for the independent tracker.

One frozen `TrackerConfig` holds a nested, frozen config per pipeline stage
(`video`, `court`, `calibration`, `models`, `players`, `ball` so far). Later
phases add their own stage config as a new field with a default, so nothing
already built has to change when they do.

Every pixel threshold is expressed as a **fraction of the image** (its width
or its diagonal), never as a raw pixel count, so the same config works on a
1920x1080 recording and on a 640x360 test fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
class ModelConfig:
    """Which learned models are used, and where their weights are kept.

    Attributes:
        person_model: Name of the pinned person-detector entry in
            `tracking.models.REGISTRY`.
        weights_dir: Folder weights are downloaded to and loaded from, or
            None for a per-user cache folder **outside the repository**
            (`tracking.models.default_weights_dir`). Weights are never
            committed.
        verify_checksum: Refuse weights whose SHA-256 differs from the
            pinned one (when the registry entry pins one).
    """

    person_model: str = "yolo11n"
    weights_dir: Path | None = None
    verify_checksum: bool = True


@dataclass(frozen=True)
class PlayerConfig:
    """Settings for finding, following and naming the two players.

    Attributes:
        stride: Run the person detector on every `stride`-th frame handed
            to the tracker; positions between are interpolated on request.
            12 on a 60fps recording is five detections a second. Measured
            on the real recording against detections every 6th frame,
            interpolating from every 12th is off by a median 3cm (near
            end) and 14cm (far end, where one pixel is 28cm) - so the
            detector's cost is halved for nothing that matters.
        detector_size: Longest image side, in pixels, the learned detector
            is run at. The far player is only tens of pixels tall at full
            size, so this cannot be small: at 640 the far player's
            confidence on the real recording was 0.58, at 960 0.75.
        min_confidence: Detections below this are discarded outright.
        high_confidence: Detections at or above this may start a track and
            are matched first; those between the two thresholds can only
            continue an existing track (the ByteTrack idea).
        match_iou: Smallest overlap, between a track's predicted box and a
            detection (both padded by `match_pad`), that links them.
        match_pad: Each box is grown by this fraction of its size before
            the overlap is measured, so a small, fast far player still
            overlaps their own last position.
        max_track_age_seconds: A track with no detection for this long is
            closed.
        min_track_hits: Detections a track needs before it may be chosen
            as a player.
        lateral_margin_m: How far outside the doubles sidelines a player's
            feet may be and still count as on this court. Neighbouring
            courts start roughly 3.7m out.
        behind_margin_m: How far behind a baseline a player's feet may be.
        net_band_m: A player whose feet are within this distance of the
            net keeps the end they were already at, so a net rush never
            reads as a change of ends.
        max_interpolation_seconds: Longest stretch between two detections
            of the same player that is bridged by interpolation; a longer
            one yields no position.
        foot_edge_margin: A box whose lower edge is within this fraction of
            the image height of the bottom of the frame has its feet cut
            off, and is flagged so.
        swap_settle_seconds: A player must have been at both ends, without
            a break, for this long before that arrangement is compared
            with the one before it. Players wander across the net during
            a changeover; only an arrangement that holds is judged.
        swap_absence_seconds: When no track was followed across, the
            previous arrangement must have gone unconfirmed for at least
            this long before a change of ends is concluded from
            appearance - players do not swap mid-rally.
        swap_margin: How much better (0..1 histogram similarity) the
            "they swapped" reading must match than the "they did not"
            reading before an end swap is recorded.
        appearance_bins: Hue and saturation bins of the appearance
            histogram.
        appearance_rate: How fast a player's stored appearance follows
            new sightings (0..1 per detection).
        motion_width: Width the classic motion fallback detector works at.
        motion_learning_rate: How fast its background model follows the
            scene; a still player fades into it at this rate.
        motion_threshold: Luminance change (of 255) that counts as
            foreground for it.
        motion_min_height_m: Shortest thing, in metres at its own
            distance, it reports as a person.
        motion_max_height_m: Tallest.
        motion_min_area: Smallest foreground blob it considers, as a
            fraction of the (reduced) image area, when uncalibrated.
    """

    stride: int = 12
    detector_size: int = 960
    min_confidence: float = 0.1
    high_confidence: float = 0.35
    match_iou: float = 0.15
    match_pad: float = 0.5
    max_track_age_seconds: float = 1.5
    min_track_hits: int = 3
    lateral_margin_m: float = 3.0
    behind_margin_m: float = 8.0
    net_band_m: float = 1.0
    max_interpolation_seconds: float = 0.6
    foot_edge_margin: float = 0.01
    swap_settle_seconds: float = 6.0
    swap_absence_seconds: float = 5.0
    swap_margin: float = 0.15
    appearance_bins: tuple[int, int] = (12, 4)
    appearance_rate: float = 0.1
    motion_width: int = 480
    motion_learning_rate: float = 0.02
    motion_threshold: float = 28.0
    motion_min_height_m: float = 0.9
    motion_max_height_m: float = 2.6
    motion_min_area: float = 0.0004


@dataclass(frozen=True)
class BallConfig:
    """Settings for finding ball candidates and linking them into a track.

    Lengths in pixels are fractions of the image width unless they say
    metres, in which case the calibration converts them.

    Attributes:
        change_threshold: How much (of 255) a pixel's "yellowness" must
            exceed both the frame before and the frame after to count as
            a moving ball-coloured thing.
        difference_seconds: How long before and after a frame the frames
            it is compared with are, when the detector is told the frame
            rate (4 frames at 60fps, 2 at 30fps). A far-court ball moves
            under a pixel a frame along the line of sight and must be
            given time to clear its own earlier position: on six rallies
            of the real 60fps recording, 4 frames instead of 2 raised the
            share of far-half frames with a ball position from 69% to 77%
            (near half: 87% to 89%; measured before `min_blob_ratio`
            existed, which took the neighbouring court's ball back out).
        difference_frames: The same spacing in frames, used only when the
            detector is not told the frame rate.
        ball_diameter_m: A tennis ball's diameter.
        size_slack: A candidate may be this many times longer than the
            ball should look (motion blur stretches it).
        min_blob_ratio: Smallest a moving-pixel blob is taken to be, in
            ball diameters. Differencing, blur and the lens smear the ball:
            over every segment that was the real ball on four rallies of
            the real recording the blob was a median 1.7 to 6 times the
            smallest the ball could look along its line of sight, while
            the neighbouring court's ball - further away than this court's
            space reaches along that line - came out at 0.76 to 0.94.
        resolved_ball_px: Width in pixels at which the ball is taken to be
            fully resolved, so that `min_blob_ratio` applies in full; at
            half this width or less only "no smaller than the ball" is
            asked. Real far-baseline balls on the recording are 2.5px
            wide and leave blobs of as little as 2px.
        size_tolerance: How far off (as a ratio) a candidate's size may be
            from anything a ball above this court could look like before
            its score reaches zero.
        max_blob: Largest candidate extent looked at, as a fraction of
            image width.
        min_blob_px: Smallest candidate area, in pixels.
        max_candidates: Candidates kept per frame, best first. More than
            one, always: the tracker, not the detector, decides.
        court_margin_m: How far behind each baseline the space a ball in
            play can be in reaches.
        side_margin_m: How far beside the doubles sidelines it reaches.
            Kept small: the neighbouring court starts about 3.7m out.
        flight_height_m: How far above the court it reaches. A lob goes
            well above 6m, and a ceiling that low made a real lob on the
            recording look too small for its line of sight.
        clutter_block: Side, in pixels, of the blocks the "always moving"
            map is kept at.
        clutter_decay: Per-frame decay of that map (swaying trees, a
            flapping windscreen).
        clutter_limit: A block whose decayed movement count is above this
            is ignored.
        player_box_weight: Factor applied to the score of a candidate
            inside a player's box - a limb or racket far more often than
            the ball. 0 discards such candidates altogether.
        player_box_pad: A player's box is grown on every side by this
            fraction of its height before that test, to take in the arm
            and racket.
        max_speed_mps: Fastest a ball is allowed to travel. A step
            implying more is an impossible jump and is never linked.
        speed_slack: Allowance on that bound for the airborne ball being
            nearer the camera than the ground at its image row.
        max_step: The same bound without a calibration, as a fraction of
            image width per second.
        min_step: Slowest movement, as a fraction of image width per
            second, a seed pair may show: slower is scenery, not a ball.
        gate: Radius within which a detection may continue a track, as a
            fraction of image width, before the speed-dependent part.
        gate_speed_factor: The gate grows by this fraction of the track's
            current per-frame displacement.
        follow_gain: While a path is being grown, how far (0..1) its
            position estimate moves toward each detection.
        velocity_gain: ...and how far its velocity estimate does.
        max_gap_seconds: Longest run of frames with no detection that is
            bridged (tagged `interpolated`). A longer silence ends the
            segment and is reported as unknown - never filled.
        min_segment_frames: Fewest measured detections a segment needs.
        min_segment_score: Least evidence (sum of detection scores,
            discounted by roughness) a segment needs to be kept at all.
        max_roughness: A segment whose detections sit, on average, further
            than this fraction of the gate from where smooth motion
            predicted them is jitter, not flight.
        min_travel: Least distance, as a fraction of image width, a
            segment must cover end to end; a twitching leaf does not.
        process_noise: Acceleration the filter allows, as a fraction of
            image width per second squared.
        measurement_noise: Detection jitter, as a fraction of image width.
        smoothness_weight: How strongly a segment's score is reduced by
            its residual from smooth motion.
        colour_weight: Share of a candidate's look that comes from being
            ball-coloured. The look is then multiplied by how well its
            size fits (`ball.FlightVolume.size_fit`).
        contrast_weight: Share that comes from how strongly it changed.
        weak_coverage: Coverage below which `BallQuality.weak` is True.
        weak_confidence: Mean confidence below which it is True. The
            classic detector's confidence is a score, not a probability:
            on six rallies of the real recording that were mostly tracked
            correctly its mean was 0.37 to 0.46, so the floor sits below
            that range rather than at a "50%" it does not mean. Revisit
            when a learned detector with calibrated confidence exists.
        scene_sample_stride: Every this-many-th frame feeds the scene
            brightness and colour summary.
        heatmap_threshold: Smallest heatmap value (0..1) a learned
            detector's peak needs to become a candidate.
    """

    change_threshold: float = 22.0
    difference_seconds: float = 0.067
    difference_frames: int = 2
    ball_diameter_m: float = 0.067
    size_slack: float = 4.0
    min_blob_ratio: float = 1.5
    resolved_ball_px: float = 6.0
    size_tolerance: float = 1.5
    max_blob: float = 0.03
    min_blob_px: int = 2
    max_candidates: int = 10
    court_margin_m: float = 4.0
    side_margin_m: float = 1.5
    flight_height_m: float = 10.0
    clutter_block: int = 4
    clutter_decay: float = 0.97
    clutter_limit: float = 6.0
    player_box_weight: float = 0.25
    player_box_pad: float = 0.15
    max_speed_mps: float = 70.0
    speed_slack: float = 2.0
    max_step: float = 3.0
    min_step: float = 0.02
    gate: float = 0.006
    gate_speed_factor: float = 0.6
    follow_gain: float = 0.85
    velocity_gain: float = 0.5
    max_gap_seconds: float = 0.1
    min_segment_frames: int = 5
    min_segment_score: float = 2.5
    max_roughness: float = 0.5
    min_travel: float = 0.02
    process_noise: float = 4.0
    measurement_noise: float = 0.0015
    smoothness_weight: float = 1.0
    colour_weight: float = 0.6
    contrast_weight: float = 0.4
    weak_coverage: float = 0.5
    weak_confidence: float = 0.25
    scene_sample_stride: int = 30
    heatmap_threshold: float = 0.5


@dataclass(frozen=True)
class TrackerConfig:
    """Everything the tracker is configured by, one field per stage.

    Attributes:
        video: Decode and timestamp-index settings.
        court: Court-line detection settings.
        calibration: Homography fit and validation settings.
        models: Learned-model choice and weight storage.
        players: Person detection, tracking and identity settings.
        ball: Ball candidate detection and track linking settings.
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    players: PlayerConfig = field(default_factory=PlayerConfig)
    ball: BallConfig = field(default_factory=BallConfig)

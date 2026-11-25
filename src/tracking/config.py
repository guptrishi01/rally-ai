"""Configuration for the independent tracker.

One frozen `TrackerConfig` holds a nested, frozen config per pipeline stage
(`video`, `court`, `calibration`, `models`, `players`, `ball`, `events`,
`strokes`, `segment`, `assemble`, `pipeline`, `retraining`). Later
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
        truncation_seconds: A file whose frames stop this much short of
            the duration its header declares is reported as cut short.
    """

    reject_vfr: bool = False
    vfr_jitter: float = 0.15
    checksum_chunk_bytes: int = 4 * 1024 * 1024
    decode_timeout_seconds: float = 30.0
    truncation_seconds: float = 2.0


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
        pose_model: Name of the pinned pose-model entry there.
        ball_model: Name of the learned ball model's entry there (Track
            B); used only when a learned ball detector is asked for.
        weights_dir: Folder weights are downloaded to and loaded from, or
            None for a per-user cache folder **outside the repository**
            (`tracking.models.default_weights_dir`). Weights are never
            committed.
        verify_checksum: Refuse weights whose SHA-256 differs from the
            pinned one (when the registry entry pins one).
    """

    person_model: str = "yolo11n"
    pose_model: str = "yolo11n-pose"
    ball_model: str = "ball-heatmap-v0"
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
class EventConfig:
    """Settings for turning the ball's track into hits, bounces and shots.

    Lengths are metres on the court, pixel lengths fractions of the image
    width, and times seconds.

    Attributes:
        fit_seconds: Stretch of track fitted on each side of a frame when
            the velocity before and after it are compared.
        min_fit_points: Fewest positions such a fit needs.
        min_kink_mps: Smallest change of velocity, in metres per second on
            the ground at that image row, that counts as a kink. A ball
            dropped from 0.4m changes its vertical speed by about 5 m/s
            when it bounces; less than this is tracking wobble.
        min_kink_ratio: The change must also be at least this fraction of
            the ball's own speed there - a fast ball's image speed drifts
            smoothly with perspective, and that drift is not an event.
        kink_separation_seconds: Two kinks closer than this are one.
        confusion_fraction: A ball segment at least this much inside a
            player's box is taken to be their racket or hand, not the
            ball, and is left out of the trajectory.
        max_junction_seconds: Longest unseen stretch between two segments
            across which their paths are extended to look for the one
            event - a hit or a bounce - that joins them. Across a longer
            one nothing is placed in the gap: the later segment is an
            appearance, and a bounce in the gap is not reported.
        max_lookback_seconds: How far back, at most, the path of a ball
            that appears mid-flight is followed to find when it left a
            player.
        junction_miss: How far apart, as a fraction of image width, the
            two extended paths may pass and still count as meeting...
        junction_miss_seconds: ...plus this many seconds of the ball's own
            speed there.
        reach_m: How far from a player's box, in metres at their own
            distance, the ball may be and still be within reach of their
            racket.
        player_lookup_seconds: When the players' track has no position at
            a moment, a detection at most this far away in time is used.
        min_flight_seconds: Least time between one player's hit and the
            other's.
        rehit_seconds: Least time between two hits by the same player
            with no hit by the other seen in between.
        max_return_seconds: Longest time after a hit in which the other
            player's hit still counts as a return of it.
        max_exchange_seconds: Two hits by one player no further apart than
            this, with no hit by the other seen between them, mean the
            other player's hit was missed; it is added, tagged inferred.
        visit_gap_seconds: Events within reach of one player and no
            further apart than this belong to one visit of the ball, in
            which at most one is the hit.
        min_hit_mps: How hard a kink near a player must be to be their
            hit: the smaller of the change in the ball's image velocity
            and the speed it leaves at, in metres per second at the
            player's own distance.
        ball_diameter_m: A tennis ball's diameter.
        min_size_ratio: The ball's blob must look at least this many times
            as wide as a ball at a player's distance would, to be within
            that player's reach...
        max_size_ratio: ...and at most this many times. On the real
            recording a ball at the near player looked 0.9 to 1.9 times
            the expected width and one at the far player 0.7 to 2.0,
            while a far-court ball seen "at" the near player in the
            picture looked 0.25 to 0.5 times it.
        ground_level_m: A ball no more than this far above the far
            player's feet in the image is on the ground in front of them.
        bounce_sideways: Largest sideways change of image velocity, as a
            fraction of the vertical change, a bounce may show.
        bounce_reversal: A kink that reverses the ball's sideways image
            motion, with more than this fraction of the vertical change on
            both sides, is not a bounce.
        bounce_min_keep: A ball leaving a bounce moves at least this
            fraction as fast, in the image, as it arrived.
        bounce_margin_m: How far outside the doubles court a bounce may
            lie and still be reported (and called out).
        pixel_sigma: Uncertainty of the ball's position at a bounce, as a
            fraction of image width, before it is turned into metres.
        bounce_floor_m: Uncertainty every bounce position carries however
            fine the pixels are, added in quadrature: the frame at which
            the path turns is not exactly the contact, and the ball's
            centre is not on the ground. On the real recording near-end
            bounces differed from the SwingVision sheet by a median 0.5m
            where the pixels alone would allow 0.2m.
        call_sigmas: A bounce is called in or out only when it is at
            least this many uncertainties from the nearest line.
        net_stop_ratio: A ball that keeps less than this fraction of its
            image speed, in front of the net, has hit it.
        net_pad: The net's outline is grown by this fraction of image
            width for that test.
        tape_pad: The net's white tape is grown by this fraction of image
            width - about a ball's radius at that distance - when asking
            whether a kink sits on it. A kink there is ignored.
        min_net_approach_mps: Slowest a ball must have been moving for a
            stop to count as hitting the net.
        audio_window_seconds: A sharp sound this close to a hit's
            estimated time (after allowing for its travel) is taken to be
            that hit and confirms it.
        audio_retime_seconds: Such a sound also replaces the hit's time
            when the picture's own estimate is less certain than this.
        audio_min_strength: Weakest sound onset used, in units of the
            recording's 99th-percentile onset. Near-end hits on the real
            recording were 0.8 to 3.0; far-end hits mostly 0.3 to 0.8,
            no louder than footsteps and the neighbouring court.
        audio_separation_seconds: Two onsets closer than this are one.
        speed_of_sound_mps: Speed of sound.
        max_speed_mph: An average shot speed above this is not reported.
        min_speed_seconds: Shortest contact-to-bounce time a speed is
            computed over.
        inferred_confidence: Confidence given to a result concluded from
            what happened next rather than seen.
        court: Lines shots are judged against: "singles" or "doubles".
    """

    fit_seconds: float = 0.17
    min_fit_points: int = 4
    min_kink_mps: float = 4.5
    min_kink_ratio: float = 0.35
    kink_separation_seconds: float = 0.12
    confusion_fraction: float = 0.9
    max_junction_seconds: float = 0.25
    max_lookback_seconds: float = 0.5
    junction_miss: float = 0.02
    junction_miss_seconds: float = 0.08
    reach_m: float = 1.2
    player_lookup_seconds: float = 1.0
    min_flight_seconds: float = 0.3
    rehit_seconds: float = 1.0
    max_return_seconds: float = 3.5
    max_exchange_seconds: float = 4.5
    visit_gap_seconds: float = 1.0
    min_hit_mps: float = 1.0
    ball_diameter_m: float = 0.067
    min_size_ratio: float = 0.7
    max_size_ratio: float = 3.0
    ground_level_m: float = 0.25
    bounce_sideways: float = 0.8
    bounce_reversal: float = 0.15
    bounce_min_keep: float = 0.3
    bounce_margin_m: float = 4.0
    pixel_sigma: float = 0.001
    bounce_floor_m: float = 0.3
    call_sigmas: float = 1.0
    net_stop_ratio: float = 0.3
    net_pad: float = 0.004
    tape_pad: float = 0.001
    min_net_approach_mps: float = 3.0
    audio_window_seconds: float = 0.06
    audio_retime_seconds: float = 0.05
    audio_min_strength: float = 1.0
    audio_separation_seconds: float = 0.08
    speed_of_sound_mps: float = 343.0
    max_speed_mph: float = 160.0
    min_speed_seconds: float = 0.1
    inferred_confidence: float = 0.5
    court: str = "singles"


@dataclass(frozen=True)
class StrokeConfig:
    """Settings for naming each hit's stroke from the hitter's pose.

    Attributes:
        handedness: The hand each person plays with, "right" or "left",
            as (person, hand) pairs for "A" - whoever was at the near end
            when tracking began - and "B". **An assumption until intake
            asks**: a wrong value swaps every forehand and backhand of
            that player.
        window_before: Pose is estimated on frames from this long before
            a hit...
        window_after: ...to this long after it. Mostly before: which side
            the racket hand is on is read from the backswing and swing,
            not the follow-through.
        pose_frames: How many frames inside that window are looked at.
        pose_size: Longest side, in pixels, the hitter's crop is scaled
            to for the pose model.
        crop_pad: The hitter's box is grown by this fraction of its size
            before cropping, to take in the racket arm.
        min_keypoint_confidence: A keypoint below this is treated as not
            seen.
        min_player_height: A hitter whose box is shorter than this
            fraction of the image height is too small for pose to mean
            anything; their strokes are judged without it. The far
            player on the real recording is about 0.08 and pose there
            was still usable.
        serve_behind_m: A serve is hit from no further inside the
            baseline than this.
        serve_quiet_seconds: A serve follows at least this long with no
            hit by the other player.
        overhead_margin: Contact counts as overhead when the hitting
            wrist is above the head by at least this fraction of the
            shoulder-to-hip length - and as *not* overhead when the ball
            was at least this fraction of the box's height below its top.
        volley_line_m: A volley is hit from no further from the net than
            the service line plus this.
        side_margin: The hitting wrist (or the ball) must be this
            fraction of the hitter's box height to one side of the body's
            centre before a forehand or backhand is named.
        min_confidence: A stroke scored below this is reported as
            `unknown`.
    """

    handedness: tuple[tuple[str, str], ...] = (("A", "right"), ("B", "right"))
    window_before: float = 0.3
    window_after: float = 0.1
    pose_frames: int = 6
    pose_size: int = 320
    crop_pad: float = 0.35
    min_keypoint_confidence: float = 0.4
    min_player_height: float = 0.05
    serve_behind_m: float = 0.6
    serve_quiet_seconds: float = 4.0
    overhead_margin: float = 0.25
    volley_line_m: float = 0.5
    side_margin: float = 0.08
    min_confidence: float = 0.4


@dataclass(frozen=True)
class SegmentConfig:
    """Settings for splitting the shots into serve attempts and points.

    Every time below was measured on the one real export available (the
    2026-08-21 sheet, 683 shots): hits inside a point are at most 4.9s
    apart (99% within 3.5s); a first-serve fault and its second serve are
    5.6-34s apart (median 9.2s, 90% within 16.5s), while consecutive points
    are at least 8.9s apart (median 30.6s); a server's feet are within
    1.02m of the centre mark on every one of 174 serves, a returner's at
    least 1.2m from it on all but a few.

    Attributes:
        rally_gap_seconds: Hits further apart than this are in different
            rallies.
        serve_quiet_seconds: A serve follows at least this long without a
            confident hit by the other player.
        quiet_breaker_confidence: A hit by the other player below this
            confidence does not break the quiet. The events stage infers a
            hit at 0.2 when the server's own ball-bounce before serving
            looks like "the ball went and came back"; that must not cost
            the real serve its quiet.
        serve_behind_near_m: The near server's feet are no further inside
            the baseline than this.
        serve_behind_far_m: The same at the far end, wider because a far
            footpoint is a metre-scale estimate (0.28m of depth per pixel
            on the real recording).
        serve_max_x_m: The server's feet are no further than this from the
            centre line - which tells a serve from a return hit after a
            serve nobody saw.
        serve_strokes: Stroke labels a serve may carry. A serve is struck
            overhead, so an `Overhead` after a quiet spell from behind the
            baseline is a serve; a named forehand or backhand is not.
        serve_lead_hits: A serve is preceded, inside its rally (since the
            last silence or the other player's last confident hit), by at
            most this many hits: the pre-serve bounce and the weak "return"
            inferred from it. More means a rally is going on - its weak
            inferred hits are real returns, timed badly.
        serve_cluster_seconds: Serve candidates by the same player this
            close together, with no confident hit by the other player
            between, are one serve: the strongest is kept. The ball bounced
            before serving is taken for a hit on the real recording.
        second_serve_max_seconds: A serve attempt by the same server from
            the same side within this long of a first serve that was not
            played on makes that first serve a fault.
        second_serve_unzoned_seconds: The same when either serving side is
            unknown - shorter, since only the time is left to go on.
        min_unserved_hits: A rally with no serve seen is taken for a point
            (whose serve was missed) only with at least this many hits by
            both players; anything less is play between points.
        match_start: Seconds on the video's clock before which play is
            warm-up and never a point. None: no warm-up is assumed.
        zone_doubt_m: When the server's feet are within this of the centre
            line - or were not seen - a seen serve bounce decides the
            serving side instead (a serve lands diagonally opposite).
        return_delay_seconds: How long before a return its serve was
            struck, for a serve inferred from the return: the sheet's
            median (90% of returns came 1.0-1.6s after the serve).
    """

    rally_gap_seconds: float = 5.0
    serve_quiet_seconds: float = 4.0
    quiet_breaker_confidence: float = 0.4
    serve_behind_near_m: float = 0.6
    serve_behind_far_m: float = 2.0
    serve_max_x_m: float = 1.8
    serve_strokes: tuple[str, ...] = ("Serve", "Overhead", "unknown")
    serve_lead_hits: int = 2
    serve_cluster_seconds: float = 4.0
    second_serve_max_seconds: float = 20.0
    second_serve_unzoned_seconds: float = 12.0
    min_unserved_hits: int = 2
    match_start: float | None = None
    zone_doubt_m: float = 1.0
    return_delay_seconds: float = 1.14


@dataclass(frozen=True)
class AssembleConfig:
    """Settings for turning segmented shots into `RawShotRow`s.

    Attributes:
        confident_hit: A hit seen or bridged (not inferred) at least this
            confident counts as "tracked with confidence" in the coverage
            note.
        center_line_m: A rally shot hit with the feet this close to the
            centre line is in the sheet's `center_line` zone. Serves never
            are: the sheet names them by the side served from.
        unknown_stroke: What is emitted for a stroke the classifier could
            not name with any confidence. Not one of SwingVision's
            strokes, on purpose: the review screens show it to a person,
            and a made-up "Forehand" would be read as a finding.
        far_weak_ratio: The far-end note is given when the far end has
            fewer than this share of the near end's confident hits.
        ad_scoring: Whether games are played with advantage (two clear
            after deuce) - the rule unseen endings are settled by.
    """

    confident_hit: float = 0.4
    center_line_m: float = 0.1
    unknown_stroke: str = "Unclear"
    far_weak_ratio: float = 0.6
    ad_scoring: bool = True


@dataclass(frozen=True)
class RetrainingConfig:
    """When to tell the user a ball model retrained on their footage would help.

    Two separate questions, each answered against a **measured** reference
    (`tracking.conditions.ConditionsReference`, stored at
    `reference_path`; docs/ball-model-training.md records how it was
    measured):

    1. *Does this footage look unlike what the detector was tuned on?* A
       run is outside the reference when its court brightness lies beyond
       the whole reference match's range, or its court colour
       (chromaticity) further from the reference's than any reference
       frame was, by more than the larger of `sigmas` times the
       reference's own spread and a floor. The `sigmas` margin is
       measured; the floors are a judgement - no footage in other
       conditions exists yet to show where the detector starts failing.
    2. *Was the ball tracked worse than on the reference footage?* Rally
       coverage below `relative_coverage` times the reference's measured
       rally coverage. With no measured baseline the absolute
       `weak_coverage` / `weak_confidence` fallbacks apply, and the note
       says they are provisional.

    Attributes:
        reference_path: The conditions reference file; None for none (no
            conditions are then compared, and the note says so).
        sigmas: Margin beyond the reference's range, in its standard
            deviations.
        brightness_floor: The smallest brightness margin, as a fraction of
            the reference's median brightness.
        chroma_floor: The smallest colour margin, in chromaticity units
            (each channel's share of R + G + B).
        relative_coverage: Weak tracking is rally coverage below this
            share of the reference's measured rally coverage.
        relative_confidence: ...or mean ball confidence below this share
            of the reference's.
        weak_coverage: Absolute fallback when the reference has no
            tracking baseline. Mirrors `BallConfig.weak_coverage`.
        weak_confidence: Absolute fallback for mean ball confidence.
        weak_point_share: Weak tracking is also concluded when more than
            this share of the points had weak ball tracking.
        max_abstention: More than this share of shots with no result
            counts as weak event reading.
    """

    reference_path: Path | None = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "models"
        / "ball_conditions_reference.json"
    )
    sigmas: float = 3.0
    brightness_floor: float = 0.25
    chroma_floor: float = 0.03
    relative_coverage: float = 0.6
    relative_confidence: float = 0.6
    weak_coverage: float = 0.5
    weak_confidence: float = 0.25
    weak_point_share: float = 0.5
    max_abstention: float = 0.6


@dataclass(frozen=True)
class DeadTimeConfig:
    """The dead-time pre-pass: skipping stretches with no ball-strike sounds.

    Off by default. Measured on the one real recording (docs/ball-model-
    training.md): these settings skip 15.6% of the 95-minute match and
    none of the 683 shots SwingVision recorded falls in a skipped stretch -
    but with 4s of padding instead of 5, one does, and the settings were
    chosen on that same recording. Treat the margin as thin.

    Attributes:
        enabled: Run the pre-pass and skip what it finds.
        min_strength: Weakest strike, in units of the recording's
            `percentile` onset.
        percentile: The onset percentile strengths are scaled by.
        separation_seconds: Two peaks closer than this are one strike.
        join_seconds: Strikes this close belong to one burst of play.
        min_strikes: A burst needs this many strikes to count as play.
        pad_before: Tracked before a burst's first strike, seconds (a
            serve's toss and bounce come before its strike).
        pad_after: Tracked after its last strike (a ball still in flight,
            bouncing out).
        min_dead_seconds: Only quiet stretches at least this long are
            skipped; shorter ones are tracked.
    """

    enabled: bool = False
    min_strength: float = 1.0
    percentile: float = 99.0
    separation_seconds: float = 0.08
    join_seconds: float = 6.0
    min_strikes: int = 2
    pad_before: float = 5.0
    pad_after: float = 5.0
    min_dead_seconds: float = 10.0


@dataclass(frozen=True)
class PipelineConfig:
    """How a whole recording is walked: chunks, overlap, checkpoints.

    The tracker costs about 80ms a 1080p frame on the development CPU -
    some 7.5 hours for a 95-minute match - and the machine is short on
    memory. So the video is processed in bounded chunks, each written to
    disk as it finishes, and a run that is killed resumes from the first
    unfinished chunk.

    Attributes:
        chunk_seconds: Length of the stretch each chunk owns.
        overlap_seconds: Extra video read on each side of a chunk, so the
            ball linker, the event rules and the motion detector's clutter
            map have context at the chunk's edges. Events are only kept
            from the stretch a chunk owns (see `stitch_margin_seconds`).
            Must exceed `stitch_margin_seconds` and should exceed the
            longest gap between hits in a point (4.9s measured).
        stitch_margin_seconds: Each chunk also keeps the events it found
            this close beyond either edge of its own stretch. Where two
            chunks saw the same event near their shared edge, one copy is
            kept; this is what stops a hit timed 149.99s by one chunk and
            150.01s by the next being kept twice or dropped.
        stitch_tolerance_seconds: Two events of the same kind (and side)
            from neighbouring chunks closer than this are the same event.
            Below `EventConfig.min_flight_seconds` (0.3s), so two real
            hits are never merged.
        work_dir: Where checkpoints are kept; `<pending_dir>/tracking_work`
            when None.
        keep_checkpoints: Keep the chunk files after the match is staged,
            so a re-stage after changing segmentation settings costs no
            tracking. They are small (a few MB per chunk).
        check_drift: Re-calibrate at the start of every chunk and note
            any chunk where the court has moved in the picture.
        drift_warn_px: Movement, in pixels, above which that note is made.
        recalibrate: Adopt a chunk's re-calibration when the court has
            moved and the new calibration lies closer to the painted lines
            (see `pipeline.TrackerPipeline._chunk_calibration`).
        recalibrate_px: Movement from the calibration in use, in pixels,
            above which a re-calibration is considered. Above the measured
            noise: two re-calibrations 20s apart on the real recording
            differed by 0.7-3.4px (median 1.9px, 9 pairs), while the camera
            itself crept 5-6.5px over the match.
        alignment_frames: Frames, from the middle of the chunk, the two
            calibrations are compared on (blended by median).
        decode_hole_frames: A run of at least this many frames that the
            timestamp index lists but decoding never produced is reported
            as a hole in the footage.
        calibrate_at: Where in the run the court is calibrated from,
            seconds after its start.
    """

    chunk_seconds: float = 150.0
    overlap_seconds: float = 8.0
    stitch_margin_seconds: float = 1.0
    stitch_tolerance_seconds: float = 0.2
    work_dir: Path | None = None
    keep_checkpoints: bool = True
    check_drift: bool = True
    drift_warn_px: float = 6.0
    recalibrate: bool = True
    recalibrate_px: float = 4.0
    alignment_frames: int = 5
    decode_hole_frames: int = 3
    calibrate_at: float = 0.0


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
        events: Hit, bounce, net and shot-result settings.
        strokes: Stroke classification settings.
        segment: Serve-attempt and point segmentation settings.
        assemble: `RawShotRow` emission settings.
        pipeline: Chunking, overlap and checkpoint settings.
        retraining: When the "retraining recommended" note is given.
        dead_time: The dead-time pre-pass.
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    players: PlayerConfig = field(default_factory=PlayerConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    events: EventConfig = field(default_factory=EventConfig)
    strokes: StrokeConfig = field(default_factory=StrokeConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    assemble: AssembleConfig = field(default_factory=AssembleConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    dead_time: DeadTimeConfig = field(default_factory=DeadTimeConfig)

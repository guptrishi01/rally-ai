"""Tests for fitting, checking and using the image<->court homography.

The synthetic courts come from a known camera, so every calibration is
compared with the truth - and every way a fit can be untrustworthy is
checked to end in an abstention with a reason, never in a made-up answer.
"""

import json
import subprocess

import cv2
import numpy as np
import pytest

from swingvision_import import footage
from tracking import calibrate, court
from tracking.config import CalibrationConfig, TrackerConfig
from tracking.provenance import Provenance

from .conftest import SIZE, ground_homography, make_camera, render_court

TEMPLATE = court.court_template()
MARKS = np.array(list(TEMPLATE.keypoints.values()))


def _clicks(camera, names, wobble=0.0, seed=0):
    """Image positions of named keypoints under a camera, optionally jittered."""
    truth = ground_homography(camera)
    rng = np.random.default_rng(seed)
    return {
        name: tuple(
            court.project(truth, np.array(TEMPLATE.keypoints[name])) + rng.normal(0, wobble, 2)
        )
        for name in names
    }


EIGHT = [
    "near_baseline:left_singles",
    "near_baseline:right_singles",
    "near_service:left_singles",
    "near_service:center",
    "near_service:right_singles",
    "far_service:left_singles",
    "far_service:right_singles",
    "far_baseline:center",
]


@pytest.fixture(scope="module")
def result(court_image):
    return calibrate.calibrate_frame(court_image)


# --------------------------------------------------------------------------
# Automatic calibration
# --------------------------------------------------------------------------


def test_a_clear_court_calibrates_to_the_cameras_own_homography(result, camera):
    assert result.ok and result.reason is None
    calibration = result.require()

    error = np.linalg.norm(
        calibration.to_image(MARKS) - court.project(ground_homography(camera), MARKS), axis=1
    )

    assert error.max() < 2.0
    assert calibration.provenance is Provenance.MEASURED
    assert calibration.template_kind == "doubles"
    assert calibration.image_size == SIZE


def test_image_points_map_back_onto_the_court_in_metres(result, camera):
    calibration = result.require()
    truth = ground_homography(camera)
    near_half = MARKS[MARKS[:, 1] < 12]

    recovered = calibration.to_court(court.project(truth, near_half))

    assert np.abs(recovered - near_half).max() < 0.05  # five centimetres, near half


def test_mapping_there_and_back_is_the_identity(result):
    calibration = result.require()
    points = np.array([[0.0, 0.0], [-3.2, 7.5], [4.0, 22.0]])

    assert calibration.to_court(calibration.to_image(points)) == pytest.approx(points, abs=1e-6)
    assert calibration.to_image([(0.0, 0.0)]).shape == (1, 2)


def test_quality_reports_residuals_resolution_and_lines(result):
    quality = result.quality

    assert quality is result.require().quality
    assert quality.points == quality.inliers == 16
    assert quality.residual_px_median < 1.0
    assert quality.residual_px_max >= quality.residual_px_median
    assert quality.residual_m_median < 0.05
    assert quality.near_depth_m_per_px < 0.05 < quality.far_depth_m_per_px
    assert quality.far_lateral_m_per_px < quality.far_depth_m_per_px
    assert quality.far_baseline_found is True
    assert len(quality.lines_found) == 9
    assert quality.support > 0.95
    assert 0 <= quality.setup_score <= 100


def test_a_calibration_serialises_to_json(result):
    stored = json.loads(json.dumps(result.require().to_dict()))

    assert stored["provenance"] == "measured"
    assert stored["template_kind"] == "doubles"
    assert np.array(stored["homography"]).shape == (3, 3)
    assert stored["quality"]["lines_found"] == list(result.quality.lines_found)
    assert isinstance(stored["quality"]["warnings"], list)


def test_a_low_camera_calibrates_but_says_the_far_end_is_weak():
    camera = make_camera(height=2.47, back=5.0, aim_y=6.0)

    outcome = calibrate.calibrate_frame(render_court(camera))

    assert outcome.ok
    quality = outcome.quality
    assert quality.far_baseline_found is False
    warnings = " ".join(quality.warnings)
    assert "far baseline was not located" in warnings and "net tape" in warnings
    assert "Raise the camera" in warnings
    assert (
        quality.setup_score
        < calibrate.calibrate_frame(render_court(make_camera(height=6))).quality.setup_score
    )


def test_several_frames_are_blended_so_a_player_does_not_hide_a_line(camera):
    frames = [
        render_court(camera, occluders=[(left, 250, left + 90, 500)], seed=left)
        for left in (150, 420, 700)
    ]

    outcome = calibrate.calibrate_frames(frames)

    assert outcome.ok
    assert len(outcome.quality.lines_found) == 9


def test_a_frame_with_no_court_abstains_with_the_detections_reason():
    blank = np.full((SIZE[1], SIZE[0], 3), 80, dtype=np.uint8)

    outcome = calibrate.calibrate_frame(blank, TrackerConfig())

    assert not outcome.ok and outcome.calibration is None and outcome.quality is None
    assert "No straight painted lines" in outcome.reason
    assert outcome.detection is not None and not outcome.detection.ok
    with pytest.raises(calibrate.CalibrationAbstained, match="No straight painted lines"):
        outcome.require()


def test_a_video_is_calibrated_from_frames_spread_over_a_few_seconds(tmp_path, court_image, camera):
    still = tmp_path / "court.png"
    cv2.imwrite(str(still), court_image[:, :, ::-1])
    clip = tmp_path / "court.mp4"
    subprocess.run(
        [
            footage.ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(still),
            "-t",
            "2",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "12",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    config = TrackerConfig(
        calibration=CalibrationConfig(frames_for_median=3, median_span_seconds=1.5)
    )

    outcome = calibrate.calibrate_video(clip, at=0.2, config=config)

    assert outcome.ok
    placed = outcome.require().to_image(MARKS)
    assert (
        np.linalg.norm(placed - court.project(ground_homography(camera), MARKS), axis=1).max() < 3
    )


# --------------------------------------------------------------------------
# The manual fallback
# --------------------------------------------------------------------------


def test_clicked_points_calibrate_and_are_tagged_as_a_persons(camera):
    outcome = calibrate.calibrate_from_points(_clicks(camera, EIGHT, wobble=0.5), SIZE)

    calibration = outcome.require()
    truth = court.project(ground_homography(camera), MARKS)
    assert np.linalg.norm(calibration.to_image(MARKS) - truth, axis=1).max() < 4.0
    assert calibration.provenance is Provenance.USER_ENTERED
    assert outcome.quality.lines_found is None
    assert outcome.quality.far_baseline_found is None
    assert outcome.quality.support is None
    assert outcome.detection is None


def test_four_clicks_are_enough_but_come_with_a_caution(camera):
    four = [
        "near_baseline:left_singles",
        "near_baseline:right_singles",
        "far_service:left_singles",
        "far_service:right_singles",
    ]

    outcome = calibrate.calibrate_from_points(_clicks(camera, four), SIZE)

    assert outcome.ok
    assert outcome.quality.residual_px_max == pytest.approx(0.0, abs=1e-3)
    assert any("exactly four points" in warning for warning in outcome.quality.warnings)


def test_a_singles_court_can_be_clicked(camera):
    names = [name for name in court.keypoint_names("singles") if "center" not in name]

    outcome = calibrate.calibrate_from_points(_clicks(camera, names), SIZE, template_kind="singles")

    assert outcome.require().template_kind == "singles"


def test_an_unknown_keypoint_name_is_an_error_that_shows_real_names(camera):
    clicks = _clicks(camera, EIGHT[:4]) | {"net:post": (10.0, 10.0)}

    with pytest.raises(ValueError, match="net:post.*near_baseline:left_doubles"):
        calibrate.calibrate_from_points(clicks, SIZE)
    with pytest.raises(ValueError, match="left_doubles"):
        calibrate.calibrate_from_points(
            {"near_baseline:left_doubles": (1.0, 2.0)}, SIZE, template_kind="singles"
        )


def test_one_stray_click_is_ignored_and_named(camera):
    clicks = _clicks(camera, EIGHT)
    x, y = clicks["near_service:center"]
    clicks["near_service:center"] = (x + 60, y - 40)

    outcome = calibrate.calibrate_from_points(clicks, SIZE)

    assert outcome.ok
    assert outcome.quality.inliers == 7 and outcome.quality.points == 8
    assert any("near_service:center" in warning for warning in outcome.quality.warnings)
    truth = court.project(ground_homography(camera), MARKS)
    assert np.linalg.norm(outcome.require().to_image(MARKS) - truth, axis=1).max() < 1.0


def test_points_outside_the_frame_are_flagged_as_extrapolated(camera):
    names = EIGHT + ["near_baseline:left_doubles"]

    outcome = calibrate.calibrate_from_points(_clicks(camera, names), SIZE)

    assert outcome.ok
    assert any("outside the frame" in warning for warning in outcome.quality.warnings)


# --------------------------------------------------------------------------
# Refusing a fit that cannot be trusted
# --------------------------------------------------------------------------


def _reason(clicks, config=None, kind="doubles"):
    outcome = calibrate.calibrate_from_points(clicks, SIZE, config, kind)
    assert not outcome.ok and outcome.calibration is None
    return outcome.reason


def test_too_few_points_abstain(camera):
    assert "at least 4 court points; 3 were given" in _reason(_clicks(camera, EIGHT[:3]))
    assert "at least 6" in _reason(_clicks(camera, EIGHT[:5]), CalibrationConfig(min_points=6))


def test_points_along_one_line_abstain(camera):
    baseline = [name for name in TEMPLATE.keypoints if name.startswith("near_baseline")]

    assert "one line" in _reason(_clicks(camera, baseline))


def test_all_clicks_on_one_spot_abstain(camera):
    assert "one line" in _reason(dict.fromkeys(EIGHT[:4], (100.0, 100.0)))


def test_a_position_that_is_not_a_number_abstains(camera):
    clicks = _clicks(camera, EIGHT)
    clicks[EIGHT[0]] = (float("nan"), 10.0)

    assert "not a number" in _reason(clicks)


def test_left_and_right_swapped_is_caught_as_a_mirrored_court(camera):
    clicks = _clicks(camera, EIGHT)
    swapped = {
        name.replace("left", "LEFT").replace("right", "left").replace("LEFT", "right"): spot
        for name, spot in clicks.items()
    }

    assert "mirrored" in _reason(swapped)


def test_near_and_far_swapped_is_caught(camera):
    clicks = _clicks(camera, EIGHT)
    swapped = {}
    for name, spot in clicks.items():
        across, along = name.split(":")
        across = across.replace("near", "NEAR").replace("far", "near").replace("NEAR", "far")
        along = along.replace("left", "LEFT").replace("right", "left").replace("LEFT", "right")
        swapped[f"{across}:{along}"] = spot  # the court turned end for end: not mirrored

    assert "wrong way round" in _reason(swapped)


def test_clicks_that_agree_on_no_court_abstain(camera):
    scattered = dict(
        zip(
            EIGHT,
            [
                (100, 100),
                (800, 120),
                (400, 400),
                (150, 500),
                (900, 300),
                (500, 50),
                (300, 200),
                (700, 480),
            ],
        )
    )

    reason = _reason(scattered)

    assert "agree" in reason or "No consistent court" in reason


def test_a_fit_that_puts_the_court_behind_the_camera_abstains():
    crossed = {
        "near_baseline:left_singles": (100.0, 500.0),
        "near_baseline:right_singles": (860.0, 500.0),
        "far_baseline:left_singles": (760.0, 100.0),
        "far_baseline:right_singles": (200.0, 100.0),
    }

    reason = _reason(crossed)

    assert "behind the camera" in reason or "mirrored" in reason


def test_the_shape_check_passes_a_real_view_and_names_what_is_wrong_with_others(camera):
    truth = ground_homography(camera)
    config = CalibrationConfig()
    flipped = np.array([[-1.0, 0, SIZE[0]], [0, 1, 0], [0, 0, 1]]) @ truth
    horizon_low = truth.copy()
    horizon_low[2] = [0.0, -0.2, 1.0]

    assert calibrate._shape_problem(truth, TEMPLATE, config) is None
    assert "mirrored" in calibrate._shape_problem(flipped, TEMPLATE, config)
    assert "behind the camera" in calibrate._shape_problem(horizon_low, TEMPLATE, config)
    assert "wrong way round" in calibrate._shape_problem(
        truth, TEMPLATE, CalibrationConfig(min_near_far_ratio=50.0)
    )


def test_a_large_residual_abstains_but_still_reports_how_large(camera):
    clicks = _clicks(camera, list(TEMPLATE.keypoints), wobble=6.0, seed=4)
    lenient_ransac = CalibrationConfig(ransac_threshold=0.05, max_residual_px=0.001)

    outcome = calibrate.calibrate_from_points(clicks, SIZE, lenient_ransac)

    assert not outcome.ok
    assert "pixels (median)" in outcome.reason
    assert outcome.quality is not None and outcome.quality.residual_px_median > 1.1


def test_a_large_residual_in_metres_abstains(camera):
    clicks = _clicks(camera, list(TEMPLATE.keypoints), wobble=2.0, seed=4)

    reason = _reason(clicks, CalibrationConfig(ransac_threshold=0.05, max_residual_m=0.001))

    assert "m on the court" in reason


def test_a_camera_too_low_to_see_the_far_end_abstains_and_says_so():
    grazing = make_camera(height=0.5, back=5.0, aim_y=6.0)
    names = [name for name in TEMPLATE.keypoints if "doubles" not in name]

    outcome = calibrate.calibrate_from_points(_clicks(grazing, names), SIZE)

    assert not outcome.ok
    assert "too low" in outcome.reason and "Raise the camera" in outcome.reason
    assert outcome.quality.far_depth_m_per_px > 1.0
    assert outcome.quality.setup_score < 40


def test_a_fit_opencv_cannot_make_abstains(camera, monkeypatch):
    monkeypatch.setattr(calibrate.cv2, "findHomography", lambda *args, **kwargs: (None, None))

    assert "No consistent court" in _reason(_clicks(camera, EIGHT))


def test_a_fit_that_flattens_the_court_abstains(camera, monkeypatch):
    flat = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]])
    monkeypatch.setattr(
        calibrate.cv2, "findHomography", lambda *args, **kwargs: (flat, np.ones((8, 1), np.uint8))
    )

    assert "onto a line" in _reason(_clicks(camera, EIGHT))


def test_too_few_agreeing_points_abstain(camera, monkeypatch):
    truth = ground_homography(camera)
    few = np.array([[1], [1], [1], [1], [0], [0], [0], [0]], np.uint8)
    monkeypatch.setattr(calibrate.cv2, "findHomography", lambda *args, **kwargs: (truth, few))

    assert "Only 4 of the 8" in _reason(_clicks(camera, EIGHT))


# --------------------------------------------------------------------------
# Scoring, drift and the overlay
# --------------------------------------------------------------------------


def test_the_setup_score_rewards_resolution_fit_and_lines():
    best = calibrate._setup_score(0.03, 0.0, 4.0, 1.0, worst=1.0)
    coarse = calibrate._setup_score(0.5, 0.0, 4.0, 1.0, worst=1.0)
    loose = calibrate._setup_score(0.03, 3.0, 4.0, 1.0, worst=1.0)
    sparse = calibrate._setup_score(0.03, 0.0, 4.0, 0.5, worst=1.0)
    hopeless = calibrate._setup_score(5.0, 9.0, 4.0, 0.0, worst=1.0)

    assert best == 100.0 and hopeless == 0.0
    assert coarse < best and loose < best and sparse < best
    assert calibrate._setup_score(0.03, 0.0, 4.0, None, worst=1.0) == 100.0


def test_resolution_of_a_degenerate_map_is_infinite():
    collapse = np.array([[0.0, 0, 5], [0, 0, 5], [0, 0, 1]])

    assert calibrate._metres_per_pixel(collapse, (0.0, 0.0), (0.0, 0.5)) == float("inf")


def test_a_fixed_camera_does_not_drift_and_a_moved_one_does(result, camera):
    calibration = result.require()
    nudged = make_camera(lateral=0.3)
    moved = calibrate.calibrate_from_points(_clicks(nudged, EIGHT), SIZE).require()
    same = calibrate.calibrate_from_points(_clicks(camera, EIGHT), SIZE).require()

    assert calibrate.drift_px(calibration, calibration) == 0.0
    assert calibrate.drift_px(calibration, same) < 2.0
    assert calibrate.drift_px(calibration, moved) > 10.0


def test_drift_falls_back_to_the_corners_when_little_of_the_court_is_in_frame(result):
    calibration = result.require()
    cropped = calibrate.Calibration(
        calibration.homography,
        calibration.inverse,
        (40, 40),
        "doubles",
        calibration.provenance,
        calibration.quality,
    )

    assert calibrate.drift_px(cropped, calibration) == pytest.approx(0.0)


def test_the_overlay_draws_the_template_on_a_copy(result, court_image):
    before = court_image.copy()

    drawn = calibrate.draw_overlay(court_image, result.require(), result.detection)
    plain = calibrate.draw_overlay(court_image, result.require())

    assert np.array_equal(court_image, before)
    assert drawn.shape == court_image.shape
    red = (drawn[..., 0] > 200) & (drawn[..., 1] < 90) & (drawn[..., 2] < 90)
    green = (drawn[..., 1] > 200) & (drawn[..., 0] < 90) & (drawn[..., 2] < 90)
    assert red.sum() > 1000 and green.sum() > 50
    plain_green = (plain[..., 1] > 200) & (plain[..., 0] < 90) & (plain[..., 2] < 90)
    assert plain_green.sum() == 0
    column, row = np.round(result.require().to_image([(0.0, 8.0)])[0]).astype(int)
    assert red[row - 2 : row + 3, column - 2 : column + 3].any()  # on the centre line


# --------------------------------------------------------------------------
# Line alignment (judging a re-calibration) and setup advice
# --------------------------------------------------------------------------


def test_a_calibration_that_fits_lies_on_the_lines_and_a_moved_one_does_not(
    result, camera, court_image
):
    fitted = calibrate.line_alignment_px(result.calibration, court_image)
    moved_camera = make_camera(lateral=0.6)
    moved = calibrate.calibrate_from_points(
        {
            name: tuple(court.project(ground_homography(moved_camera), np.array(point)))
            for name, point in TEMPLATE.keypoints.items()
        },
        SIZE,
    ).calibration
    shifted = calibrate.line_alignment_px(moved, court_image)
    assert fitted < 1.0 < shifted
    assert calibrate.line_alignment_px(moved, render_court(moved_camera)) < 1.0


def test_alignment_needs_lines_and_a_court_in_the_frame(result):
    blank = np.full((SIZE[1], SIZE[0], 3), 60, dtype=np.uint8)
    assert calibrate.line_alignment_px(result.calibration, blank) is None
    away = calibrate.Calibration(
        homography=np.array([[1.0, 0, -1e6], [0, 1.0, -1e6], [0, 0, 1.0]]),
        inverse=np.array([[1.0, 0, 1e6], [0, 1.0, 1e6], [0, 0, 1.0]]),
        image_size=SIZE,
        template_kind="doubles",
        provenance=Provenance.MEASURED,
        quality=result.calibration.quality,
    )
    lined = np.full((SIZE[1], SIZE[0], 3), 60, dtype=np.uint8)
    lined[100:103, :] = 255
    assert calibrate.line_alignment_px(away, lined) is None


def _quality(score, far_depth=0.28, near=0.014, far_found=False, residual=0.6):
    return calibrate.CalibrationQuality(
        points=11,
        inliers=11,
        residual_px_median=residual,
        residual_px_max=2 * residual,
        residual_m_median=0.01,
        residual_m_max=0.1,
        near_depth_m_per_px=near,
        far_depth_m_per_px=far_depth,
        far_lateral_m_per_px=0.02,
        lines_found=None,
        far_baseline_found=far_found,
        support=None,
        setup_score=score,
        warnings=(),
    )


def test_setup_advice_says_what_a_weak_far_end_costs_and_what_to_do():
    advice = calibrate.setup_advice(_quality(66.5))
    assert advice.rating == "fair" and advice.reshoot
    assert advice.headline == (
        "Camera setup score 66/100 (fair): the far end is the weak part of this recording."
    )
    assert "0.28 m of court depth (20 times the near end's 0.014 m)" in advice.consequences[0]
    assert "behind the net tape" in advice.consequences[1]
    assert "Re-shooting from higher up" in advice.note()
    assert advice.to_dict()["far_depth_m_per_px"] == 0.28


def test_a_good_setup_needs_no_reshoot_and_a_loose_fit_is_named():
    good = calibrate.setup_advice(_quality(88.0, far_depth=0.08, far_found=True))
    assert good.rating == "good" and not good.reshoot and good.consequences == ()
    assert good.headline.endswith("the whole court is well placed from this position.")
    assert "Re-shooting" not in good.note()
    loose = calibrate.setup_advice(_quality(40.0, far_depth=0.08, far_found=None, residual=4.0))
    assert loose.rating == "poor" and not loose.reshoot
    assert loose.headline == "Camera setup score 40/100 (poor)."
    assert "fit loosely (4.0 px median)" in loose.consequences[0]
    flat = calibrate.setup_advice(_quality(50.0, far_depth=0.3, near=0.0, far_found=True))
    assert "inf times" in flat.consequences[0]

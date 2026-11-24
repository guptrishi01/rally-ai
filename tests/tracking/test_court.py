"""Tests for the court template and for finding the court's lines in a frame.

Detection is tested on courts rendered through a known camera (see
conftest.py), so "found the court" means "put its keypoints where the
camera really put them", not merely "returned something".
"""

import math

import cv2
import numpy as np
import pytest

from tracking import court
from tracking.config import CourtConfig

from .conftest import SIZE, ground_homography, make_camera, render_court

# --------------------------------------------------------------------------
# The template and its coordinate frame
# --------------------------------------------------------------------------


def test_the_frame_starts_at_the_near_centre_mark_in_metres():
    template = court.court_template("doubles")

    assert template.keypoints["near_baseline:center"] == (0.0, 0.0)
    assert template.keypoints["near_baseline:right_doubles"] == pytest.approx((5.485, 0.0))
    assert template.keypoints["near_baseline:left_singles"] == pytest.approx((-4.115, 0.0))
    assert template.keypoints["near_service:center"] == pytest.approx((0.0, 5.485))
    assert template.keypoints["far_service:right_singles"] == pytest.approx((4.115, 18.285))
    assert template.keypoints["far_baseline:left_doubles"] == pytest.approx((-5.485, 23.77))
    assert court.NET_Y_M == pytest.approx(11.885)


def test_a_doubles_court_has_nine_lines_and_sixteen_keypoints():
    template = court.court_template("doubles")

    assert len(template.lines) == 9
    assert len(template.keypoints) == 16
    assert "near_service:left_doubles" not in template.keypoints  # the service line stops short
    assert template.half_width == pytest.approx(5.485)


def test_a_singles_court_has_no_alleys():
    template = court.court_template("singles")

    assert len(template.lines) == 7
    assert len(template.keypoints) == 12
    assert template.line("near_baseline").end == pytest.approx((4.115, 0.0))
    assert template.corners[2] == pytest.approx((4.115, 23.77))
    assert court.keypoint_names("singles") == tuple(template.keypoints)


def test_families_are_ordered_near_to_far_and_left_to_right():
    template = court.court_template()

    assert [ln.name for ln in template.family(court.TRANSVERSE)] == [
        "near_baseline",
        "near_service",
        "far_service",
        "far_baseline",
    ]
    assert [ln.name for ln in template.family(court.LONGITUDINAL)] == [
        "left_doubles",
        "left_singles",
        "center",
        "right_singles",
        "right_doubles",
    ]
    assert template.line("center").start == pytest.approx((0.0, 5.485))


def test_unknown_templates_and_lines_are_errors():
    with pytest.raises(ValueError, match="mixed"):
        court.court_template("mixed")
    with pytest.raises(KeyError):
        court.court_template().line("tramline")


# --------------------------------------------------------------------------
# ImageLine
# --------------------------------------------------------------------------


def test_a_line_measures_distance_angle_and_crossings():
    flat = court.ImageLine.through((0, 10), (100, 10))
    steep = court.ImageLine.through((50, 0), (50, 80))

    assert flat.angle_deg == pytest.approx(0.0)
    assert steep.angle_deg == pytest.approx(90.0)
    assert flat.length == pytest.approx(100.0) and flat.support == pytest.approx(100.0)
    assert abs(float(flat.distance(np.array([3.0, 14.0])))) == pytest.approx(4.0)
    assert flat.intersect(steep) == pytest.approx((50.0, 10.0))
    assert flat.y_at(999) == pytest.approx(10.0)
    assert steep.x_at(-5) == pytest.approx(50.0)


def test_a_lines_angle_does_not_depend_on_which_end_comes_first():
    assert court.ImageLine.through((100, 10), (0, 10)).angle_deg == pytest.approx(0.0)
    assert court.ImageLine.through((50, 80), (50, 0)).angle_deg == pytest.approx(90.0)
    rising = court.ImageLine.through((0, 0), (10, -10)).angle_deg
    assert court.ImageLine.through((10, -10), (0, 0)).angle_deg == pytest.approx(rising)


def test_parallel_lines_do_not_cross_and_a_point_is_not_a_line():
    one = court.ImageLine.through((0, 0), (10, 0))
    two = court.ImageLine.through((0, 5), (10, 5))

    assert one.intersect(two) is None
    with pytest.raises(ValueError):
        court.ImageLine.through((3, 3), (3, 3))


# --------------------------------------------------------------------------
# Frames to line pixels to lines
# --------------------------------------------------------------------------


def test_the_median_of_several_frames_removes_what_moved():
    base = np.full((40, 60, 3), 90, dtype=np.uint8)
    frames = []
    for column in (5, 25, 45):
        frame = base.copy()
        frame[10:30, column : column + 8] = 255  # a "player" somewhere else each time
        frames.append(frame)

    assert np.array_equal(court.median_frame(frames), base)


def test_median_frame_needs_matching_frames():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert np.array_equal(court.median_frame([frame]), frame)
    assert court.median_frame([frame]) is not frame
    with pytest.raises(ValueError, match="at least one"):
        court.median_frame([])
    with pytest.raises(ValueError, match="one size"):
        court.median_frame([frame, np.zeros((5, 4, 3), dtype=np.uint8)])


def _canvas(colour=(40, 80, 175)):
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    image[:] = colour
    return image


def test_the_mask_keeps_thin_white_lines_and_nothing_else():
    image = _canvas()
    cv2.line(image, (100, 300), (800, 320), (245, 245, 245), 3)  # a painted line
    cv2.rectangle(image, (300, 20), (700, 200), (250, 250, 250), -1)  # sky: bright but broad
    cv2.line(image, (100, 400), (800, 400), (255, 230, 40), 3)  # thin but yellow
    cv2.line(image, (100, 450), (800, 450), (95, 95, 95), 3)  # thin but dim

    mask = court.line_mask(image)

    assert mask.dtype == np.uint8 and mask.shape == (SIZE[1], SIZE[0])
    assert mask[310, 450] == 255
    assert mask[100, 500] == 0
    assert mask[400, 450] == 0
    assert mask[450, 450] == 0


def test_segments_are_found_on_a_line_and_not_on_nothing():
    image = _canvas()
    cv2.line(image, (100, 300), (800, 320), (245, 245, 245), 3)

    segments = court.find_segments(court.line_mask(image))

    assert segments.ndim == 2 and segments.shape[1] == 4 and len(segments) >= 1
    assert court.find_segments(court.line_mask(_canvas())).shape == (0, 4)


def test_segments_on_one_line_merge_and_others_stay_apart():
    segments = np.array(
        [
            [100, 200, 300, 200],  # the same line, in two pieces...
            [340, 201, 600, 201],
            [120, 203, 280, 203],  # ...and its other edge
            [100, 260, 600, 262],  # a different line
            [400, 50, 420, 400],  # and a steep one
        ],
        dtype=float,
    )

    lines = court.merge_segments(segments, SIZE)

    assert len(lines) == 3
    assert lines[0].support >= lines[1].support >= lines[2].support
    longest = next(ln for ln in lines if abs(ln.start[1] - 201) < 3)
    assert longest.angle_deg == pytest.approx(0.0, abs=0.5)
    assert min(longest.start[0], longest.end[0]) == pytest.approx(100, abs=1)
    assert max(longest.start[0], longest.end[0]) == pytest.approx(600, abs=1)
    assert longest.support == pytest.approx(462, abs=3)  # the 40px gap is not evidence
    assert court.merge_segments(np.zeros((0, 4)), SIZE) == []


def test_evidence_maps_keep_flat_and_steep_segments_apart():
    maps = court.evidence_maps(np.array([[100, 200, 300, 200], [400, 50, 420, 400]]), SIZE)

    assert maps.shape == (2, SIZE[1], SIZE[0]) and maps.dtype == bool
    assert maps[0, 200, 200] and not maps[1, 200, 200]
    assert maps[1, 225, 410] and not maps[0, 225, 410]


# --------------------------------------------------------------------------
# Line families
# --------------------------------------------------------------------------


def test_court_lines_split_into_two_families_around_a_vanishing_point(court_image, camera):
    lines = court.merge_segments(court.find_segments(court.line_mask(court_image)), SIZE)

    families = court.classify_lines(lines, SIZE)

    truth = ground_homography(camera)
    expected = court.project(truth, np.array([0.0, 1e7]))  # where "down the court" ends up
    assert families.vanishing_point == pytest.approx(tuple(expected), abs=6)
    assert len(families.longitudinal) >= 5
    assert len(families.transverse) >= 4
    assert all(abs(ln.angle_deg) <= 12 for ln in families.transverse)


def test_without_converging_lines_there_is_no_longitudinal_family():
    flat = [court.ImageLine.through((0, y), (900, y + 3)) for y in (300, 400)]
    upright = [court.ImageLine.through((x, 100), (x, 500)) for x in (200, 600)]  # parallel
    diverging = [  # these meet below themselves, as no sideline pair can
        court.ImageLine.through((200, 100), (300, 500)),
        court.ImageLine.through((700, 100), (600, 500)),
    ]

    assert court.classify_lines(flat, SIZE).longitudinal == ()
    assert court.classify_lines(flat + upright[:1], SIZE).vanishing_point is None
    assert court.classify_lines(flat + upright, SIZE).vanishing_point is None
    assert court.classify_lines(flat + diverging, SIZE).vanishing_point is None
    assert len(court.classify_lines(flat, SIZE).transverse) == 2


def test_lines_above_the_horizon_cannot_be_court_lines():
    sidelines = [
        court.ImageLine.through((100, 500), (400, 200)),
        court.ImageLine.through((860, 500), (560, 200)),
    ]
    roof = court.ImageLine.through((100, 60), (800, 62))
    baseline = court.ImageLine.through((50, 480), (900, 482))

    families = court.classify_lines([*sidelines, roof, baseline], SIZE)

    assert families.vanishing_point == pytest.approx((480, 120), abs=1)
    assert [ln.start[1] for ln in families.transverse] == pytest.approx([480], abs=1)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def test_project_maps_points_and_survives_infinity(camera):
    truth = ground_homography(camera)
    point = court.project(truth, np.array([[0.0, 0.0], [4.115, 5.485]]))

    assert point.shape == (2, 2)
    assert 0 < point[0, 0] < SIZE[0]
    horizon = np.array([[1.0, 0, 0], [0, 1, 0], [0, 1, 0]])
    assert not np.isfinite(court.project(horizon, np.array([3.0, 0.0]))).all()


def test_only_focal_lengths_a_camera_could_have_are_plausible(camera):
    truth = ground_homography(camera)
    fisheye = ground_homography(make_camera(focal=0.1))
    telephoto = ground_homography(make_camera(focal=8.0, back=120, height=40))

    verdict = court.plausible_camera(np.stack([truth, fisheye, telephoto]), SIZE)

    assert verdict.tolist() == [True, False, False]


def test_the_whole_camera_is_recovered_from_the_ground_homography():
    camera = make_camera(lateral=1.0, roll_deg=2.0)

    recovered = court.camera_from_homography(ground_homography(camera), SIZE)

    above_ground = np.array([[0.0, 11.885, 1.07, 1.0], [-3.0, 20.0, 2.0, 1.0]])
    wanted = above_ground @ camera.T
    got = above_ground @ recovered.T
    assert got[:, :2] / got[:, 2:] == pytest.approx(wanted[:, :2] / wanted[:, 2:], abs=0.05)


def test_no_camera_is_recovered_from_a_homography_no_camera_makes():
    assert court.camera_from_homography(np.eye(3), SIZE) is None  # a flat scan, no depth
    stretched = np.array([[40.0, 0, 480], [0, 900.0, 270], [0, 0.001, 1]])
    assert court.camera_from_homography(stretched, SIZE) is None
    assert court.net_tape_zone(np.eye(3), SIZE, 2.0) is None


def test_the_net_tape_zone_covers_the_tape_and_not_the_court_lines(camera):
    truth = ground_homography(camera)

    zone = court.net_tape_zone(truth, SIZE, 1.0)

    tape = np.array([[0.0, 11.885, 1.0, 1.0]]) @ camera.T
    column, row = np.round(tape[0, :2] / tape[0, 2]).astype(int)
    assert zone[row, column]
    for mark in ((0.0, 5.485), (0.0, 23.77), (4.115, 0.0)):
        column, row = np.round(court.project(truth, np.array(mark))).astype(int)
        assert not zone[row, column]


def test_a_camera_standing_inside_the_court_has_no_tape_zone():
    behind = ground_homography(make_camera(back=-5.0, aim_y=30.0))  # the baseline is behind it

    assert court.net_tape_zone(behind, SIZE, 2.0) is None


# --------------------------------------------------------------------------
# Finding the court
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detection(court_image):
    return court.detect_court(court_image)


def _worst_keypoint_error(found, camera):
    truth = ground_homography(camera)
    return max(
        math.dist(kp.image, court.project(truth, np.array(kp.court))) for kp in found.keypoints
    )


def test_every_line_of_a_clear_court_is_found_and_named(detection):
    assert detection.ok and detection.reason is None
    assert detection.template_kind == "doubles"
    assert {line.name for line in detection.lines} == {
        line.name for line in court.court_template().lines
    }
    assert detection.rejected == {}
    assert detection.support > 0.95
    assert detection.runner_up < 0.9
    assert detection.image_size == SIZE


def test_keypoints_land_where_the_camera_put_them(detection, camera):
    assert len(detection.keypoints) == 16
    assert _worst_keypoint_error(detection, camera) < 2.0
    outside = [kp.name for kp in detection.keypoints if not kp.in_frame]
    assert set(outside) <= {"near_baseline:left_doubles", "near_baseline:right_doubles"}


def test_the_working_homography_matches_the_camera(detection, camera):
    truth = ground_homography(camera)
    marks = np.array(list(court.court_template().keypoints.values()))

    placed = court.project(detection.homography, marks)
    error = np.linalg.norm(placed - court.project(truth, marks), axis=1)

    assert error.max() < 2.5


def test_a_detected_line_can_be_looked_up(detection):
    baseline = detection.line("near_baseline")

    assert baseline is not None and baseline.coverage > 0.9
    assert baseline.thickness_px < 8
    assert detection.line("tramline") is None


def test_a_court_is_found_despite_noise_players_and_an_off_centre_camera():
    camera = make_camera(lateral=1.5, roll_deg=1.5)
    image = render_court(
        camera, noise=7.0, occluders=[(380, 300, 440, 470), (500, 130, 520, 170)], seed=3
    )

    found = court.detect_court(image)

    assert found.ok
    assert len(found.lines) >= 8
    assert _worst_keypoint_error(found, camera) < 3.0


def test_a_singles_court_is_recognised_as_one():
    camera = make_camera()

    found = court.detect_court(render_court(camera, kind="singles"))

    assert found.ok and found.template_kind == "singles"
    assert not any("doubles" in line.name for line in found.lines)
    assert _worst_keypoint_error(found, camera) < 2.0


def test_the_net_tape_is_not_accepted_as_the_far_baseline():
    # From 2.47m up, 5m back, the top of the net lines up with the far
    # baseline - as on the real recording this was built against.
    camera = make_camera(height=2.47, back=5.0, aim_y=6.0)

    found = court.detect_court(render_court(camera))

    assert found.ok
    assert found.line("far_baseline") is None
    assert "behind the net tape" in found.rejected["far_baseline"]
    assert found.line("far_service") is not None
    assert _worst_keypoint_error(found, camera) < 3.5


def test_detection_is_deterministic(court_image, detection):
    again = court.detect_court(court_image)

    assert np.array_equal(again.homography, detection.homography)
    assert again.keypoints == detection.keypoints


# --------------------------------------------------------------------------
# Abstaining
# --------------------------------------------------------------------------


def test_a_blank_frame_has_no_court():
    found = court.detect_court(_canvas())

    assert not found.ok
    assert "No straight painted lines" in found.reason
    assert found.keypoints == () and found.homography is None


def test_pure_noise_has_no_court():
    noise = np.random.default_rng(1).integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)

    assert not court.detect_court(noise).ok


def test_a_camera_too_low_to_separate_the_lines_abstains():
    image = render_court(make_camera(height=0.45, back=5.0, aim_y=6.0))

    found = court.detect_court(image)

    assert not found.ok
    assert "camera" in found.reason


def test_lines_across_with_no_sidelines_abstain():
    image = _canvas()
    for row in (200, 300, 420):
        cv2.line(image, (60, row), (900, row + 4), (245, 245, 245), 3)

    found = court.detect_court(image)

    assert "two sidelines" in found.reason


def test_sidelines_with_nothing_across_abstain():
    image = _canvas()
    cv2.line(image, (100, 520), (400, 200), (245, 245, 245), 3)
    cv2.line(image, (860, 520), (560, 200), (245, 245, 245), 3)

    found = court.detect_court(image)

    assert "across the court" in found.reason


def test_four_bare_lines_could_be_any_four_lines_so_nothing_is_claimed():
    image = _canvas()
    cv2.line(image, (100, 520), (400, 200), (245, 245, 245), 3)
    cv2.line(image, (860, 520), (560, 200), (245, 245, 245), 3)
    cv2.line(image, (60, 480), (900, 480), (245, 245, 245), 3)
    cv2.line(image, (300, 260), (660, 260), (245, 245, 245), 3)

    found = court.detect_court(image)

    assert not found.ok
    assert found.keypoints == ()


def test_lines_that_fit_no_court_abstain():
    image = _canvas()
    cv2.line(image, (100, 520), (400, 200), (245, 245, 245), 3)
    cv2.line(image, (860, 520), (560, 200), (245, 245, 245), 3)
    cv2.line(image, (60, 480), (900, 480), (245, 245, 245), 3)
    cv2.line(image, (60, 470), (900, 470), (245, 245, 245), 3)  # ten pixels from the other

    found = court.detect_court(image)

    assert not found.ok


def test_stricter_settings_turn_a_detection_into_an_abstention(court_image):
    unsupported = court.detect_court(court_image, CourtConfig(min_support=1.01))
    unconfirmed = court.detect_court(court_image, CourtConfig(min_line_coverage=1.5))
    undecided = court.detect_court(court_image, CourtConfig(max_runner_up=0.1))

    assert "too little to trust" in unsupported.reason
    assert "could be confirmed" in unconfirmed.reason
    assert "almost equally well" in undecided.reason


# --------------------------------------------------------------------------
# The search and refit steps, one at a time
# --------------------------------------------------------------------------


def _families(transverse, longitudinal):
    return court.LineFamilies(tuple(transverse), tuple(longitudinal), None)


_ACROSS = [court.ImageLine.through((0, y), (960, y)) for y in (480.0, 300.0)]
_ALONG = [
    court.ImageLine.through((100, 520), (400, 200)),
    court.ImageLine.through((860, 520), (560, 200)),
]


def test_candidates_need_two_lines_of_each_family():
    template = court.court_template()

    assert court._candidates(_families(_ACROSS[:1], _ALONG), template, SIZE).shape == (0, 3, 3)
    assert len(court._candidates(_families(_ACROSS, _ALONG), template, SIZE)) == 6 * 10


def test_candidates_skip_lines_that_never_cross_or_cross_the_wrong_way():
    template = court.court_template()
    sideways = [court.ImageLine.through((0, y), (960, y)) for y in (100.0, 200.0)]
    crossed = [  # these two swap sides between the near and the far line
        court.ImageLine.through((100, 520), (860, 200)),
        court.ImageLine.through((860, 520), (100, 200)),
    ]

    assert len(court._candidates(_families(_ACROSS, sideways), template, SIZE)) == 0
    assert len(court._candidates(_families(_ACROSS, crossed), template, SIZE)) == 0


def test_a_search_with_no_candidates_or_no_evidence_finds_nothing():
    template = court.court_template()
    nothing = np.zeros((2, SIZE[1], SIZE[0]), dtype=bool)
    config = CourtConfig()

    assert court._best_fit(_families(_ACROSS[:1], _ALONG), template, nothing, config) is None
    assert court._best_fit(_families(_ACROSS, _ALONG), template, nothing, config) is None


def test_clipping_a_segment_to_the_frame():
    inside = court._clip_to_frame(np.array([-50.0, 100.0]), np.array([50.0, 100.0]), 200, 200)

    assert inside[0] == pytest.approx((0.0, 100.0)) and inside[1] == pytest.approx((50.0, 100.0))
    above = court._clip_to_frame(np.array([0.0, -5.0]), np.array([100.0, -5.0]), 200, 200)
    beside = court._clip_to_frame(np.array([300.0, 0.0]), np.array([400.0, 100.0]), 200, 200)
    assert above is None and beside is None


def _refit(line_name, homography, pixels, **settings):
    line = court.court_template().line(line_name)
    return court._refit_line(
        line, homography, np.linalg.inv(homography), pixels, SIZE, CourtConfig(**settings)
    )


def test_a_template_line_is_refitted_to_the_pixels_near_it(camera):
    truth = ground_homography(camera)
    left, right = court.project(truth, np.array([[-5.485, 0.0], [5.485, 0.0]]))
    columns = np.arange(40.0, 900.0)
    rows = left[1] + (columns - left[0]) * (right[1] - left[1]) / (right[0] - left[0]) + 2.0
    pixels = np.stack([columns, rows], axis=1)

    found = _refit("near_baseline", truth, pixels)

    assert isinstance(found, court.DetectedLine)
    assert float(np.abs(found.line.distance(pixels)).max()) < 0.2  # two pixels off was corrected
    assert found.coverage > 0.85


def test_a_line_is_refused_with_a_reason(camera):
    truth = ground_homography(camera)
    none = np.zeros((0, 2))
    left, right = court.project(truth, np.array([[-5.485, 0.0], [5.485, 0.0]]))
    row = (left[1] + right[1]) / 2
    patchy = np.stack([np.arange(300.0, 420.0), np.full(120, row)], axis=1)
    fat = np.array([(x, row + dy) for x in range(0, 960) for dy in (-4, -3, -2, 0, 2, 3, 4)], float)
    horizon_low = truth.copy()
    horizon_low[2] = [0.0, -0.2, 1.0]  # the far half of the court is behind this camera
    shifted = np.array([[1.0, 0, 5000], [0, 1, 0], [0, 0, 1]]) @ truth

    assert "no line pixels" in _refit("near_baseline", truth, none)
    assert "of its visible length" in _refit("near_baseline", truth, patchy)
    lift = court.project(truth, np.array([0, 23.77]))[1] - row
    assert "too thick" in _refit("far_baseline", truth, fat + [0, lift])
    assert _refit("far_baseline", horizon_low, none) == "projects behind the camera"
    assert _refit("near_baseline", shifted, none) == "outside the frame"


def test_keypoints_come_only_from_lines_that_cross_near_the_frame():
    template = court.court_template()

    def located(name, start, end):
        return court.DetectedLine(name, court.ImageLine.through(start, end), 1.0, 2.0)

    found = {
        "near_baseline": located("near_baseline", (0, 480), (960, 480)),
        "near_service": located("near_service", (0, 300), (960, 300)),
        "left_singles": located("left_singles", (100, 520), (400, 200)),
        "center": located("center", (0, 100), (960, 100)),  # never crosses the baselines
        "right_singles": located("right_singles", (5000, 520), (5000, 200)),  # far off-frame
        "left_doubles": located("left_doubles", (200, 520), (200, 200)),
        "far_baseline": located("far_baseline", (0, -3000), (960, -3000)),  # far above it
    }

    keypoints = court._keypoints(found, template, SIZE, margin=0.35)

    assert [kp.name for kp in keypoints] == [
        "near_baseline:left_doubles",
        "near_baseline:left_singles",
        "near_service:left_singles",
    ]
    assert all(kp.in_frame for kp in keypoints)


def test_refinement_stops_if_the_lines_stop_fixing_a_court(court_image, monkeypatch):
    monkeypatch.setattr(court.cv2, "findHomography", lambda *args, **kwargs: (None, None))

    found = court.detect_court(court_image)

    assert found.ok  # the search's own fit stands, refitted once
    assert len(found.lines) >= 8

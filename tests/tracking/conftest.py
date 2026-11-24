"""Shared fixtures for the tracker's tests: synthetic courts and tiny videos.

The court tests' backbone is `render_court`: a regulation court drawn
through a known pinhole camera, so the homography a test recovers can be
compared against the one the picture was made with. Videos are tiny real
clips built with ffmpeg's lavfi sources, as in
`tests/swingvision_import/test_footage.py`.
"""

from __future__ import annotations

import subprocess

import cv2
import numpy as np
import pytest

from swingvision_import import footage
from tracking import court

SIZE = (960, 540)


def make_camera(
    size=SIZE,
    focal=0.68,
    height=3.0,
    back=6.0,
    lateral=0.0,
    aim_y=9.0,
    roll_deg=0.0,
):
    """Builds a 3x4 pinhole projection for a camera behind the near baseline.

    Args:
        size: (width, height) of the image.
        focal: Focal length as a fraction of image width.
        height: Camera height above the court, metres.
        back: Distance behind the near baseline, metres.
        lateral: Sideways offset from the centre line, metres.
        aim_y: The court y the camera is aimed at (on the centre line).
        roll_deg: Roll about the optical axis.

    Returns:
        The projection matrix taking (x, y, z, 1) court metres to pixels.
    """
    width, tall = size
    centre = np.array([lateral, -back, height], dtype=float)
    forward = np.array([0.0, aim_y, 0.0]) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    roll = np.radians(roll_deg)
    right, down = (
        np.cos(roll) * right + np.sin(roll) * down,
        -np.sin(roll) * right + np.cos(roll) * down,
    )
    rotation = np.stack([right, down, forward])
    intrinsics = np.array(
        [[focal * width, 0, width / 2], [0, focal * width, tall / 2], [0, 0, 1]], dtype=float
    )
    return intrinsics @ np.hstack([rotation, (-rotation @ centre)[:, None]])


def ground_homography(camera):
    """Returns the court-plane (z = 0) homography of a projection matrix."""
    homography = camera[:, [0, 1, 3]]
    return homography / homography[2, 2]


def _pixels(camera, points):
    """Projects (N, 3) court points to (N, 2) integer-ready pixels."""
    lifted = np.hstack([np.asarray(points, dtype=float), np.ones((len(points), 1))])
    mapped = lifted @ camera.T
    return mapped[:, :2] / mapped[:, 2:]


def _fill(image, camera, corners, colour):
    """Fills a court-space polygon, subdividing edges so perspective is honoured."""
    dense = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        dense.extend(np.linspace(start, end, 12, endpoint=False))
    mapped = _pixels(camera, dense)
    cv2.fillPoly(image, [np.round(mapped * 16).astype(np.int32)], colour, cv2.LINE_AA, shift=4)


def render_court(
    camera,
    size=SIZE,
    kind="doubles",
    net=True,
    noise=3.0,
    occluders=(),
    paint=0.06,
    seed=0,
):
    """Draws a court as the given camera sees it.

    Args:
        camera: 3x4 projection from `make_camera`.
        size: (width, height) of the image.
        kind: "doubles" or "singles".
        net: Draw a net - a dim mesh with a white tape along its top.
        noise: Standard deviation of the pixel noise added.
        occluders: (x0, y0, x1, y1) image rectangles painted skin-coloured,
            standing in for players.
        paint: Width of the painted lines in metres.
        seed: Seed for the noise.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    width, tall = size
    template = court.court_template(kind)
    image = np.zeros((tall, width, 3), dtype=np.uint8)
    image[:] = (70, 120, 90)
    half = template.half_width
    _fill(
        image,
        camera,
        [(-half, 0, 0), (half, 0, 0), (half, 23.77, 0), (-half, 23.77, 0)],
        (40, 80, 175),
    )
    for line in template.lines:
        (x0, y0), (x1, y1) = line.start, line.end
        grow = paint / 2
        if line.family == court.TRANSVERSE:
            quad = [(x0, y0 - grow, 0), (x1, y0 - grow, 0), (x1, y0 + grow, 0), (x0, y0 + grow, 0)]
        else:
            quad = [(x0 - grow, y0, 0), (x0 + grow, y0, 0), (x0 + grow, y1, 0), (x0 - grow, y1, 0)]
        _fill(image, camera, quad, (245, 245, 245))
    if net:
        y = court.NET_Y_M
        mesh = image.copy()
        _fill(
            mesh, camera, [(-6.4, y, 0), (6.4, y, 0), (6.4, y, 1.0), (-6.4, y, 1.0)], (20, 20, 20)
        )
        image = cv2.addWeighted(image, 0.65, mesh, 0.35, 0)
        _fill(
            image,
            camera,
            [(-6.4, y, 0.97), (6.4, y, 0.97), (6.4, y, 1.07), (-6.4, y, 1.07)],
            (240, 240, 240),
        )
    for x0, y0, x1, y1 in occluders:
        cv2.rectangle(image, (x0, y0), (x1, y1), (190, 140, 110), -1)
    if noise:
        rng = np.random.default_rng(seed)
        image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
    return image


@pytest.fixture(scope="session")
def camera():
    """The standard test camera: 3m up, 6m behind the baseline."""
    return make_camera()


@pytest.fixture(scope="session")
def court_image(camera):
    """A doubles court with a net, seen by the standard camera."""
    return render_court(camera)


def _lavfi(tmp_path_factory, name, source, *extra):
    """Encodes a tiny clip from a lavfi source and returns its path."""
    path = tmp_path_factory.mktemp("video") / name
    subprocess.run(
        [
            footage.ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            *extra,
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def steady_video(tmp_path_factory):
    """Three seconds at a constant 30fps, 320x180."""
    return _lavfi(tmp_path_factory, "steady.mp4", "testsrc=size=320x180:rate=30:duration=3")


@pytest.fixture(scope="session")
def vfr_video(tmp_path_factory):
    """Two seconds of 30fps with every fifth frame dropped: irregular timing."""
    return _lavfi(
        tmp_path_factory,
        "vfr.mp4",
        "testsrc=size=320x180:rate=30:duration=2",
        "-vf",
        "select='not(eq(mod(n,5),3))'",
        "-fps_mode",
        "vfr",
    )


@pytest.fixture(scope="session")
def offset_video(tmp_path_factory):
    """Three seconds at 30fps whose timestamps start at 5s, not 0."""
    return _lavfi(
        tmp_path_factory,
        "offset.mp4",
        "testsrc=size=320x180:rate=30:duration=3",
        "-output_ts_offset",
        "5",
    )


@pytest.fixture(scope="session")
def rotated_video(tmp_path_factory, steady_video):
    """The steady clip, flagged for a player to rotate a quarter turn."""
    path = tmp_path_factory.mktemp("video") / "rotated.mp4"
    subprocess.run(
        [
            footage.ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(steady_video),
            "-c",
            "copy",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path

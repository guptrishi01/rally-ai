"""Tests for reading a recording: header, timestamp index, frames, checksum.

Every video here is a real clip encoded by ffmpeg, so decoding is tested
against ffmpeg rather than a mock of it.
"""

import hashlib
import io
import queue
from fractions import Fraction

import numpy as np
import pytest

from tracking import video
from tracking.config import VideoConfig

# --------------------------------------------------------------------------
# The header
# --------------------------------------------------------------------------


def test_probe_reads_size_rate_and_duration(steady_video):
    info = video.probe(steady_video)

    assert (info.width, info.height) == (320, 180)
    assert info.rotation == 0
    assert info.codec == "h264"
    assert info.nominal_fps == pytest.approx(30.0)
    assert info.average_fps == pytest.approx(30.0)
    assert info.duration == pytest.approx(3.0, abs=0.1)
    assert info.start_time == pytest.approx(0.0)
    assert info.frame_count is None and info.is_vfr is None


def test_probe_reports_rotation_and_the_displayed_size(rotated_video):
    info = video.probe(rotated_video)

    assert info.rotation == 270
    assert (info.width, info.height) == (180, 320)


def test_decoded_frames_come_out_display_oriented(rotated_video):
    frame = next(video.iter_frames(rotated_video))

    assert frame.image.shape == (320, 180, 3)


def test_probe_reads_a_start_offset(offset_video):
    assert video.probe(offset_video).start_time == pytest.approx(5.0)


def test_a_missing_file_is_reported_as_unreadable(tmp_path):
    missing = tmp_path / "nope.mp4"
    for call in (
        video.probe,
        video.build_pts_index,
        video.file_checksum,
        video.load_video,
        lambda path: list(video.iter_frames(path)),
    ):
        with pytest.raises(video.VideoUnreadableError, match="nope.mp4"):
            call(missing)


def test_a_file_that_is_not_a_video_is_reported_as_unreadable(tmp_path):
    junk = tmp_path / "notes.mp4"
    junk.write_text("this is not a video")

    with pytest.raises(video.VideoUnreadableError):
        video.probe(junk)
    with pytest.raises(video.VideoUnreadableError):
        video.build_pts_index(junk)
    with pytest.raises(video.VideoUnreadableError, match="could not decode"):
        list(video.iter_frames(junk))


def _fake_ffmpeg(monkeypatch, stderr="", stdout="", returncode=0):
    """Makes every ffmpeg call return canned output."""

    class Result:
        pass

    def run(*args, **kwargs):
        result = Result()
        result.stderr, result.stdout, result.returncode = stderr, stdout, returncode
        return result

    monkeypatch.setattr(video.subprocess, "run", run)


def test_an_audio_only_file_has_no_video_stream(monkeypatch, steady_video):
    _fake_ffmpeg(
        monkeypatch,
        stderr="  Duration: 00:00:03.00, start: 0.000000\n  Stream #0:0: Audio: aac, 48000 Hz\n",
    )

    with pytest.raises(video.VideoUnreadableError, match="no video stream"):
        video.probe(steady_video)


def test_a_stream_declaring_no_frame_rate_is_refused(monkeypatch, steady_video):
    _fake_ffmpeg(
        monkeypatch,
        stderr="  Duration: 00:00:03.00\n  Stream #0:0: Video: h264, 320x180, 9 tbn\n",
    )

    with pytest.raises(video.VideoUnreadableError, match="no frame rate"):
        video.probe(steady_video)


def test_probe_reads_an_old_style_rotate_tag_and_a_thousand_tbr(monkeypatch, steady_video):
    _fake_ffmpeg(
        monkeypatch,
        stderr=(
            "  Duration: 00:00:03.00, bitrate: 5 kb/s\n"
            "  Stream #0:0(und): Video: hevc (Main), yuv420p, 1920x1080, 1k tbr, 600 tbn\n"
            "    Metadata:\n"
            "      rotate          : 90\n"
        ),
    )

    info = video.probe(steady_video)

    assert info.rotation == 90
    assert (info.width, info.height) == (1080, 1920)
    assert info.nominal_fps == pytest.approx(1000.0)
    assert info.average_fps is None
    assert info.start_time == 0.0


# --------------------------------------------------------------------------
# The timestamp index
# --------------------------------------------------------------------------


def test_the_index_holds_every_frame_in_display_order(steady_video):
    index = video.build_pts_index(steady_video)

    assert len(index) == 90
    assert index.pts_of(0) == pytest.approx(0.0)
    assert index.pts_of(89) == pytest.approx(89 / 30)
    assert (np.diff(index.pts) > 0).all()  # sorted even though the codec reorders
    assert index.time_base == Fraction(1, 15360)


def test_pts_and_frame_number_round_trip(vfr_video):
    index = video.build_pts_index(vfr_video)

    for number in range(len(index)):
        assert index.frame_at(index.pts_of(number)) == number
        assert index.frame_on_screen(index.pts_of(number)) == number


def test_frame_at_picks_the_nearest_frame_and_clamps():
    index = video.PtsIndex([0.0, 0.1, 0.5])

    assert index.frame_at(0.04) == 0
    assert index.frame_at(0.06) == 1
    assert index.frame_at(0.31) == 2
    assert index.frame_at(-3.0) == 0
    assert index.frame_at(99.0) == 2


def test_frame_on_screen_is_the_last_frame_not_after_the_moment():
    index = video.PtsIndex([0.0, 0.1, 0.5])

    assert index.frame_on_screen(0.49) == 1
    assert index.frame_on_screen(0.5) == 2
    assert index.frame_on_screen(-1.0) == 0


def test_frames_between_is_a_closed_range():
    index = video.PtsIndex([0.0, 0.1, 0.2, 0.3])

    assert list(index.frames_between(0.1, 0.2)) == [1, 2]
    assert list(index.frames_between(0.11, 0.19)) == []
    assert list(index.frames_between(5.0, 1.0)) == []


def test_the_index_is_read_only_and_refuses_bad_lookups():
    index = video.PtsIndex([0.2, 0.0, 0.1])

    assert index.pts.tolist() == [0.0, 0.1, 0.2]
    with pytest.raises(ValueError):
        index.pts[0] = 9.0
    with pytest.raises(IndexError):
        index.pts_of(3)
    with pytest.raises(ValueError):
        video.PtsIndex([])


def test_the_index_refuses_frames_without_timestamps(monkeypatch, steady_video):
    _fake_ffmpeg(
        monkeypatch, stdout="#tb 0: 1/600\n0,          0,      NOPTS,       10,   5, 0x1\n"
    )
    with pytest.raises(video.VideoUnreadableError, match="no presentation timestamp"):
        video.build_pts_index(steady_video)

    _fake_ffmpeg(monkeypatch, stdout="#tb 0: 1/600\n\n")
    with pytest.raises(video.VideoUnreadableError, match="no frames"):
        video.build_pts_index(steady_video)


# --------------------------------------------------------------------------
# Irregular frame timing
# --------------------------------------------------------------------------


def test_a_steady_clip_is_not_variable_frame_rate(steady_video):
    stats = video.build_pts_index(steady_video).stats()

    assert not stats.is_vfr
    assert stats.irregular == 0
    assert stats.median_gap == pytest.approx(1 / 30)
    assert stats.effective_fps == pytest.approx(30.0)


def test_dropped_frames_are_measured(vfr_video):
    stats = video.build_pts_index(vfr_video).stats()

    assert stats.is_vfr
    assert stats.frames == 48
    assert stats.irregular == 12
    assert stats.max_gap == pytest.approx(2 / 30)
    assert stats.irregular_fraction == pytest.approx(12 / 47)
    assert stats.effective_fps < 30


def test_time_base_rounding_is_not_mistaken_for_dropped_frames():
    ticks = np.cumsum([10, 9, 11, 10, 10, 9, 11]) / 600

    assert not video.PtsIndex(ticks).stats(jitter=0.15).is_vfr
    assert video.PtsIndex(ticks).stats(jitter=0.05).is_vfr


def test_a_single_frame_has_no_gaps():
    stats = video.PtsIndex([1.0]).stats()

    assert (stats.frames, stats.irregular, stats.effective_fps) == (1, 0, 0.0)


def test_load_video_flags_irregular_timing_but_accepts_it(vfr_video, caplog):
    with caplog.at_level("WARNING", logger="tracking.video"):
        source = video.load_video(vfr_video)

    assert source.info.is_vfr is True
    assert source.info.frame_count == 48
    assert "irregular" in caplog.text


def test_load_video_rejects_irregular_timing_when_told_to(vfr_video):
    with pytest.raises(video.VariableFrameRateError, match="12 of 47 frame gaps"):
        video.load_video(vfr_video, VideoConfig(reject_vfr=True))


def test_load_video_bundles_header_index_and_checksum(steady_video):
    source = video.load_video(steady_video, VideoConfig(reject_vfr=True))

    assert source.info.frame_count == len(source.index) == 90
    assert source.info.is_vfr is False
    assert source.checksum == hashlib.sha256(steady_video.read_bytes()).hexdigest()
    assert video.load_video(steady_video, checksum=False).checksum is None


def test_the_checksum_is_the_same_whatever_the_chunk_size(steady_video):
    small = video.file_checksum(steady_video, VideoConfig(checksum_chunk_bytes=1000))

    assert small == video.file_checksum(steady_video)


# --------------------------------------------------------------------------
# Frames with their timestamps
# --------------------------------------------------------------------------


def test_every_frame_is_yielded_with_its_indexed_timestamp(vfr_video):
    index = video.build_pts_index(vfr_video)

    frames = list(video.iter_frames(vfr_video, index=index))

    assert [frame.index for frame in frames] == list(range(len(index)))
    assert [frame.pts for frame in frames] == pytest.approx(index.pts.tolist(), abs=1e-9)
    assert frames[0].image.shape == (180, 320, 3)
    assert frames[0].image.dtype == np.uint8


def test_without_an_index_frames_carry_no_frame_number(steady_video):
    frame = next(video.iter_frames(steady_video))

    assert frame.index is None
    assert frame.pts == pytest.approx(0.0)


def test_a_time_range_is_exact_at_both_ends(steady_video):
    index = video.build_pts_index(steady_video)

    frames = list(video.iter_frames(steady_video, start=2.0, end=2.2, index=index))

    assert [frame.index for frame in frames] == [60, 61, 62, 63, 64, 65, 66]
    assert frames[0].pts == pytest.approx(2.0)


def test_a_range_seeks_correctly_when_timestamps_do_not_start_at_zero(offset_video):
    index = video.build_pts_index(offset_video)
    assert index.pts_of(0) == pytest.approx(5.0)

    frames = list(video.iter_frames(offset_video, start=7.0, end=7.1, index=index))

    assert [frame.index for frame in frames] == [60, 61, 62, 63]
    assert frames[0].pts == pytest.approx(7.0)


def test_stride_keeps_every_nth_frame_from_the_first_in_range(steady_video):
    index = video.build_pts_index(steady_video)

    frames = list(video.iter_frames(steady_video, start=1.0, end=1.5, stride=5, index=index))

    assert [frame.index for frame in frames] == [30, 35, 40, 45]


def test_frames_can_be_scaled(steady_video):
    frame = next(video.iter_frames(steady_video, width=160))

    assert frame.image.shape == (90, 160, 3)


def test_the_frames_differ_from_one_another(steady_video):
    first, second = list(video.iter_frames(steady_video, end=0.04))[:2]

    assert not np.array_equal(first.image, second.image)


def test_bad_ranges_are_refused(steady_video):
    with pytest.raises(ValueError, match="stride"):
        list(video.iter_frames(steady_video, stride=0))
    with pytest.raises(ValueError, match="before start"):
        list(video.iter_frames(steady_video, start=2.0, end=1.0))


def test_stopping_early_ends_the_decode_cleanly(steady_video):
    frames = video.iter_frames(steady_video)
    next(frames)
    frames.close()  # must not hang or leak the ffmpeg process


def test_frame_at_returns_the_first_frame_at_or_after_a_moment(vfr_video):
    index = video.build_pts_index(vfr_video)
    moment = index.pts_of(10) + 0.001

    frame = video.frame_at(vfr_video, moment)

    assert frame.pts == pytest.approx(index.pts_of(11))


def test_frame_at_past_the_end_is_an_error(steady_video):
    with pytest.raises(video.VideoUnreadableError, match="no frame at"):
        video.frame_at(steady_video, 60.0)


def test_frames_at_decodes_one_frame_per_moment(steady_video):
    frames = video.frames_at(steady_video, [0.5, 2.0, 1.0], width=160)

    assert [frame.pts for frame in frames] == pytest.approx([0.5, 2.0, 1.0])
    assert all(frame.image.shape == (90, 160, 3) for frame in frames)


def test_a_frame_without_a_timestamp_is_an_error_not_a_guess(monkeypatch, steady_video):
    def silent(stream, stamps, log):
        stamps.put(None)
        stream.read()

    monkeypatch.setattr(video, "_drain_timestamps", silent)

    with pytest.raises(video.VideoUnreadableError, match="no timestamp"):
        list(video.iter_frames(steady_video))


def test_a_timestamp_that_never_arrives_is_an_error_not_a_hang(monkeypatch, steady_video):
    monkeypatch.setattr(video, "_drain_timestamps", lambda stream, stamps, log: stream.read())

    with pytest.raises(video.VideoUnreadableError, match="no timestamp"):
        list(video.iter_frames(steady_video, config=VideoConfig(decode_timeout_seconds=0.2)))


def test_timestamps_are_exact_ticks_once_the_time_base_is_known():
    stamps: queue.Queue = queue.Queue()
    log: list[str] = []
    lines = (
        b"[Parsed_showinfo_0 @ 0x1] n:   0 pts:     30 pts_time:0.05 duration: 10\n"
        b"[Parsed_showinfo_0 @ 0x1] config in time_base: 1/600, frame_rate: 60/1\n"
        b"[Parsed_showinfo_0 @ 0x1] n:   1 pts:     40 pts_time:0.0666667 duration: 10\n"
        b"[Parsed_showinfo_0 @ 0x1] color_range:unknown\n"
        b"\n"
        b"Something went wrong\n"
    )

    video._drain_timestamps(io.BytesIO(lines), stamps, log)

    assert stamps.get() == pytest.approx(0.05)  # no time base yet: the printed time
    assert stamps.get() == 40 / 600  # exact, from ticks, not the rounded text
    assert stamps.get() is None
    assert log == ["Something went wrong"]

from __future__ import annotations

import json
import subprocess

import anthropic
import httpx
import pytest

from swingvision_import import footage
from swingvision_import.raw import RawShotRow
from swingvision_import.records import PointRecord
from tests.ai.conftest import FakeMessage, FakeTextBlock

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _verdict_json(**overrides) -> str:
    payload = {
        "footage_usable": True,
        "server_seen": "tracked",
        "winner": "tracked",
        "point_end_type": "winner",
        "tracked_at_net": "unclear",
        "confidence": "high",
        "reasoning": "The opponent never reached the ball.",
    }
    payload.update(overrides)
    return json.dumps(payload)


class _FakeUsage:
    def __init__(self, input_tokens: int = 12000, output_tokens: int = 900):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeBetaMessages:
    """Stands in for client.beta.messages, which is what footage calls."""

    def __init__(self, response_text: str | None = None, errors: list[Exception] | None = None,
                 stop_reason: str = "end_turn"):
        self.response_text = response_text if response_text is not None else _verdict_json()
        self.errors = list(errors or [])
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        message = FakeMessage(content=[FakeTextBlock(text=self.response_text)])
        message.usage = _FakeUsage()
        message.stop_reason = self.stop_reason
        return message


class _FakeBeta:
    def __init__(self, messages):
        self.messages = messages


class _FakeClient:
    def __init__(self, **kwargs):
        self.messages = _FakeBetaMessages(**kwargs)
        self.beta = _FakeBeta(self.messages)


def _window(**overrides) -> footage.PointWindow:
    base = dict(
        part=0,
        first_shot=10.0,
        prev_shot=13.0,
        last_shot=14.0,
        tracked_serving=True,
        tracked_side="near",
        shot_count=4,
    )
    base.update(overrides)
    return footage.PointWindow(**base)


def _frames(n: int = 3) -> list[footage.Frame]:
    return [footage.Frame("finish", -0.5 + i * 0.4, b"\xff\xd8fake-jpeg") for i in range(n)]


def _point(**overrides) -> PointRecord:
    base = dict(
        game_number=1,
        point_number=1,
        is_serving=True,
        point_won=True,
        point_end_type="winner",
        net_approach=False,
        needs_review=True,
        source_point_number=1,
    )
    base.update(overrides)
    return PointRecord(**base)


# ---------------------------------------------------------------------------
# Alignment - the guard that makes the whole feature trustworthy
# ---------------------------------------------------------------------------


def test_condensed_footage_is_refused_before_anything_is_spent():
    """SwingVision's rallies-only export is ~1/5 the length of the recording
    its Video Time values refer to, so seeking into it lands on the wrong
    rally - which would still produce a confident-sounding verdict.

    The message has to name the cause, not just the mismatch: this export is
    what the share sheet produces when the app's rally filter is left on, so
    a refusal that does not say that leaves nothing to act on."""
    with pytest.raises(footage.FootageAlignmentError) as exc_info:
        footage.check_alignment({0: 2190.0}, [448.75])

    message = str(exc_info.value)
    assert "2190" in message and "449" in message
    assert "rally filter" in message


def test_full_length_footage_passes_alignment():
    footage.check_alignment({0: 2190.0, 1: 3012.8}, [2223.0, 3100.0])


def test_a_part_with_no_video_is_refused():
    with pytest.raises(footage.FootageAlignmentError, match="no matching video"):
        footage.check_alignment({0: 10.0, 1: 20.0}, [500.0])


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _shot(point, number, player, shot_type, result, video_time, side, part=0) -> RawShotRow:
    return RawShotRow(
        point_number=point,
        shot_number=number,
        player=player,
        shot_type=shot_type,
        stroke="Forehand",
        result=result,
        video_time=video_time,
        hit_side=side,
        source_part=part,
    )


def test_plan_point_locates_the_point_and_who_is_where():
    shots = [
        _shot(1, 1, "Me", "first_serve", "In", 100.0, "near"),
        _shot(1, 2, "Them", "first_return", "In", 101.2, "far"),
        _shot(1, 3, "Me", "in_play", "Out", 102.5, "near"),
    ]

    window = footage.plan_point(shots, "Me")

    assert window.first_shot == 100.0
    assert window.prev_shot == 101.2  # the shot the final one answered
    assert window.last_shot == 102.5
    assert window.tracked_serving is True
    assert window.tracked_side == "near"
    assert window.shot_count == 3


def test_plan_point_infers_our_end_when_only_the_opponent_hit():
    shots = [_shot(1, 1, "Them", "first_serve", "In", 5.0, "far")]

    window = footage.plan_point(shots, "Me")

    assert window.tracked_serving is False
    assert window.tracked_side == "near"  # the other end from the opponent
    assert window.prev_shot is None


def test_plan_point_returns_none_without_timing():
    shots = [_shot(1, 1, "Me", "first_serve", "In", None, None)]

    assert footage.plan_point(shots, "Me") is None


def test_the_part_index_decides_which_recording_a_point_is_in():
    shots = [_shot(94, 1, "Me", "first_serve", "In", 26.8, "near", part=1)]

    assert footage.plan_point(shots, "Me").part == 1


# ---------------------------------------------------------------------------
# Independence of the second opinion
# ---------------------------------------------------------------------------


def test_the_judge_is_never_given_the_recorded_outcome():
    """The whole claim is "two independent sources agreed". If the model
    could see the recorded answer it would be invited to agree, and the
    claim would mean nothing - so the outcome isn't reachable from here at
    all: neither judge_point nor the prompt takes the point."""
    import inspect

    judge_params = set(inspect.signature(footage.judge_point).parameters)
    assert "point" not in judge_params and "record" not in judge_params
    prompt_params = set(inspect.signature(footage._prompt).parameters)
    # An exact set, not a subset check: this is an allowlist, so anything
    # new reaching the prompt has to be argued for here rather than slipping
    # in. player_appearance says which player to look at, never what
    # happened to them, so it cannot carry the outcome.
    assert prompt_params == {"window", "reviewer_note", "player_appearance"}


def test_the_prompt_gives_context_but_asserts_no_outcome():
    client = _FakeClient()

    footage.judge_point(client, footage.FootageConfig(), _window(), _frames())

    sent = client.messages.calls[0]
    text = " ".join(
        block["text"] for block in sent["messages"][0]["content"] if block["type"] == "text"
    )
    # Context it may have: who served, and which end each player is at.
    assert "serves this point" in text
    assert "NEAR end" in text
    # But nothing claiming how the point actually went.
    for leak in ("best guess", "point_won", "was won by", "the record"):
        assert leak not in text.lower()


def test_frames_are_sent_as_images_before_the_instructions():
    client = _FakeClient()

    footage.judge_point(client, footage.FootageConfig(), _window(), _frames(4))

    content = client.messages.calls[0]["messages"][0]["content"]
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 4
    assert images[0]["source"]["media_type"] == "image/jpeg"
    assert content[-1]["type"] == "text"  # instructions come last


def test_a_verdict_is_parsed_with_its_token_usage():
    client = _FakeClient(response_text=_verdict_json(confidence="low", winner="opponent"))

    verdict = footage.judge_point(client, footage.FootageConfig(), _window(), _frames())

    assert verdict.winner == "opponent"
    assert verdict.confidence == "low"
    assert verdict.input_tokens == 12000
    assert verdict.output_tokens == 900


def test_a_refusal_becomes_an_unusable_verdict_not_an_exception():
    """A refused point just goes to a human - it shouldn't abort the run."""
    client = _FakeClient(stop_reason="refusal")

    verdict = footage.judge_point(client, footage.FootageConfig(), _window(), _frames())

    assert verdict.refused is True
    assert verdict.usable is False


def _api_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    if status == 429:
        return anthropic.RateLimitError(message="slow down", response=response, body=None)
    return anthropic.InternalServerError(message="boom", response=response, body=None)


def test_a_transient_failure_is_retried():
    client = _FakeClient(errors=[_api_error(429), _api_error(500)])
    config = footage.FootageConfig(max_attempts=3, retry_base_delay=0.0)

    verdict = footage.judge_point(client, config, _window(), _frames())

    assert verdict.usable is True
    assert len(client.messages.calls) == 3


def test_running_out_of_attempts_raises():
    client = _FakeClient(errors=[_api_error(429), _api_error(429)])
    config = footage.FootageConfig(max_attempts=2, retry_base_delay=0.0)

    with pytest.raises(footage.FootageError):
        footage.judge_point(client, config, _window(), _frames())


def test_rejected_fallbacks_are_dropped_and_the_run_continues():
    """If the API won't take the refusal-fallback parameter, that must not
    cost every remaining point in the run."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    rejection = anthropic.BadRequestError(
        message="fallbacks: unsupported", response=httpx.Response(400, request=request), body=None
    )
    client = _FakeClient(errors=[rejection])
    switch = footage.FallbackSwitch(True)

    verdict = footage.judge_point(
        client, footage.FootageConfig(), _window(), _frames(), None, switch
    )

    assert verdict.usable is True
    assert switch.enabled is False
    assert "fallbacks" in client.messages.calls[0]
    assert "fallbacks" not in client.messages.calls[1]  # retried without it


def test_an_unreadable_verdict_raises():
    client = _FakeClient(response_text="not json")

    with pytest.raises(footage.FootageError):
        footage.judge_point(client, footage.FootageConfig(), _window(), _frames())


# ---------------------------------------------------------------------------
# When footage counts as corroboration
# ---------------------------------------------------------------------------


def _verdict(**overrides) -> footage.FootageVerdict:
    base = dict(
        usable=True,
        server_seen="tracked",
        winner="tracked",
        point_end_type="winner",
        tracked_at_net="unclear",
        confidence="high",
        reasoning="clear",
    )
    base.update(overrides)
    return footage.FootageVerdict(**base)


def test_matching_outcome_at_high_confidence_agrees():
    assert footage.compare(_verdict(), _point(), _window()) == "agrees"


def test_a_different_outcome_disagrees():
    verdict = _verdict(winner="opponent", point_end_type="unforced_error")

    assert footage.compare(verdict, _point(), _window()) == "disagrees"


def test_the_same_winner_but_a_different_type_disagrees():
    """reconstruct.py never emits forced_error, so this is exactly the
    judgement call a human should still make."""
    verdict = _verdict(winner="opponent", point_end_type="forced_error")
    point = _point(point_won=False, point_end_type="unforced_error")

    assert footage.compare(verdict, point, _window()) == "disagrees"


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": "medium"},
        {"usable": False},
        {"winner": "unclear"},
        {"point_end_type": "unclear"},
        {"refused": True},
    ],
)
def test_anything_short_of_a_confident_reading_is_inconclusive(overrides):
    assert footage.compare(_verdict(**overrides), _point(), _window()) == "inconclusive"


def test_seeing_a_different_server_is_inconclusive_not_agreement():
    """The strongest per-point signal that the frames are from another
    rally - it must never be allowed to auto-confirm."""
    verdict = _verdict(server_seen="opponent")

    assert footage.compare(verdict, _point(), _window(tracked_serving=True)) == "inconclusive"


def test_a_self_contradicting_verdict_is_inconclusive():
    verdict = _verdict(winner="opponent", point_end_type="winner")

    assert footage.compare(verdict, _point(), _window()) == "inconclusive"


def test_seeing_a_net_approach_that_was_not_recorded_disagrees():
    verdict = _verdict(tracked_at_net="yes")

    assert footage.compare(verdict, _point(net_approach=False), _window()) == "disagrees"


def test_not_seeing_a_net_approach_is_not_evidence_against_one():
    """The frames skip the middle of a rally, so a volley can happen
    off-camera - only a positive sighting contradicts the record."""
    verdict = _verdict(tracked_at_net="no")

    assert footage.compare(verdict, _point(net_approach=True), _window()) == "agrees"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_frame_tokens_follows_the_documented_patch_formula():
    assert footage.frame_tokens(1280, 720) == 46 * 26


def test_the_estimate_scales_with_points_and_names_its_model():
    config = footage.FootageConfig()
    one = footage.estimate_cost(1, config)
    many = footage.estimate_cost(100, config)

    assert many.usd > one.usd
    assert many.model == "claude-opus-5"
    assert one.input_tokens_per_point > config.end_frames * footage.frame_tokens(1280, 720)


def test_a_cheaper_model_estimates_cheaper():
    opus = footage.estimate_cost(50, footage.FootageConfig())
    sonnet = footage.estimate_cost(50, footage.FootageConfig(model="claude-sonnet-5"))

    assert sonnet.usd < opus.usd


# ---------------------------------------------------------------------------
# Frame extraction, against a real video
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_video(tmp_path):
    """A real 6-second video, so extraction is tested against ffmpeg rather
    than a mock of it."""
    exe = footage.ffmpeg_exe()
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=640x360:rate=30:duration=6", "-pix_fmt", "yuv420p", str(path)],
        check=True,
        capture_output=True,
    )
    return path


def test_probe_duration_reads_a_real_file(tiny_video):
    assert footage.probe_duration(tiny_video) == pytest.approx(6.0, abs=0.3)


def test_extract_frames_returns_jpegs_at_the_requested_rate(tiny_video):
    frames = footage.extract_frames(tiny_video, start=1.0, duration=2.0, count=4, width=320)

    assert len(frames) == 4
    assert all(f.startswith(b"\xff\xd8") for f in frames)  # JPEG magic


def test_extracting_past_the_end_returns_what_exists_rather_than_failing(tiny_video):
    frames = footage.extract_frames(tiny_video, start=5.5, duration=3.0, count=6, width=320)

    assert len(frames) < 6


def test_a_missing_video_is_reported_clearly(tmp_path):
    with pytest.raises(footage.FootageError):
        footage.probe_duration(tmp_path / "nope.mp4")


# ---------------------------------------------------------------------------
# Identifying the tracked player by appearance
# ---------------------------------------------------------------------------


def test_the_prompt_describes_the_tracked_player_when_appearance_is_given():
    client = _FakeClient()

    footage.judge_point(
        client,
        footage.FootageConfig(),
        _window(),
        _frames(),
        player_appearance="grey shirt, black shorts",
    )

    sent = client.messages.calls[0]
    text = " ".join(
        block["text"] for block in sent["messages"][0]["content"] if block["type"] == "text"
    )
    assert "grey shirt, black shorts" in text
    assert "tracked_seen_at" in text


def test_appearance_is_absent_from_the_prompt_when_it_was_not_supplied():
    """Nothing is invented: no appearance means no description and no claim
    to report one, rather than a placeholder the model might answer anyway."""
    client = _FakeClient()

    footage.judge_point(client, footage.FootageConfig(), _window(), _frames())

    sent = client.messages.calls[0]
    text = " ".join(
        block["text"] for block in sent["messages"][0]["content"] if block["type"] == "text"
    )
    assert "described as" not in text
    assert "tracked_seen_at" not in text


def test_seeing_the_tracked_player_at_the_other_end_is_inconclusive():
    """Ends swap at changeovers, so the described player turning up at the
    end the sheet did not expect means these frames are of the wrong person.
    That is a reason to stop, not to judge harder."""
    verdict = _verdict(tracked_seen_at="far")

    assert footage.compare(verdict, _point(), _window(tracked_side="near")) == "inconclusive"


def test_seeing_the_tracked_player_at_the_expected_end_still_agrees():
    verdict = _verdict(tracked_seen_at="near")

    assert footage.compare(verdict, _point(), _window(tracked_side="near")) == "agrees"


def test_an_unclear_sighting_does_not_block_agreement():
    """Absence of an identification is not a contradiction - the check only
    fires on a positive sighting at the wrong end."""
    verdict = _verdict(tracked_seen_at="unclear")

    assert footage.compare(verdict, _point(), _window(tracked_side="near")) == "agrees"


# ---------------------------------------------------------------------------
# Content-addressed retrieval for the condensed rallies-only export
# ---------------------------------------------------------------------------


def _retrieval_json(**overrides) -> str:
    payload = {
        "rally_present": True,
        "offset_seconds": 10.0,
        "confidence": "high",
        "reasoning": "Shot sequence matches - serve, return, forehand.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _fingerprint_shot(
    point: int,
    number: int,
    player: str,
    stroke: str,
    video_time: float,
    spin: str = "Flat",
    speed: float = 60.0,
) -> RawShotRow:
    return RawShotRow(
        point_number=point,
        shot_number=number,
        player=player,
        shot_type="",
        stroke=stroke,
        result="In",
        video_time=video_time,
        hit_side=None,
        spin=spin,
        speed_mph=speed,
    )


def test_fingerprint_point_marks_the_tracked_player_and_strips_outcome():
    """The fingerprint carries only sheet fields that identify the shots
    (who hit it, stroke, speed) - never the outcome. The judge call
    forms its outcome independently later, so if "Result" leaked into
    the retrieval call it would stop being a clean independent check."""
    shots = [
        _fingerprint_shot(1, 1, "Me", "Serve", 10.0, speed=70.0),
        _fingerprint_shot(1, 2, "You", "Forehand", 11.0, spin="Topspin", speed=55.0),
        _fingerprint_shot(1, 3, "Me", "Backhand", 12.0, spin="Slice", speed=42.0),
    ]

    prints = footage.fingerprint_point(shots, "Me")

    assert [f.who for f in prints] == ["tracked", "opponent", "tracked"]
    assert [f.stroke for f in prints] == ["Serve", "Forehand", "Backhand"]
    assert [f.speed_mph for f in prints] == [70, 55, 42]
    # ShotFingerprint has no field named "result". Pinning the exact
    # field set makes any future drift (leaking Result in) an obvious
    # structural failure rather than a quiet regression.
    assert set(footage.ShotFingerprint.__dataclass_fields__) == {
        "who",
        "stroke",
        "spin",
        "speed_mph",
    }


def test_predict_condensed_center_falls_inside_the_video():
    """The arithmetic model solves padding so the concatenated clips
    total the video length; every predicted center is inside [0, length]
    by construction."""
    shots_by_point = {
        1: [
            _fingerprint_shot(1, 1, "Me", "Serve", 10.0),
            _fingerprint_shot(1, 2, "You", "Forehand", 13.0),
        ],
        2: [_fingerprint_shot(2, 1, "Me", "Serve", 25.0)],
        3: [
            _fingerprint_shot(3, 1, "Me", "Serve", 40.0),
            _fingerprint_shot(3, 2, "You", "Backhand", 44.0),
            _fingerprint_shot(3, 3, "Me", "Forehand", 48.0),
        ],
    }
    length = 60.0

    center = footage.predict_condensed_center(shots_by_point, 2, length, min_padding=4.0)

    assert center is not None
    assert 0.0 <= center <= length


def test_locate_point_in_condensed_parses_a_found_verdict():
    client = _FakeClient(response_text=_retrieval_json())
    prints = [footage.ShotFingerprint("tracked", "Serve", "Flat", 70)]
    frames = _frames()

    verdict = footage.locate_point_in_condensed(
        client, footage.FootageConfig(), prints, frames, window_seconds=50.0
    )

    assert verdict.found is True
    assert verdict.offset_s == 10.0
    assert verdict.confidence == "high"


def test_locate_point_in_condensed_parses_a_not_found_verdict():
    """The "rally not present in these frames" answer is the honest
    outcome for points the condensed export doesn't contain. Downstream
    must never turn this into a judgement."""
    client = _FakeClient(
        response_text=_retrieval_json(
            rally_present=False,
            offset_seconds=0.0,
            reasoning="None of these frames show the described shot sequence.",
        )
    )

    verdict = footage.locate_point_in_condensed(
        client,
        footage.FootageConfig(),
        [footage.ShotFingerprint("tracked", "Serve", "Flat", 70)],
        _frames(),
        window_seconds=50.0,
    )

    assert verdict.found is False


def test_the_retrieval_prompt_never_sees_the_point_outcome():
    """Same rule as the judge call's allowlist: anything new reaching
    the retrieval prompt has to be argued for in this test, so the
    outcome cannot be silently added. Point outcome fields would turn
    "found" into "told where to find it", which is no longer an
    independent location check."""
    import inspect

    locate_params = set(inspect.signature(footage.locate_point_in_condensed).parameters)
    assert "point" not in locate_params and "record" not in locate_params

    retrieval_prompt_params = set(inspect.signature(footage._retrieval_prompt).parameters)
    assert retrieval_prompt_params == {"fingerprints", "window_seconds", "frame_count"}

    fingerprint_params = set(inspect.signature(footage.fingerprint_point).parameters)
    assert fingerprint_params == {"shots", "host_name"}
    # The fingerprint type's field names are the allowlist of what the
    # retrieval prompt may see about each shot. Pin them exactly.
    assert set(footage.ShotFingerprint.__dataclass_fields__) == {
        "who",
        "stroke",
        "spin",
        "speed_mph",
    }

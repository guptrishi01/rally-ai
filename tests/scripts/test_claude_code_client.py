"""Tests for judging through the Claude Code CLI. No test runs the real CLI."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import anthropic
import pytest

from scripts import claude_code_client
from scripts.claude_code_client import (
    ClaudeCodeClient,
    ClaudeCodeConfig,
    read_answer,
    write_request,
)
from scripts.client import get_footage_client
from swingvision_import import footage

_SCHEMA = {"type": "object", "properties": {"winner": {"type": "string"}}}


def _image(data: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


def _request(**overrides) -> dict:
    request = {
        "model": "claude-opus-5",
        "max_tokens": 16000,
        "output_config": {"effort": "high", "format": {"type": "json_schema", "schema": _SCHEMA}},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 1 (serve, t=-4.00s):"},
                    _image(b"first"),
                    {"type": "text", "text": "Frame 2 (finish, t=+0.00s):"},
                    _image(b"second"),
                    {"type": "text", "text": "Judge the point."},
                ],
            }
        ],
    }
    request.update(overrides)
    return request


class _FakeCli:
    """Stands in for subprocess.run, recording what the CLI was asked."""

    def __init__(self, stdout=None, returncode=0, raises=None):
        self.stdout = stdout if stdout is not None else json.dumps(
            {"is_error": False, "structured_output": {"winner": "tracked"}, "total_cost_usd": 0.4}
        )
        self.returncode = returncode
        self.raises = raises
        self.command: list[str] = []
        self.kwargs: dict = {}
        self.files: dict[str, bytes] = {}

    def __call__(self, command, **kwargs):
        self.command, self.kwargs = command, kwargs
        folder = Path(kwargs["cwd"])
        self.files = {path.name: path.read_bytes() for path in folder.iterdir()}
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, "boom")


@pytest.fixture
def cli(monkeypatch):
    fake = _FakeCli()
    monkeypatch.setattr(claude_code_client.subprocess, "run", fake)
    return fake


def _client() -> ClaudeCodeClient:
    return ClaudeCodeClient(ClaudeCodeConfig(executable="claude"))


def test_images_are_written_out_and_named_in_place(tmp_path):
    prompt = write_request(_request()["messages"][0]["content"], tmp_path)

    assert (tmp_path / "frame_01.jpg").read_bytes() == b"first"
    assert (tmp_path / "frame_02.jpg").read_bytes() == b"second"
    body = prompt.split("\n\n", 1)[1].splitlines()
    assert body == [
        "Frame 1 (serve, t=-4.00s):",
        "[image file: frame_01.jpg]",
        "Frame 2 (finish, t=+0.00s):",
        "[image file: frame_02.jpg]",
        "Judge the point.",
    ]
    assert "frame_01.jpg, frame_02.jpg" in prompt.split("\n\n", 1)[0]


def test_a_request_with_no_images_is_passed_through_as_text(tmp_path):
    assert write_request([{"type": "text", "text": "Just this."}], tmp_path) == "Just this."


def test_the_cli_is_asked_for_the_same_judgement_the_api_would_be(cli):
    response = _client().beta.messages.create(**_request())

    assert json.loads(response.content[0].text) == {"winner": "tracked"}
    assert response.stop_reason == "end_turn"
    command = cli.command
    assert command[:2] == ["claude", "--print"]
    assert json.loads(command[command.index("--json-schema") + 1]) == _SCHEMA
    assert command[command.index("--effort") + 1] == "high"
    assert command[command.index("--model") + 1] == "claude-opus-5"
    assert command[command.index("--tools") + 1] == "Read"
    assert set(cli.files) == {"frame_01.jpg", "frame_02.jpg"}
    assert "Judge the point." in cli.kwargs["input"]


def test_the_judge_runs_in_an_empty_folder_without_the_api_key(cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

    _client().messages.create(**_request())

    assert "ANTHROPIC_API_KEY" not in cli.kwargs["env"]
    assert "rallyai-judge-" in cli.kwargs["cwd"]
    assert not Path(cli.kwargs["cwd"]).exists()  # cleaned up afterwards


def test_nothing_is_billed_so_no_tokens_are_reported_but_list_price_is_kept(cli):
    client = _client()

    response = client.messages.create(**_request())
    client.messages.create(**_request())

    assert (response.usage.input_tokens, response.usage.output_tokens) == (0, 0)
    assert client.calls == 2
    assert client.list_price_usd == pytest.approx(0.8)


def test_the_model_can_be_left_to_the_subscriptions_default(monkeypatch):
    cli = _FakeCli(stdout=json.dumps({"result": "plain answer"}))
    monkeypatch.setattr(claude_code_client.subprocess, "run", cli)
    client = ClaudeCodeClient(ClaudeCodeConfig(executable="claude", pass_model=False))

    client.messages.create(**_request(output_config={}))

    assert "--model" not in cli.command
    assert "--json-schema" not in cli.command
    assert "--effort" not in cli.command


@pytest.mark.parametrize(
    "raises",
    [subprocess.TimeoutExpired("claude", 600), OSError("not executable")],
)
def test_a_cli_that_hangs_or_cannot_start_is_a_retryable_failure(monkeypatch, raises):
    monkeypatch.setattr(claude_code_client.subprocess, "run", _FakeCli(raises=raises))

    with pytest.raises(anthropic.APIConnectionError):
        _client().messages.create(**_request())


def test_a_missing_cli_says_what_to_do(monkeypatch):
    monkeypatch.setattr(claude_code_client.shutil, "which", lambda name: None)

    with pytest.raises(anthropic.APIConnectionError, match="was not found"):
        ClaudeCodeClient().messages.create(**_request())


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (1, ""),
        (0, "not json"),
        (0, json.dumps({"is_error": True, "result": "usage limit reached"})),
        (0, json.dumps({"is_error": False, "result": ""})),
    ],
)
def test_a_failed_or_empty_run_is_a_retryable_failure(returncode, stdout):
    with pytest.raises(anthropic.APIConnectionError):
        read_answer(returncode, stdout, "boom", structured=True)


def test_an_unstructured_answer_is_the_result_text():
    answer = read_answer(0, json.dumps({"result": "hello", "total_cost_usd": 0.1}), "", False)

    assert (answer.text, answer.list_price_usd) == ("hello", 0.1)


def test_the_footage_judge_cannot_tell_the_two_clients_apart(monkeypatch):
    verdict = {
        "footage_usable": True,
        "server_seen": "tracked",
        "winner": "tracked",
        "point_end_type": "winner",
        "tracked_at_net": "no",
        "confidence": "medium",
        "reasoning": "seen",
    }
    fake = _FakeCli(stdout=json.dumps({"structured_output": verdict, "total_cost_usd": 0.3}))
    monkeypatch.setattr(claude_code_client.subprocess, "run", fake)
    window = footage.PointWindow(0, 1.0, None, 2.0, True, "near", 3)
    frames = [footage.Frame("finish", 0.0, b"\xff\xd8jpeg")]

    result = footage.judge_point(_client(), footage.FootageConfig(), window, frames)

    assert (result.winner, result.confidence, result.input_tokens) == ("tracked", "medium", 0)
    assert json.loads(fake.command[fake.command.index("--json-schema") + 1]) == (
        footage.VERDICT_SCHEMA
    )
    assert fake.files == {"frame_01.jpg": b"\xff\xd8jpeg"}


def test_the_backend_is_chosen_by_argument_or_environment(monkeypatch):
    monkeypatch.setattr("scripts.client.load_dotenv", lambda: None)
    monkeypatch.setattr("scripts.client.get_anthropic_client", lambda: "api client")
    monkeypatch.delenv("RALLY_AI_FOOTAGE_BACKEND", raising=False)

    # Claude Code is the default, so nothing reaches for an API key unasked.
    assert isinstance(get_footage_client(), ClaudeCodeClient)
    assert get_footage_client("api") == "api client"
    monkeypatch.setenv("RALLY_AI_FOOTAGE_BACKEND", "API")
    assert get_footage_client() == "api client"
    assert isinstance(get_footage_client("claude-code"), ClaudeCodeClient)
    with pytest.raises(ValueError, match="Unknown footage backend"):
        get_footage_client("carrier-pigeon")

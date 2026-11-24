"""A stand-in Anthropic client that judges through the local Claude Code CLI.

The footage check sends one request per point: a run of labelled images
and a prompt, answered as schema-checked JSON. This module makes the same
request by running `claude -p` (Claude Code's headless mode) instead of
calling the API, so the work is drawn from the signed-in Claude
subscription rather than billed against an API key.

It is shaped like the part of `anthropic.Anthropic` that
`swingvision_import.footage` uses (`client.beta.messages.create(...)`), so
the pipeline, its prompts, its independence rule and its agreement rule
all run unchanged - only the transport differs.

What is not the same as the API path, stated plainly:

- The images are written to a temporary folder and read by the agent with
  its Read tool, rather than attached to the request. Each run is a fresh
  process in that empty folder, so it sees no project files, no memory and
  nothing about the match beyond the request itself.
- The model is whatever `--model` resolves to for the subscription; a
  model id the plan does not offer fails the call.
- Nothing is billed per token, so the response reports zero token usage
  and the run's dollar figure stays at $0.00. What the calls would have
  cost at list price is kept on the client (`list_price_usd`) for
  reference only.

Like scripts/client.py, this is entry-point code: nothing under `ai/` or
`src/` constructs it.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx

logger = logging.getLogger(__name__)

_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}
_REQUEST = httpx.Request("POST", "claude-code://local/print")


@dataclass
class ClaudeCodeConfig:
    """Settings for judging through the Claude Code CLI.

    Attributes:
        executable: The `claude` binary; found on PATH if not given.
        timeout: Seconds one judgement may take before it counts as a
            transient failure and is retried by the caller.
        pass_model: Send the request's model id as `--model`. Off uses
            the subscription's default model instead.
    """

    executable: str | None = None
    timeout: float = 600.0
    pass_model: bool = True


def _failure(message: str) -> anthropic.APIConnectionError:
    """A failure the footage check already knows how to retry."""
    return anthropic.APIConnectionError(message=message, request=_REQUEST)


class _Messages:
    """`messages.create`, answered by one headless Claude Code run."""

    def __init__(self, owner: ClaudeCodeClient) -> None:
        self._owner = owner

    def create(self, **kwargs):
        """Runs one request through `claude -p`.

        Args:
            **kwargs: The Messages API arguments. Used: `model`,
                `messages` (the first user turn's text and image blocks),
                and `output_config` (`effort`, and `format.schema`). The
                rest (`max_tokens`, `betas`, `fallbacks`) have no CLI
                counterpart and are ignored.

        Returns:
            An object with `.content` (one text block holding the JSON
            answer), `.usage` (zero tokens - nothing is billed per token)
            and `.stop_reason`.

        Raises:
            anthropic.APIConnectionError: If the CLI is missing, times
                out, exits non-zero, or returns no answer.
        """
        return self._owner._run(kwargs)


class ClaudeCodeClient:
    """Answers footage-check requests with the local Claude Code CLI.

    Attributes:
        calls: Requests answered so far.
        list_price_usd: What those requests would have cost at API list
            price, as reported by the CLI. Reference only - nothing was
            charged.
    """

    def __init__(self, config: ClaudeCodeConfig | None = None) -> None:
        self.config = config or ClaudeCodeConfig()
        self.calls = 0
        self.list_price_usd = 0.0
        self._lock = threading.Lock()
        self.messages = _Messages(self)
        self.beta = SimpleNamespace(messages=self.messages)

    def _executable(self) -> str:
        found = self.config.executable or shutil.which("claude")
        if not found:
            raise _failure(
                "The `claude` command was not found. Install Claude Code and sign in, "
                "or use the API instead."
            )
        return found

    def _run(self, kwargs: dict):
        content = kwargs["messages"][0]["content"]
        output = kwargs.get("output_config") or {}
        schema = (output.get("format") or {}).get("schema")
        with tempfile.TemporaryDirectory(prefix="rallyai-judge-") as folder:
            prompt = write_request(content, Path(folder))
            command = [
                self._executable(),
                "--print",
                "--output-format",
                "json",
                "--tools",
                "Read",
                "--allowed-tools",
                "Read",
                "--strict-mcp-config",
                "--no-session-persistence",
            ]
            if schema:
                command += ["--json-schema", json.dumps(schema)]
            if output.get("effort"):
                command += ["--effort", str(output["effort"])]
            if self.config.pass_model and kwargs.get("model"):
                command += ["--model", str(kwargs["model"])]
            # With an API key in the environment the CLI would bill that
            # key, which is exactly what this client exists to avoid.
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            try:
                done = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=folder,
                    env=env,
                    timeout=self.config.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise _failure(f"Claude Code took over {self.config.timeout:.0f}s") from exc
            except OSError as exc:
                raise _failure(f"Claude Code could not be started: {exc}") from exc
        answer = read_answer(done.returncode, done.stdout, done.stderr, bool(schema))
        with self._lock:
            self.calls += 1
            self.list_price_usd += answer.list_price_usd
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=answer.text)],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            stop_reason="end_turn",
        )


def write_request(content: list[dict], folder: Path) -> str:
    """Saves a request's images and turns it into a prompt that names them.

    Args:
        content: The user turn's blocks: text and base64 images, in order.
        folder: Where to write the images.

    Returns:
        The prompt: an instruction to read every image, then the original
        text with each image replaced by a line naming its file.
    """
    lines: list[str] = []
    names: list[str] = []
    for block in content:
        if block.get("type") == "image":
            source = block["source"]
            extension = _EXTENSIONS.get(source.get("media_type", ""), "jpg")
            name = f"frame_{len(names) + 1:02d}.{extension}"
            (folder / name).write_bytes(base64.standard_b64decode(source["data"]))
            names.append(name)
            lines.append(f"[image file: {name}]")
        elif block.get("type") == "text":
            lines.append(block["text"])
    header = (
        f"This request comes with {len(names)} image file(s) in the current directory: "
        f"{', '.join(names)}. Read every one of them with the Read tool, in that order, "
        "before answering - the answer must come from the images. Use no other tool and "
        "read no other file. Below, each image's place in the request is marked by a line "
        "naming its file.\n\n"
        if names
        else ""
    )
    return header + "\n".join(lines)


@dataclass(frozen=True)
class _Answer:
    text: str
    list_price_usd: float


def read_answer(returncode: int, stdout: str, stderr: str, structured: bool) -> _Answer:
    """Pulls the answer out of `claude --print --output-format json`.

    Args:
        returncode: The CLI's exit code.
        stdout: What it printed.
        stderr: Its error output, for the failure message.
        structured: Whether a JSON schema was sent, in which case the
            answer is taken from `structured_output`.

    Returns:
        The answer text (JSON when structured) and its list-price cost.

    Raises:
        anthropic.APIConnectionError: If the run failed or gave no answer.
    """
    try:
        result = json.loads(stdout)
    except ValueError:
        result = None
    if returncode != 0 or not isinstance(result, dict) or result.get("is_error"):
        detail = ""
        if isinstance(result, dict):
            detail = str(result.get("result") or result.get("subtype") or "")
        detail = detail or stderr.strip() or stdout.strip()
        raise _failure(f"Claude Code run failed (exit {returncode}): {detail[:300]}")
    structured_output = result.get("structured_output")
    if structured and structured_output is not None:
        text = json.dumps(structured_output)
    else:
        text = str(result.get("result") or "")
    if not text:
        raise _failure("Claude Code returned no answer")
    return _Answer(text=text, list_price_usd=float(result.get("total_cost_usd") or 0.0))

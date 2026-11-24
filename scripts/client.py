"""Constructs the real Anthropic client for scripts.

ai/client.py's AnthropicClientLike is always injected, never constructed
inside ai/ or src/swingvision_import/ themselves, so tests never touch the
real API or spend real money. This module is the one place that actually
builds a real client - for scripts that intentionally want to.
"""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv


def get_anthropic_client() -> anthropic.Anthropic:
    """Loads .env (if present) and constructs a real Anthropic client.

    Returns:
        An anthropic.Anthropic client authenticated from the
        ANTHROPIC_API_KEY environment variable.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY isn't set after loading .env.
    """
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill in "
            "a real key, or export it in your shell before running this script."
        )
    return anthropic.Anthropic(api_key=api_key)


FOOTAGE_BACKENDS = frozenset({"api", "claude-code"})


def get_footage_client(backend: str | None = None):
    """Builds the client the footage check judges through.

    Two ways to get the same judgement: the Anthropic API, billed per
    token against ANTHROPIC_API_KEY, or the local Claude Code CLI, drawn
    from the signed-in subscription (scripts/claude_code_client.py). The
    pipeline cannot tell them apart.

    Args:
        backend: "api" or "claude-code". When not given, the
            RALLY_AI_FOOTAGE_BACKEND environment variable decides, and
            "claude-code" is the default - the footage check judges
            through the signed-in subscription unless the API is asked
            for by name, so no API credit is spent by default.

    Returns:
        A client with `beta.messages.create`.

    Raises:
        ValueError: If the backend is not one of FOOTAGE_BACKENDS.
        RuntimeError: If the API is chosen and ANTHROPIC_API_KEY isn't set.
    """
    load_dotenv()
    requested = backend or os.environ.get("RALLY_AI_FOOTAGE_BACKEND") or "claude-code"
    chosen = requested.strip().lower()
    if chosen not in FOOTAGE_BACKENDS:
        raise ValueError(
            f"Unknown footage backend {chosen!r}; expected one of {sorted(FOOTAGE_BACKENDS)}."
        )
    if chosen == "claude-code":
        from scripts.claude_code_client import ClaudeCodeClient

        return ClaudeCodeClient()
    return get_anthropic_client()

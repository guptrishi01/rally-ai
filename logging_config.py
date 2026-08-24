"""Shared logging setup for RallyAI's future entry points (CLI, scripts).

Library modules only ever call `logging.getLogger(__name__)` per CLAUDE.md's
per-module logging convention and never configure handlers themselves -
that's an application entry point's job, not a library's. configure_logging()
is that one call, meant to run once at the start of a script.
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configures the root logger with a single stream handler.

    Safe to call more than once - repeated calls don't add duplicate
    handlers, so it's safe to call from a script's entry point even if
    something else (a test, a REPL) already called it in the same process.

    Args:
        level: The minimum log level to emit. Defaults to INFO.
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)

"""Flask entry point: the RallyAI dashboard (frontend/) + standalone reports.

Two responsibilities, kept deliberately separate:

- The built React SPA (frontend/dist/, see api.py's JSON routes under
  /api) is the interactive Overview/Journal/Statistics/Film Review
  dashboard - it's served as static files from "/".
- /report/<match_id> is unchanged from before the SPA existed: a
  self-contained, standalone HTML report (reports.render.render_match_report),
  reused as-is rather than folded into the SPA, since it's meant to be
  viewable on its own (downloaded, opened without a server running).
  Viewing it never spends API money - it only reads whatever AI coaching
  report is already cached on disk, never constructs a client.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from flask import Flask, send_from_directory  # noqa: E402

from ai.config import AICoachConfig  # noqa: E402
from ai.records import CoachingReport  # noqa: E402
from logging_config import configure_logging  # noqa: E402
from reports.render import render_match_report  # noqa: E402
from stats.queries import match_stats  # noqa: E402
from swingvision_import.config import ImportConfig  # noqa: E402
from swingvision_import.db import get_connection  # noqa: E402
from swingvision_import.pipeline import SwingVisionImportPipeline  # noqa: E402
from webapp.api import create_api_blueprint, register_media_route  # noqa: E402
from webapp.config import WebAppConfig  # noqa: E402

_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"


def _load_cached_coaching_report(match_id: int) -> CoachingReport | None:
    """Best-effort loads a match's AI coaching report from disk, if cached.

    Never constructs a client or calls the API - viewing a report must
    never spend real money. Degrades to None on a missing or unreadable
    file, same as every other optional-enrichment lookup in this codebase.

    Args:
        match_id: The match to look up a cached report for.

    Returns:
        The cached CoachingReport, or None if there isn't one.
    """
    report_path = AICoachConfig().reports_dir / f"{match_id}.json"
    if not report_path.exists():
        return None
    try:
        return CoachingReport.from_dict(json.loads(report_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def create_app(
    import_config: ImportConfig | None = None,
    webapp_config: WebAppConfig | None = None,
) -> Flask:
    """Builds the Flask app.

    Args:
        import_config: SwingVision import pipeline settings. Defaults to
            ImportConfig() if not given.
        webapp_config: Upload/media directory settings. Defaults to
            WebAppConfig() if not given.

    Returns:
        A configured, ready-to-run Flask app.
    """
    configure_logging()
    app = Flask(__name__, static_folder=str(_FRONTEND_DIST), static_url_path="")

    import_config = import_config or ImportConfig()
    webapp_config = webapp_config or WebAppConfig()
    app.config["MAX_CONTENT_LENGTH"] = webapp_config.max_content_length
    pipeline = SwingVisionImportPipeline(import_config)

    app.register_blueprint(
        create_api_blueprint(pipeline, import_config, webapp_config), url_prefix="/api"
    )
    register_media_route(app, webapp_config.media_dir)

    @app.get("/report/<int:match_id>")
    def view_report(match_id: int):
        connection = get_connection(import_config.db_path, import_config.schema_path)
        try:
            try:
                stats = match_stats(connection, match_id)
            except ValueError:
                return "Match not found.", 404
        finally:
            connection.close()
        coaching_report = _load_cached_coaching_report(match_id)
        return render_match_report(stats, coaching_report)

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    return app


if __name__ == "__main__":
    # Binds to 127.0.0.1 by default (Flask's own default) - a local,
    # single-user tool, never meant to be exposed beyond localhost.
    create_app().run(debug=True)

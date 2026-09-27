"""JSON API for the Rally AI dashboard (frontend/): Overview, Journal,
Statistics, and Film Review.

Every route here is a thin wrapper over an existing pipeline/query
function — no parallel business logic. This is what closes the gap
CLAUDE.md flags as TBD ("a full in-browser review/editing UI for
resolving needs_review flags"): the /pending/... routes let the frontend
walk a staged match's flagged points and confirm each one directly,
instead of hand-editing the pending JSON or going through the CLI. The
safety gate itself is unchanged — finalize() still refuses a match with
any needs_review=True point, exactly as before.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from flask import Blueprint, Flask, jsonify, request, send_from_directory

from ai.config import AICoachConfig
from ai.journal import generate_journal_feedback
from scripts.client import get_anthropic_client
from shot_map.embeddings import build_shot_embeddings
from stats.queries import (
    all_match_ids,
    all_match_stats,
    career_stats_from_matches,
    match_stats,
    update_journal_fields,
)
from swingvision_import import footage
from swingvision_import.config import ImportConfig
from swingvision_import.db import get_connection
from swingvision_import.pipeline import SwingVisionImportPipeline
from swingvision_import.review import ConfirmationError, load_pending, unresolved_flags

from .config import WebAppConfig
from .jobs import JobRegistry
from .uploads import match_slug, pair_uploads, save_paired_videos, save_uploaded_xlsx

_SET_NUMBERS = (1, 2, 3)
_OVERRIDE_TEXT_FIELDS = ("pros", "cons", "notes", "location", "reviewer_note")
_OVERRIDE_RATING_FIELDS = ("energy_rating", "mental_rating")

# Footage-check settings a caller may override; anything else stays at
# the module default so a request can't, say, aim it at another model
# by accident.
_FOOTAGE_OVERRIDES = frozenset({"model", "workers", "frame_width", "end_frames"})

# Footage checks run for minutes, so they run on a thread and the UI
# polls. Process-local by design - see webapp/jobs.py.
_footage_jobs = JobRegistry()


def _parse_first_server_by_set(form) -> dict[int, str] | None:
    """Builds the set_number -> "me"/"opponent" map from the intake form.

    Args:
        form: The submitted form (request.form).

    Returns:
        The map, or None if no set's question was answered.
    """
    first_server_by_set = {}
    for set_number in _SET_NUMBERS:
        value = form.get(f"first_server_set{set_number}")
        if value:
            first_server_by_set[set_number] = value
    return first_server_by_set or None


def _parse_match_overrides(form) -> dict[str, object]:
    """Builds the ingest() match_overrides kwargs from the intake form.

    Args:
        form: The submitted form (request.form).

    Returns:
        Only the fields the user actually filled in - ingest() defaults
        handle the rest.
    """
    overrides: dict[str, object] = {}
    for field in _OVERRIDE_RATING_FIELDS:
        value = form.get(field)
        if value:
            overrides[field] = int(value)
    for field in _OVERRIDE_TEXT_FIELDS:
        value = form.get(field)
        if value:
            overrides[field] = value
    return overrides


def create_api_blueprint(
    pipeline: SwingVisionImportPipeline,
    import_config: ImportConfig,
    webapp_config: WebAppConfig,
) -> Blueprint:
    """Builds the /api Blueprint.

    Args:
        pipeline: The shared SwingVision import pipeline instance (same
            one app.py constructs from import_config).
        import_config: SwingVision import pipeline settings.
        webapp_config: Upload/media directory settings.

    Returns:
        The configured Blueprint, ready for app.register_blueprint.
    """
    api = Blueprint("api", __name__)

    def _connection():
        return get_connection(import_config.db_path, import_config.schema_path)

    def _resolve_pending_path(json_filename: str) -> Path | None:
        """Resolves a pending-JSON filename to a real path inside pending_dir.

        Mirrors the path-traversal guard app.py's do_suggest route already
        used, now shared by every /api/pending/... route.

        Returns:
            The resolved path, or None if it doesn't exist or would
            escape pending_dir.
        """
        pending_dir = import_config.pending_dir.resolve()
        json_path = (pending_dir / json_filename).resolve()
        if pending_dir not in json_path.parents or not json_path.exists():
            return None
        return json_path

    @api.get("/overview")
    def overview():
        connection = _connection()
        try:
            match_ids = all_match_ids(connection)
        finally:
            connection.close()
        stats = all_match_stats(import_config.db_path, match_ids)
        return jsonify(asdict(career_stats_from_matches(stats)))

    @api.get("/matches")
    def matches():
        connection = _connection()
        try:
            match_ids = all_match_ids(connection)
        finally:
            connection.close()
        stats = all_match_stats(import_config.db_path, match_ids)
        return jsonify([asdict(m) for m in stats])

    @api.get("/matches/<int:match_id>")
    def match_detail(match_id: int):
        connection = _connection()
        try:
            try:
                stats = match_stats(connection, match_id)
            except ValueError:
                return jsonify({"error": "Match not found."}), 404
        finally:
            connection.close()
        return jsonify(asdict(stats))

    @api.put("/matches/<int:match_id>/journal")
    def update_journal(match_id: int):
        body = request.get_json(force=True, silent=True) or {}
        connection = _connection()
        try:
            try:
                update_journal_fields(
                    connection,
                    match_id,
                    pros=body.get("pros"),
                    cons=body.get("cons"),
                    notes=body.get("notes"),
                )
            except ValueError:
                return jsonify({"error": "Match not found."}), 404
            stats = match_stats(connection, match_id)
        finally:
            connection.close()
        return jsonify(asdict(stats))

    @api.post("/matches/<int:match_id>/coach")
    def coach(match_id: int):
        body = request.get_json(force=True, silent=True) or {}
        journal_text = body.get("journal_text", "")
        force = bool(body.get("force", False))

        connection = _connection()
        try:
            try:
                stats = match_stats(connection, match_id)
            except ValueError:
                return jsonify({"error": "Match not found."}), 404
        finally:
            connection.close()

        client = get_anthropic_client()
        feedback = generate_journal_feedback(
            stats, client, journal_text, config=AICoachConfig(), force=force
        )
        return jsonify(asdict(feedback))

    @api.get("/matches/<int:match_id>/media")
    def media(match_id: int):
        connection = _connection()
        try:
            try:
                stats = match_stats(connection, match_id)
            except ValueError:
                return jsonify({"error": "Match not found."}), 404
        finally:
            connection.close()

        match_dir = webapp_config.media_dir / match_slug(stats.date, stats.opponent)
        if not match_dir.is_dir():
            return jsonify({"videos": []})
        videos = sorted(
            f"/media/{match_dir.name}/{path.name}"
            for path in match_dir.iterdir()
            if path.is_file()
        )
        return jsonify({"videos": videos})

    @api.get("/shots/embeddings")
    def shot_embeddings():
        raw_ids = request.args.get("match_ids")
        connection = _connection()
        try:
            match_ids = (
                [int(x) for x in raw_ids.split(",") if x]
                if raw_ids
                else all_match_ids(connection)
            )
        finally:
            connection.close()

        points = build_shot_embeddings(
            import_config.db_path, import_config.pending_dir, match_ids
        )
        return jsonify([asdict(p) for p in points])

    @api.post("/import")
    def do_import():
        # "xlsx_files" is the current field (one or many, a multi-part
        # recording's exports); "xlsx_file" stays accepted so anything
        # posting the older single-file field keeps working.
        xlsx_files = [
            f
            for f in (*request.files.getlist("xlsx_files"), *request.files.getlist("xlsx_file"))
            if f and f.filename
        ]
        if not xlsx_files:
            return jsonify({"error": "Please choose at least one SwingVision .xlsx export."}), 400

        date = request.form["date"]
        opponent = request.form["opponent"]
        result = request.form["result"]

        video_files = [f for f in request.files.getlist("video_files") if f and f.filename]
        pairing = pair_uploads(xlsx_files, video_files)

        xlsx_paths = [
            save_uploaded_xlsx(
                part.xlsx, date=date, opponent=opponent, uploads_dir=webapp_config.uploads_dir
            )
            for part in pairing.parts
        ]
        save_paired_videos(
            pairing, date=date, opponent=opponent, media_dir=webapp_config.media_dir
        )

        ingest_kwargs = dict(
            date=date,
            opponent=opponent,
            result=result,
            first_server_by_set=_parse_first_server_by_set(request.form),
            tracked_identity=request.form.get("tracked_identity") or None,
            **_parse_match_overrides(request.form),
        )
        try:
            # Two or more exports are one interrupted recording, not two
            # matches: merged into a single continuous reconstruction so a
            # set spanning the file boundary gets one correct score.
            json_path = (
                pipeline.ingest_multi_part(xlsx_paths, **ingest_kwargs)
                if pairing.is_multi_part
                else pipeline.ingest(xlsx_paths[0], **ingest_kwargs)
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        # Staging succeeded, so the loose .xlsx copies in data/uploads/ have
        # done their job: pack them into one zip (measured ~40% smaller) and
        # re-point the record at it. Footage is left alone on purpose - it's
        # already entropy-coded, and Film Review streams it from disk.
        pipeline.archive_sources(json_path)

        record = load_pending(json_path)
        connection = _connection()
        try:
            staged_label = len(all_match_ids(connection)) + 1
        finally:
            connection.close()

        return jsonify(
            {
                "json_filename": json_path.name,
                "staged_label": staged_label,
                "date": record.date,
                "opponent": record.opponent,
                "flags": unresolved_flags(record),
                "import_notes": record.import_notes,
                "parts": [
                    {
                        "index": part.index,
                        "export": part.xlsx.filename,
                        "videos": [video.filename for video in part.videos],
                    }
                    for part in pairing.parts
                ],
                "upload_notes": pairing.notes,
            }
        )

    @api.get("/pending")
    def pending_matches():
        """Lists staged matches still waiting on review.

        Without this the review modal is only reachable in the same browser
        session as the upload that staged it - a staged match whose tab was
        closed could only be revisited by re-uploading the whole thing,
        footage included.
        """
        pending_dir = import_config.pending_dir
        if not pending_dir.is_dir():
            return jsonify([])
        staged = []
        for json_path in sorted(pending_dir.glob("*.json")):
            try:
                record = load_pending(json_path)
            except (OSError, ValueError):
                continue  # a half-written or hand-edited file isn't fatal here
            points = [p for set_record in record.sets for p in set_record.points]
            staged.append(
                {
                    "json_filename": json_path.name,
                    "date": record.date,
                    "opponent": record.opponent,
                    "total_points": len(points),
                    "flags_remaining": sum(1 for p in points if p.needs_review),
                    "confirmed_by_footage": sum(1 for p in points if p.confirmed_by == "footage"),
                }
            )
        return jsonify(staged)

    @api.get("/pending/<path:json_filename>")
    def pending_detail(json_filename: str):
        json_path = _resolve_pending_path(json_filename)
        if json_path is None:
            return jsonify({"error": "Pending match not found."}), 404

        record = load_pending(json_path)
        shots_by_point = pipeline.shots_by_point(record) or {}
        points = []
        for set_record in record.sets:
            for point in set_record.points:
                if not point.needs_review:
                    continue
                shots = (
                    shots_by_point.get(point.source_point_number, [])
                    if point.source_point_number is not None
                    else []
                )
                points.append(
                    {
                        "set_number": set_record.set_number,
                        "game_number": point.game_number,
                        "point_number": point.point_number,
                        "point_end_type": point.point_end_type,
                        "point_won": point.point_won,
                        "net_approach": point.net_approach,
                        "ai_suggested_point_end_type": point.ai_suggested_point_end_type,
                        "ai_suggestion_reasoning": point.ai_suggestion_reasoning,
                        "footage_check": point.footage_check,
                        "shots": [
                            {
                                "shot_number": s.shot_number,
                                "player": s.player,
                                "type": s.shot_type,
                                "stroke": s.stroke,
                                "result": s.result,
                            }
                            for s in shots
                        ],
                    }
                )

        return jsonify(
            {
                "json_filename": json_path.name,
                "date": record.date,
                "opponent": record.opponent,
                "import_notes": record.import_notes,
                # Read back to the reviewer while they confirm points: it's
                # the note they wrote about this very import.
                "reviewer_note": record.reviewer_note,
                "points": points,
            }
        )

    def _match_videos(record) -> tuple[list[Path], str | None]:
        """Finds one video per part for a staged match, in part order.

        save_paired_videos names a multi-part match's clips `partNN__...`,
        which is exactly the order footage checks need. A part with no clip -
        or more than one, where a timestamp would be ambiguous - is reported
        rather than guessed at.

        Args:
            record: The staged match record.

        Returns:
            (videos, reason). `videos` is empty when a check isn't possible,
            and `reason` then says why in one sentence.
        """
        match_dir = webapp_config.media_dir / match_slug(record.date, record.opponent)
        if not match_dir.is_dir():
            return [], "No footage has been uploaded for this match."
        clips = sorted(p for p in match_dir.iterdir() if p.is_file())
        parts = len(record.source_files or [record.source_file or ""])
        if parts <= 1:
            usable = [p for p in clips if not p.name.startswith("unpaired__")]
            if len(usable) != 1:
                return [], (
                    f"Expected one video for this match, found {len(usable)} - a footage "
                    "check needs to know which recording the timestamps belong to."
                )
            return usable, None

        videos: list[Path] = []
        for index in range(parts):
            matching = [p for p in clips if p.name.startswith(f"part{index + 1:02d}__")]
            if len(matching) != 1:
                return [], (
                    f"Part {index + 1} has {len(matching)} videos - a footage check needs "
                    "exactly one recording per export to place its timestamps."
                )
            videos.append(matching[0])
        return videos, None

    def _footage_status(record, json_path: Path) -> dict:
        """Describes whether this match can be checked against footage.

        Everything here is read-only and cheap (a duration probe per file),
        so the review UI can show the true state - including *why* it can't
        run - before anyone spends anything.

        Args:
            record: The staged match record.
            json_path: Its pending JSON path.

        Returns:
            A dict for the review UI.
        """
        eligible = sum(
            1
            for set_record in record.sets
            for point in set_record.points
            if point.needs_review and point.source_point_number is not None
        )
        already = sum(
            1
            for set_record in record.sets
            for point in set_record.points
            if point.footage_check is not None
        )
        status = {
            "available": False,
            "reason": None,
            "eligible_points": eligible,
            "already_checked": already,
            "estimate_usd": None,
            "model": footage.FootageConfig().model,
        }
        videos, reason = _match_videos(record)
        if reason:
            status["reason"] = reason
            return status

        try:
            exe = footage.ffmpeg_exe()
            durations = [footage.probe_duration(video, exe) for video in videos]
            shots_by_point = pipeline.shots_by_point(record) or {}
            host_name = pipeline._host_name(record)
            latest: dict[int, float] = {}
            for shots in shots_by_point.values():
                window = footage.plan_point(shots, host_name)
                if window is not None:
                    latest[window.part] = max(latest.get(window.part, 0.0), window.last_shot)
            if not latest:
                status["reason"] = (
                    "This export has no per-shot video timestamps, so points can't be "
                    "located in the footage."
                )
                return status
            footage.check_alignment(latest, durations)
        except footage.FootageAlignmentError as exc:
            status["reason"] = str(exc)
            return status
        except footage.FootageError as exc:
            status["reason"] = str(exc)
            return status

        remaining = max(0, eligible - already)
        status["available"] = remaining > 0
        status["estimate_usd"] = footage.estimate_cost(
            remaining, footage.FootageConfig()
        ).usd
        if remaining == 0:
            status["reason"] = "Every eligible point has already been checked against footage."
        return status

    @api.get("/pending/<path:json_filename>/footage")
    def footage_status(json_filename: str):
        json_path = _resolve_pending_path(json_filename)
        if json_path is None:
            return jsonify({"error": "Pending match not found."}), 404
        status = _footage_status(load_pending(json_path), json_path)
        status["job"] = _footage_jobs.snapshot(json_path.name)
        return jsonify(status)

    @api.post("/pending/<path:json_filename>/footage")
    def start_footage_check(json_filename: str):
        json_path = _resolve_pending_path(json_filename)
        if json_path is None:
            return jsonify({"error": "Pending match not found."}), 404
        record = load_pending(json_path)
        videos, reason = _match_videos(record)
        if reason:
            return jsonify({"error": reason}), 400

        body = request.get_json(force=True, silent=True) or {}
        config = footage.FootageConfig(
            **{k: v for k, v in body.items() if k in _FOOTAGE_OVERRIDES}
        )
        client = get_anthropic_client()

        def run():
            return pipeline.verify_footage(
                client,
                json_path,
                videos,
                footage_config=config,
                progress=lambda s: _footage_jobs.update(json_path.name, s),
            )

        if not _footage_jobs.start(json_path.name, run):
            return jsonify({"error": "A footage check is already running for this match."}), 409
        return jsonify(_footage_jobs.snapshot(json_path.name)), 202

    @api.post("/pending/<path:json_filename>/suggest")
    def suggest(json_filename: str):
        json_path = _resolve_pending_path(json_filename)
        if json_path is None:
            return jsonify({"error": "Pending match not found."}), 404

        client = get_anthropic_client()
        pipeline.suggest(client, json_path)
        return pending_detail(json_filename)

    @api.post("/pending/<path:json_filename>/confirm-point")
    def confirm_point(json_filename: str):
        json_path = _resolve_pending_path(json_filename)
        if json_path is None:
            return jsonify({"error": "Pending match not found."}), 404

        body = request.get_json(force=True, silent=True) or {}
        try:
            record = pipeline.confirm_point(
                json_path,
                set_number=int(body["set_number"]),
                game_number=int(body["game_number"]),
                point_number=int(body["point_number"]),
                point_end_type=body["point_end_type"],
                point_won=bool(body["point_won"]),
                net_approach=bool(body["net_approach"]),
            )
        except ConfirmationError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError as exc:
            return jsonify({"error": f"Missing field: {exc}"}), 400

        return jsonify({"flags_remaining": len(unresolved_flags(record))})

    @api.post("/pending/<path:json_filename>/finalize")
    def finalize(json_filename: str):
        json_path = _resolve_pending_path(json_filename)
        if json_path is None:
            return jsonify({"error": "Pending match not found."}), 404

        record = load_pending(json_path)
        flags = unresolved_flags(record)
        if flags:
            return jsonify({"error": "Unresolved review flags remain.", "flags": flags}), 409

        try:
            match_id = pipeline.finalize(json_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

        return jsonify({"match_id": match_id})

    return api


def register_media_route(app: Flask, media_dir: Path) -> None:
    """Registers the static route Film Review's <video> tags load from.

    A thin, explicit wrapper (not folded into the API blueprint, which is
    JSON-only) so app.py's static-file responsibilities stay in one place.

    Args:
        app: The Flask app to register the route on.
        media_dir: Base directory uploaded match video is stored under
            (webapp_config.media_dir) - one subdirectory per match.
    """

    @app.get("/media/<path:filename>")
    def media_file(filename: str):
        return send_from_directory(media_dir, filename)

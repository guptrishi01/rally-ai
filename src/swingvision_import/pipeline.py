"""Orchestrates the SwingVision import flow: ingest, optional AI-assisted
review suggestions, then finalize.

ingest() never touches SQL — it only stages a reviewable JSON file, routing
through either the direct Points-sheet parse (transform.py, when a Pro
export has real Points rows) or the Shots-based reconstruction fallback
(reconstruct.py, confirmed the actual path for two real non-Pro exports).
suggest() is a separate, explicit, opt-in step — it spends real API money,
so it never runs automatically inside ingest(). finalize() is the only path
into the database, and delegates the review-flag gate to
load.finalize_and_load; suggestions never bypass that gate.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ai.client import AnthropicClientLike

from . import db, load, quality_check, reconstruct, review, review_assist, review_resolve
from .config import ImportConfig
from .parse import SwingVisionParser
from .raw import RawSetRow, RawSettings, RawShotRow
from .records import MatchRecord
from .review_assist import SuggestionConfig
from .review_resolve import ResolutionConfig
from .transform import transform

logger = logging.getLogger(__name__)


class SwingVisionImportPipeline:
    """Orchestrates the SwingVision import flow.

    ingest() never touches SQL — it only stages a reviewable JSON file.
    finalize() is the only path into the database, and delegates the
    review-flag gate to load.finalize_and_load.
    """

    def __init__(self, config: ImportConfig | None = None) -> None:
        """Initializes the pipeline.

        Args:
            config: Sheet/column aliases and file paths to use. Defaults to
                ImportConfig() if not given.
        """
        self.config = config or ImportConfig()
        self._parser = SwingVisionParser(self.config)

    def ingest(
        self,
        xlsx_path: Path,
        *,
        date: str,
        opponent: str,
        result: str,
        first_server_by_set: dict[int, str] | None = None,
        tracked_identity: str | None = None,
        **match_overrides: object,
    ) -> Path:
        """Parses a SwingVision export and stages it for review.

        Args:
            xlsx_path: Path to the SwingVision .xlsx export.
            date: ISO-format date of the match.
            opponent: Opponent's name.
            result: Match result, "W" or "L".
            first_server_by_set: Optional set_number -> "me"/"opponent"
                ground truth (from the intake UI) cross-checked against the
                reconstructed serve order; see quality_check.check_serve_order.
            tracked_identity: Optional self-reported name of "who I am in
                this recording", cross-checked against the Settings sheet's
                host name; see quality_check.check_tracked_identity.
            **match_overrides: Additional MatchRecord fields (e.g.
                energy_rating, pros, cons, location).

        Returns:
            Path to the staged pending-review JSON file.

        Raises:
            ValueError: If the Points sheet is empty (no Pro rollup) and
                there's also no Settings sheet to identify the tracked
                player for Shots-based reconstruction.
        """
        raw = self._parser.parse(xlsx_path)

        reconstruction = None
        if raw.points:
            record = transform(
                raw,
                date=date,
                opponent=opponent,
                result=result,
                config=self.config,
                source_file=str(xlsx_path),
                **match_overrides,
            )
        else:
            if raw.settings is None:
                raise ValueError(
                    f"{xlsx_path}: Points sheet is empty (no Pro rollup) and no "
                    "Settings sheet was found to identify the tracked player for "
                    "Shots-based reconstruction."
                )
            logger.info(
                "Points sheet is empty for %s — reconstructing from Shots instead "
                "(SwingVision's point rollup needs Pro; shot tracking doesn't).",
                xlsx_path,
            )
            # reconstruct_all logs its own skipped/excluded warnings; no need
            # to re-log here.
            reconstruction = reconstruct.reconstruct_all(raw.shots, raw.settings.host_name)
            shots_by_point = reconstruct.group_shots_by_point(raw.shots)
            shot_pattern_summary = reconstruct.build_shot_pattern_summary(
                reconstruction.points, shots_by_point
            )
            record = MatchRecord(
                date=date,
                opponent=opponent,
                result=result,
                source_file=str(xlsx_path),
                sets=reconstruction.sets,
                shot_pattern_summary=shot_pattern_summary,
                **match_overrides,
            )

        record.import_notes = self._build_import_notes(
            record, raw.sets, raw.settings, reconstruction, first_server_by_set, tracked_identity
        )

        json_path = review.save_pending(record, self.config.pending_dir)
        flags = review.unresolved_flags(record)
        logger.info(
            "Staged %s vs %s at %s (%d point(s) need review)",
            date, opponent, json_path, len(flags),
        )
        return json_path

    def ingest_multi_part(
        self,
        xlsx_paths: list[Path],
        *,
        date: str,
        opponent: str,
        result: str,
        first_server_by_set: dict[int, str] | None = None,
        tracked_identity: str | None = None,
        **match_overrides: object,
    ) -> Path:
        """Merges multiple exports of one interrupted match and stages it.

        For a match whose recording was cut and resumed as separate
        SwingVision files (each with its own restarting Point/Set
        numbering) — reconstructing each file independently would produce
        two wrong, incomplete scores (e.g. a set that spans the file
        boundary looks like it never finished in either file). This merges
        every file's Shots rows into one continuous sequence
        (reconstruct.merge_shots) before reconstructing once, so a set
        spanning the boundary gets one correct combined score.

        Every file must be the Shots-reconstruction case (no Points-sheet
        Pro rollup) with a matching Settings sheet — a Pro export in the
        middle of an otherwise-reconstructed match isn't supported.

        Args:
            xlsx_paths: The exports, in play order (earliest first).
            date: ISO-format date of the match.
            opponent: Opponent's name.
            result: Match result, "W" or "L".
            first_server_by_set: See ingest().
            tracked_identity: See ingest().
            **match_overrides: Additional MatchRecord fields.

        Returns:
            Path to the staged pending-review JSON file.

        Raises:
            ValueError: If fewer than 2 paths are given, any file has a
                populated Points sheet (Pro rollup — use ingest() instead),
                any file is missing a Settings sheet, or the files disagree
                on the tracked player's host name.
        """
        if len(xlsx_paths) < 2:
            raise ValueError("ingest_multi_part needs at least 2 files - use ingest() for one.")

        raw_exports = [self._parser.parse(p) for p in xlsx_paths]
        for path, raw in zip(xlsx_paths, raw_exports, strict=True):
            if raw.points:
                raise ValueError(
                    f"{path}: has a populated Points sheet (Pro rollup) - "
                    "ingest_multi_part only supports Shots-based reconstruction."
                )
            if raw.settings is None:
                raise ValueError(f"{path}: no Settings sheet to identify the tracked player.")
        host_names = {raw.settings.host_name for raw in raw_exports}
        if len(host_names) > 1:
            raise ValueError(f"Files disagree on the tracked player's host name: {host_names}")
        host_name = raw_exports[0].settings.host_name

        logger.info(
            "Merging %d files into one continuous reconstruction for %s vs %s",
            len(xlsx_paths), date, opponent,
        )
        merged_shots = reconstruct.merge_shots([raw.shots for raw in raw_exports])
        reconstruction = reconstruct.reconstruct_all(merged_shots, host_name)
        shots_by_point = reconstruct.group_shots_by_point(merged_shots)
        shot_pattern_summary = reconstruct.build_shot_pattern_summary(
            reconstruction.points, shots_by_point
        )
        record = MatchRecord(
            date=date,
            opponent=opponent,
            result=result,
            source_files=[str(p) for p in xlsx_paths],
            sets=reconstruction.sets,
            shot_pattern_summary=shot_pattern_summary,
            **match_overrides,
        )
        record.import_notes = self._build_import_notes(
            record,
            raw_sets=[],  # no single file's Sets-sheet summary is a valid reference here
            settings=raw_exports[0].settings,
            reconstruction=reconstruction,
            first_server_by_set=first_server_by_set,
            tracked_identity=tracked_identity,
        )
        record.import_notes.append(
            f"Merged from {len(xlsx_paths)} files: assumes no points were lost exactly "
            "at a file boundary (no shot data exists there to check)."
        )

        json_path = review.save_pending(record, self.config.pending_dir)
        flags = review.unresolved_flags(record)
        logger.info(
            "Staged %s vs %s at %s (%d point(s) need review)",
            date, opponent, json_path, len(flags),
        )
        return json_path

    def _build_import_notes(
        self,
        record: MatchRecord,
        raw_sets: list[RawSetRow],
        settings: RawSettings | None,
        reconstruction: reconstruct.ReconstructionResult | None,
        first_server_by_set: dict[int, str] | None,
        tracked_identity: str | None,
    ) -> list[str]:
        """Assembles this ingest's informational data-quality notes.

        Args:
            record: The MatchRecord just built (direct-parse or reconstructed).
            raw_sets: Raw Sets-sheet rows to cross-check the reconstructed
                score against; empty for a merged multi-part match, where
                no single file's Sets summary is a valid reference.
            settings: Parsed Settings-sheet data, for the identity check.
            reconstruction: The reconstruction result, if the Shots-based
                fallback path was used; None on the direct-parse path (no
                gap/exclusion counts to report).
            first_server_by_set: See ingest().
            tracked_identity: See ingest().

        Returns:
            Every note gathered — never blocks finalize(), just surfaces
            things worth a human's attention.
        """
        notes = []
        if reconstruction is not None:
            if reconstruction.skipped_points:
                notes.append(
                    f"{len(reconstruction.skipped_points)} point(s) had no shot data "
                    f"and were skipped (likely a recording gap): "
                    f"{reconstruction.skipped_points}"
                )
            if reconstruction.excluded_points:
                notes.append(
                    f"{len(reconstruction.excluded_points)} point(s) were excluded as "
                    f"non-match activity (e.g. a fed ball between points): "
                    f"{reconstruction.excluded_points}"
                )
        notes.extend(quality_check.check_score_against_sets_sheet(record.sets, raw_sets))
        notes.extend(quality_check.check_serve_order(record.sets, first_server_by_set))
        notes.extend(quality_check.check_tracked_identity(settings, tracked_identity))
        return notes

    def shots_by_point(self, record: MatchRecord) -> dict[int, list[RawShotRow]] | None:
        """Re-derives point_number -> shots for a staged record.

        Re-merges multi-part files the same way ingest_multi_part() did, so
        source_point_number (assigned from the merged sequence at ingest
        time) still resolves correctly.

        Args:
            record: The staged match record.

        Returns:
            The lookup, or None if the record has no source file(s) at all
            to re-parse (distinct from a source that parses to zero shots).
        """
        if record.source_files:
            raw_exports = [self._parser.parse(Path(p)) for p in record.source_files]
            return reconstruct.group_shots_by_point(
                reconstruct.merge_shots([raw.shots for raw in raw_exports])
            )
        if record.source_file is not None:
            raw = self._parser.parse(Path(record.source_file))
            return reconstruct.group_shots_by_point(raw.shots)
        return None

    def suggest(
        self,
        client: AnthropicClientLike,
        json_path: Path,
        *,
        suggestion_config: SuggestionConfig | None = None,
    ) -> MatchRecord:
        """Generates Claude-assisted suggestions for a staged match's flagged points.

        Opt-in and separate from ingest()/finalize() on purpose — this
        spends real API money, so it never runs automatically. Only
        applies to points that came from Shots-based reconstruction
        (source_point_number is not None); a direct Points-sheet parse has
        no raw shot sequence to re-fetch and reason over. Never clears
        needs_review — the user still confirms every point by hand.

        Args:
            client: An anthropic.Anthropic-shaped client (injected so
                tests never hit the real API or spend real money).
            json_path: Path to a pending JSON file previously written by
                ingest().
            suggestion_config: Model/token settings. Defaults
                to SuggestionConfig() if not given.

        Returns:
            The record with ai_suggested_point_end_type/
            ai_suggestion_reasoning filled in where a suggestion succeeded.
            Also re-saved to the same pending JSON path so the suggestions
            persist across the actual manual review.

        Raises:
            ValueError: If the staged record has neither source_file nor
                source_files to re-parse.
        """
        record = review.load_pending(json_path)
        shots_by_point = self.shots_by_point(record)
        if shots_by_point is None:
            raise ValueError(f"{json_path}: record has no source_file(s) to re-parse.")

        suggestion_config = suggestion_config or SuggestionConfig()

        suggested_count = 0
        for set_record in record.sets:
            for point in set_record.points:
                if point.source_point_number is None:
                    continue
                shot_context = shots_by_point.get(point.source_point_number, [])
                try:
                    suggestion = review_assist.suggest_point_resolution(
                        client, suggestion_config, point, shot_context
                    )
                except review_assist.SuggestionError:
                    logger.exception(
                        "Suggestion failed for set %d game %d point %d",
                        set_record.set_number,
                        point.game_number,
                        point.point_number,
                    )
                    continue
                point.ai_suggested_point_end_type = suggestion.point_end_type
                point.ai_suggestion_reasoning = suggestion.reasoning
                suggested_count += 1

        review.save_pending(record, self.config.pending_dir)
        logger.info("Generated %d suggestion(s) for %s", suggested_count, json_path)
        return record

    def resolve(
        self,
        client: AnthropicClientLike,
        json_path: Path,
        *,
        resolution_config: ResolutionConfig | None = None,
    ) -> MatchRecord:
        """Parses every flagged point's human review_answer into resolved_* fields.

        Opt-in and separate from ingest()/suggest()/finalize() — spends
        real API money, so it never runs automatically. Run this after
        hand-editing review_answer onto the points you've actually
        reviewed (any point, not just reconstructed ones — a direct-parse
        point can carry a review_answer too, just without raw shot context
        to enrich it). Never clears needs_review or touches the real
        point_end_type/point_won/net_approach fields — apply_resolutions()
        is the only thing that does that, as one more explicit human
        checkpoint.

        Args:
            client: An anthropic.Anthropic-shaped client (injected so
                tests never hit the real API or spend real money).
            json_path: Path to a pending JSON file previously written by
                ingest().
            resolution_config: Model/token settings. Defaults to
                ResolutionConfig() if not given.

        Returns:
            The record with resolved_point_end_type/resolved_point_won/
            resolved_net_approach/resolution_reasoning filled in for every
            point that had a review_answer and parsed successfully. Also
            re-saved to the same pending JSON path.
        """
        record = review.load_pending(json_path)
        shots_by_point = self.shots_by_point(record) or {}
        resolution_config = resolution_config or ResolutionConfig()

        resolved_count = 0
        for set_record in record.sets:
            for point in set_record.points:
                if not point.review_answer:
                    continue
                shot_context = (
                    shots_by_point.get(point.source_point_number, [])
                    if point.source_point_number is not None
                    else []
                )
                try:
                    resolution = review_resolve.resolve_point_answer(
                        client, resolution_config, point, shot_context
                    )
                except review_resolve.ResolutionError:
                    logger.exception(
                        "Failed to parse review_answer for set %d game %d point %d",
                        set_record.set_number,
                        point.game_number,
                        point.point_number,
                    )
                    continue
                point.resolved_point_end_type = resolution.point_end_type
                point.resolved_point_won = resolution.point_won
                point.resolved_net_approach = resolution.net_approach
                point.resolution_reasoning = resolution.reasoning
                resolved_count += 1

        review.save_pending(record, self.config.pending_dir)
        logger.info("Parsed %d review answer(s) for %s", resolved_count, json_path)
        return record

    def apply_resolutions(self, json_path: Path) -> MatchRecord:
        """Applies every point's parsed resolution and clears needs_review for it.

        The explicit human checkpoint resolve() defers to: only points
        with a resolved_point_end_type (i.e. resolve() successfully parsed
        their review_answer) are touched. A point whose review_answer
        failed to parse, or was never written, is untouched and stays
        needs_review=True — finalize() will keep refusing the match until
        it's addressed some other way (a corrected review_answer + another
        resolve() pass, or a direct hand-edit).

        Args:
            json_path: Path to a pending JSON file, after resolve() has run.

        Returns:
            The record with resolved fields applied and needs_review
            cleared for every point that had one. Also re-saved to the
            same pending JSON path.
        """
        record = review.load_pending(json_path)
        applied_count = 0
        for set_record in record.sets:
            for point in set_record.points:
                if point.resolved_point_end_type is None:
                    continue
                point.point_end_type = point.resolved_point_end_type
                point.point_won = bool(point.resolved_point_won)
                if point.resolved_net_approach is not None:
                    point.net_approach = point.resolved_net_approach
                point.needs_review = False
                applied_count += 1

        review.save_pending(record, self.config.pending_dir)
        logger.info("Applied %d resolution(s) for %s", applied_count, json_path)
        return record

    def confirm_point(
        self,
        json_path: Path,
        *,
        set_number: int,
        game_number: int,
        point_number: int,
        point_end_type: str,
        point_won: bool,
        net_approach: bool,
    ) -> MatchRecord:
        """Directly applies a human's manual confirmation for one flagged point.

        The browser review UI's fast path onto review.confirm_point() - the
        human picks the point's real classification straight from controls
        in the UI rather than writing a review_answer for Claude to parse.
        Loads the pending record, delegates the actual field update and
        consistency check to review.confirm_point(), then re-saves.

        Args:
            json_path: Path to a pending JSON file previously written by
                ingest().
            set_number: The point's set number.
            game_number: The point's game number within that set.
            point_number: The point's number within that game.
            point_end_type: The confirmed outcome.
            point_won: Whether the tracked player won the point.
            net_approach: Whether the tracked player approached the net.

        Returns:
            The record with the matching point's fields updated and
            needs_review cleared. Also re-saved to the same pending JSON
            path.

        Raises:
            review.ConfirmationError: See review.confirm_point.
        """
        record = review.load_pending(json_path)
        review.confirm_point(
            record,
            set_number=set_number,
            game_number=game_number,
            point_number=point_number,
            point_end_type=point_end_type,
            point_won=point_won,
            net_approach=net_approach,
        )
        review.save_pending(record, self.config.pending_dir)
        return record

    def finalize(self, json_path: Path) -> int:
        """Loads a staged match and writes it to SQL if fully reviewed.

        Args:
            json_path: Path to a pending JSON file previously written by
                ingest() (and possibly hand-edited, or annotated by
                suggest(), to resolve review flags).

        Returns:
            The newly inserted match's match_id.

        Raises:
            UnresolvedReviewError: If any point still needs review.
            ValueError: If this match is already loaded.
        """
        record = review.load_pending(json_path)
        connection = db.get_connection(self.config.db_path, self.config.schema_path)
        try:
            match_id = load.finalize_and_load(connection, record)
        finally:
            connection.close()
        logger.info("Loaded match_id=%d (%s vs %s)", match_id, record.date, record.opponent)
        return match_id

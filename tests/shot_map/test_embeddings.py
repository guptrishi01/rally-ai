from __future__ import annotations

import numpy as np

from shot_map.embeddings import ShotPoint, _feature_vector, _pca_3d, build_shot_embeddings
from swingvision_import.pipeline import SwingVisionImportPipeline
from swingvision_import.raw import RawShotRow
from swingvision_import.review import load_pending, save_pending


def _clear_review_flags(pipeline: SwingVisionImportPipeline, json_path) -> None:
    record = load_pending(json_path)
    for set_record in record.sets:
        for point in set_record.points:
            point.needs_review = False
    save_pending(record, pipeline.config.pending_dir)


# --- _pca_3d -----------------------------------------------------------


def test_pca_3d_returns_empty_for_no_samples():
    result = _pca_3d(np.zeros((0, 5)))
    assert result.shape == (0, 3)


def test_pca_3d_pads_with_zeros_when_fewer_than_three_usable_dimensions():
    # A single sample: centering makes it the zero vector, so every
    # projected axis is 0 - the degenerate case shouldn't raise.
    result = _pca_3d(np.array([[1.0, 2.0, 3.0]]))
    assert result.shape == (1, 3)
    assert np.allclose(result, 0.0)


def test_pca_3d_produces_finite_coordinates_for_varied_samples():
    rng = np.random.default_rng(0)
    features = rng.random((10, 8))

    result = _pca_3d(features)

    assert result.shape == (10, 3)
    assert np.all(np.isfinite(result))


# --- _feature_vector -----------------------------------------------------


def test_feature_vector_one_hot_encodes_known_categories():
    shot = RawShotRow(
        point_number=1, shot_number=1, player="Test Player",
        shot_type="first_serve", stroke="Serve", result="In",
    )

    vector = _feature_vector(shot, is_serving=True, rally_position=0)

    assert len(vector) == 8 + 7 + 3 + 2  # shot_types + strokes + results + is_serving + position
    assert sum(vector[:8]) == 1.0  # exactly one shot_type bucket
    assert sum(vector[8:15]) == 1.0  # exactly one stroke bucket
    assert sum(vector[15:18]) == 1.0  # exactly one result bucket
    assert vector[18] == 1.0  # is_serving
    assert vector[19] == 0.0  # rally_position 0 -> normalized 0


def test_feature_vector_caps_rally_position_at_the_max():
    shot = RawShotRow(1, 1, "Test Player", "in_play", "Forehand", "In")

    vector = _feature_vector(shot, is_serving=False, rally_position=99)

    assert vector[-1] == 1.0  # capped at _MAX_RALLY_POSITION / _MAX_RALLY_POSITION


# --- build_shot_embeddings -------------------------------------------------


def test_build_shot_embeddings_includes_only_tracked_player_shots(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    _clear_review_flags(pipeline, json_path)
    match_id = pipeline.finalize(json_path)

    points = build_shot_embeddings(import_config.db_path, import_config.pending_dir, [match_id])

    assert len(points) == 3  # 1 shot from point 1 + 2 shots from point 2 (see conftest)
    assert all(isinstance(p, ShotPoint) for p in points)
    assert all(p.match_id == match_id for p in points)
    assert {p.stroke for p in points} <= {"Serve", "Forehand"}  # never "Backhand" (opponent's)


def test_build_shot_embeddings_reflects_each_points_real_outcome(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    _clear_review_flags(pipeline, json_path)
    match_id = pipeline.finalize(json_path)

    points = build_shot_embeddings(import_config.db_path, import_config.pending_dir, [match_id])

    won_shots = [p for p in points if p.point_won]
    lost_shots = [p for p in points if not p.point_won]
    assert len(won_shots) == 1  # point 1's single serve
    assert len(lost_shots) == 2  # point 2's serve + serve_plus_one


def test_build_shot_embeddings_skips_a_match_with_no_pending_json_left(
    synthetic_non_pro_xlsx, import_config
):
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(
        synthetic_non_pro_xlsx, date="2026-08-06", opponent="Alex", result="W"
    )
    _clear_review_flags(pipeline, json_path)
    match_id = pipeline.finalize(json_path)
    json_path.unlink()  # simulate a cleaned-up pending directory

    points = build_shot_embeddings(import_config.db_path, import_config.pending_dir, [match_id])

    assert points == []


def test_build_shot_embeddings_skips_direct_parse_matches_with_no_shot_provenance(
    synthetic_xlsx, import_config
):
    # synthetic_xlsx takes the direct Points-sheet parse path - its points
    # have no source_point_number, so there's no shot<->point link at all.
    pipeline = SwingVisionImportPipeline(import_config)
    json_path = pipeline.ingest(synthetic_xlsx, date="2026-08-06", opponent="Alex", result="W")
    _clear_review_flags(pipeline, json_path)
    match_id = pipeline.finalize(json_path)

    points = build_shot_embeddings(import_config.db_path, import_config.pending_dir, [match_id])

    assert points == []


def test_build_shot_embeddings_ignores_an_unknown_match_id(import_config):
    from swingvision_import.db import get_connection

    get_connection(import_config.db_path, import_config.schema_path).close()

    points = build_shot_embeddings(import_config.db_path, import_config.pending_dir, [999])

    assert points == []

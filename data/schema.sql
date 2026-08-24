-- ============================================
-- RallyAI - Tennis Match Tracker
-- Database Schema
-- ============================================

-- ============================================
-- MATCH TABLE
-- Top-level entity representing a single match
-- ============================================
CREATE TABLE match (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date            DATE NOT NULL,
    opponent        TEXT NOT NULL,
    result          TEXT NOT NULL CHECK (result IN ('W', 'L')),
    match_type      TEXT NOT NULL DEFAULT 'competitive' CHECK (match_type IN ('competitive', 'practice')),
    location        TEXT,
    energy_rating   INTEGER CHECK (energy_rating BETWEEN 1 AND 5),
    mental_rating   INTEGER CHECK (mental_rating BETWEEN 1 AND 5),
    pros            TEXT,
    cons            TEXT,
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- SET TABLE
-- Each match has 2-3 sets
-- ============================================
CREATE TABLE "set" (
    set_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL,
    set_number      INTEGER NOT NULL CHECK (set_number BETWEEN 1 AND 3),
    games_won       INTEGER NOT NULL CHECK (games_won >= 0),
    games_lost      INTEGER NOT NULL CHECK (games_lost >= 0),
    is_tiebreak_set BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE,
    UNIQUE (match_id, set_number)
);

-- ============================================
-- POINT TABLE
-- Individual point-by-point data logged from
-- recorded match footage
-- ============================================
CREATE TABLE point (
    point_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id          INTEGER NOT NULL,
    game_number     INTEGER NOT NULL CHECK (game_number >= 1),
    point_number    INTEGER NOT NULL CHECK (point_number >= 1),
    is_serving      BOOLEAN NOT NULL,
    first_serve_in  BOOLEAN,
    second_serve_in BOOLEAN,
    point_won       BOOLEAN NOT NULL,
    point_end_type  TEXT NOT NULL CHECK (point_end_type IN (
                        'winner',
                        'unforced_error',
                        'forced_error',
                        'ace',
                        'double_fault',
                        'return_winner',
                        'return_error'
                    )),
    net_approach     BOOLEAN NOT NULL DEFAULT FALSE,
    is_tiebreak_game BOOLEAN NOT NULL DEFAULT FALSE,
    notes            TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- point_end_type functionally determines point_won (an ace/winner/
    -- return_winner always means the tracked player won the point; a
    -- double_fault/unforced_error/forced_error/return_error always means
    -- they lost it). point_won stays a stored column rather than a derived
    -- value on purpose - it's independently reviewable during manual
    -- review, matching the app's "no single auto-tag is trusted alone"
    -- philosophy - but this CHECK keeps the two from ever silently
    -- disagreeing. (net_point_won was dropped entirely: it was always
    -- exactly equal to point_won whenever net_approach was true, pure
    -- redundant storage with no independent value - see the Net Approach
    -- Effectiveness example query below for the derived equivalent.)
    CHECK (
        (point_end_type IN ('ace', 'winner', 'return_winner') AND point_won = TRUE)
        OR
        (point_end_type IN ('double_fault', 'unforced_error', 'forced_error', 'return_error')
            AND point_won = FALSE)
    ),

    FOREIGN KEY (set_id) REFERENCES "set"(set_id) ON DELETE CASCADE
);

-- ============================================
-- INDEXES
-- Optimizes common query patterns
-- ============================================

-- Fetch all sets for a match
CREATE INDEX idx_set_match_id ON "set"(match_id);

-- Fetch all points for a set
CREATE INDEX idx_point_set_id ON point(set_id);

-- Filter points by serve/receive
CREATE INDEX idx_point_serving ON point(set_id, is_serving);

-- Filter points by outcome type
CREATE INDEX idx_point_end_type ON point(set_id, point_end_type);

-- Query matches by date range
CREATE INDEX idx_match_date ON match(date);

-- ============================================
-- EXAMPLE DERIVED STAT QUERIES
-- These demonstrate how all stats are computed
-- from the raw point data rather than stored
-- ============================================

-- Total Points Played & Won (match level)
-- SELECT
--     COUNT(*) AS total_points_played,
--     SUM(CASE WHEN point_won = TRUE THEN 1 ELSE 0 END) AS total_points_won
-- FROM point p
-- JOIN "set" s ON p.set_id = s.set_id
-- WHERE s.match_id = ?;

-- First Serve Percentage (match level)
-- SELECT
--     SUM(CASE WHEN first_serve_in = TRUE THEN 1 ELSE 0 END) AS first_serves_in,
--     COUNT(*) AS first_serves_total,
--     ROUND(
--         SUM(CASE WHEN first_serve_in = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
--         1
--     ) AS first_serve_pct
-- FROM point p
-- JOIN "set" s ON p.set_id = s.set_id
-- WHERE s.match_id = ? AND p.is_serving = TRUE;

-- Aces & Double Faults (match level)
-- SELECT
--     SUM(CASE WHEN point_end_type = 'ace' THEN 1 ELSE 0 END) AS aces,
--     SUM(CASE WHEN point_end_type = 'double_fault' THEN 1 ELSE 0 END) AS double_faults
-- FROM point p
-- JOIN "set" s ON p.set_id = s.set_id
-- WHERE s.match_id = ?;

-- Winners & Errors breakdown (match level)
-- SELECT
--     SUM(CASE WHEN point_end_type = 'winner' THEN 1 ELSE 0 END) AS winners,
--     SUM(CASE WHEN point_end_type = 'unforced_error' THEN 1 ELSE 0 END) AS unforced_errors,
--     SUM(CASE WHEN point_end_type = 'forced_error' THEN 1 ELSE 0 END) AS forced_errors,
--     SUM(CASE WHEN point_end_type = 'return_winner' THEN 1 ELSE 0 END) AS return_winners,
--     SUM(CASE WHEN point_end_type = 'return_error' THEN 1 ELSE 0 END) AS return_errors
-- FROM point p
-- JOIN "set" s ON p.set_id = s.set_id
-- WHERE s.match_id = ?;

-- Net Approach Effectiveness (match level)
-- net_point_won isn't a stored column - a net approach's own success is
-- always just point_won, restricted to net_approach points.
-- SELECT
--     SUM(CASE WHEN net_approach = TRUE THEN 1 ELSE 0 END) AS net_approaches,
--     SUM(CASE WHEN net_approach = TRUE AND point_won = TRUE THEN 1 ELSE 0 END) AS net_points_won
-- FROM point p
-- JOIN "set" s ON p.set_id = s.set_id
-- WHERE s.match_id = ?;

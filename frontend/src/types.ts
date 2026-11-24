export interface ServingStats {
  first_serves_total: number
  first_serves_in: number
  first_serve_pct: number
  second_serves_total: number
  second_serves_in: number
  second_serve_pct: number
  aces: number
  double_faults: number
  service_games_won: number
  service_games_total: number
  service_hold_pct: number
}

export interface ReceivingStats {
  break_points_total: number
  break_points_converted: number
  break_point_conversion_pct: number
  return_games_won: number
  return_games_total: number
  return_win_pct: number
}

export interface PointOutcomeStats {
  total_points_played: number
  total_points_won: number
  points_won_pct: number
  winners: number
  unforced_errors: number
  forced_errors: number
  return_winners: number
  return_errors: number
  winner_to_ue_ratio: number
}

export interface NetStats {
  net_approaches: number
  net_points_won: number
  net_success_pct: number
}

export interface ClutchStats {
  deuce_points_played: number
  deuces_converted: number
  deuce_conversion_pct: number
}

export interface SelfAssessment {
  energy_rating: number | null
  mental_rating: number | null
  pros: string | null
  cons: string | null
  notes: string | null
}

export interface MatchStats {
  match_id: number
  date: string
  opponent: string
  result: 'W' | 'L'
  serving: ServingStats
  receiving: ReceivingStats
  point_outcomes: PointOutcomeStats
  net: NetStats
  clutch: ClutchStats
  self_assessment: SelfAssessment
}

export interface CareerHighlight {
  match_id: number
  date: string
  opponent: string
  value: number
  label: string
}

export interface CareerStats {
  total_matches: number
  wins: number
  losses: number
  win_pct: number
  current_streak_result: 'W' | 'L' | null
  current_streak_count: number
  avg_first_serve_pct: number
  avg_points_won_pct: number
  best_match_by_points_won_pct: CareerHighlight | null
  most_aces_in_a_match: CareerHighlight | null
}

/** One export and the footage paired with it by filename, as the server
 *  actually resolved it (the client-side preview is only a preview). */
export interface MatchPart {
  index: number
  export: string
  videos: string[]
}

export interface ImportResult {
  json_filename: string
  staged_label: number
  date: string
  opponent: string
  flags: string[]
  import_notes: string[]
  parts: MatchPart[]
  upload_notes: string[]
}

/** A match staged by an earlier upload and still awaiting review. */
export interface StagedMatch {
  json_filename: string
  date: string
  opponent: string
  total_points: number
  flags_remaining: number
  confirmed_by_footage: number
}

export interface FlaggedPointShot {
  shot_number: number
  player: string
  type: string
  stroke: string
  result: string
}

/** What the footage check concluded for one point. A "disagrees" verdict
 *  never overwrites anything - it is context for the human, exactly like
 *  ai_suggested_point_end_type. "not_in_footage" means the point's rally
 *  is not present in the condensed export (the structurally-missing
 *  serve/return-decided cases); the point stays flagged honestly rather
 *  than being judged against a nearby rally. */
export interface FootageCheck {
  verdict: 'agrees' | 'disagrees' | 'inconclusive' | 'not_in_footage'
  point_end_type: string
  point_won: boolean
  confidence: string
  reasoning: string
  at_net: string
  server_seen: string
}

/** What a footage run did. Every coverage figure here must be shown with
 *  its denominator (`points_played`), never alone - a "20 confirmed" read
 *  without "of 123 points played" context misleads about the tool's reach. */
export interface FootageRunSummary {
  points_played: number
  /** Rallies the footage itself holds, as the uploader counted them - the
   *  ceiling on what a check can reach. Null when they did not say. */
  footage_rally_count: number | null
  checked: number
  agreed: number
  disagreed: number
  inconclusive: number
  not_in_footage: number
  failed: number
  server_mismatches: number
  input_tokens: number
  output_tokens: number
  usd: number
  notes: string[]
}

export interface FootageJob {
  state: 'running' | 'done' | 'failed'
  progress: FootageRunSummary | null
  summary: FootageRunSummary | null
  error: string | null
}

/** Whether this staged match can be checked against its footage, and what
 *  that would cost - shown before anything is spent. */
export interface FootageStatus {
  available: boolean
  reason: string | null
  eligible_points: number
  already_checked: number
  /** Denominator for every coverage figure the UI shows. */
  points_played: number
  /** Rallies the footage itself holds, as the uploader counted them. */
  footage_rally_count: number | null
  /** Points already marked as structurally absent from the condensed
   *  export - shown so "N confirmed" never reads without what is being
   *  left out. */
  not_in_footage: number
  estimate_usd: number | null
  model: string
  job: FootageJob | null
}

export interface FlaggedPoint {
  set_number: number
  game_number: number
  point_number: number
  point_end_type: string
  point_won: boolean
  net_approach: boolean
  ai_suggested_point_end_type: string | null
  ai_suggestion_reasoning: string | null
  footage_check: FootageCheck | null
  shots: FlaggedPointShot[]
}

export interface PendingDetail {
  json_filename: string
  date: string
  opponent: string
  import_notes: string[]
  /** What the uploader wrote about this import's own quirks, echoed back
   *  so it's in front of them while they confirm the flagged points. */
  reviewer_note: string | null
  /** How the tracked player looked, so a footage check can pick them out
   *  of the frame. Ends swap at changeovers, so "near"/"far" alone is not
   *  a stable way to say which player is you. */
  player_appearance: string | null
  /** Rallies in the footage, as the uploader counted them at intake. */
  footage_rally_count: number | null
  points: FlaggedPoint[]
}

export interface JournalFeedback {
  match_id: number
  generated_at: string
  model: string
  journal_text: string
  feedback: string
}

export const POINT_END_TYPES = [
  'winner',
  'unforced_error',
  'forced_error',
  'ace',
  'double_fault',
  'return_winner',
  'return_error',
] as const

export type PointEndType = (typeof POINT_END_TYPES)[number]

export const WINNING_END_TYPES: ReadonlySet<string> = new Set(['ace', 'winner', 'return_winner'])

export interface ShotPoint {
  match_id: number
  point_won: boolean
  shot_type: string
  stroke: string
  result: string
  x: number
  y: number
  z: number
}

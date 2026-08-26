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

export interface ImportResult {
  json_filename: string
  staged_label: number
  date: string
  opponent: string
  flags: string[]
  import_notes: string[]
}

export interface FlaggedPointShot {
  shot_number: number
  player: string
  type: string
  stroke: string
  result: string
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
  shots: FlaggedPointShot[]
}

export interface PendingDetail {
  json_filename: string
  date: string
  opponent: string
  import_notes: string[]
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

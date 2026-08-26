import type {
  CareerStats,
  ImportResult,
  JournalFeedback,
  MatchStats,
  PendingDetail,
} from './types'

export class ApiError extends Error {
  status: number
  flags?: string[]

  constructor(message: string, status: number, flags?: string[]) {
    super(message)
    this.status = status
    this.flags = flags
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  const contentType = response.headers.get('content-type') ?? ''
  const body = contentType.includes('application/json') ? await response.json() : null

  if (!response.ok) {
    throw new ApiError(body?.error ?? response.statusText, response.status, body?.flags)
  }
  return body as T
}

function json(body: unknown): RequestInit {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

export interface ConfirmPointInput {
  set_number: number
  game_number: number
  point_number: number
  point_end_type: string
  point_won: boolean
  net_approach: boolean
}

export const api = {
  overview: () => request<CareerStats>('/api/overview'),

  matches: () => request<MatchStats[]>('/api/matches'),

  matchDetail: (matchId: number) => request<MatchStats>(`/api/matches/${matchId}`),

  updateJournal: (matchId: number, fields: { pros: string; cons: string; notes: string }) =>
    request<MatchStats>(`/api/matches/${matchId}/journal`, { ...json(fields), method: 'PUT' }),

  coach: (matchId: number, journalText: string, force = false) =>
    request<JournalFeedback>(
      `/api/matches/${matchId}/coach`,
      json({ journal_text: journalText, force }),
    ),

  media: (matchId: number) => request<{ videos: string[] }>(`/api/matches/${matchId}/media`),

  importMatch: (formData: FormData) =>
    request<ImportResult>('/api/import', { method: 'POST', body: formData }),

  pendingDetail: (jsonFilename: string) =>
    request<PendingDetail>(`/api/pending/${encodeURIComponent(jsonFilename)}`),

  suggest: (jsonFilename: string) =>
    request<PendingDetail>(`/api/pending/${encodeURIComponent(jsonFilename)}/suggest`, {
      method: 'POST',
    }),

  confirmPoint: (jsonFilename: string, point: ConfirmPointInput) =>
    request<{ flags_remaining: number }>(
      `/api/pending/${encodeURIComponent(jsonFilename)}/confirm-point`,
      json(point),
    ),

  finalize: (jsonFilename: string) =>
    request<{ match_id: number }>(
      `/api/pending/${encodeURIComponent(jsonFilename)}/finalize`,
      { method: 'POST' },
    ),
}

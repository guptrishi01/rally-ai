import { useState } from 'react'
import { api } from '../api'
import { POINT_END_TYPES, WINNING_END_TYPES, type FlaggedPoint, type PendingDetail } from '../types'
import GlassCard from './GlassCard'

interface ReviewModalProps {
  detail: PendingDetail
  onAllConfirmed: () => void
  onClose: () => void
}

function pointKey(p: Pick<FlaggedPoint, 'set_number' | 'game_number' | 'point_number'>) {
  return `${p.set_number}-${p.game_number}-${p.point_number}`
}

/**
 * The browser review UI CLAUDE.md flags as TBD: every flagged point must
 * be explicitly confirmed here before the match can be finalized. point_won
 * is never asked for directly - it's derived from the chosen point_end_type
 * (see WINNING_END_TYPES), the same consistency data/schema.sql's CHECK
 * constraint enforces server-side, so the UI can't submit an invalid pair.
 */
export default function ReviewModal({ detail, onAllConfirmed, onClose }: ReviewModalProps) {
  const [points, setPoints] = useState(detail.points)
  const [choices, setChoices] = useState<Record<string, { endType: string; netApproach: boolean }>>(
    () =>
      Object.fromEntries(
        detail.points.map((p) => [
          pointKey(p),
          {
            endType: p.ai_suggested_point_end_type ?? p.point_end_type,
            netApproach: p.net_approach,
          },
        ]),
      ),
  )
  const [confirming, setConfirming] = useState<string | null>(null)
  const [suggesting, setSuggesting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSuggest() {
    setSuggesting(true)
    setError(null)
    try {
      const updated = await api.suggest(detail.json_filename)
      setPoints(updated.points)
      setChoices((prev) => {
        const next = { ...prev }
        for (const p of updated.points) {
          if (p.ai_suggested_point_end_type) {
            next[pointKey(p)] = {
              ...next[pointKey(p)],
              endType: p.ai_suggested_point_end_type,
            }
          }
        }
        return next
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get suggestions.')
    } finally {
      setSuggesting(false)
    }
  }

  async function handleConfirm(point: FlaggedPoint) {
    const key = pointKey(point)
    const choice = choices[key]
    setConfirming(key)
    setError(null)
    try {
      const result = await api.confirmPoint(detail.json_filename, {
        set_number: point.set_number,
        game_number: point.game_number,
        point_number: point.point_number,
        point_end_type: choice.endType,
        point_won: WINNING_END_TYPES.has(choice.endType),
        net_approach: choice.netApproach,
      })
      setPoints((prev) => prev.filter((p) => pointKey(p) !== key))
      if (result.flags_remaining === 0) {
        onAllConfirmed()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to confirm point.')
    } finally {
      setConfirming(null)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <GlassCard className="max-h-[85vh] w-full max-w-2xl overflow-y-auto p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-[var(--font-display)] text-xl font-bold">
              Review flagged points
            </h2>
            <p className="text-sm text-[var(--color-ink-secondary)]">
              {detail.date} vs {detail.opponent} — {points.length} point(s) need confirmation
            </p>
          </div>
          <button
            onClick={onClose}
            className="shrink-0 text-lg text-[var(--color-ink-muted)] hover:text-[var(--color-ink-primary)]"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <button
          onClick={handleSuggest}
          disabled={suggesting || points.length === 0}
          className="mt-4 rounded-lg border border-[var(--color-border)] px-3 py-1.5 text-sm text-[var(--color-ink-secondary)] transition hover:border-[var(--color-brand-blue)] hover:text-[var(--color-ink-primary)] disabled:opacity-50"
        >
          {suggesting ? 'Asking Claude…' : 'Get Claude suggestions (spends API money)'}
        </button>

        {error && <p className="mt-3 text-sm text-[var(--color-critical)]">{error}</p>}

        <div className="mt-4 space-y-3">
          {points.length === 0 && (
            <p className="text-sm font-medium text-[var(--color-good)]">
              All points confirmed — ready to load.
            </p>
          )}
          {points.map((point) => {
            const key = pointKey(point)
            const choice = choices[key]
            const won = WINNING_END_TYPES.has(choice.endType)
            return (
              <div key={key} className="rounded-xl border border-[var(--color-border-soft)] p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-sm font-medium">
                    Set {point.set_number} · Game {point.game_number} · Point{' '}
                    {point.point_number}
                  </span>
                  <span
                    className="rounded-full px-2 py-0.5 text-xs font-semibold"
                    style={{
                      color: won ? 'var(--color-good)' : 'var(--color-critical)',
                      background: won
                        ? 'color-mix(in srgb, var(--color-good) 18%, transparent)'
                        : 'color-mix(in srgb, var(--color-critical) 18%, transparent)',
                    }}
                  >
                    {won ? 'Won' : 'Lost'}
                  </span>
                </div>

                {point.ai_suggested_point_end_type && (
                  <p className="mt-2 text-xs text-[var(--color-brand-cyan)]">
                    Claude suggests <strong>{point.ai_suggested_point_end_type}</strong>:{' '}
                    {point.ai_suggestion_reasoning}
                  </p>
                )}

                {point.shots.length > 0 && (
                  <div className="mt-2 overflow-x-auto">
                    <table className="w-full text-left text-xs text-[var(--color-ink-secondary)]">
                      <thead>
                        <tr className="text-[var(--color-ink-muted)]">
                          <th className="pr-3 font-normal">#</th>
                          <th className="pr-3 font-normal">Player</th>
                          <th className="pr-3 font-normal">Type</th>
                          <th className="pr-3 font-normal">Stroke</th>
                          <th className="font-normal">Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        {point.shots.map((s) => (
                          <tr key={s.shot_number}>
                            <td className="pr-3">{s.shot_number}</td>
                            <td className="pr-3">{s.player}</td>
                            <td className="pr-3">{s.type}</td>
                            <td className="pr-3">{s.stroke}</td>
                            <td>{s.result}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <select
                    value={choice.endType}
                    onChange={(e) =>
                      setChoices((prev) => ({
                        ...prev,
                        [key]: { ...prev[key], endType: e.target.value },
                      }))
                    }
                    className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-1 text-sm"
                  >
                    {POINT_END_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t.replace(/_/g, ' ')}
                      </option>
                    ))}
                  </select>

                  <label className="flex items-center gap-1.5 text-sm text-[var(--color-ink-secondary)]">
                    <input
                      type="checkbox"
                      checked={choice.netApproach}
                      onChange={(e) =>
                        setChoices((prev) => ({
                          ...prev,
                          [key]: { ...prev[key], netApproach: e.target.checked },
                        }))
                      }
                    />
                    Net approach
                  </label>

                  <button
                    onClick={() => handleConfirm(point)}
                    disabled={confirming === key}
                    className="ml-auto rounded-lg bg-gradient-brand px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
                  >
                    {confirming === key ? 'Confirming…' : 'Confirm'}
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      </GlassCard>
    </div>
  )
}

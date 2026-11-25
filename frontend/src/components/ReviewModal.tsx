import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { api } from '../api'
import FootagePanel from './FootagePanel'
import {
  MatchTrackingNote,
  PointTrackingPanel,
  ShotProvenanceLine,
  StrokeCell,
  traceFor,
} from './TrackerProvenance'
import { POINT_END_TYPES, WINNING_END_TYPES, type FlaggedPoint, type PendingDetail } from '../types'

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
 *
 * Presented as one point at a time rather than a scrolling list: the whole
 * premise is that each of these is a judgement call, and a list invites
 * skimming past the ones that look obvious.
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

  const total = detail.points.length
  const done = total - points.length
  const current = points[0] ?? null

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

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

  const choice = current ? choices[pointKey(current)] : null
  const won = choice ? WINNING_END_TYPES.has(choice.endType) : false

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      style={{ background: 'color-mix(in srgb, var(--color-ink) 42%, transparent)' }}
      onClick={onClose}
    >
      <motion.div
        className="panel flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden"
        initial={{ opacity: 0, y: 24, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: 'spring', stiffness: 280, damping: 28 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-[var(--color-line)] px-8 py-6">
          <div className="flex items-start justify-between gap-6">
            <div>
              <p className="eyebrow">Confirm before it counts</p>
              <h2 className="h2 mt-3">
                {points.length === 0 ? 'All points confirmed' : 'Was this point yours?'}
              </h2>
              <p className="num mt-2 text-[13px] text-[var(--color-dim)]">
                {detail.date} · {detail.opponent}
              </p>
            </div>
            <button
              onClick={onClose}
              className="shrink-0 text-[var(--color-dim)] transition-colors hover:text-[var(--color-ink)]"
              aria-label="Close"
            >
              &#10005;
            </button>
          </div>

          {/* Progress: the whole point of this modal is that it ends, so it
              says exactly how far off that is. */}
          <div className="mt-6 flex items-center gap-4">
            <div className="h-px flex-1 bg-[var(--color-line-strong)]">
              <motion.div
                className="h-px bg-[var(--color-accent)]"
                animate={{ width: `${total === 0 ? 100 : (done / total) * 100}%` }}
                transition={{ type: 'spring', stiffness: 140, damping: 22 }}
              />
            </div>
            <span className="eyebrow num shrink-0">
              {done} / {total}
            </span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-8 py-7">
          <button
            onClick={handleSuggest}
            disabled={suggesting || points.length === 0}
            className="btn btn-ghost btn-sm"
          >
            {suggesting ? 'Asking Claude…' : 'Ask Claude for a read (spends API credit)'}
          </button>

          {error && (
            <p className="mt-4 text-[14px]" style={{ color: 'var(--color-loss)' }}>
              {error}
            </p>
          )}

          <FootagePanel
            jsonFilename={detail.json_filename}
            onFinished={async () => {
              // Points the footage confirmed are no longer flagged, so pull
              // the queue again rather than walking ones already settled.
              const fresh = await api.pendingDetail(detail.json_filename)
              setPoints(fresh.points)
              if (fresh.points.length === 0) onAllConfirmed()
            }}
          />

          {/* The uploader's own note about this import, kept in view for the
              whole walk-through - it usually explains the very points that
              got flagged. */}
          {detail.reviewer_note && (
            <div className="panel-sunk mt-5 p-4">
              <p className="eyebrow">Your note on this import</p>
              <p className="mt-2 whitespace-pre-line text-[14px] leading-relaxed">
                {detail.reviewer_note}
              </p>
            </div>
          )}

          {/* What the import already knows is off about this match. Shown
              here, not only once review is finished: "host/guest may be
              reversed for this set" is worth knowing before confirming a
              hundred points, not after. Informational - never a gate. */}
          {detail.tracking && <MatchTrackingNote tracking={detail.tracking} />}

          {detail.import_notes.length > 0 && (
            <details className="panel-sunk mt-5 p-4" data-testid="import-notes">
              <summary className="eyebrow cursor-pointer">
                What the import noticed ({detail.import_notes.length})
              </summary>
              {detail.import_notes.map((note) => (
                <p
                  key={note}
                  className="mt-3 break-words text-[13px] leading-relaxed text-[var(--color-dim)]"
                >
                  {note}
                </p>
              ))}
            </details>
          )}

          <AnimatePresence mode="wait">
            {current && choice ? (
              <motion.div
                key={pointKey(current)}
                initial={{ opacity: 0, x: 24 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -24 }}
                transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                className="mt-7"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-3 border-t border-[var(--color-line)] pt-5">
                  <span className="num text-[14px]">
                    Set {current.set_number} · Game {current.game_number} · Point{' '}
                    {current.point_number}
                  </span>
                  <motion.span
                    key={String(won)}
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="eyebrow"
                    style={{ color: won ? 'var(--color-win)' : 'var(--color-loss)' }}
                  >
                    {won ? 'Counts as won' : 'Counts as lost'}
                  </motion.span>
                </div>

                {current.footage_check && (
                  <div className="panel-sunk mt-5 p-4">
                    <div className="flex flex-wrap items-baseline justify-between gap-3">
                      <p className="eyebrow">From the footage</p>
                      <p
                        className="eyebrow"
                        style={{
                          color:
                            current.footage_check.verdict === 'disagrees'
                              ? 'var(--color-loss)'
                              : 'var(--color-dim)',
                        }}
                      >
                        {current.footage_check.verdict === 'disagrees'
                          ? 'Contradicts the tracker'
                          : current.footage_check.verdict === 'not_in_footage'
                          ? 'Not in this footage'
                          : `Couldn't settle it (${current.footage_check.confidence})`}
                      </p>
                    </div>
                    <p className="mt-2 text-[14px] leading-relaxed">
                      {current.footage_check.verdict === 'disagrees' && (
                        <strong>
                          {current.footage_check.point_end_type.replace(/_/g, ' ')} —{' '}
                        </strong>
                      )}
                      {current.footage_check.reasoning}
                    </p>
                  </div>
                )}

                {current.tracking && <PointTrackingPanel tracking={current.tracking} />}

                {current.ai_suggested_point_end_type && (
                  <div className="panel-sunk mt-5 p-4">
                    <p className="eyebrow">Claude&rsquo;s read</p>
                    <p className="mt-2 text-[14px] leading-relaxed">
                      <strong>{current.ai_suggested_point_end_type.replace(/_/g, ' ')}</strong> —{' '}
                      {current.ai_suggestion_reasoning}
                    </p>
                  </div>
                )}

                {current.shots.length > 0 && (
                  <div className="mt-6 overflow-x-auto">
                    <table className="w-full border-collapse text-left text-[13px]">
                      <thead>
                        <tr className="border-b border-[var(--color-line)]">
                          {[
                            '#',
                            'Player',
                            'Type',
                            'Stroke',
                            'Result',
                            ...(current.tracking ? ['How it is known'] : []),
                          ].map((h) => (
                            <th key={h} className="eyebrow py-2 pr-4 font-normal">
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {current.shots.map((s) => (
                          <tr
                            key={s.shot_number}
                            className="border-b border-[var(--color-line)] transition-colors hover:bg-[var(--color-paper-2)]"
                          >
                            <td className="num py-2 pr-4 text-[var(--color-dim)]">
                              {s.shot_number}
                            </td>
                            <td className="py-2 pr-4">{s.player}</td>
                            <td className="py-2 pr-4 text-[var(--color-dim)]">{s.type}</td>
                            <td className="py-2 pr-4">
                              <StrokeCell stroke={s.stroke} />
                            </td>
                            <td className="py-2 pr-4">{s.result}</td>
                            {current.tracking && (
                              <td className="min-w-[12rem] py-2">
                                <ShotProvenanceLine
                                  shot={traceFor(current.tracking, s.shot_number)}
                                />
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* End type as a set of chips rather than a <select>: with
                    seven options it's one click instead of two, and the whole
                    vocabulary stays visible while deciding. */}
                <div className="mt-7">
                  <span className="field-label">How did it end?</span>
                  <div className="flex flex-wrap gap-2">
                    {POINT_END_TYPES.map((t) => {
                      const on = choice.endType === t
                      return (
                        <button
                          key={t}
                          onClick={() =>
                            setChoices((prev) => ({
                              ...prev,
                              [pointKey(current)]: { ...prev[pointKey(current)], endType: t },
                            }))
                          }
                          className="rounded-full border px-4 py-2 text-[13px] transition-colors"
                          style={{
                            borderColor: on ? 'var(--color-accent)' : 'var(--color-line-strong)',
                            background: on
                              ? 'color-mix(in srgb, var(--color-accent) 10%, transparent)'
                              : 'transparent',
                            color: on ? 'var(--color-accent)' : 'var(--color-dim)',
                          }}
                        >
                          {t.replace(/_/g, ' ')}
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className="mt-7 flex flex-wrap items-center justify-between gap-4 border-t border-[var(--color-line)] pt-6">
                  <label className="flex cursor-pointer items-center gap-2.5 text-[14px]">
                    <input
                      type="checkbox"
                      checked={choice.netApproach}
                      onChange={(e) =>
                        setChoices((prev) => ({
                          ...prev,
                          [pointKey(current)]: {
                            ...prev[pointKey(current)],
                            netApproach: e.target.checked,
                          },
                        }))
                      }
                      className="h-4 w-4 accent-[var(--color-accent)]"
                    />
                    Came to the net
                  </label>

                  <button
                    onClick={() => handleConfirm(current)}
                    disabled={confirming === pointKey(current)}
                    className="btn"
                  >
                    {confirming === pointKey(current) ? 'Confirming…' : 'Confirm point'}
                    <span aria-hidden>&#8594;</span>
                  </button>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-8 border-t border-[var(--color-line)] pt-8"
              >
                <p className="h3" style={{ color: 'var(--color-win)' }}>
                  Every flagged point has a decision on it.
                </p>
                <p className="lede mt-3 text-[15px]">
                  Close this and finalize the match to write it into your stats.
                </p>
                <button onClick={onClose} className="btn mt-6">
                  Back to the form
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </motion.div>
  )
}

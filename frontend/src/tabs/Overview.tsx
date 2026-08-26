import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { api, ApiError } from '../api'
import type { CareerStats, ImportResult, PendingDetail } from '../types'
import GlassCard from '../components/GlassCard'
import StatTile from '../components/StatTile'
import Skeleton from '../components/Skeleton'
import ReviewModal from '../components/ReviewModal'
import TennisBallProgress from '../components/TennisBallProgress'

type Phase = 'idle' | 'uploading' | 'reviewing' | 'ready' | 'finalizing' | 'done'

interface OverviewProps {
  onMatchFinalized: (matchId: number) => void
}

export default function Overview({ onMatchFinalized }: OverviewProps) {
  const [overview, setOverview] = useState<CareerStats | null>(null)
  const [loadingOverview, setLoadingOverview] = useState(true)

  const [phase, setPhase] = useState<Phase>('idle')
  const [importResult, setImportResult] = useState<ImportResult | null>(null)
  const [pendingDetail, setPendingDetail] = useState<PendingDetail | null>(null)
  const [finalizedMatchId, setFinalizedMatchId] = useState<number | null>(null)
  const [formError, setFormError] = useState<string | null>(null)
  const formRef = useRef<HTMLFormElement>(null)

  async function loadOverview() {
    setLoadingOverview(true)
    try {
      setOverview(await api.overview())
    } finally {
      setLoadingOverview(false)
    }
  }

  useEffect(() => {
    loadOverview()
  }, [])

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    // Only a fresh, idle form is actually ready for a new upload - once a
    // submission is staged/reviewing/finalizing/done, any further submit
    // event on this form (e.g. a stray re-fire from the Finalize button,
    // which shares the form but is type="button") must be a no-op rather
    // than re-staging the same match a second time.
    if (phase !== 'idle') return
    if (!formRef.current) return
    setFormError(null)
    setPhase('uploading')
    try {
      const result = await api.importMatch(new FormData(formRef.current))
      setImportResult(result)
      if (result.flags.length > 0) {
        const detail = await api.pendingDetail(result.json_filename)
        setPendingDetail(detail)
        setPhase('reviewing')
      } else {
        setPhase('ready')
      }
    } catch (err) {
      setFormError(err instanceof ApiError ? err.message : 'Failed to stage this match.')
      setPhase('idle')
    }
  }

  async function handleFinalize() {
    if (phase !== 'ready' || !importResult) return
    setPhase('finalizing')
    setFormError(null)
    try {
      const { match_id } = await api.finalize(importResult.json_filename)
      setFinalizedMatchId(match_id)
      setPhase('done')
      loadOverview()
    } catch (err) {
      setFormError(err instanceof ApiError ? err.message : 'Failed to finalize this match.')
      setPhase('ready')
    }
  }

  function resetForm() {
    formRef.current?.reset()
    setImportResult(null)
    setPendingDetail(null)
    setFinalizedMatchId(null)
    setFormError(null)
    setPhase('idle')
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="font-[var(--font-display)] text-3xl font-extrabold">
          Welcome back to <span className="text-gradient">RallyAI</span>
        </h1>
        <p className="mt-1 text-[var(--color-ink-secondary)]">
          Your career at a glance, and the fastest way to log a new match.
        </p>
      </section>

      {loadingOverview ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : overview && overview.total_matches > 0 ? (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
            <StatTile
              label="Record"
              value={`${overview.wins}–${overview.losses}`}
              sublabel={`${overview.total_matches} matches`}
              accent="brand"
            />
            <StatTile
              label="Win %"
              value={`${overview.win_pct}%`}
              accent={overview.win_pct >= 50 ? 'good' : 'critical'}
              delay={0.04}
            />
            <StatTile
              label="Current streak"
              value={
                overview.current_streak_result
                  ? `${overview.current_streak_count}${overview.current_streak_result}`
                  : '—'
              }
              sublabel={overview.current_streak_result === 'W' ? 'wins in a row' : 'losses in a row'}
              accent={overview.current_streak_result === 'W' ? 'good' : 'critical'}
              delay={0.08}
            />
            <StatTile
              label="Avg first serve %"
              value={`${overview.avg_first_serve_pct}%`}
              delay={0.12}
            />
            <StatTile
              label="Avg points won %"
              value={`${overview.avg_points_won_pct}%`}
              delay={0.16}
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {overview.best_match_by_points_won_pct && (
              <GlassCard className="p-5" glow>
                <p className="text-xs font-medium uppercase tracking-wider text-[var(--color-ink-muted)]">
                  🏆 {overview.best_match_by_points_won_pct.label}
                </p>
                <p className="mt-1 font-[var(--font-display)] text-xl font-bold">
                  {overview.best_match_by_points_won_pct.value}% vs{' '}
                  {overview.best_match_by_points_won_pct.opponent}
                </p>
                <p className="text-sm text-[var(--color-ink-secondary)]">
                  {overview.best_match_by_points_won_pct.date}
                </p>
              </GlassCard>
            )}
            {overview.most_aces_in_a_match && (
              <GlassCard className="p-5" glow>
                <p className="text-xs font-medium uppercase tracking-wider text-[var(--color-ink-muted)]">
                  🎯 {overview.most_aces_in_a_match.label}
                </p>
                <p className="mt-1 font-[var(--font-display)] text-xl font-bold">
                  {overview.most_aces_in_a_match.value} aces vs{' '}
                  {overview.most_aces_in_a_match.opponent}
                </p>
                <p className="text-sm text-[var(--color-ink-secondary)]">
                  {overview.most_aces_in_a_match.date}
                </p>
              </GlassCard>
            )}
          </div>
        </>
      ) : (
        <GlassCard className="p-6 text-center text-[var(--color-ink-secondary)]">
          No matches loaded yet — stage your first one below.
        </GlassCard>
      )}

      <GlassCard className="p-6">
        <h2 className="font-[var(--font-display)] text-xl font-bold">Load a match</h2>
        <p className="mt-1 text-sm text-[var(--color-ink-secondary)]">
          Upload a SwingVision export and (optionally) footage. You'll confirm any flagged
          points before it's added to your career stats.
        </p>

        {phase === 'done' && finalizedMatchId !== null ? (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-5 rounded-xl border border-[var(--color-good)]/40 bg-[color-mix(in_srgb,var(--color-good)_10%,transparent)] p-5"
          >
            <p className="font-[var(--font-display)] text-lg font-bold text-[var(--color-good)]">
              Match #{finalizedMatchId} loaded!
            </p>
            <p className="mt-1 text-sm text-[var(--color-ink-secondary)]">
              Please enter a journal entry for feedback.
            </p>
            <div className="mt-4 flex gap-3">
              <button
                onClick={() => onMatchFinalized(finalizedMatchId)}
                className="rounded-lg bg-gradient-brand px-4 py-2 text-sm font-semibold text-white"
              >
                Go to Journal
              </button>
              <button
                onClick={resetForm}
                className="rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm text-[var(--color-ink-secondary)] hover:text-[var(--color-ink-primary)]"
              >
                Load another match
              </button>
            </div>
          </motion.div>
        ) : (
          <form ref={formRef} onSubmit={handleSubmit} className="mt-5 space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">
                  SwingVision export (.xlsx)
                </span>
                <input
                  type="file"
                  name="xlsx_file"
                  accept=".xlsx"
                  required
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-[var(--color-brand-blue)] file:px-2 file:py-1 file:text-white"
                />
              </label>
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">
                  Match video (optional, multiple for a split recording)
                </span>
                <input
                  type="file"
                  name="video_files"
                  multiple
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-[var(--color-surface-raised)] file:px-2 file:py-1 file:text-[var(--color-ink-secondary)]"
                />
              </label>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">Date</span>
                <input
                  type="date"
                  name="date"
                  required
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">Opponent</span>
                <input
                  type="text"
                  name="opponent"
                  required
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">Result</span>
                <select
                  name="result"
                  required
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                >
                  <option value="W">Win</option>
                  <option value="L">Loss</option>
                </select>
              </label>
            </div>

            <label className="block text-sm">
              <span className="text-[var(--color-ink-secondary)]">
                Confirm your name as tracked by SwingVision (optional)
              </span>
              <input
                type="text"
                name="tracked_identity"
                placeholder="e.g. Rishi Gupta"
                className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
              />
            </label>

            <div className="grid gap-4 sm:grid-cols-3">
              {[1, 2, 3].map((n) => (
                <label key={n} className="block text-sm">
                  <span className="text-[var(--color-ink-secondary)]">
                    Who served first in Set {n}?
                  </span>
                  <select
                    name={`first_server_set${n}`}
                    defaultValue=""
                    className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                  >
                    <option value="">Not sure / N/A</option>
                    <option value="me">Me</option>
                    <option value="opponent">Opponent</option>
                  </select>
                </label>
              ))}
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">Energy rating (1-5)</span>
                <input
                  type="number"
                  name="energy_rating"
                  min={1}
                  max={5}
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">Mental rating (1-5)</span>
                <input
                  type="number"
                  name="mental_rating"
                  min={1}
                  max={5}
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">Location</span>
                <input
                  type="text"
                  name="location"
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
            </div>

            {formError && <p className="text-sm text-[var(--color-critical)]">{formError}</p>}

            {phase === 'ready' && importResult && (
              <div className="rounded-xl border border-[var(--color-brand-blue)]/40 bg-[color-mix(in_srgb,var(--color-brand-blue)_10%,transparent)] p-4 text-sm">
                Staged as import #{importResult.staged_label} — every flagged point is
                confirmed. Ready to load.
              </div>
            )}

            {(phase === 'finalizing' || phase === 'done') && (
              <TennisBallProgress
                status={phase === 'finalizing' ? 'running' : 'done'}
                label="Loading match into your career stats…"
              />
            )}

            <div className="flex items-center gap-3">
              {phase === 'ready' ? (
                <button
                  type="button"
                  onClick={handleFinalize}
                  className="rounded-lg bg-gradient-brand px-5 py-2.5 text-sm font-semibold text-white"
                >
                  Finalize &amp; Load
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={phase === 'uploading'}
                  className="rounded-lg bg-gradient-brand px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-50"
                >
                  {phase === 'uploading' ? 'Staging…' : 'Load Data'}
                </button>
              )}
            </div>
          </form>
        )}
      </GlassCard>

      {phase === 'reviewing' && pendingDetail && (
        <ReviewModal
          detail={pendingDetail}
          onAllConfirmed={() => setPhase('ready')}
          onClose={() => setPhase('idle')}
        />
      )}
    </div>
  )
}

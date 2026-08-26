import { useEffect, useState } from 'react'
import { api, ApiError } from '../api'
import type { JournalFeedback, MatchStats } from '../types'
import GlassCard from '../components/GlassCard'
import Skeleton from '../components/Skeleton'
import StickyNote from '../components/StickyNote'

interface JournalProps {
  initialMatchId: number | null
  onMatchIdConsumed: () => void
}

export default function Journal({ initialMatchId, onMatchIdConsumed }: JournalProps) {
  const [matches, setMatches] = useState<MatchStats[] | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const [pros, setPros] = useState('')
  const [cons, setCons] = useState('')
  const [notes, setNotes] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  const [feedback, setFeedback] = useState<JournalFeedback | null>(null)
  const [coaching, setCoaching] = useState(false)
  const [coachError, setCoachError] = useState<string | null>(null)

  useEffect(() => {
    api.matches().then((list) => {
      const byDateDesc = [...list].sort((a, b) => b.date.localeCompare(a.date))
      setMatches(byDateDesc)
      if (initialMatchId !== null) {
        setSelectedId(initialMatchId)
        onMatchIdConsumed()
      } else if (byDateDesc.length > 0) {
        setSelectedId(byDateDesc[0].match_id)
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const selected = matches?.find((m) => m.match_id === selectedId) ?? null

  useEffect(() => {
    if (!selected) return
    setPros(selected.self_assessment.pros ?? '')
    setCons(selected.self_assessment.cons ?? '')
    setNotes(selected.self_assessment.notes ?? '')
    setFeedback(null)
    setCoachError(null)
  }, [selected])

  async function handleSave() {
    if (!selected) return
    setSaving(true)
    setSaveError(null)
    try {
      const updated = await api.updateJournal(selected.match_id, { pros, cons, notes })
      setMatches((prev) => prev?.map((m) => (m.match_id === updated.match_id ? updated : m)) ?? null)
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : 'Failed to save.')
    } finally {
      setSaving(false)
    }
  }

  async function handleCoach(force: boolean) {
    if (!selected) return
    setCoaching(true)
    setCoachError(null)
    try {
      const journalText = [pros, cons, notes].filter(Boolean).join('\n\n')
      setFeedback(await api.coach(selected.match_id, journalText, force))
    } catch (err) {
      setCoachError(err instanceof ApiError ? err.message : 'Failed to get coaching feedback.')
    } finally {
      setCoaching(false)
    }
  }

  if (matches === null) {
    return (
      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <Skeleton className="h-96" />
        <Skeleton className="h-96" />
      </div>
    )
  }

  if (matches.length === 0) {
    return (
      <GlassCard className="p-6 text-center text-[var(--color-ink-secondary)]">
        No matches loaded yet — head to Overview to load your first one.
      </GlassCard>
    )
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
      <div className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--color-ink-muted)]">
          Sticky notes
        </h2>
        {matches.map((m) => (
          <StickyNote
            key={m.match_id}
            match={m}
            selected={m.match_id === selectedId}
            onClick={() => setSelectedId(m.match_id)}
          />
        ))}
      </div>

      {selected && (
        <div className="grid gap-6 lg:grid-cols-2">
          <GlassCard className="p-6">
            <h2 className="font-[var(--font-display)] text-lg font-bold">
              {selected.opponent} — {selected.date}
            </h2>
            <div className="mt-4 space-y-4">
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">What went right</span>
                <textarea
                  value={pros}
                  onChange={(e) => setPros(e.target.value)}
                  rows={3}
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">What went wrong</span>
                <textarea
                  value={cons}
                  onChange={(e) => setCons(e.target.value)}
                  rows={3}
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
              <label className="block text-sm">
                <span className="text-[var(--color-ink-secondary)]">Other thoughts</span>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={3}
                  className="mt-1 block w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
                />
              </label>
              {saveError && <p className="text-sm text-[var(--color-critical)]">{saveError}</p>}
              <button
                onClick={handleSave}
                disabled={saving}
                className="rounded-lg bg-gradient-brand px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
              >
                {saving ? 'Saving…' : 'Save journal entry'}
              </button>
            </div>
          </GlassCard>

          <GlassCard className="p-6" glow={Boolean(feedback)}>
            <h2 className="font-[var(--font-display)] text-lg font-bold">Coach's feedback</h2>
            <p className="mt-1 text-sm text-[var(--color-ink-secondary)]">
              Grounded in this match's stats and what you wrote above.
            </p>

            {coachError && (
              <p className="mt-3 text-sm text-[var(--color-critical)]">{coachError}</p>
            )}

            {feedback ? (
              <div className="mt-4 space-y-3 whitespace-pre-line text-sm leading-relaxed text-[var(--color-ink-primary)]">
                {feedback.feedback}
              </div>
            ) : (
              <button
                onClick={() => handleCoach(false)}
                disabled={coaching}
                className="mt-4 rounded-lg border border-[var(--color-border)] px-4 py-2 text-sm text-[var(--color-ink-secondary)] transition hover:border-[var(--color-brand-blue)] hover:text-[var(--color-ink-primary)] disabled:opacity-50"
              >
                {coaching
                  ? 'Coach is reviewing…'
                  : 'Get coaching feedback (uses a cached result if available, otherwise spends API money)'}
              </button>
            )}

            {feedback && (
              <button
                onClick={() => handleCoach(true)}
                disabled={coaching}
                className="mt-4 text-xs text-[var(--color-ink-muted)] underline hover:text-[var(--color-ink-secondary)]"
              >
                Regenerate (spends API money)
              </button>
            )}
          </GlassCard>
        </div>
      )}
    </div>
  )
}

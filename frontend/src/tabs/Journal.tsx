import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { api, ApiError } from '../api'
import type { JournalFeedback, MatchStats } from '../types'
import Panel from '../components/Panel'
import Skeleton from '../components/Skeleton'
import StickyNote from '../components/StickyNote'
import Reveal from '../components/Reveal'
import Magnetic from '../components/Magnetic'

/** One journal prompt: a labelled textarea with a live character count, so
 *  the three of them read as one entry with three parts. */
function Prompt({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  placeholder: string
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className="field-label">{label}</span>
        <span className="eyebrow num">{value.trim() ? `${value.trim().length}` : '—'}</span>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
        placeholder={placeholder}
        className="field resize-y leading-relaxed"
      />
    </label>
  )
}

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
  const [saved, setSaved] = useState(false)
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
    setSaved(false)
  }, [selected])

  async function handleSave() {
    if (!selected) return
    setSaving(true)
    setSaveError(null)
    try {
      const updated = await api.updateJournal(selected.match_id, { pros, cons, notes })
      setMatches((prev) => prev?.map((m) => (m.match_id === updated.match_id ? updated : m)) ?? null)
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2400)
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
      <div className="grid gap-8 lg:grid-cols-[300px_1fr]">
        <Skeleton className="h-96" />
        <Skeleton className="h-96" />
      </div>
    )
  }

  if (matches.length === 0) {
    return (
      <div className="border-y border-[var(--color-line)] py-14 text-center">
        <p className="h2">No matches to write about yet.</p>
        <p className="lede mx-auto mt-4 text-center">
          Load one on the Overview tab and it will show up here as a card.
        </p>
      </div>
    )
  }

  return (
    <div className="grid gap-10 lg:grid-cols-[300px_1fr]">
      <div>
        <div className="flex items-baseline justify-between border-t border-[var(--color-line)] pt-5">
          <span className="eyebrow">The stack</span>
          <span className="eyebrow num">{matches.length}</span>
        </div>
        <p className="mt-3 text-[13px] text-[var(--color-dim)]">
          Pick a match. Cards can be pulled out of the stack — they always spring back.
        </p>
        <div className="mt-6 space-y-4">
          {matches.map((m) => (
            <StickyNote
              key={m.match_id}
              match={m}
              selected={m.match_id === selectedId}
              onClick={() => setSelectedId(m.match_id)}
            />
          ))}
        </div>
      </div>

      {selected && (
        <AnimatePresence mode="wait">
          <motion.div
            key={selected.match_id}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="space-y-10"
          >
            <div className="border-t border-[var(--color-line)] pt-6">
              <div className="flex flex-wrap items-baseline justify-between gap-4">
                <h2 className="h2">{selected.opponent}</h2>
                <span className="eyebrow num">
                  {selected.date} &middot; {selected.result === 'W' ? 'Won' : 'Lost'}
                </span>
              </div>
            </div>

            <div className="grid gap-10 xl:grid-cols-2">
              <div className="space-y-6">
                <Prompt
                  label="What went right"
                  value={pros}
                  onChange={setPros}
                  placeholder="The serve pattern that kept working, the return position that finally stuck…"
                />
                <Prompt
                  label="What went wrong"
                  value={cons}
                  onChange={setCons}
                  placeholder="Where the points actually got away from you."
                />
                <Prompt
                  label="Other thoughts"
                  value={notes}
                  onChange={setNotes}
                  placeholder="Conditions, body, nerves, anything the tracking can't see."
                />

                {saveError && (
                  <p className="text-[14px]" style={{ color: 'var(--color-clay)' }}>
                    {saveError}
                  </p>
                )}

                <div className="flex items-center gap-4">
                  <Magnetic>
                    <button className="btn" onClick={handleSave} disabled={saving}>
                      {saving ? 'Saving…' : 'Save entry'}
                    </button>
                  </Magnetic>
                  <AnimatePresence>
                    {saved && (
                      <motion.span
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0 }}
                        className="eyebrow"
                        style={{ color: 'var(--color-teal)' }}
                      >
                        Saved
                      </motion.span>
                    )}
                  </AnimatePresence>
                </div>
              </div>

              <Reveal>
                <Panel spotlight className="h-full p-7">
                  <div className="flex items-baseline justify-between gap-4">
                    <h3 className="h3">Coach's read</h3>
                    <span className="eyebrow">Claude</span>
                  </div>
                  <p className="lede mt-3 text-[14px]">
                    Grounded in this match's confirmed stats and what you just wrote — nothing
                    else.
                  </p>

                  {coachError && (
                    <p className="mt-4 text-[14px]" style={{ color: 'var(--color-clay)' }}>
                      {coachError}
                    </p>
                  )}

                  <AnimatePresence mode="wait">
                    {feedback ? (
                      <motion.div
                        key="feedback"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-6 whitespace-pre-line border-t border-[var(--color-line)] pt-6 text-[15px] leading-[1.7]"
                      >
                        {feedback.feedback}
                      </motion.div>
                    ) : (
                      <motion.div key="ask" exit={{ opacity: 0 }} className="mt-6">
                        <button
                          className="btn btn-ghost"
                          onClick={() => handleCoach(false)}
                          disabled={coaching}
                        >
                          {coaching ? 'Reading the match…' : 'Ask the coach'}
                        </button>
                        <p className="mt-3 text-[12px] text-[var(--color-dim)]">
                          Uses a cached read if one exists; otherwise this spends API credit.
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {feedback && (
                    <button
                      onClick={() => handleCoach(true)}
                      disabled={coaching}
                      className="link mt-6 text-[12px] text-[var(--color-dim)]"
                    >
                      Regenerate (spends API credit)
                    </button>
                  )}
                </Panel>
              </Reveal>
            </div>
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { api, ApiError } from '../api'
import type { CareerStats, ImportResult, PendingDetail, StagedMatch } from '../types'
import Panel from '../components/Panel'
import StatTile from '../components/StatTile'
import Skeleton from '../components/Skeleton'
import ReviewModal from '../components/ReviewModal'
import TennisBallProgress from '../components/TennisBallProgress'
import Reveal from '../components/Reveal'
import Magnetic from '../components/Magnetic'
import Marquee from '../components/Marquee'
import Dropzone from '../components/Dropzone'
import PartPairing from '../components/PartPairing'

type Phase = 'idle' | 'uploading' | 'reviewing' | 'ready' | 'finalizing' | 'done'

interface OverviewProps {
  onMatchFinalized: (matchId: number) => void
}

/** A numbered rule + title, the section opener used down the whole page. */
function SectionHead({ index, title, lede }: { index: string; title: string; lede?: string }) {
  return (
    <div className="border-t border-[var(--color-line)] pt-6">
      <div className="flex items-baseline justify-between gap-6">
        <h2 className="h2 max-w-[20ch]">{title}</h2>
        <span className="eyebrow num shrink-0">{index}</span>
      </div>
      {lede && <p className="lede mt-4 text-[15px]">{lede}</p>}
    </div>
  )
}

export default function Overview({ onMatchFinalized }: OverviewProps) {
  const [overview, setOverview] = useState<CareerStats | null>(null)
  const [loadingOverview, setLoadingOverview] = useState(true)

  const [phase, setPhase] = useState<Phase>('idle')
  const [importResult, setImportResult] = useState<ImportResult | null>(null)
  const [pendingDetail, setPendingDetail] = useState<PendingDetail | null>(null)
  const [finalizedMatchId, setFinalizedMatchId] = useState<number | null>(null)
  const [formError, setFormError] = useState<string | null>(null)
  // Bumped on reset: remounting the form is what clears the dropzones' own
  // filename state, which a plain form.reset() can't reach.
  const [formKey, setFormKey] = useState(0)
  // Matches staged by an earlier upload that still need reviewing - without
  // this they'd be reachable only in the browser session that staged them.
  const [staged, setStaged] = useState<StagedMatch[]>([])
  // Mirrors of what's currently selected in the two dropzones, purely so
  // the pairing preview can react as files are picked - the inputs
  // themselves still own the files, and FormData still reads them.
  const [exportNames, setExportNames] = useState<string[]>([])
  const [videoNames, setVideoNames] = useState<string[]>([])
  const formRef = useRef<HTMLFormElement>(null)

  async function loadOverview() {
    setLoadingOverview(true)
    try {
      setOverview(await api.overview())
    } finally {
      setLoadingOverview(false)
    }
  }

  async function loadStaged() {
    try {
      setStaged(await api.pendingMatches())
    } catch {
      setStaged([]) // a listing failure shouldn't block the upload form
    }
  }

  useEffect(() => {
    loadOverview()
    loadStaged()
  }, [])

  async function resumeReview(jsonFilename: string) {
    setFormError(null)
    try {
      const detail = await api.pendingDetail(jsonFilename)
      setPendingDetail(detail)
      // A resumed review has no upload response to read from, so what the
      // staged file itself records stands in for it. The match number is
      // the one thing it cannot know (0 = unknown, and not displayed).
      setImportResult({
        json_filename: jsonFilename,
        staged_label: 0,
        date: detail.date,
        opponent: detail.opponent,
        flags: [],
        import_notes: detail.import_notes,
        parts: [],
        upload_notes: [],
      })
      setPhase('reviewing')
    } catch (err) {
      setFormError(
        err instanceof ApiError ? err.message : 'Failed to reopen that staged match.',
      )
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    // Only a fresh, idle form is actually ready for a new upload - once a
    // submission is staged/reviewing/finalizing/done, any further submit
    // event on this form (e.g. a stray re-fire from the Finalize button,
    // which shares the form but is type="button") must be a no-op rather
    // than re-staging the same match a second time.
    if (phase !== 'idle') return
    if (!formRef.current) return
    const data = new FormData(formRef.current)
    const exports = data.getAll('xlsx_files').filter((f) => f instanceof File && f.size > 0)
    if (exports.length === 0) {
      setFormError('Add at least one SwingVision .xlsx export before staging.')
      return
    }
    setFormError(null)
    setPhase('uploading')
    try {
      const result = await api.importMatch(data)
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
    setImportResult(null)
    setPendingDetail(null)
    setFinalizedMatchId(null)
    setFormError(null)
    setPhase('idle')
    loadStaged()
    setExportNames([])
    setVideoNames([])
    setFormKey((k) => k + 1)
  }

  const hasMatches = Boolean(overview && overview.total_matches > 0)

  const tickerItems =
    overview && hasMatches
      ? [
          `${overview.wins} won`,
          `${overview.losses} lost`,
          `${overview.win_pct}% win rate`,
          `${overview.avg_first_serve_pct}% first serves in`,
          `${overview.avg_points_won_pct}% of points won`,
          `${overview.total_matches} matches on record`,
        ]
      : []

  return (
    <div className="space-y-16">
      {loadingOverview ? (
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      ) : hasMatches && overview ? (
        <section className="space-y-10">
          <div className="grid grid-cols-2 gap-x-8 gap-y-10 sm:grid-cols-3 lg:grid-cols-5">
            <StatTile
              label="Record"
              value={`${overview.wins}–${overview.losses}`}
              sublabel={`${overview.total_matches} matches`}
            />
            <StatTile
              label="Win rate"
              value={overview.win_pct}
              suffix="%"
              decimals={1}
              accent={overview.win_pct >= 50 ? 'good' : 'critical'}
              delay={0.05}
            />
            <StatTile
              label="Current streak"
              value={
                overview.current_streak_result
                  ? `${overview.current_streak_count}${overview.current_streak_result}`
                  : '—'
              }
              sublabel={
                overview.current_streak_result === 'W'
                  ? 'wins in a row'
                  : overview.current_streak_result === 'L'
                    ? 'losses in a row'
                    : undefined
              }
              accent={
                overview.current_streak_result === 'W'
                  ? 'good'
                  : overview.current_streak_result === 'L'
                    ? 'critical'
                    : 'ink'
              }
              delay={0.1}
            />
            <StatTile
              label="First serve"
              value={overview.avg_first_serve_pct}
              suffix="%"
              decimals={1}
              sublabel="match average"
              delay={0.15}
            />
            <StatTile
              label="Points won"
              value={overview.avg_points_won_pct}
              suffix="%"
              decimals={1}
              sublabel="match average"
              delay={0.2}
            />
          </div>

          <Marquee items={tickerItems} />

          <div className="grid gap-6 sm:grid-cols-2">
            {overview.best_match_by_points_won_pct && (
              <Reveal>
                <Panel tilt spotlight className="h-full p-7">
                  <p className="eyebrow">{overview.best_match_by_points_won_pct.label}</p>
                  <p className="h2 mt-4">
                    <span className="num">{overview.best_match_by_points_won_pct.value}%</span>
                  </p>
                  <p className="mt-3 text-[15px] text-[var(--color-dim)]">
                    versus {overview.best_match_by_points_won_pct.opponent}
                    <span className="num ml-2 text-[13px]">
                      {overview.best_match_by_points_won_pct.date}
                    </span>
                  </p>
                </Panel>
              </Reveal>
            )}
            {overview.most_aces_in_a_match && (
              <Reveal delay={0.08}>
                <Panel tilt spotlight className="h-full p-7">
                  <p className="eyebrow">{overview.most_aces_in_a_match.label}</p>
                  <p className="h2 mt-4">
                    <span className="num">{overview.most_aces_in_a_match.value}</span> aces
                  </p>
                  <p className="mt-3 text-[15px] text-[var(--color-dim)]">
                    versus {overview.most_aces_in_a_match.opponent}
                    <span className="num ml-2 text-[13px]">
                      {overview.most_aces_in_a_match.date}
                    </span>
                  </p>
                </Panel>
              </Reveal>
            )}
          </div>
        </section>
      ) : (
        <Reveal>
          <div className="border-y border-[var(--color-line)] py-14 text-center">
            <p className="h2 mx-auto max-w-[18ch]">Nothing on record yet.</p>
            <p className="lede mx-auto mt-4 text-center">
              Stage your first SwingVision export below and it will appear here the moment you
              finish confirming its flagged points.
            </p>
          </div>
        </Reveal>
      )}

      {staged.length > 0 && (
        <Reveal>
          <section>
            <SectionHead
              index="Awaiting review"
              title="Staged, not yet loaded"
              lede="These were uploaded but still have points to confirm. Nothing reaches your career stats until every flag is cleared."
            />
            <div className="mt-6 space-y-3">
              {staged.map((match) => (
                <div
                  key={match.json_filename}
                  className="flex flex-wrap items-baseline justify-between gap-4 border-t border-[var(--color-line)] pt-4"
                >
                  <div>
                    <p className="h3">{match.opponent || match.json_filename}</p>
                    <p className="num mt-1 text-[12px] text-[var(--color-dim)]">
                      {match.date} &middot; {match.flags_remaining} of {match.total_points} points
                      still flagged
                      {match.confirmed_by_footage > 0 &&
                        ` · ${match.confirmed_by_footage} confirmed by footage`}
                    </p>
                  </div>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => resumeReview(match.json_filename)}
                  >
                    Resume review
                  </button>
                </div>
              ))}
            </div>
          </section>
        </Reveal>
      )}

      <section id="load" className="scroll-mt-24">
        <Reveal>
          <SectionHead
            index="Step 01"
            title="Load a match"
            lede="The export goes to a staging file first, never straight into your stats. Anything the reconstruction wasn't sure about comes back to you for a decision."
          />
        </Reveal>

        <AnimatePresence mode="wait">
          {phase === 'done' && finalizedMatchId !== null ? (
            <motion.div
              key="done"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              className="mt-8"
            >
              <Panel className="p-8" spotlight>
                <p className="eyebrow" style={{ color: 'var(--color-win)' }}>
                  On the record
                </p>
                <p className="h2 mt-4">
                  Match <span className="num">#{finalizedMatchId}</span> is in your career stats.
                </p>
                <p className="lede mt-4 text-[15px]">
                  Write the journal entry while it's fresh — the reflection reads what you
                  wrote, not just what the tracking saw.
                </p>
                <div className="mt-7 flex flex-wrap items-center gap-3">
                  <Magnetic>
                    <button className="btn" onClick={() => onMatchFinalized(finalizedMatchId)}>
                      Write the entry
                      <span aria-hidden>&#8594;</span>
                    </button>
                  </Magnetic>
                  <button className="btn btn-ghost" onClick={resetForm}>
                    Load another
                  </button>
                </div>
              </Panel>
            </motion.div>
          ) : (
            <motion.form
              key={`form-${formKey}`}
              ref={formRef}
              onSubmit={handleSubmit}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mt-8 space-y-9"
            >
              <div className="grid gap-6 sm:grid-cols-2">
                <Dropzone
                  name="xlsx_files"
                  label="SwingVision exports"
                  accept=".xlsx"
                  multiple
                  onNamesChange={setExportNames}
                  hint="Required &middot; one, or several for a split recording"
                />
                <Dropzone
                  name="video_files"
                  label="Match footage"
                  multiple
                  onNamesChange={setVideoNames}
                  hint="Optional &middot; name each clip after its export"
                />
              </div>

              {exportNames.length > 0 && (
                <PartPairing exports={exportNames} videos={videoNames} />
              )}

              <div className="grid gap-6 sm:grid-cols-3">
                <label>
                  <span className="field-label">Date</span>
                  <input type="date" name="date" required className="field num" />
                </label>
                <label>
                  <span className="field-label">Opponent</span>
                  <input type="text" name="opponent" required className="field" />
                </label>
                <label>
                  <span className="field-label">Result</span>
                  <select name="result" required className="field" defaultValue="W">
                    <option value="W">Win</option>
                    <option value="L">Loss</option>
                  </select>
                </label>
              </div>

              <div className="grid gap-6 sm:grid-cols-3">
                <label className="sm:col-span-2">
                  <span className="field-label">Your name as SwingVision tracked it</span>
                  <input
                    type="text"
                    name="tracked_identity"
                    placeholder="e.g. Rishi Gupta"
                    className="field"
                  />
                </label>
                <label>
                  <span className="field-label">Location</span>
                  <input type="text" name="location" className="field" />
                </label>
              </div>

              <div className="grid gap-6 sm:grid-cols-3">
                {[1, 2, 3].map((n) => (
                  <label key={n}>
                    <span className="field-label">Served first, set {n}</span>
                    <select name={`first_server_set${n}`} defaultValue="" className="field">
                      <option value="">Not sure / N/A</option>
                      <option value="me">Me</option>
                      <option value="opponent">Opponent</option>
                    </select>
                  </label>
                ))}
              </div>

              {/* The one field that exists for the AI rather than the record:
                  whatever the reconstruction can't see for itself. It matters
                  most exactly when several exports are involved, so the
                  prompt says so and the placeholder changes to match. */}
              <label className="block">
                <div className="flex items-baseline justify-between">
                  <span className="field-label">What you were wearing</span>
                  <span className="eyebrow num">optional</span>
                </div>
                <input
                  name="player_appearance"
                  type="text"
                  className="field"
                  placeholder="e.g. grey shirt, black shorts"
                />
                <p className="mt-2 text-[12px] leading-relaxed text-[var(--color-dim)]">
                  Used only to pick you out of the frame during a footage check. You swap
                  ends at changeovers, so which side of the court you are on is not a
                  stable way to say which player is you &mdash; what you wore is.
                </p>
              </label>

              <label className="block">
                <div className="flex items-baseline justify-between">
                  <span className="field-label">
                    Total rallies in the match (from the footage)
                  </span>
                  <span className="eyebrow num">optional</span>
                </div>
                <input
                  name="footage_rally_count"
                  type="number"
                  min={1}
                  step={1}
                  inputMode="numeric"
                  className="field num"
                  placeholder="e.g. 123"
                />
                <p className="mt-2 text-[12px] leading-relaxed text-[var(--color-dim)]">
                  SwingVision&rsquo;s &ldquo;all rallies&rdquo; export shows this number. A
                  rallies-only video holds fewer rallies than the export has points, so
                  this is the most a footage check can ever reach &mdash; it is shown beside
                  every &ldquo;of N points played&rdquo; figure.
                </p>
              </label>

              <label className="block">
                <div className="flex items-baseline justify-between">
                  <span className="field-label">
                    Notes for the review &mdash; anything odd about this import
                  </span>
                  <span className="eyebrow num">
                    {exportNames.length > 1 ? `${exportNames.length} parts` : 'optional'}
                  </span>
                </div>
                <textarea
                  name="reviewer_note"
                  rows={3}
                  className="field resize-y leading-relaxed"
                  placeholder={
                    exportNames.length > 1
                      ? 'e.g. part 2 restarts mid-game; I switched ends between files; the last game of part 1 is repeated at the start of part 2.'
                      : 'e.g. the recording cut out at 3-2; a let was replayed but tracked as a point; I was tracked as the guest, not the host.'
                  }
                />
                <p className="mt-2 text-[12px] leading-relaxed text-[var(--color-dim)]">
                  Read by Claude when it suggests a call on a flagged point, and shown back
                  to you during review. It never resolves a flag on its own.
                </p>
              </label>

              <div className="grid gap-6 sm:grid-cols-2">
                <label>
                  <span className="field-label">Energy rating (1&ndash;5)</span>
                  <input type="number" name="energy_rating" min={1} max={5} className="field num" />
                </label>
                <label>
                  <span className="field-label">Mental rating (1&ndash;5)</span>
                  <input type="number" name="mental_rating" min={1} max={5} className="field num" />
                </label>
              </div>

              <AnimatePresence>
                {formError && (
                  <motion.p
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="text-[14px]"
                    style={{ color: 'var(--color-loss)' }}
                  >
                    {formError}
                  </motion.p>
                )}
              </AnimatePresence>

              {phase === 'ready' && importResult && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="panel-sunk p-5 text-[14px]"
                >
                  <p>
                    {importResult.staged_label > 0 ? (
                      <>
                        Staged as <span className="num">{importResult.staged_label}</span>
                      </>
                    ) : (
                      'Staged'
                    )}{' '}
                    — every flagged point is confirmed. Ready to load.
                  </p>
                  {importResult.parts.length > 1 && (
                    <p className="mt-2 text-[13px] text-[var(--color-dim)]">
                      Merged{' '}
                      <span className="num">{importResult.parts.length}</span> exports into one
                      continuous match.
                    </p>
                  )}
                  {/* What the server actually did with the filenames, and
                      anything the reconstruction wants the reader to know -
                      both informational, neither blocks finalizing. */}
                  {[...importResult.upload_notes, ...importResult.import_notes].map((note) => (
                    <p
                      key={note}
                      className="mt-2 border-t border-[var(--color-line)] pt-2 text-[12px] leading-relaxed text-[var(--color-dim)]"
                    >
                      {note}
                    </p>
                  ))}
                </motion.div>
              )}

              {(phase === 'finalizing' || phase === 'done') && (
                <TennisBallProgress
                  status={phase === 'finalizing' ? 'running' : 'done'}
                  label="Writing the match into your career stats…"
                />
              )}

              <div className="flex items-center gap-3 border-t border-[var(--color-line)] pt-8">
                {phase === 'ready' ? (
                  <Magnetic>
                    <button type="button" className="btn" onClick={handleFinalize}>
                      Finalize &amp; load
                      <span aria-hidden>&#8594;</span>
                    </button>
                  </Magnetic>
                ) : (
                  <Magnetic>
                    <button type="submit" className="btn" disabled={phase === 'uploading'}>
                      {phase === 'uploading' ? 'Staging…' : 'Stage this match'}
                      <span aria-hidden>&#8594;</span>
                    </button>
                  </Magnetic>
                )}
              </div>
            </motion.form>
          )}
        </AnimatePresence>
      </section>

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

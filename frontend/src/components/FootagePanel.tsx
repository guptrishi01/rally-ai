import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { api, ApiError } from '../api'
import type { FootageStatus } from '../types'

interface FootagePanelProps {
  jsonFilename: string
  /** Called when a run finishes, so the caller can reload the points it
   *  auto-confirmed out of the queue. */
  onFinished: () => void
}

/**
 * Checking a staged match's flagged points against its own footage.
 *
 * Deliberately never automatic: a run costs real money, so the estimate is
 * shown first and nothing starts until it's clicked. When footage can't be
 * used — most often SwingVision's condensed export, whose timestamps point
 * at the full recording — this says so plainly instead of offering a button
 * that would check the wrong rallies.
 */
export default function FootagePanel({ jsonFilename, onFinished }: FootagePanelProps) {
  const [status, setStatus] = useState<FootageStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const finishedRef = useRef(false)

  async function load() {
    try {
      setStatus(await api.footageStatus(jsonFilename))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to read footage status.')
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jsonFilename])

  // Poll only while a run is actually going, and tell the caller once, so a
  // finished run doesn't keep refetching the whole review queue.
  const running = status?.job?.state === 'running'
  useEffect(() => {
    if (!running) {
      if (status?.job?.state === 'done' && !finishedRef.current) {
        finishedRef.current = true
        onFinished()
      }
      return
    }
    const id = window.setInterval(load, 2000)
    return () => window.clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, status?.job?.state])

  async function start() {
    setStarting(true)
    setError(null)
    finishedRef.current = false
    try {
      await api.startFootageCheck(jsonFilename)
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to start the footage check.')
    } finally {
      setStarting(false)
    }
  }

  if (!status) return null

  const job = status.job
  const progress = job?.progress
  const summary = job?.summary
  const total = status.eligible_points - status.already_checked

  return (
    <div className="panel-sunk mt-5 p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <span className="eyebrow">Check against footage</span>
        {status.available && !job && (
          <span className="eyebrow num">
            {total} points &middot; ~${status.estimate_usd?.toFixed(2)}
          </span>
        )}
      </div>

      {/* Why it can't run, when it can't. */}
      {!status.available && !job && (
        <p className="mt-3 text-[13px] leading-relaxed text-[var(--color-dim)]">
          {status.reason ?? 'Footage checking is unavailable for this match.'}
        </p>
      )}

      {status.available && !job && (
        <>
          <p className="mt-3 text-[13px] leading-relaxed text-[var(--color-dim)]">
            Claude watches the video around each flagged point and judges it without being
            told what the tracker recorded. Points where both agree are confirmed for you;
            anything it can&rsquo;t settle still comes to you, with its reading attached.
          </p>
          <button className="btn btn-sm mt-4" onClick={start} disabled={starting}>
            {starting ? 'Starting…' : `Check ${total} points (~$${status.estimate_usd?.toFixed(2)})`}
          </button>
        </>
      )}

      <AnimatePresence>
        {running && progress && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4">
            <div className="h-px w-full bg-[var(--color-line-strong)]">
              <motion.div
                className="h-px bg-[var(--color-accent)]"
                animate={{ width: `${total ? (progress.checked / total) * 100 : 0}%` }}
                transition={{ type: 'spring', stiffness: 120, damping: 24 }}
              />
            </div>
            <p className="num mt-3 text-[12px] text-[var(--color-dim)]">
              {progress.checked} / {total} checked &middot; {progress.agreed} confirmed
              {progress.not_in_footage > 0 && (
                <>
                  {' '}
                  &middot; {progress.not_in_footage} not in this footage
                </>
              )}{' '}
              &middot; ${progress.usd.toFixed(2)} so far
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {running && !progress && (
        <p className="mt-4 text-[13px] text-[var(--color-dim)]">Pulling frames…</p>
      )}

      {summary && (
        <div className="mt-4 border-t border-[var(--color-line)] pt-4">
          {/* The denominator always goes first. "Confirmed" alone invites
              reading it as coverage; "confirmed, of N points played" makes
              the missing ones visible instead of invisible. */}
          <p className="num mb-2 text-[12px] text-[var(--color-dim)]">
            Of {summary.points_played} points played
          </p>
          <p className="text-[14px]">
            <span className="num" style={{ color: 'var(--color-win)' }}>
              {summary.agreed}
            </span>{' '}
            confirmed by footage &middot;{' '}
            <span className="num">{summary.disagreed}</span> disagreed &middot;{' '}
            <span className="num">{summary.inconclusive}</span> unclear
            {summary.not_in_footage > 0 && (
              <>
                {' '}
                &middot; <span className="num">{summary.not_in_footage}</span> not in this
                footage
              </>
            )}
            {summary.failed > 0 && (
              <>
                {' '}
                &middot; <span className="num">{summary.failed}</span> failed
              </>
            )}
          </p>
          <p className="num mt-2 text-[12px] text-[var(--color-dim)]">
            Cost ${summary.usd.toFixed(2)}
          </p>
          {summary.not_in_footage > 0 && (
            <p className="mt-2 text-[12px] leading-relaxed text-[var(--color-dim)]">
              SwingVision&rsquo;s rallies-only export does not contain points that
              ended on a serve or return. Those stay flagged for you to settle - we
              can&rsquo;t verify what isn&rsquo;t in the file.
            </p>
          )}
          {summary.notes.map((note) => (
            <p key={note} className="mt-3 text-[13px]" style={{ color: 'var(--color-loss)' }}>
              {note}
            </p>
          ))}
        </div>
      )}

      {job?.state === 'failed' && (
        <p className="mt-3 text-[13px]" style={{ color: 'var(--color-loss)' }}>
          {job.error}
        </p>
      )}

      {error && (
        <p className="mt-3 text-[13px]" style={{ color: 'var(--color-loss)' }}>
          {error}
        </p>
      )}
    </div>
  )
}

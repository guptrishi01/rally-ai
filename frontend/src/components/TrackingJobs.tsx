import { useCallback, useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { api, ApiError } from '../api'
import type { TrackingJob } from '../types'

interface TrackingJobsProps {
  /** Bumped by the parent after it starts a job, to fetch at once. */
  refreshKey: number
  /** Opens the review modal on a staged match. */
  onReview: (jsonFilename: string) => void
  /** Called when a job is first seen staged, so the staged list refreshes. */
  onStaged: () => void
}

const POLL_MS = 5000

const STAGES: Record<string, string> = {
  opening: 'Reading the frame timestamps',
  calibrating: 'Calibrating the court',
  tracking: 'Tracking',
  assembling: 'Assembling points',
  staged: 'Staged',
  skipped: 'Already staged',
}

/** h:mm:ss, or "unknown" before the tracker has timed a chunk. */
function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return 'unknown'
  const whole = Math.max(0, Math.round(seconds))
  const h = Math.floor(whole / 3600)
  const m = Math.floor((whole % 3600) / 60)
  const s = whole % 60
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function stateLabel(job: TrackingJob): { text: string; color: string } {
  switch (job.state) {
    case 'running':
      return { text: 'Tracking', color: 'var(--color-accent-deep)' }
    case 'staged':
      return { text: 'Ready to review', color: 'var(--color-win)' }
    case 'failed':
      return { text: 'Stopped with an error', color: 'var(--color-loss)' }
    default:
      return { text: 'Interrupted', color: 'var(--color-loss)' }
  }
}

/**
 * Video-tracking jobs: each one is its own OS process, hours long, so this
 * only reads how far it has got. A stopped job is never restarted on its
 * own - the person presses Resume, and finished chunks are kept.
 */
export default function TrackingJobs({ refreshKey, onReview, onStaged }: TrackingJobsProps) {
  const [jobs, setJobs] = useState<TrackingJob[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const seenStaged = useRef<Set<string>>(new Set())

  const load = useCallback(async () => {
    try {
      const fresh = await api.trackingJobs()
      setJobs(fresh)
      let newlyStaged = false
      for (const job of fresh) {
        if (job.state === 'staged' && !seenStaged.current.has(job.json_filename)) {
          seenStaged.current.add(job.json_filename)
          newlyStaged = true
        }
      }
      if (newlyStaged) onStaged()
    } catch {
      // A failed poll is not worth interrupting anyone for; the next one
      // will try again.
    }
  }, [onStaged])

  useEffect(() => {
    load()
  }, [load, refreshKey])

  const anyRunning = jobs.some((job) => job.state === 'running')
  useEffect(() => {
    if (!anyRunning) return
    const timer = window.setInterval(load, POLL_MS)
    return () => window.clearInterval(timer)
  }, [anyRunning, load])

  async function act(job: TrackingJob, action: 'resume' | 'dismiss') {
    setBusy(job.json_filename)
    setError(null)
    try {
      if (action === 'resume') await api.resumeTracking(job.json_filename)
      else await api.dismissTracking(job.json_filename)
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'That did not work - try again.')
    } finally {
      setBusy(null)
    }
  }

  if (jobs.length === 0) return null

  return (
    <section data-testid="tracking-jobs">
      <div className="border-t border-[var(--color-line)] pt-6">
        <div className="flex items-baseline justify-between gap-6">
          <h2 className="h2 max-w-[20ch]">Tracking from video</h2>
          <span className="eyebrow num shrink-0">Long-running</span>
        </div>
        <p className="lede mt-4 text-[15px]">
          Each video is tracked in its own process &mdash; hours for a full match. You can
          close this tab or the browser and it keeps going. If the server or the computer
          restarts, come back and press Resume: finished chunks are kept.
        </p>
      </div>

      {error && (
        <p className="mt-4 text-[14px]" style={{ color: 'var(--color-loss)' }}>
          {error}
        </p>
      )}

      <div className="mt-6 space-y-4">
        {jobs.map((job) => {
          const label = stateLabel(job)
          const p = job.progress
          const share = p && p.chunks_total > 0 ? p.chunks_done / p.chunks_total : 0
          return (
            <div
              key={job.json_filename}
              className="panel-sunk p-5"
              data-testid="tracking-job"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-3">
                <div className="min-w-0">
                  <p className="h3 break-words">{job.opponent}</p>
                  <p className="num mt-1 break-all text-[12px] text-[var(--color-dim)]">
                    {job.date} &middot; {job.video}
                    {job.launches > 1 && ` · launch ${job.launches}`}
                  </p>
                </div>
                <span className="eyebrow" style={{ color: label.color }}>
                  {label.text}
                </span>
              </div>

              {p && job.state !== 'staged' && (
                <div className="mt-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-2 text-[13px]">
                    <span>{STAGES[p.stage] ?? p.stage}</span>
                    {p.chunks_total > 0 && (
                      <span className="num text-[var(--color-dim)]">
                        chunk {p.chunks_done} of {p.chunks_total}
                        {p.chunks_resumed > 0 && ` (${p.chunks_resumed} from checkpoints)`}
                      </span>
                    )}
                  </div>
                  <div className="mt-2 h-1 overflow-hidden rounded-full bg-[var(--color-line)]">
                    <motion.div
                      className="h-1 rounded-full bg-[var(--color-accent)]"
                      animate={{ width: `${share * 100}%` }}
                      transition={{ type: 'spring', stiffness: 120, damping: 24 }}
                    />
                  </div>
                  <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-[12px] sm:grid-cols-3">
                    <div>
                      <dt className="eyebrow">Video done</dt>
                      <dd className="num mt-1">
                        {(p.video_seconds_done / 60).toFixed(1)} of{' '}
                        {(p.video_seconds / 60).toFixed(1)} min
                      </dd>
                    </div>
                    <div>
                      <dt className="eyebrow">Time left</dt>
                      <dd className="num mt-1">{clock(p.eta_seconds)}</dd>
                    </div>
                    <div>
                      <dt className="eyebrow">Speed</dt>
                      <dd className="num mt-1">
                        {p.seconds_per_video_minute === null
                          ? 'timed after the first chunk'
                          : `${Math.round(p.seconds_per_video_minute)}s per video minute`}
                      </dd>
                    </div>
                  </dl>
                  {p.message && (
                    <p className="mt-3 text-[12px] leading-relaxed text-[var(--color-dim)]">
                      {p.message}
                    </p>
                  )}
                </div>
              )}

              {!p && job.state === 'running' && (
                <p className="mt-4 text-[13px] text-[var(--color-dim)]">
                  Starting up &mdash; progress appears once the tracker has opened the video.
                </p>
              )}

              {job.error && (
                <p className="mt-3 break-words text-[13px]" style={{ color: 'var(--color-loss)' }}>
                  {job.error}
                </p>
              )}
              {job.state === 'interrupted' && (
                <p className="mt-3 text-[13px] text-[var(--color-dim)]">
                  The tracking process is no longer running and did not finish. Resume carries
                  on from the last finished chunk.
                </p>
              )}

              <div className="mt-4 flex flex-wrap gap-3">
                {job.state === 'staged' && (
                  <button className="btn btn-sm" onClick={() => onReview(job.json_filename)}>
                    Review the points
                  </button>
                )}
                {(job.state === 'interrupted' || job.state === 'failed') && (
                  <button
                    className="btn btn-sm"
                    disabled={busy === job.json_filename}
                    onClick={() => act(job, 'resume')}
                  >
                    Resume tracking
                  </button>
                )}
                {job.state !== 'running' && (
                  <button
                    className="btn btn-ghost btn-sm"
                    disabled={busy === job.json_filename}
                    onClick={() => act(job, 'dismiss')}
                  >
                    Dismiss
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

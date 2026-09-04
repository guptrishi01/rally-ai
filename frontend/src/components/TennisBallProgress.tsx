import { useEffect, useState } from 'react'
import { motion } from 'motion/react'

interface TennisBallProgressProps {
  status: 'running' | 'done'
  label: string
}

/** A small drawn ball - seam and all - rather than the 🎾 emoji, which
 *  rendered as a different picture on every platform and clashed with the
 *  rest of the type. */
function Ball({ spin }: { spin: boolean }) {
  return (
    <motion.svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      aria-hidden
      animate={spin ? { rotate: 360 } : { rotate: 0 }}
      transition={spin ? { duration: 1.6, ease: 'linear', repeat: Infinity } : { duration: 0.4 }}
    >
      <circle cx="12" cy="12" r="10" fill="var(--color-accent)" />
      <path
        d="M4 5c4 3 4 11 0 14M20 5c-4 3-4 11 0 14"
        fill="none"
        stroke="var(--color-paper-3)"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </motion.svg>
  )
}

/**
 * A ball travelling a hairline track from 0% to 100%.
 *
 * Eases toward ~90% while `status` is "running" (an indeterminate animation,
 * since /finalize is one synchronous request with no incremental progress
 * signal) and snaps to 100% only once the caller flips `status` to "done" -
 * tied to the real request, never to a fixed fake duration.
 */
export default function TennisBallProgress({ status, label }: TennisBallProgressProps) {
  const [percent, setPercent] = useState(4)

  useEffect(() => {
    if (status === 'done') {
      setPercent(100)
      return
    }
    setPercent(4)
    const id = window.setInterval(() => {
      setPercent((p) => (p >= 90 ? p : p + (90 - p) * 0.12))
    }, 180)
    return () => window.clearInterval(id)
  }, [status])

  return (
    <div className="w-full py-2">
      <div className="relative h-px w-full bg-[var(--color-line-strong)]">
        <motion.div
          className="absolute inset-y-0 left-0 bg-[var(--color-accent)]"
          animate={{ width: `${percent}%` }}
          transition={{ type: 'spring', stiffness: 90, damping: 20 }}
        />
        <motion.div
          className="absolute top-1/2 -mt-[9px]"
          animate={{ left: `calc(${percent}% - 9px)` }}
          transition={{ type: 'spring', stiffness: 90, damping: 20 }}
        >
          <Ball spin={status === 'running'} />
        </motion.div>
      </div>
      <p className="mt-4 flex items-center gap-2 text-[13px] text-[var(--color-dim)]">
        <span className="num tabular-nums">{Math.round(percent)}%</span>
        <span>{status === 'done' ? 'Loaded.' : label}</span>
      </p>
    </div>
  )
}

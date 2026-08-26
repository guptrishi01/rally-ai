import { useEffect, useState } from 'react'
import { motion } from 'motion/react'

interface TennisBallProgressProps {
  status: 'running' | 'done'
  label: string
}

/**
 * A tennis ball racing along a track from 0% to 100%. Eases toward ~90%
 * while `status` is "running" (an indeterminate animation, since /finalize
 * is one synchronous request with no incremental progress signal) and
 * snaps to 100% only once the caller flips `status` to "done" - tied to
 * the real request, not a fixed fake duration.
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
      <div className="relative h-3 w-full rounded-full bg-[var(--color-border)]">
        <motion.div
          className="h-3 rounded-full bg-gradient-brand"
          animate={{ width: `${percent}%` }}
          transition={{ type: 'spring', stiffness: 90, damping: 20 }}
        />
        <motion.div
          className="absolute top-1/2 text-xl leading-none"
          style={{ marginTop: '-13px' }}
          animate={{ left: `calc(${percent}% - 13px)` }}
          transition={{ type: 'spring', stiffness: 90, damping: 20 }}
          aria-hidden
        >
          🎾
        </motion.div>
      </div>
      <p className="mt-3 text-sm text-[var(--color-ink-secondary)]">
        {status === 'done' ? 'Done!' : label}
      </p>
    </div>
  )
}

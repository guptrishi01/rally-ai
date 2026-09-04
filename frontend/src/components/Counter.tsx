import { useEffect, useRef, useState } from 'react'
import { animate, useInView, useReducedMotion } from 'motion/react'

interface CounterProps {
  value: number
  /** Decimal places to hold steady while counting, so the width never jitters. */
  decimals?: number
  prefix?: string
  suffix?: string
  duration?: number
  className?: string
}

/**
 * Counts a number up from zero the first time it scrolls into view.
 *
 * Only ever used on figures where the count itself is meaningful (a
 * percentage, a total) - never on an id, a date, or a score line, where
 * animating through wrong intermediate values would be actively misleading.
 */
export default function Counter({
  value,
  decimals = 0,
  prefix = '',
  suffix = '',
  duration = 1.1,
  className = '',
}: CounterProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true })
  const reduced = useReducedMotion()
  const [shown, setShown] = useState(reduced ? value : 0)

  useEffect(() => {
    if (reduced) {
      setShown(value)
      return
    }
    if (!inView) return
    const controls = animate(0, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => setShown(v),
    })
    return () => controls.stop()
  }, [inView, value, duration, reduced])

  return (
    <span ref={ref} className={className}>
      {prefix}
      {shown.toFixed(decimals)}
      {suffix}
    </span>
  )
}

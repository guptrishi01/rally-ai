import { useRef, type ReactNode } from 'react'
import { motion, useInView, useReducedMotion } from 'motion/react'

interface RevealProps {
  children: ReactNode
  /** Seconds to hold before this element starts, for staggering siblings. */
  delay?: number
  /** Distance in px the element travels up into place. */
  y?: number
  className?: string
}

/**
 * Reveals its children once, the first time they scroll into view.
 *
 * `once: true` matters: an element that re-animates every time it re-enters
 * the viewport turns scrolling back up into a flicker, which reads as a bug
 * rather than as polish. Falls back to a plain, static wrapper when the OS
 * asks for reduced motion.
 */
export default function Reveal({ children, delay = 0, y = 22, className = '' }: RevealProps) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '0px 0px -12% 0px' })
  const reduced = useReducedMotion()

  if (reduced) {
    return (
      <div ref={ref} className={className}>
        {children}
      </div>
    )
  }

  return (
    <motion.div
      ref={ref}
      className={className}
      initial={{ opacity: 0, y }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y }}
      transition={{ duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  )
}

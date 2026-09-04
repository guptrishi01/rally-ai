import { useRef, type ReactNode } from 'react'
import { motion, useMotionValue, useReducedMotion, useSpring } from 'motion/react'

interface MagneticProps {
  children: ReactNode
  /** How far the element is allowed to chase the cursor, as a fraction of
   *  the cursor's offset from its centre. 0.3 is a nudge; 1 would stick. */
  strength?: number
  className?: string
}

/**
 * Pulls its child toward the cursor while the cursor is over it, then springs
 * it back on leave.
 *
 * Applied to primary actions only. Every element doing this at once turns a
 * page into soup - the effect works because it marks the one thing that is
 * worth clicking here.
 */
export default function Magnetic({ children, strength = 0.32, className = '' }: MagneticProps) {
  const ref = useRef<HTMLDivElement>(null)
  const reduced = useReducedMotion()

  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const springX = useSpring(x, { stiffness: 220, damping: 18, mass: 0.4 })
  const springY = useSpring(y, { stiffness: 220, damping: 18, mass: 0.4 })

  function handleMove(event: React.MouseEvent<HTMLDivElement>) {
    const el = ref.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    x.set((event.clientX - (rect.left + rect.width / 2)) * strength)
    y.set((event.clientY - (rect.top + rect.height / 2)) * strength)
  }

  function handleLeave() {
    x.set(0)
    y.set(0)
  }

  if (reduced) {
    return <div className={className}>{children}</div>
  }

  return (
    <motion.div
      ref={ref}
      className={className}
      style={{ x: springX, y: springY, display: 'inline-flex' }}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {children}
    </motion.div>
  )
}

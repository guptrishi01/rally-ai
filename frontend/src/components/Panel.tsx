import { useRef, type ReactNode } from 'react'
import { motion, useMotionValue, useReducedMotion, useSpring, useTransform } from 'motion/react'

interface PanelProps {
  children?: ReactNode
  className?: string
  /** Tips slightly toward the cursor in 3D. For hero/summary cards only. */
  tilt?: boolean
  /** Paints a soft terracotta highlight that follows the cursor. */
  spotlight?: boolean
  /** Sunken (paper-2) rather than raised (paper-3) - for nested blocks. */
  sunk?: boolean
  onClick?: () => void
}

const MAX_TILT = 5 // degrees

/**
 * The one card surface in the app: a hairline-bordered sheet of warm paper,
 * optionally cursor-reactive.
 *
 * Replaces the old translucent GlassCard - blur-over-dark stops meaning
 * anything on a light paper ground, where depth comes from a single hairline
 * and a very slight lift instead.
 */
export default function Panel({
  children,
  className = '',
  tilt = false,
  spotlight = false,
  sunk = false,
  onClick,
}: PanelProps) {
  const ref = useRef<HTMLDivElement>(null)
  const reduced = useReducedMotion()
  const interactive = (tilt || spotlight) && !reduced

  const px = useMotionValue(0.5)
  const py = useMotionValue(0.5)
  const rotateX = useSpring(useTransform(py, [0, 1], [MAX_TILT, -MAX_TILT]), {
    stiffness: 260,
    damping: 24,
  })
  const rotateY = useSpring(useTransform(px, [0, 1], [-MAX_TILT, MAX_TILT]), {
    stiffness: 260,
    damping: 24,
  })

  function handleMove(event: React.MouseEvent<HTMLDivElement>) {
    const el = ref.current
    if (!el || !interactive) return
    const rect = el.getBoundingClientRect()
    const nx = (event.clientX - rect.left) / rect.width
    const ny = (event.clientY - rect.top) / rect.height
    px.set(nx)
    py.set(ny)
    if (spotlight) {
      // Written straight to CSS custom properties so the highlight gradient
      // itself stays declared in index.css rather than rebuilt per frame.
      el.style.setProperty('--mx', `${nx * 100}%`)
      el.style.setProperty('--my', `${ny * 100}%`)
    }
  }

  function handleLeave() {
    px.set(0.5)
    py.set(0.5)
  }

  return (
    <motion.div
      ref={ref}
      onClick={onClick}
      onMouseMove={interactive ? handleMove : undefined}
      onMouseLeave={interactive ? handleLeave : undefined}
      className={`${sunk ? 'panel-sunk' : 'panel'} ${spotlight ? 'spotlight overflow-hidden' : ''} ${className}`}
      style={
        tilt && !reduced
          ? { rotateX, rotateY, transformPerspective: 900, transformStyle: 'preserve-3d' }
          : undefined
      }
      whileHover={interactive || onClick ? { y: -2 } : undefined}
      transition={{ type: 'spring', stiffness: 300, damping: 26 }}
    >
      {children}
    </motion.div>
  )
}

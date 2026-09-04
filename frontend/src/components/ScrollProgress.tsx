import { motion, useScroll, useSpring } from 'motion/react'

/**
 * A hairline terracotta rule across the very top of the page, filling as the
 * document scrolls. Sits flush under the sticky header's own bottom border,
 * so it reads as that rule coming to life rather than as a second element.
 */
export default function ScrollProgress() {
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 140, damping: 26, restDelta: 0.001 })

  return (
    <motion.div
      aria-hidden
      className="fixed left-0 right-0 top-0 z-50 h-[2px] origin-left"
      style={{ scaleX, background: 'var(--color-accent)' }}
    />
  )
}

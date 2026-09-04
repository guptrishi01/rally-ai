import { useEffect, useState } from 'react'
import { motion, useMotionValue, useSpring } from 'motion/react'

const INTERACTIVE = 'a, button, input, select, textarea, label, [data-cursor="grow"]'

/**
 * A trailing ring that follows the pointer and swells over anything
 * clickable, plus a dot that tracks it exactly.
 *
 * Deliberately additive: the real system cursor is never hidden, so nothing
 * about pointing and clicking depends on this rendering. It mounts only for
 * fine pointers (an actual mouse) and never on touch, where there is no
 * cursor to decorate and the listeners would just cost battery.
 */
export default function Cursor() {
  const [enabled, setEnabled] = useState(false)
  const [hot, setHot] = useState(false)
  const [pressed, setPressed] = useState(false)

  const x = useMotionValue(-100)
  const y = useMotionValue(-100)
  const ringX = useSpring(x, { stiffness: 320, damping: 28, mass: 0.5 })
  const ringY = useSpring(y, { stiffness: 320, damping: 28, mass: 0.5 })

  useEffect(() => {
    const fine = window.matchMedia('(pointer: fine)')
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
    if (!fine.matches || reduced.matches) return
    setEnabled(true)

    function onMove(event: MouseEvent) {
      x.set(event.clientX)
      y.set(event.clientY)
      const target = event.target as Element | null
      setHot(Boolean(target?.closest?.(INTERACTIVE)))
    }
    const onDown = () => setPressed(true)
    const onUp = () => setPressed(false)

    window.addEventListener('mousemove', onMove, { passive: true })
    window.addEventListener('mousedown', onDown)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('mouseup', onUp)
    }
  }, [x, y])

  if (!enabled) return null

  return (
    <>
      <motion.div
        aria-hidden
        className="pointer-events-none fixed left-0 top-0 z-[100] rounded-full border"
        style={{
          x: ringX,
          y: ringY,
          translateX: '-50%',
          translateY: '-50%',
          borderColor: 'var(--color-accent)',
        }}
        animate={{
          width: hot ? 46 : 26,
          height: hot ? 46 : 26,
          opacity: pressed ? 1 : 0.55,
          scale: pressed ? 0.82 : 1,
        }}
        transition={{ type: 'spring', stiffness: 340, damping: 26 }}
      />
      <motion.div
        aria-hidden
        className="pointer-events-none fixed left-0 top-0 z-[100] h-1 w-1 rounded-full"
        style={{
          x,
          y,
          translateX: '-50%',
          translateY: '-50%',
          background: 'var(--color-accent)',
        }}
      />
    </>
  )
}

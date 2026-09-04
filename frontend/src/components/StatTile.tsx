import { motion, useReducedMotion } from 'motion/react'
import Counter from './Counter'

interface StatTileProps {
  label: string
  /** A number animates up from zero; a string (a score line, a dash) doesn't. */
  value: number | string
  suffix?: string
  decimals?: number
  sublabel?: string
  accent?: 'ink' | 'good' | 'critical' | 'accent'
  delay?: number
}

const ACCENT_COLORS: Record<NonNullable<StatTileProps['accent']>, string> = {
  ink: 'var(--color-ink)',
  good: 'var(--color-teal)',
  critical: 'var(--color-clay)',
  accent: 'var(--color-accent)',
}

/**
 * One figure, set the way a printed stats page sets it: a mono eyebrow under
 * a hairline, then the number itself large in the serif.
 *
 * There is no card around it on purpose - a row of these reads as a table of
 * figures, which is what it is, rather than as five floating widgets. Hover
 * warms the rule, which is the whole of the affordance.
 */
export default function StatTile({
  label,
  value,
  suffix = '',
  decimals = 0,
  sublabel,
  accent = 'ink',
  delay = 0,
}: StatTileProps) {
  const reduced = useReducedMotion()

  return (
    <motion.div
      className="group relative pt-5"
      initial={reduced ? undefined : { opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* The tile's whole frame is this one rule: it draws itself in from the
          left on mount, then warms to terracotta while hovered. */}
      <motion.span
        className="absolute left-0 right-0 top-0 h-px origin-left bg-[var(--color-line-strong)] transition-colors duration-300 group-hover:bg-[var(--color-accent)]"
        initial={reduced ? undefined : { scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.8, delay: delay + 0.08, ease: [0.16, 1, 0.3, 1] }}
      />
      <p className="eyebrow">{label}</p>
      <p
        className="num mt-3 text-[clamp(30px,3.6vw,46px)] leading-[0.95]"
        style={{
          color: ACCENT_COLORS[accent],
          fontFamily: 'var(--font-serif)',
          fontWeight: 460,
        }}
      >
        {typeof value === 'number' ? (
          <Counter value={value} decimals={decimals} suffix={suffix} />
        ) : (
          <>
            {value}
            {suffix}
          </>
        )}
      </p>
      {sublabel && <p className="mt-2 text-[13px] text-[var(--color-dim)]">{sublabel}</p>}
    </motion.div>
  )
}

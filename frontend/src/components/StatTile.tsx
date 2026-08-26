import { motion } from 'motion/react'
import GlassCard from './GlassCard'

interface StatTileProps {
  label: string
  value: string
  sublabel?: string
  accent?: 'brand' | 'good' | 'critical' | 'neutral'
  delay?: number
}

const accentClasses: Record<NonNullable<StatTileProps['accent']>, string> = {
  brand: 'text-gradient',
  good: 'text-[var(--color-good)]',
  critical: 'text-[var(--color-critical)]',
  neutral: 'text-[var(--color-ink-primary)]',
}

export default function StatTile({
  label,
  value,
  sublabel,
  accent = 'neutral',
  delay = 0,
}: StatTileProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay }}
    >
      <GlassCard className="p-5">
        <p className="text-xs font-medium uppercase tracking-wider text-[var(--color-ink-muted)]">
          {label}
        </p>
        <p
          className={`mt-2 font-[var(--font-display)] text-3xl font-bold ${accentClasses[accent]}`}
        >
          {value}
        </p>
        {sublabel && (
          <p className="mt-1 text-sm text-[var(--color-ink-secondary)]">{sublabel}</p>
        )}
      </GlassCard>
    </motion.div>
  )
}

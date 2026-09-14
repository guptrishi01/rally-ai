import { motion, useReducedMotion } from 'motion/react'
import type { MatchStats } from '../types'

interface StickyNoteProps {
  match: MatchStats
  selected: boolean
  onClick: () => void
}

// Deterministic per match id, so a given match's card always sits at the
// same angle across reloads instead of reshuffling under the reader.
const TILTS = [-1.4, 0.9, -0.5, 1.6, 0, -1.1]

/**
 * One match as an index card in the journal stack.
 *
 * Draggable, with `dragSnapToOrigin` - it can be pulled out of the stack and
 * always returns, so the gesture is discoverable without ever putting the
 * list into a state the user has to undo. Selection is committed on `onTap`
 * rather than `onClick`, which is what keeps the end of a drag from also
 * counting as a click.
 */
export default function StickyNote({ match, selected, onClick }: StickyNoteProps) {
  const reduced = useReducedMotion()
  const tilt = TILTS[match.match_id % TILTS.length]
  const won = match.result === 'W'

  return (
    <motion.button
      type="button"
      onTap={onClick}
      drag={!reduced}
      dragSnapToOrigin
      dragElastic={0.14}
      dragMomentum={false}
      whileDrag={{ scale: 1.04, rotate: 0, zIndex: 30, cursor: 'grabbing' }}
      whileHover={reduced ? undefined : { rotate: 0, y: -3 }}
      initial={reduced ? undefined : { rotate: tilt }}
      animate={{ rotate: tilt }}
      transition={{ type: 'spring', stiffness: 300, damping: 24 }}
      className="relative w-full cursor-grab rounded-[10px] border p-4 text-left"
      style={{
        background: selected ? 'var(--color-paper-3)' : 'var(--color-paper-2)',
        borderColor: selected ? 'var(--color-accent)' : 'var(--color-line)',
        boxShadow: selected
          ? '0 8px 22px -14px rgba(11,21,21,0.4)'
          : '0 4px 14px -12px rgba(11,21,21,0.35)',
      }}
    >
      <span
        aria-hidden
        className="absolute left-0 top-4 h-6 w-[3px] rounded-r"
        style={{ background: selected ? 'var(--color-accent)' : 'transparent' }}
      />
      <div className="flex items-baseline justify-between gap-2">
        <span className="h3 truncate">{match.opponent}</span>
        <span
          className="eyebrow shrink-0"
          style={{ color: won ? 'var(--color-win)' : 'var(--color-loss)' }}
        >
          {won ? 'Won' : 'Lost'}
        </span>
      </div>
      <p className="num mt-1 text-[11px] text-[var(--color-dim)]">{match.date}</p>
      <p className="mt-3 line-clamp-3 text-[13px] leading-relaxed text-[var(--color-dim)]">
        {match.self_assessment.pros || match.self_assessment.cons || 'No journal entry yet.'}
      </p>
    </motion.button>
  )
}

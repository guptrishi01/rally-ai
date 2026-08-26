import type { MatchStats } from '../types'

interface StickyNoteProps {
  match: MatchStats
  selected: boolean
  onClick: () => void
}

const TILTS = ['-rotate-1', 'rotate-1', '-rotate-2', 'rotate-2', 'rotate-0', '-rotate-1']
const SHADES = ['#241f14', '#1a2414', '#141f24', '#241420', '#1f1424']

export default function StickyNote({ match, selected, onClick }: StickyNoteProps) {
  const tilt = TILTS[match.match_id % TILTS.length]
  const shade = SHADES[match.match_id % SHADES.length]

  return (
    <button
      onClick={onClick}
      className={`w-full rounded-xl p-4 text-left shadow-lg transition-transform hover:-translate-y-0.5 hover:rotate-0 ${tilt} ${
        selected ? 'ring-2 ring-[var(--color-brand-blue)]' : ''
      }`}
      style={{ background: shade }}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-sm font-semibold">{match.opponent}</span>
        <span
          className={`shrink-0 text-xs font-bold ${
            match.result === 'W' ? 'text-[var(--color-good)]' : 'text-[var(--color-critical)]'
          }`}
        >
          {match.result}
        </span>
      </div>
      <p className="mt-1 text-xs text-[var(--color-ink-muted)]">{match.date}</p>
      <p className="mt-2 line-clamp-3 text-xs text-[var(--color-ink-secondary)]">
        {match.self_assessment.pros || match.self_assessment.cons || 'No journal entry yet.'}
      </p>
    </button>
  )
}

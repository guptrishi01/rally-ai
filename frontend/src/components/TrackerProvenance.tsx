import type { FieldProvenance, MatchTracking, PointTracking, ShotProvenance } from '../types'

/** The tracker's word for a stroke it could not name with any confidence. */
export const UNCLEAR_STROKE = 'Unclear'

/** A stroke cell: "Unclear" is shown as unknown, not as a stroke name. */
export function StrokeCell({ stroke }: { stroke: string }) {
  if (stroke !== UNCLEAR_STROKE) return <>{stroke}</>
  return (
    <span
      className="italic text-[var(--color-faint)]"
      title="The tracker could not name this stroke with any confidence. Treat it as unknown."
    >
      Unclear
    </span>
  )
}

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? '?' : `${Math.round(value * 100)}%`
}

function group(fields: FieldProvenance[]) {
  return {
    measured: fields.filter((f) => !f.guess && (f.provenance === 'measured' || f.provenance === 'user_entered')),
    bridged: fields.filter((f) => !f.guess && f.provenance === 'interpolated'),
    inferred: fields.filter((f) => !f.guess && f.provenance === 'inferred'),
    guessed: fields.filter((f) => f.guess),
  }
}

/** One shot's provenance in a line: which values were seen and which were not. */
export function ShotProvenanceLine({ shot }: { shot: ShotProvenance | undefined }) {
  if (!shot) return <span className="text-[var(--color-faint)]">&mdash;</span>
  const g = group(shot.fields)
  const parts: { label: string; names: string; color: string }[] = []
  if (g.measured.length)
    parts.push({ label: 'seen', names: g.measured.map((f) => f.label).join(', '), color: 'var(--color-win)' })
  if (g.bridged.length)
    parts.push({ label: 'bridged', names: g.bridged.map((f) => f.label).join(', '), color: 'var(--color-dim)' })
  if (g.inferred.length)
    parts.push({ label: 'inferred', names: g.inferred.map((f) => f.label).join(', '), color: 'var(--color-dim)' })
  if (g.guessed.length)
    parts.push({ label: 'guessed', names: g.guessed.map((f) => f.label).join(', '), color: 'var(--color-loss)' })
  return (
    <span className="text-[12px] leading-relaxed">
      {parts.map((p, i) => (
        <span key={p.label} className="block">
          <span className="eyebrow" style={{ color: p.color }}>
            {p.label}
          </span>{' '}
          <span className="text-[var(--color-dim)]">{p.names}</span>
          {i === parts.length - 1 && !shot.tracked && (
            <span className="text-[var(--color-faint)]"> &middot; hit not tracked with confidence</span>
          )}
        </span>
      ))}
    </span>
  )
}

/** The match's coverage line, shown once at the top of a tracked review. */
export function MatchTrackingNote({ tracking }: { tracking: MatchTracking }) {
  return (
    <div className="panel-sunk mt-5 p-4" data-testid="tracker-coverage">
      <p className="eyebrow">Tracked from video, not imported</p>
      {tracking.notes[0] && <p className="mt-2 text-[14px] leading-relaxed">{tracking.notes[0]}</p>}
      <p className="num mt-2 text-[12px] leading-relaxed text-[var(--color-dim)]">
        {tracking.guessed_results ?? 0} results and {tracking.guessed_strokes ?? 0} strokes of{' '}
        {tracking.shots ?? 0} shots were guessed from context
        {tracking.inferred_serves ? `; ${tracking.inferred_serves} serves were inferred, not seen` : ''}.
        Each point below says which of its values the tracker measured. None of it settles a point:
        every one still needs your decision.
      </p>
    </div>
  )
}

/** How the tracker arrived at one point - context for the reviewer only. */
export function PointTrackingPanel({ tracking }: { tracking: PointTracking }) {
  const fault = tracking.shots.filter((s) => s.part === 'folded_fault')
  return (
    <div className="panel-sunk mt-5 p-4" data-testid="tracker-point">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <p className="eyebrow">How the tracker read this point</p>
        <p className="eyebrow num">segmentation confidence {percent(tracking.confidence)}</p>
      </div>
      <p className="mt-2 text-[13px] leading-relaxed">
        {tracking.kind === 'serve_unseen'
          ? 'The serve was not seen; it was inferred from the return.'
          : 'The serve was seen.'}
        {tracking.server_side && ` Served from the ${tracking.server_side} end.`}
        {tracking.second_serve && ' A second serve.'}
      </p>
      {tracking.reasons.length > 0 && (
        <p className="mt-1 text-[12px] leading-relaxed text-[var(--color-dim)]">
          {tracking.reasons.join(' · ')}
        </p>
      )}
      {fault.length > 0 && (
        <div className="mt-3 border-t border-[var(--color-line)] pt-3">
          <p className="text-[12px]">A first-serve fault was folded into this point:</p>
          {fault.map((s) => (
            <ShotProvenanceLine key={`fault-${s.shot_number}`} shot={s} />
          ))}
        </div>
      )}
    </div>
  )
}

/** The trace for one of the point's own shots, by shot number. */
export function traceFor(tracking: PointTracking | null, shotNumber: number) {
  return tracking?.shots.find((s) => s.part === 'point' && s.shot_number === shotNumber)
}

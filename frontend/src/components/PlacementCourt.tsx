import { useEffect, useMemo, useState } from 'react'
import { api, ApiError } from '../api'
import type { Bounce, Placement } from '../types'
import Skeleton from './Skeleton'

// Court geometry in metres (ITF). The origin is the centre of the hitter's
// baseline: every bounce arrives already turned to face the hitter's way.
const LENGTH = 23.77
const NET = LENGTH / 2
const HALF_DOUBLES = 5.485
const HALF_SINGLES = 4.115
const SERVICE = 6.4
// What is drawn around the court, so long and wide balls still show.
const MARGIN_X = 2.2
const MARGIN_Y = 3.2
const SCALE = 20 // viewBox units per metre
const VIEW_W = (HALF_DOUBLES + MARGIN_X) * 2 * SCALE
const VIEW_H = (LENGTH + MARGIN_Y * 2) * SCALE

/** Court metres -> viewBox units; the hitter's baseline is at the bottom. */
function sx(x: number) {
  return (x + HALF_DOUBLES + MARGIN_X) * SCALE
}
function sy(y: number) {
  return (LENGTH + MARGIN_Y - y) * SCALE
}

type Who = 'me' | 'opponent'
type Kind = 'all' | 'serve' | 'rally'

const PROVENANCE_WORDS: Record<string, string> = {
  measured: 'measured',
  interpolated: 'bridged across a gap',
  inferred: 'inferred',
  unknown: 'of unknown origin',
}

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  testid,
}: {
  label: string
  value: T
  options: readonly (readonly [T, string])[]
  onChange: (value: T) => void
  testid: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap items-center gap-2">
      <span className="eyebrow mr-1">{label}</span>
      {options.map(([v, text]) => {
        const on = v === value
        return (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={on}
            data-testid={`${testid}-${v}`}
            onClick={() => onChange(v)}
            className="rounded-full border px-3 py-1.5 text-[12px] transition-colors"
            style={{
              borderColor: on ? 'var(--color-accent)' : 'var(--color-line-strong)',
              background: on ? 'color-mix(in srgb, var(--color-accent) 10%, transparent)' : 'transparent',
              color: on ? 'var(--color-accent-deep)' : 'var(--color-dim)',
            }}
          >
            {text}
          </button>
        )
      })}
    </div>
  )
}

function isIn(b: Bounce) {
  return b.result === 'In'
}

function describe(b: Bounce, tracked: boolean): string {
  const parts = [`Point ${b.point_number}, shot ${b.shot_number}`, b.stroke, b.result]
  if (tracked) parts.push(`bounce ${PROVENANCE_WORDS[b.provenance ?? 'unknown']}`)
  if (b.rotated) parts.push('hit from the far end, turned to face up')
  return parts.join(' · ')
}

/** Counts in the target half, by depth and the hitter's left/right. */
function areaRows(bounces: Bounce[]) {
  const rows = [
    { name: 'Short, left of centre', test: (b: Bounce) => b.y > NET && b.y <= NET + SERVICE && b.x < 0 },
    { name: 'Short, right of centre', test: (b: Bounce) => b.y > NET && b.y <= NET + SERVICE && b.x >= 0 },
    { name: 'Deep, left of centre', test: (b: Bounce) => b.y > NET + SERVICE && b.x < 0 },
    { name: 'Deep, right of centre', test: (b: Bounce) => b.y > NET + SERVICE && b.x >= 0 },
    { name: 'Own half (into the net)', test: (b: Bounce) => b.y <= NET },
  ]
  return rows.map((row) => {
    const hits = bounces.filter(row.test)
    return { name: row.name, in: hits.filter(isIn).length, out: hits.filter((b) => !isIn(b)).length }
  })
}

function Court() {
  const line = { stroke: 'var(--color-line-strong)', strokeWidth: 1.5, vectorEffect: 'non-scaling-stroke' as const }
  return (
    <g>
      <rect
        x={sx(-HALF_DOUBLES)}
        y={sy(LENGTH)}
        width={HALF_DOUBLES * 2 * SCALE}
        height={LENGTH * SCALE}
        fill="var(--color-paper-3)"
        {...line}
      />
      {[-HALF_SINGLES, HALF_SINGLES].map((x) => (
        <line key={x} x1={sx(x)} x2={sx(x)} y1={sy(0)} y2={sy(LENGTH)} {...line} />
      ))}
      {[NET - SERVICE, NET + SERVICE].map((y) => (
        <line key={y} x1={sx(-HALF_SINGLES)} x2={sx(HALF_SINGLES)} y1={sy(y)} y2={sy(y)} {...line} />
      ))}
      <line x1={sx(0)} x2={sx(0)} y1={sy(NET - SERVICE)} y2={sy(NET + SERVICE)} {...line} />
      {[0, LENGTH].map((y) => (
        <line key={y} x1={sx(0)} x2={sx(0)} y1={sy(y)} y2={sy(y) + (y === 0 ? -6 : 6)} {...line} />
      ))}
      <line
        x1={sx(-HALF_DOUBLES - 0.4)}
        x2={sx(HALF_DOUBLES + 0.4)}
        y1={sy(NET)}
        y2={sy(NET)}
        stroke="var(--color-ink-2)"
        strokeWidth={2}
        vectorEffect="non-scaling-stroke"
      />
    </g>
  )
}

/**
 * Where shots landed, on a court drawn to scale. Every bounce is shown from
 * the hitter's end (the hitter at the bottom), so shots hit from either end
 * of the court read the same way. Dot form rather than binned cells: a
 * match is a few hundred shots, too few for a density grid to say more
 * than the dots themselves.
 */
export default function PlacementCourt({ matchId }: { matchId: number }) {
  const [placement, setPlacement] = useState<Placement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [who, setWho] = useState<Who>('me')
  const [kind, setKind] = useState<Kind>('all')
  const [hover, setHover] = useState<Bounce | null>(null)

  useEffect(() => {
    let live = true
    setPlacement(null)
    setError(null)
    api
      .placement(matchId)
      .then((p) => live && setPlacement(p))
      .catch((err) => live && setError(err instanceof ApiError ? err.message : 'Could not load placement.'))
    return () => {
      live = false
    }
  }, [matchId])

  const shown = useMemo(
    () =>
      (placement?.bounces ?? []).filter(
        (b) => b.by === who && (kind === 'all' || b.kind === kind),
      ),
    [placement, who, kind],
  )

  if (error) {
    return (
      <p className="text-[14px]" style={{ color: 'var(--color-loss)' }}>
        {error}
      </p>
    )
  }
  if (!placement) return <Skeleton className="h-96" />

  if (!placement.available) {
    return (
      <div className="panel-sunk p-6" data-testid="placement-empty">
        <p className="h3">No placement data for this match.</p>
        <p className="mt-2 text-[14px] leading-relaxed text-[var(--color-dim)]">{placement.reason}</p>
      </div>
    )
  }

  const tracked = placement.source === 'tracker'
  const notMeasured = (b: Bounce) => tracked && b.provenance !== 'measured'
  const provenanceText = Object.entries(placement.provenance)
    .map(([k, n]) => `${n} ${PROVENANCE_WORDS[k] ?? k}`)
    .join(', ')

  return (
    <div className="grid gap-8 md:grid-cols-[minmax(0,360px)_minmax(0,1fr)]" data-testid="placement">
      <div className="relative mx-auto w-full max-w-[360px]">
        <svg
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          className="block h-auto w-full"
          role="img"
          aria-label={`Bounce positions: ${shown.length} shots by ${who === 'me' ? 'you' : 'your opponent'}`}
          onPointerLeave={() => setHover(null)}
        >
          <rect width={VIEW_W} height={VIEW_H} fill="var(--color-paper-2)" />
          <Court />
          <text x={sx(0)} y={sy(-1.6)} textAnchor="middle" className="placement-label">
            hitter&rsquo;s baseline
          </text>
          <text x={sx(HALF_DOUBLES + 0.25)} y={sy(NET) - 5} textAnchor="end" className="placement-label">
            net
          </text>
          {shown.map((b) => {
            const cx = sx(b.x)
            const cy = sy(b.y)
            const active = hover === b
            return (
              <g
                key={`${b.point_number}-${b.shot_number}-${b.by}`}
                onPointerEnter={() => setHover(b)}
                onFocus={() => setHover(b)}
                onBlur={() => setHover(null)}
                tabIndex={0}
                aria-label={describe(b, tracked)}
                style={{ outline: 'none' }}
              >
                <circle cx={cx} cy={cy} r={11} fill="transparent" />
                <circle
                  cx={cx}
                  cy={cy}
                  r={active ? 6.5 : 5}
                  fill={isIn(b) ? 'var(--color-blue)' : 'var(--color-paper-3)'}
                  fillOpacity={isIn(b) ? (notMeasured(b) ? 0.35 : 0.75) : 1}
                  stroke={isIn(b) ? 'var(--color-paper-3)' : 'var(--color-ink-2)'}
                  strokeWidth={isIn(b) ? 2 : 1.75}
                  strokeOpacity={notMeasured(b) ? 0.5 : 1}
                  strokeDasharray={!isIn(b) && notMeasured(b) ? '2 2' : undefined}
                />
              </g>
            )
          })}
        </svg>
        {hover && (
          <div
            className="panel pointer-events-none absolute z-10 max-w-[15rem] px-3 py-2 text-[12px] leading-snug"
            style={{
              left: `${Math.min(70, Math.max(4, (sx(hover.x) / VIEW_W) * 100))}%`,
              top: `${(sy(hover.y) / VIEW_H) * 100}%`,
              transform: 'translate(-30%, calc(-100% - 12px))',
            }}
            role="tooltip"
          >
            {describe(hover, tracked)}
          </div>
        )}
      </div>

      <div className="min-w-0 space-y-6">
        <div className="space-y-3">
          <Segmented
            label="Hit by"
            value={who}
            onChange={setWho}
            testid="placement-who"
            options={[
              ['me', 'Me'],
              ['opponent', 'Opponent'],
            ]}
          />
          <Segmented
            label="Shots"
            value={kind}
            onChange={setKind}
            testid="placement-kind"
            options={[
              ['all', 'All'],
              ['serve', 'Serves'],
              ['rally', 'Rally'],
            ]}
          />
        </div>

        <div className="flex flex-wrap gap-x-6 gap-y-2" aria-label="Legend">
          <span className="eyebrow flex items-center gap-2">
            <svg width="12" height="12" aria-hidden>
              <circle cx="6" cy="6" r="5" fill="var(--color-blue)" fillOpacity={0.75} />
            </svg>
            In
          </span>
          <span className="eyebrow flex items-center gap-2">
            <svg width="12" height="12" aria-hidden>
              <circle cx="6" cy="6" r="4.5" fill="none" stroke="var(--color-ink-2)" strokeWidth={1.5} />
            </svg>
            Out or net
          </span>
          {tracked && (
            <span className="eyebrow flex items-center gap-2">
              <svg width="12" height="12" aria-hidden>
                <circle cx="6" cy="6" r="5" fill="var(--color-blue)" fillOpacity={0.35} />
              </svg>
              Faint: bounce not measured
            </span>
          )}
        </div>

        <div className="text-[13px] leading-relaxed" data-testid="placement-coverage">
          <p>
            <span className="num">{shown.length}</span> shots shown.{' '}
            <span className="num">{placement.with_bounce}</span> of{' '}
            <span className="num">{placement.shots}</span> shots in this match have a bounce
            position; the rest are not drawn.
          </p>
          <p className="mt-2 text-[var(--color-dim)]">
            {tracked
              ? `Tracked from video by Rally AI: of the bounces drawn, ${provenanceText}. A bridged or inferred bounce is an estimate, not a sighting.`
              : 'Positions come from the SwingVision export.'}{' '}
            Every shot is drawn from the hitter&rsquo;s end, so shots hit from the far end are
            turned to face up the page.
          </p>
        </div>

        <details className="panel-sunk p-4">
          <summary className="eyebrow cursor-pointer">Counts by area (table)</summary>
          <table className="mt-3 w-full border-collapse text-left text-[13px]">
            <thead>
              <tr className="border-b border-[var(--color-line)]">
                <th className="eyebrow py-2 pr-4 font-normal">Where it bounced</th>
                <th className="eyebrow py-2 pr-4 font-normal">In</th>
                <th className="eyebrow py-2 font-normal">Out or net</th>
              </tr>
            </thead>
            <tbody>
              {areaRows(shown).map((row) => (
                <tr key={row.name} className="border-b border-[var(--color-line)]">
                  <td className="py-2 pr-4">{row.name}</td>
                  <td className="num py-2 pr-4">{row.in}</td>
                  <td className="num py-2">{row.out}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </div>
    </div>
  )
}

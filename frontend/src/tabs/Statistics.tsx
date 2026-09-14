import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'
import type { MatchStats } from '../types'
import Skeleton from '../components/Skeleton'
import StatTile from '../components/StatTile'
import Reveal from '../components/Reveal'

// Lazy: three.js/@react-three/fiber only load once someone actually views
// the Statistics tab, keeping the rest of the app's bundle unaffected.
const ShotMap = lazy(() => import('../components/ShotMap'))

const AXIS_STYLE = {
  fontSize: 11,
  fill: 'var(--color-dim)',
  fontFamily: 'var(--font-mono)',
}

/** Recharts' default tooltip is a white box with a grey border; this one is
 *  the same paper card the rest of the page uses. */
function PaperTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: { name?: string; value?: number | string; color?: string }[]
  label?: string | number
}) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="panel px-3.5 py-2.5 shadow-sm">
      {label !== undefined && <p className="eyebrow num">{label}</p>}
      <div className="mt-1.5 space-y-1">
        {payload.map((entry, i) => (
          <p key={i} className="flex items-baseline gap-3 text-[13px]">
            <span
              aria-hidden
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ background: entry.color }}
            />
            <span className="text-[var(--color-dim)]">{entry.name}</span>
            <span className="num ml-auto text-[var(--color-ink)]">{entry.value}</span>
          </p>
        ))}
      </div>
    </div>
  )
}

export default function Statistics() {
  const [matches, setMatches] = useState<MatchStats[] | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  useEffect(() => {
    api.matches().then((list) => {
      const byDateAsc = [...list].sort((a, b) => a.date.localeCompare(b.date))
      setMatches(byDateAsc)
      if (byDateAsc.length > 0) setSelectedId(byDateAsc[byDateAsc.length - 1].match_id)
    })
  }, [])

  const trendData = useMemo(
    () =>
      (matches ?? []).map((m) => ({
        date: m.date,
        opponent: m.opponent,
        'Points won %': m.point_outcomes.points_won_pct,
        'First serve %': m.serving.first_serve_pct,
      })),
    [matches],
  )

  const selected = matches?.find((m) => m.match_id === selectedId) ?? null

  const servingBars = selected
    ? [
        { name: 'First serve', value: selected.serving.first_serve_pct },
        { name: 'Second serve', value: selected.serving.second_serve_pct },
        { name: 'Holds', value: selected.serving.service_hold_pct },
      ]
    : []

  const outcomeBars = selected
    ? [
        { name: 'Winners', value: selected.point_outcomes.winners },
        { name: 'Unforced', value: selected.point_outcomes.unforced_errors },
        { name: 'Forced', value: selected.point_outcomes.forced_errors },
        { name: 'Return W', value: selected.point_outcomes.return_winners },
        { name: 'Return E', value: selected.point_outcomes.return_errors },
      ]
    : []

  if (matches === null) {
    return (
      <div className="space-y-8">
        <Skeleton className="h-72" />
        <Skeleton className="h-96" />
      </div>
    )
  }

  if (matches.length === 0) {
    return (
      <div className="border-y border-[var(--color-line)] py-14 text-center">
        <p className="h2">Nothing to chart yet.</p>
        <p className="lede mx-auto mt-4 text-center">
          Statistics fill in as soon as your first match is finalized.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-16">
      <section>
        <Reveal>
          <div className="flex items-baseline justify-between gap-6 border-t border-[var(--color-line)] pt-6">
            <h2 className="h2">Across the season</h2>
            <span className="eyebrow num">{matches.length} matches</span>
          </div>
          <p className="lede mt-4 text-[15px]">
            Points won and first serves in, match by match. Hover any point to read the pair.
          </p>
        </Reveal>

        <Reveal delay={0.05}>
          {/* Direct legend rather than Recharts' <Legend>, so the two series
              are named in the page's own type instead of the library's. */}
          <div className="mt-7 flex flex-wrap items-center gap-x-6 gap-y-2">
            {[
              { name: 'Points won %', color: 'var(--color-blue)' },
              { name: 'First serve %', color: 'var(--color-win)' },
            ].map((s) => (
              <span key={s.name} className="eyebrow flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                {s.name}
              </span>
            ))}
          </div>
          <div className="mt-5 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData} margin={{ top: 8, right: 26, bottom: 0, left: -18 }}>
                <CartesianGrid stroke="var(--color-line)" vertical={false} />
                <XAxis dataKey="date" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
                <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} width={46} />
                <Tooltip
                  content={<PaperTooltip />}
                  cursor={{ stroke: 'var(--color-line-strong)', strokeDasharray: '3 3' }}
                />
                <Line
                  type="monotone"
                  dataKey="Points won %"
                  stroke="var(--color-blue)"
                  strokeWidth={1.75}
                  dot={{ r: 2.5, strokeWidth: 0, fill: 'var(--color-blue)' }}
                  activeDot={{ r: 5 }}
                  animationDuration={1100}
                />
                <Line
                  type="monotone"
                  dataKey="First serve %"
                  stroke="var(--color-win)"
                  strokeWidth={1.75}
                  dot={{ r: 2.5, strokeWidth: 0, fill: 'var(--color-win)' }}
                  activeDot={{ r: 5 }}
                  animationDuration={1100}
                  animationBegin={180}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Reveal>
      </section>

      <section>
        <Reveal>
          <div className="flex flex-wrap items-baseline justify-between gap-4 border-t border-[var(--color-line)] pt-6">
            <h2 className="h2">One match, in full</h2>
            <select
              value={selectedId ?? ''}
              onChange={(e) => setSelectedId(Number(e.target.value))}
              className="field num w-auto"
            >
              {matches
                .slice()
                .reverse()
                .map((m) => (
                  <option key={m.match_id} value={m.match_id}>
                    {m.date} · {m.opponent} ({m.result})
                  </option>
                ))}
            </select>
          </div>
        </Reveal>

        {selected && (
          <AnimatePresence mode="wait">
            <motion.div
              key={selected.match_id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="mt-10 space-y-12"
            >
              <div className="grid grid-cols-2 gap-x-8 gap-y-10 lg:grid-cols-3">
                <StatTile
                  label="Points won"
                  value={selected.point_outcomes.points_won_pct}
                  suffix="%"
                  decimals={1}
                />
                <StatTile
                  label="Break points converted"
                  value={selected.receiving.break_point_conversion_pct}
                  suffix="%"
                  decimals={1}
                  delay={0.05}
                />
                <StatTile
                  label="Net success"
                  value={selected.net.net_success_pct}
                  suffix="%"
                  decimals={1}
                  delay={0.1}
                />
                <StatTile
                  label="Aces / double faults"
                  value={`${selected.serving.aces} / ${selected.serving.double_faults}`}
                  delay={0.15}
                />
                <StatTile
                  label="Winner : unforced"
                  value={selected.point_outcomes.winner_to_ue_ratio}
                  decimals={2}
                  delay={0.2}
                />
                <StatTile
                  label="Deuces converted"
                  value={selected.clutch.deuce_conversion_pct}
                  suffix="%"
                  decimals={1}
                  delay={0.25}
                />
              </div>

              <div className="grid gap-10 lg:grid-cols-2">
                <div>
                  <p className="eyebrow border-t border-[var(--color-line)] pt-5">Serving</p>
                  <div className="mt-5 h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={servingBars} margin={{ top: 8, right: 8, left: -18 }}>
                        <CartesianGrid stroke="var(--color-line)" vertical={false} />
                        <XAxis dataKey="name" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
                        <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} width={46} />
                        <Tooltip
                          content={<PaperTooltip />}
                          cursor={{ fill: 'color-mix(in srgb, var(--color-accent) 7%, transparent)' }}
                        />
                        <Bar
                          dataKey="value"
                          name="Percent"
                          fill="var(--color-blue)"
                          radius={[3, 3, 0, 0]}
                          maxBarSize={56}
                          animationDuration={900}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div>
                  <p className="eyebrow border-t border-[var(--color-line)] pt-5">
                    How points ended
                  </p>
                  <div className="mt-5 h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={outcomeBars} margin={{ top: 8, right: 8, left: -18 }}>
                        <CartesianGrid stroke="var(--color-line)" vertical={false} />
                        <XAxis dataKey="name" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
                        <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} width={46} />
                        <Tooltip
                          content={<PaperTooltip />}
                          cursor={{ fill: 'color-mix(in srgb, var(--color-accent) 7%, transparent)' }}
                        />
                        <Bar
                          dataKey="value"
                          name="Points"
                          fill="var(--color-ink-2)"
                          radius={[3, 3, 0, 0]}
                          maxBarSize={56}
                          animationDuration={900}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        )}
      </section>

      <section>
        <Reveal>
          <div className="border-t border-[var(--color-line)] pt-6">
            <h2 className="h2">Shot map</h2>
            <p className="lede mt-4 text-[15px]">
              Every tracked shot across your matches, projected into three dimensions — teal
              belongs to rallies you won, terracotta to ones you lost. Drag to turn it, scroll to
              zoom.
            </p>
          </div>
        </Reveal>
        <div className="mt-8">
          <Suspense fallback={<Skeleton className="h-[420px]" />}>
            <ShotMap />
          </Suspense>
        </div>
      </section>
    </div>
  )
}

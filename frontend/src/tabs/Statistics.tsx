import { useEffect, useMemo, useState } from 'react'
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
import GlassCard from '../components/GlassCard'
import Skeleton from '../components/Skeleton'
import StatTile from '../components/StatTile'

const AXIS_STYLE = { fontSize: 12, fill: 'var(--color-ink-muted)' }
const TOOLTIP_STYLE = {
  background: 'var(--color-surface-raised)',
  border: '1px solid var(--color-border)',
  borderRadius: 8,
  fontSize: 12,
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
        'PW%': m.point_outcomes.points_won_pct,
        'FS%': m.serving.first_serve_pct,
      })),
    [matches],
  )

  const selected = matches?.find((m) => m.match_id === selectedId) ?? null

  const servingBars = selected
    ? [
        { name: 'FS%', value: selected.serving.first_serve_pct },
        { name: 'SS%', value: selected.serving.second_serve_pct },
        { name: 'Hold%', value: selected.serving.service_hold_pct },
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
      <div className="space-y-6">
        <Skeleton className="h-72" />
        <Skeleton className="h-96" />
      </div>
    )
  }

  if (matches.length === 0) {
    return (
      <GlassCard className="p-6 text-center text-[var(--color-ink-secondary)]">
        No matches loaded yet — head to Overview to load your first one.
      </GlassCard>
    )
  }

  return (
    <div className="space-y-6">
      <GlassCard className="p-6">
        <h2 className="font-[var(--font-display)] text-lg font-bold">Trends across matches</h2>
        <p className="mt-1 text-sm text-[var(--color-ink-secondary)]">
          Points-won % and first-serve % over time.
        </p>
        <div className="mt-4 h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trendData}>
              <CartesianGrid stroke="var(--color-border-soft)" vertical={false} />
              <XAxis dataKey="date" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
              <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} width={36} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Line
                type="monotone"
                dataKey="PW%"
                stroke="var(--color-sequential)"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="FS%"
                stroke="var(--color-categorical-2)"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </GlassCard>

      <GlassCard className="p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-[var(--font-display)] text-lg font-bold">Match breakdown</h2>
          <select
            value={selectedId ?? ''}
            onChange={(e) => setSelectedId(Number(e.target.value))}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-sm"
          >
            {matches
              .slice()
              .reverse()
              .map((m) => (
                <option key={m.match_id} value={m.match_id}>
                  {m.date} vs {m.opponent} ({m.result})
                </option>
              ))}
          </select>
        </div>

        {selected && (
          <div className="mt-5 space-y-6">
            <div className="grid gap-4 sm:grid-cols-3">
              <StatTile label="Points won" value={`${selected.point_outcomes.points_won_pct}%`} />
              <StatTile
                label="Break point conv."
                value={`${selected.receiving.break_point_conversion_pct}%`}
              />
              <StatTile label="Net success" value={`${selected.net.net_success_pct}%`} />
              <StatTile
                label="Aces / DFs"
                value={`${selected.serving.aces} / ${selected.serving.double_faults}`}
              />
              <StatTile label="W/UE ratio" value={`${selected.point_outcomes.winner_to_ue_ratio}`} />
              <StatTile
                label="Deuce conversion"
                value={`${selected.clutch.deuce_conversion_pct}%`}
              />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <h3 className="text-sm font-semibold text-[var(--color-ink-secondary)]">
                  Serving
                </h3>
                <div className="mt-2 h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={servingBars}>
                      <CartesianGrid stroke="var(--color-border-soft)" vertical={false} />
                      <XAxis dataKey="name" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
                      <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} width={36} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} />
                      <Bar dataKey="value" fill="var(--color-sequential)" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-[var(--color-ink-secondary)]">
                  Point outcomes
                </h3>
                <div className="mt-2 h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={outcomeBars}>
                      <CartesianGrid stroke="var(--color-border-soft)" vertical={false} />
                      <XAxis dataKey="name" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
                      <YAxis tick={AXIS_STYLE} axisLine={false} tickLine={false} width={36} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} />
                      <Bar
                        dataKey="value"
                        fill="var(--color-categorical-2)"
                        radius={[6, 6, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}
      </GlassCard>
    </div>
  )
}

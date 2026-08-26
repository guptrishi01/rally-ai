import { useEffect, useState } from 'react'
import { api } from '../api'
import type { MatchStats } from '../types'
import GlassCard from '../components/GlassCard'
import Skeleton from '../components/Skeleton'

interface MatchWithVideos {
  match: MatchStats
  videos: string[]
}

export default function FilmReview() {
  const [entries, setEntries] = useState<MatchWithVideos[] | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      const matches = await api.matches()
      // Independent per-match media lookups - fetched together rather than
      // one at a time so this tab doesn't get slower as match count grows.
      const withVideos = await Promise.all(
        matches.map(async (match) => ({ match, videos: (await api.media(match.match_id)).videos })),
      )
      if (!cancelled) {
        setEntries(
          withVideos
            .filter((e) => e.videos.length > 0)
            .sort((a, b) => b.match.date.localeCompare(a.match.date)),
        )
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  if (entries === null) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-56" />
        ))}
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <GlassCard className="p-6 text-center text-[var(--color-ink-secondary)]">
        No match footage uploaded yet — add video next time you load a match on Overview.
      </GlassCard>
    )
  }

  return (
    <div>
      <h1 className="font-[var(--font-display)] text-2xl font-bold">Film Review</h1>
      <p className="mt-1 text-[var(--color-ink-secondary)]">
        Footage from every match you've uploaded, newest first.
      </p>

      <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {entries.map(({ match, videos }) => (
          <GlassCard key={match.match_id} className="overflow-hidden">
            <video src={videos[0]} controls preload="metadata" className="w-full bg-black" />
            <div className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">{match.opponent}</span>
                <span
                  className={`text-xs font-bold ${
                    match.result === 'W' ? 'text-[var(--color-good)]' : 'text-[var(--color-critical)]'
                  }`}
                >
                  {match.result}
                </span>
              </div>
              <p className="mt-1 text-xs text-[var(--color-ink-muted)]">{match.date}</p>
              {videos.length > 1 && (
                <p className="mt-1 text-xs text-[var(--color-ink-secondary)]">
                  +{videos.length - 1} more clip{videos.length - 1 === 1 ? '' : 's'}
                </p>
              )}
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  )
}

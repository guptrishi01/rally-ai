import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { api } from '../api'
import type { MatchStats } from '../types'
import Skeleton from '../components/Skeleton'
import Reveal from '../components/Reveal'

interface MatchWithVideos {
  match: MatchStats
  videos: string[]
}

/**
 * One clip. Plays a silent preview on hover and stops (rewinding to the
 * poster frame) on leave, so a wall of stills becomes scannable without
 * anyone having to click into four matches to find the right one. Clicking
 * hands control back to the native player.
 */
function ClipCard({ match, videos, index }: MatchWithVideos & { index: number }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [controls, setControls] = useState(false)
  const won = match.result === 'W'

  function preview() {
    const el = videoRef.current
    if (!el || controls) return
    el.muted = true
    void el.play().catch(() => {})
  }

  function stop() {
    const el = videoRef.current
    if (!el || controls) return
    el.pause()
    el.currentTime = 0
  }

  return (
    <Reveal delay={index * 0.05}>
      <motion.div
        className="group overflow-hidden rounded-[14px] border border-[var(--color-line)] bg-[var(--color-paper-3)]"
        whileHover={{ y: -4 }}
        transition={{ type: 'spring', stiffness: 300, damping: 26 }}
        onMouseEnter={preview}
        onMouseLeave={stop}
      >
        <div className="relative aspect-video bg-[var(--color-ink)]">
          <video
            ref={videoRef}
            src={videos[0]}
            preload="metadata"
            playsInline
            controls={controls}
            onClick={() => setControls(true)}
            className="h-full w-full object-cover"
          />
          {!controls && (
            <span className="eyebrow pointer-events-none absolute bottom-3 left-3 rounded-full bg-[var(--color-paper)] px-3 py-1 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
              Click for controls
            </span>
          )}
        </div>
        <div className="flex items-baseline justify-between gap-3 p-5">
          <div>
            <p className="h3">{match.opponent}</p>
            <p className="num mt-1 text-[12px] text-[var(--color-dim)]">
              {match.date}
              {videos.length > 1 && ` · ${videos.length} clips`}
            </p>
          </div>
          <span
            className="eyebrow shrink-0"
            style={{ color: won ? 'var(--color-win)' : 'var(--color-loss)' }}
          >
            {won ? 'Won' : 'Lost'}
          </span>
        </div>
      </motion.div>
    </Reveal>
  )
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
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-64" />
        ))}
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className="border-y border-[var(--color-line)] py-14 text-center">
        <p className="h2">No footage yet.</p>
        <p className="lede mx-auto mt-4 text-center">
          Add video alongside the export next time you stage a match, and the clips will collect
          here.
        </p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-baseline justify-between border-t border-[var(--color-line)] pt-6">
        <span className="eyebrow">The tape</span>
        <span className="eyebrow num">{entries.length} matches</span>
      </div>
      <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {entries.map((entry, i) => (
          <ClipCard key={entry.match.match_id} {...entry} index={i} />
        ))}
      </div>
    </div>
  )
}

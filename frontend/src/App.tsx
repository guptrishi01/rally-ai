import { lazy, Suspense, useState } from 'react'
import { motion } from 'motion/react'
import Skeleton from './components/Skeleton'

// Code-split per tab: only the active tab's bundle loads, so the initial
// page paint isn't paying for Recharts/the review flow/video gallery code
// before the user has even picked a tab.
const Overview = lazy(() => import('./tabs/Overview'))
const Journal = lazy(() => import('./tabs/Journal'))
const Statistics = lazy(() => import('./tabs/Statistics'))
const FilmReview = lazy(() => import('./tabs/FilmReview'))

type Tab = 'overview' | 'journal' | 'statistics' | 'film'

const TABS: { id: Tab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'journal', label: 'Journal' },
  { id: 'statistics', label: 'Statistics' },
  { id: 'film', label: 'Film Review' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('overview')
  const [journalMatchId, setJournalMatchId] = useState<number | null>(null)

  function goToJournal(matchId: number) {
    setJournalMatchId(matchId)
    setTab('journal')
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-[var(--color-border-soft)] bg-[var(--color-page)]/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎾</span>
            <span className="font-[var(--font-display)] text-lg font-extrabold text-gradient">
              RallyAI
            </span>
          </div>
          <nav className="flex gap-1 rounded-full border border-[var(--color-border-soft)] bg-[var(--color-surface)] p-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className="relative rounded-full px-4 py-1.5 text-sm font-medium transition-colors"
                style={{ color: tab === t.id ? '#fff' : 'var(--color-ink-secondary)' }}
              >
                {tab === t.id && (
                  <motion.span
                    layoutId="tab-pill"
                    className="absolute inset-0 rounded-full bg-gradient-brand"
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  />
                )}
                <span className="relative">{t.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <Suspense fallback={<Skeleton className="h-96" />}>
          {tab === 'overview' && <Overview onMatchFinalized={goToJournal} />}
          {tab === 'journal' && (
            <Journal
              initialMatchId={journalMatchId}
              onMatchIdConsumed={() => setJournalMatchId(null)}
            />
          )}
          {tab === 'statistics' && <Statistics />}
          {tab === 'film' && <FilmReview />}
        </Suspense>
      </main>
    </div>
  )
}

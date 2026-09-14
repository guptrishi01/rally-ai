import { lazy, Suspense, useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import Skeleton from './components/Skeleton'
import Cursor from './components/Cursor'
import ScrollProgress from './components/ScrollProgress'
import SplitText from './components/SplitText'
import Magnetic from './components/Magnetic'

// Code-split per tab: only the active tab's bundle loads, so the initial
// page paint isn't paying for Recharts/the review flow/video gallery code
// before the user has even picked a tab.
const Overview = lazy(() => import('./tabs/Overview'))
const Journal = lazy(() => import('./tabs/Journal'))
const Statistics = lazy(() => import('./tabs/Statistics'))
const FilmReview = lazy(() => import('./tabs/FilmReview'))

type Tab = 'overview' | 'journal' | 'statistics' | 'film'

interface TabDef {
  id: Tab
  label: string
  /** The editorial masthead this tab opens with. */
  index: string
  title: string
  lede: string
}

const TABS: TabDef[] = [
  {
    id: 'overview',
    label: 'Overview',
    index: '01',
    title: 'Every match you play, kept honestly.',
    lede:
      'Upload a SwingVision export, confirm the calls its AI got wrong, and let the record build itself. Nothing reaches your career stats until you have signed off on it.',
  },
  {
    id: 'journal',
    label: 'Journal',
    index: '02',
    title: 'What you felt, beside what happened.',
    lede:
      'Your own account of a match, read back to you next to its numbers — and a coach that only argues from both.',
  },
  {
    id: 'statistics',
    label: 'Statistics',
    index: '03',
    title: 'The numbers, and where they came from.',
    lede:
      'Serve, return, net play and the clutch points, per match and across the season. Every figure traces back to a point you confirmed.',
  },
  {
    id: 'film',
    label: 'Film Review',
    index: '04',
    title: 'The footage behind the flags.',
    lede: 'Every clip you have uploaded, newest first, sitting with the match it belongs to.',
  },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('overview')
  const [journalMatchId, setJournalMatchId] = useState<number | null>(null)
  const [scrolled, setScrolled] = useState(false)

  const active = TABS.find((t) => t.id === tab) ?? TABS[0]

  // The header only earns its hairline and blur once the page has moved -
  // at rest it should read as part of the paper, not as a bar sitting on it.
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  function goToJournal(matchId: number) {
    setJournalMatchId(matchId)
    setTab('journal')
    window.scrollTo({ top: 0 })
  }

  function selectTab(next: Tab) {
    setTab(next)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen">
      <Cursor />
      <ScrollProgress />

      {/* At rest the header is part of the paper, so the masthead below it
          keeps its air. Once the page moves it becomes the ATP navy bar -
          and that dark ground is what finally makes the Aero lime legible,
          so the active-tab marker lights up in it. */}
      <header
        className="sticky top-0 z-40 transition-colors duration-300"
        style={{
          background: scrolled ? 'color-mix(in srgb, var(--color-navy) 94%, transparent)' : 'transparent',
          backdropFilter: scrolled ? 'blur(10px)' : 'none',
          borderBottom: `1px solid ${scrolled ? 'var(--color-navy-deep)' : 'transparent'}`,
        }}
      >
        <div className="wrap flex items-center justify-between gap-6 py-4">
          <button
            onClick={() => selectTab('overview')}
            className="group flex items-baseline gap-2.5"
            aria-label="Rally AI home"
          >
            <span aria-hidden className="relative block h-2.5 w-2.5 self-center">
              <span
                className="absolute inset-0 rounded-full transition-colors duration-300"
                style={{ background: scrolled ? 'var(--color-spark)' : 'var(--color-accent)' }}
              />
              <motion.span
                className="absolute inset-0 rounded-full"
                style={{ background: scrolled ? 'var(--color-spark)' : 'var(--color-accent)' }}
                animate={{ scale: [1, 2.1, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2.6, repeat: Infinity, ease: 'easeOut' }}
              />
            </span>
            <span
              className="text-[19px] tracking-[-0.02em] transition-colors duration-300"
              style={{
                fontFamily: 'var(--font-serif)',
                fontWeight: 500,
                color: scrolled ? 'var(--color-paper)' : 'var(--color-ink)',
              }}
            >
              Rally AI
            </span>
          </button>

          <nav className="flex items-center gap-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => selectTab(t.id)}
                className="relative px-3 py-2 text-[13px] transition-colors"
                style={{
                  color: scrolled
                    ? tab === t.id
                      ? 'var(--color-paper)'
                      : '#a9b0cc'
                    : tab === t.id
                      ? 'var(--color-ink)'
                      : 'var(--color-dim)',
                }}
              >
                <span className="hidden sm:inline">{t.label}</span>
                <span className="num sm:hidden">{t.index}</span>
                {tab === t.id && (
                  <motion.span
                    layoutId="nav-underline"
                    className="absolute inset-x-2 -bottom-px h-px"
                    style={{ background: scrolled ? 'var(--color-spark)' : 'var(--color-accent)' }}
                    transition={{ type: 'spring', stiffness: 400, damping: 34 }}
                  />
                )}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Masthead: re-typesets itself on every tab change, which is what makes
          switching tabs feel like turning to a new section rather than
          swapping a panel. */}
      <section className="wrap above-grain pt-[clamp(40px,9vh,96px)]">
        <AnimatePresence mode="wait">
          <motion.div
            key={active.id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <div className="flex items-center gap-4">
              <span className="eyebrow num">{active.index}</span>
              <span className="h-px flex-1 bg-[var(--color-line)]" />
              <span className="eyebrow">Rally AI</span>
            </div>
            <h1 className="display mt-7 max-w-[17ch]">
              <SplitText text={active.title} />
            </h1>
            <motion.p
              className="lede mt-6"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
            >
              {active.lede}
            </motion.p>
          </motion.div>
        </AnimatePresence>

        {tab !== 'overview' && (
          <div className="mt-9">
            <Magnetic>
              <button className="btn" onClick={() => selectTab('overview')}>
                Load a match
                <span aria-hidden>&#8594;</span>
              </button>
            </Magnetic>
          </div>
        )}
      </section>

      <main className="wrap above-grain pb-24 pt-[clamp(36px,6vh,72px)]">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
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
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="on-navy above-grain">
        <div className="wrap grid gap-10 py-14 sm:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <p
              className="text-[19px] tracking-[-0.02em]"
              style={{ fontFamily: 'var(--font-serif)', fontWeight: 500 }}
            >
              Rally AI
            </p>
            <p className="lede mt-3 max-w-[38ch] text-[14px]">
              A reviewed record of how you actually play — built from SwingVision tracking, never
              from an auto-tag nobody checked.
            </p>
          </div>
          <div>
            <p className="eyebrow">Sections</p>
            <div className="mt-4 flex flex-col items-start gap-2.5 text-[14px]">
              {TABS.map((t) => (
                <button key={t.id} className="link" onClick={() => selectTab(t.id)}>
                  {t.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <p className="eyebrow">Pipeline</p>
            <div className="mt-4 flex flex-col gap-2.5 text-[14px] text-[#a9b0cc]">
              <span>Ingest &rarr; Review</span>
              <span>Resolve &rarr; Finalize</span>
              <span>Stats &rarr; Coach</span>
            </div>
          </div>
        </div>
        <div className="wrap flex flex-wrap items-center justify-between gap-3 border-t border-white/12 py-5">
          <span className="eyebrow">Local-only &middot; 127.0.0.1</span>
          <span className="eyebrow">Nothing final until you confirm it</span>
        </div>
      </footer>
    </div>
  )
}

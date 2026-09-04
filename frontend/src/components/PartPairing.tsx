import { AnimatePresence, motion } from 'motion/react'

interface PartPairingProps {
  exports: string[]
  videos: string[]
}

/** Approximates werkzeug's `secure_filename` + stem + lowercase, which is
 *  what the server actually pairs on. Close enough for a preview; the
 *  server's own answer comes back in the import response and is the truth. */
function partKey(filename: string): string {
  const stem = filename.replace(/\.[^./\\]+$/, '')
  return stem
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

/** Orders "part2" before "part10", the way the server's natural sort does. */
function compareNatural(a: string, b: string): number {
  return partKey(a).localeCompare(partKey(b), undefined, { numeric: true })
}

export interface PairedPart {
  index: number
  exportName: string
  videos: string[]
}

/**
 * Computes the same pairing the server will, so it can be shown before
 * anything is uploaded.
 *
 * One export is the whole match, so every clip belongs to it and the names
 * don't have to agree. Two or more means an interrupted recording, and then
 * the filename is doing real work: it sets play order and says which clip
 * goes with which export.
 */
export function pairByName(
  exports: string[],
  videos: string[],
): { parts: PairedPart[]; unpaired: string[] } {
  const ordered = [...exports].sort(compareNatural)
  if (ordered.length < 2) {
    return {
      parts: ordered.map((exportName, i) => ({ index: i + 1, exportName, videos: [...videos] })),
      unpaired: [],
    }
  }

  const claimed = new Set<string>()
  const parts = ordered.map((exportName, i) => {
    const key = partKey(exportName)
    const matched = videos.filter((v) => partKey(v) === key)
    matched.forEach((v) => claimed.add(v))
    return { index: i + 1, exportName, videos: matched }
  })
  return { parts, unpaired: videos.filter((v) => !claimed.has(v)) }
}

/**
 * Shows how the selected exports and clips will be paired, live, before the
 * upload happens — so a mismatched filename is caught while it's still one
 * rename away from being fixed, rather than after a staged import.
 */
export default function PartPairing({ exports, videos }: PartPairingProps) {
  if (exports.length === 0) return null
  const { parts, unpaired } = pairByName(exports, videos)
  const multi = parts.length > 1
  const missing = multi && videos.length > 0 && parts.some((p) => p.videos.length === 0)
  const trouble = missing || unpaired.length > 0

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        className="panel-sunk p-5"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <span className="eyebrow">
            {multi ? `${parts.length} parts, in play order` : 'One export, one part'}
          </span>
          <span
            className="eyebrow"
            style={{ color: trouble ? 'var(--color-clay)' : 'var(--color-teal)' }}
          >
            {trouble ? 'Filenames disagree' : 'Paired by filename'}
          </span>
        </div>

        <div className="mt-4 space-y-2">
          {parts.map((part) => (
            <motion.div
              key={part.exportName}
              layout
              className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-[var(--color-line)] pt-2.5 text-[13px]"
            >
              <span className="eyebrow num w-14 shrink-0">
                {multi ? `Part ${part.index}` : 'Match'}
              </span>
              <span className="num">{part.exportName}</span>
              <span aria-hidden className="text-[var(--color-faint)]">
                &#8596;
              </span>
              {part.videos.length > 0 ? (
                <span className="num text-[var(--color-dim)]">{part.videos.join(', ')}</span>
              ) : (
                <span
                  className="num"
                  style={{ color: videos.length > 0 ? 'var(--color-clay)' : 'var(--color-faint)' }}
                >
                  {videos.length > 0 ? 'no matching footage' : 'no footage'}
                </span>
              )}
            </motion.div>
          ))}

          {unpaired.length > 0 && (
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-[var(--color-line)] pt-2.5 text-[13px]">
              <span className="eyebrow w-14 shrink-0">Loose</span>
              <span className="num" style={{ color: 'var(--color-clay)' }}>
                {unpaired.join(', ')}
              </span>
            </div>
          )}
        </div>

        {multi && (
          <p className="mt-4 text-[12px] leading-relaxed text-[var(--color-dim)]">
            {trouble
              ? 'Give a clip the same filename as its export (set2.xlsx / set2.mp4) and they will travel together. Anything left unmatched is still uploaded — just not tied to a part.'
              : 'These will be merged into one continuous match, in this order.'}
          </p>
        )}
      </motion.div>
    </AnimatePresence>
  )
}

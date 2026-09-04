interface MarqueeProps {
  items: string[]
  /** Seconds for one full pass. Longer = calmer. */
  duration?: number
}

/**
 * A slow horizontal ticker of short facts, bounded by hairlines.
 *
 * The list is rendered twice and translated exactly -50%, which is what makes
 * the loop seamless: at the end of a pass the second copy sits precisely
 * where the first started. Driven by a CSS keyframe rather than a JS
 * animation specifically so hovering can pause it with
 * `animation-play-state` - a passing figure stays readable - and so the
 * reduced-motion block in index.css stops it without any JS branch here.
 */
export default function Marquee({ items, duration = 40 }: MarqueeProps) {
  if (items.length === 0) return null

  const row = (
    <div className="flex shrink-0 items-center">
      {items.map((item, i) => (
        <span key={i} className="flex items-center whitespace-nowrap">
          <span className="eyebrow">{item}</span>
          <span aria-hidden className="mx-6 text-[var(--color-accent)]">
            &#9679;
          </span>
        </span>
      ))}
    </div>
  )

  return (
    <div className="marquee overflow-hidden border-y border-[var(--color-line)] py-3">
      <div className="marquee-track flex w-max" style={{ animationDuration: `${duration}s` }}>
        {row}
        <div aria-hidden>{row}</div>
      </div>
    </div>
  )
}

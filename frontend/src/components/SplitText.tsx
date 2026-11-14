import { Fragment } from 'react'
import { motion, useReducedMotion } from 'motion/react'

interface SplitTextProps {
  text: string
  className?: string
  /** Seconds before the first word moves. */
  delay?: number
  /** Seconds between consecutive words. */
  stagger?: number
}

/**
 * Renders a headline word by word, each word rising out of its own clipped
 * line box - the "type sets itself" treatment the reference site opens with.
 *
 * Split by word rather than by character on purpose: per-character staggers
 * on a 70px serif read as novelty, and they break text selection and screen
 * readers far more aggressively. The whole string is still exposed once via
 * an aria-label, with the animated pieces hidden from the accessibility tree.
 */
export default function SplitText({
  text,
  className = '',
  delay = 0,
  stagger = 0.055,
}: SplitTextProps) {
  const reduced = useReducedMotion()
  const words = text.split(' ')

  if (reduced) {
    return <span className={className}>{text}</span>
  }

  return (
    <span className={className} aria-label={text}>
      {words.map((word, i) => (
        <Fragment key={`${word}-${i}`}>
          <span
            aria-hidden
            style={{
              display: 'inline-block',
              overflow: 'hidden',
              verticalAlign: 'bottom',
              // The clip box is the line box, which at a line-height under 1
              // cuts descenders (the y in "play,"). Pad it out, then pull the
              // same amount back so the headline's leading is unchanged.
              paddingBottom: '0.18em',
              marginBottom: '-0.18em',
            }}
          >
            <motion.span
              style={{ display: 'inline-block', willChange: 'transform' }}
              initial={{ y: '110%', opacity: 0 }}
              animate={{ y: '0%', opacity: 1 }}
              transition={{
                duration: 0.9,
                delay: delay + i * stagger,
                ease: [0.16, 1, 0.3, 1],
              }}
            >
              {word}
            </motion.span>
          </span>
          {/* A real space between the clip boxes rather than a margin on
              them: the gap has to survive selecting and copying the
              headline, and a margin renders a gap while the text itself
              reads "Everymatchyouplay". */}
          {i < words.length - 1 ? ' ' : null}
        </Fragment>
      ))}
    </span>
  )
}

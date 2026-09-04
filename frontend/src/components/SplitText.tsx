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
        <span
          key={`${word}-${i}`}
          aria-hidden
          style={{
            display: 'inline-block',
            overflow: 'hidden',
            verticalAlign: 'bottom',
            // The clip box is the line box, which at line-height < 1 cuts
            // descenders (the y in "play,"). Pad it out, then pull the same
            // amount back off so the headline's leading is unchanged.
            paddingBottom: '0.18em',
            marginBottom: '-0.18em',
            // A trailing space inside an inline-block collapses away, so the
            // word gap has to be a margin rather than whitespace.
            marginRight: i < words.length - 1 ? '0.26em' : undefined,
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
      ))}
    </span>
  )
}

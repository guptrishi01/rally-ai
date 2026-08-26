import type { HTMLAttributes } from 'react'

type GlassCardProps = HTMLAttributes<HTMLDivElement> & {
  glow?: boolean
}

export default function GlassCard({ glow = false, className = '', ...rest }: GlassCardProps) {
  return (
    <div
      className={`glass-card rounded-2xl ${glow ? 'glow-violet' : ''} ${className}`}
      {...rest}
    />
  )
}

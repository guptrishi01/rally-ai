export default function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div
      className={`shimmer rounded-xl border border-[var(--color-line)] ${className}`}
      aria-hidden
    />
  )
}

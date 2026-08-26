import { useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import { api } from '../api'
import type { ShotPoint } from '../types'
import Skeleton from './Skeleton'

const COLOR_WON = '#16c95f'
const COLOR_LOST = '#f0475a'
const SCENE_RADIUS = 4.5

/** Rescales every axis independently into [-SCENE_RADIUS, SCENE_RADIUS], so
 * PCA's arbitrary output scale always fills the same visible cube. */
function normalize(points: ShotPoint[]): { x: number; y: number; z: number; point_won: boolean; label: string }[] {
  if (points.length === 0) return []
  const axis = (pick: (p: ShotPoint) => number) => {
    const values = points.map(pick)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const span = max - min || 1
    return (v: number) => ((v - min) / span) * 2 * SCENE_RADIUS - SCENE_RADIUS
  }
  const scaleX = axis((p) => p.x)
  const scaleY = axis((p) => p.y)
  const scaleZ = axis((p) => p.z)
  return points.map((p) => ({
    x: scaleX(p.x),
    y: scaleY(p.y),
    z: scaleZ(p.z),
    point_won: p.point_won,
    label: `${p.stroke} (${p.shot_type.replace(/_/g, ' ')}) — ${p.result}`,
  }))
}

function Shots({ points }: { points: ReturnType<typeof normalize> }) {
  return (
    <>
      {points.map((p, i) => (
        <mesh key={i} position={[p.x, p.y, p.z]}>
          <sphereGeometry args={[0.09, 12, 12]} />
          <meshStandardMaterial
            color={p.point_won ? COLOR_WON : COLOR_LOST}
            emissive={p.point_won ? COLOR_WON : COLOR_LOST}
            emissiveIntensity={0.5}
          />
        </mesh>
      ))}
    </>
  )
}

export default function ShotMap() {
  const [raw, setRaw] = useState<ShotPoint[] | null>(null)

  useEffect(() => {
    api.shotEmbeddings().then(setRaw)
  }, [])

  const points = useMemo(() => normalize(raw ?? []), [raw])

  if (raw === null) {
    return <Skeleton className="h-96" />
  }

  if (points.length === 0) {
    return (
      <p className="rounded-xl border border-[var(--color-border-soft)] p-6 text-center text-sm text-[var(--color-ink-secondary)]">
        Not enough shot data yet to plot — this fills in as you load matches with footage.
      </p>
    )
  }

  return (
    <div>
      <div className="h-96 overflow-hidden rounded-xl border border-[var(--color-border-soft)] bg-black/20">
        <Canvas camera={{ position: [7, 6, 9], fov: 50 }}>
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} intensity={60} />
          <Grid
            args={[SCENE_RADIUS * 2.4, SCENE_RADIUS * 2.4]}
            position={[0, -SCENE_RADIUS - 0.5, 0]}
            cellColor="#2a2a38"
            sectionColor="#3a3a4a"
            fadeDistance={30}
          />
          <Shots points={points} />
          <OrbitControls enableDamping dampingFactor={0.08} autoRotate autoRotateSpeed={0.6} />
        </Canvas>
      </div>
      <div className="mt-3 flex items-center gap-4 text-xs text-[var(--color-ink-secondary)]">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR_WON }} />
          Point won
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR_LOST }} />
          Point lost
        </span>
        <span className="ml-auto">{points.length} shots · drag to rotate, scroll to zoom</span>
      </div>
    </div>
  )
}

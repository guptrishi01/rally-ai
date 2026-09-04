import { useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import { api } from '../api'
import type { ShotPoint } from '../types'
import Skeleton from './Skeleton'

const COLOR_WON = '#1e9e8a'
const COLOR_LOST = '#bc5b45'
const SCENE_RADIUS = 4.5

type PlacedShot = { x: number; y: number; z: number; point_won: boolean; label: string }

/** Rescales every axis independently into [-SCENE_RADIUS, SCENE_RADIUS], so
 * PCA's arbitrary output scale always fills the same visible cube. */
function normalize(points: ShotPoint[]): PlacedShot[] {
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

/**
 * The point cloud. Each shot reports what it is on hover (to an HTML readout
 * outside the canvas rather than a 3D label, which would need a font atlas
 * and would fight the camera for legibility), and swells slightly while
 * hovered so it's clear which one is being described.
 */
function Shots({
  points,
  hovered,
  onHover,
}: {
  points: PlacedShot[]
  hovered: number | null
  onHover: (index: number | null) => void
}) {
  return (
    <>
      {points.map((p, i) => (
        <mesh
          key={i}
          position={[p.x, p.y, p.z]}
          scale={hovered === i ? 1.9 : 1}
          onPointerOver={(event) => {
            event.stopPropagation()
            onHover(i)
          }}
          onPointerOut={() => onHover(null)}
        >
          <sphereGeometry args={[0.085, 14, 14]} />
          <meshStandardMaterial
            color={p.point_won ? COLOR_WON : COLOR_LOST}
            roughness={0.45}
            metalness={0.05}
          />
        </mesh>
      ))}
    </>
  )
}

export default function ShotMap() {
  const [raw, setRaw] = useState<ShotPoint[] | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const [spinning, setSpinning] = useState(true)

  useEffect(() => {
    api.shotEmbeddings().then(setRaw)
  }, [])

  const points = useMemo(() => normalize(raw ?? []), [raw])

  if (raw === null) {
    return <Skeleton className="h-[420px]" />
  }

  if (points.length === 0) {
    return (
      <div className="panel-sunk p-10 text-center">
        <p className="text-[15px] text-[var(--color-dim)]">
          Not enough shot data to plot yet — this fills in as you load matches.
        </p>
      </div>
    )
  }

  return (
    <div>
      <div
        className="relative h-[420px] overflow-hidden rounded-[14px] border border-[var(--color-line)]"
        style={{ background: 'var(--color-paper-2)' }}
      >
        <Canvas camera={{ position: [7, 6, 9], fov: 50 }}>
          <ambientLight intensity={1.5} />
          <directionalLight position={[6, 10, 6]} intensity={1.6} />
          <Grid
            args={[SCENE_RADIUS * 2.4, SCENE_RADIUS * 2.4]}
            position={[0, -SCENE_RADIUS - 0.5, 0]}
            cellColor="#dad3c4"
            sectionColor="#b3aa99"
            fadeDistance={34}
          />
          <Shots points={points} hovered={hovered} onHover={setHovered} />
          <OrbitControls
            enableDamping
            dampingFactor={0.08}
            autoRotate={spinning}
            autoRotateSpeed={0.55}
            onStart={() => setSpinning(false)}
          />
        </Canvas>

        {/* Readout, pinned rather than following the pointer: a tooltip that
            chases the cursor across a rotating scene is unreadable. */}
        <div className="pointer-events-none absolute left-4 top-4 max-w-[70%]">
          <span
            className="eyebrow block transition-opacity duration-200"
            style={{ opacity: hovered === null ? 0 : 1 }}
          >
            {hovered === null ? ' ' : points[hovered].label}
          </span>
        </div>

        {!spinning && (
          <button
            onClick={() => setSpinning(true)}
            className="btn btn-ghost btn-sm absolute bottom-4 right-4 bg-[var(--color-paper)]"
          >
            Resume spin
          </button>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2">
        <span className="eyebrow flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: COLOR_WON }} />
          Point won
        </span>
        <span className="eyebrow flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: COLOR_LOST }} />
          Point lost
        </span>
        <span className="eyebrow num ml-auto">
          {points.length} shots &middot; drag to rotate, scroll to zoom
        </span>
      </div>
    </div>
  )
}

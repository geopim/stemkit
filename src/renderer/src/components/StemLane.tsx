import { useEffect, useMemo, useRef } from 'react'
import type { StemId, ParentStemId } from '../../../shared/types'
import { DownloadIcon, SplitTrackIcon, MergeTrackIcon } from './Icons'
import { PanKnob } from './PanKnob'

export interface StemMeta {
  id: StemId
  label: string
  color: string
  icon: React.ReactNode
  parent?: ParentStemId
}

const BUCKETS = 1600

function computePeaks(buffer: AudioBuffer): Float32Array {
  const out = new Float32Array(BUCKETS)
  const data = buffer.getChannelData(0)
  const size = Math.floor(data.length / BUCKETS)
  for (let b = 0; b < BUCKETS; b++) {
    let max = 0
    const start = b * size
    const end = start + size
    for (let i = start; i < end; i += 4) {
      const v = Math.abs(data[i])
      if (v > max) max = v
    }
    out[b] = max
  }
  const globalMax = out.reduce((m, v) => Math.max(m, v), 0.0001)
  for (let b = 0; b < BUCKETS; b++) out[b] = Math.min(1, out[b] / globalMax)
  return out
}

interface Props {
  meta: StemMeta
  buffer: AudioBuffer | null
  duration: number
  getPosition: () => number
  audible: boolean
  volume: number
  pan: number
  onPan: (value: number) => void
  muted: boolean
  soloed: boolean
  onToggleMute: () => void
  onToggleSolo: () => void
  onVolume: (v: number) => void
  onSeek: (seconds: number) => void
  onExport?: () => void
  onSubsplit?: () => void
  subsplitLabel?: string
  isSplitting?: boolean
  splitPct?: number
  splitMessage?: string
  onMergeParent?: () => void
  mergeLabel?: string
  loopA?: number | null
  loopB?: number | null
  isLoopActive?: boolean
}

export function StemLane({
  meta,
  buffer,
  duration,
  getPosition,
  audible,
  volume,
  pan,
  onPan,
  muted,
  soloed,
  onToggleMute,
  onToggleSolo,
  onVolume,
  onSeek,
  onExport,
  onSubsplit,
  subsplitLabel,
  isSplitting = false,
  splitPct,
  splitMessage,
  onMergeParent,
  mergeLabel,
  loopA = null,
  loopB = null,
  isLoopActive = false
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const peaks = useMemo(() => (buffer ? computePeaks(buffer) : null), [buffer])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let raf = 0

    const draw = (): void => {
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      const dpr = window.devicePixelRatio || 1
      if (w > 0 && (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr))) {
        canvas.width = Math.round(w * dpr)
        canvas.height = Math.round(h * dpr)
      }
      const ctx = canvas.getContext('2d')
      if (ctx && w > 0) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        ctx.clearRect(0, 0, w, h)
        if (peaks) {
          const frac =
            duration > 0 ? Math.min(1, Math.max(0, getPosition() / duration)) : 0
          const barW = w / peaks.length
          const barInner = Math.max(1, barW * 0.62)

          const hasLoop = Boolean(isLoopActive && loopA !== null && loopB !== null && duration > 0)
          const loopStartFrac = hasLoop ? Math.min(loopA!, loopB!) / duration : 0
          const loopEndFrac = hasLoop ? Math.max(loopA!, loopB!) / duration : 1

          for (let i = 0; i < peaks.length; i++) {
            const amp = Math.max(0.04, peaks[i])
            const barH = amp * (h - 8)
            const x = i * barW + (barW - barInner) / 2
            const y = (h - barH) / 2
            const barFrac = i / peaks.length
            const inLoop = !hasLoop || (barFrac >= loopStartFrac && barFrac <= loopEndFrac)
            const played = barFrac <= frac

            if (inLoop) {
              ctx.fillStyle = played ? meta.color : 'rgba(255,255,255,0.18)'
            } else {
              // Outside A-B region: dimmed opacity
              ctx.fillStyle = played ? `${meta.color}40` : 'rgba(255,255,255,0.04)'
            }

            ctx.beginPath()
            ctx.roundRect(x, y, barInner, barH, 1.5)
            ctx.fill()
          }

          // If loop is active, dim outside region and draw emerald boundary lines
          if (hasLoop) {
            const lx1 = loopStartFrac * w
            const lx2 = loopEndFrac * w

            // Translucent dark veil over outside regions
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)'
            ctx.fillRect(0, 0, lx1, h)
            ctx.fillRect(lx2, 0, w - lx2, h)

            // Top and bottom loop bounds
            ctx.fillStyle = 'rgba(52, 211, 153, 0.25)'
            ctx.fillRect(lx1, 0, lx2 - lx1, 1)
            ctx.fillRect(lx1, h - 1, lx2 - lx1, 1)

            // Boundaries A and B
            ctx.fillStyle = 'rgba(52, 211, 153, 0.8)'
            ctx.fillRect(lx1 - 0.75, 0, 1.5, h)
            ctx.fillRect(lx2 - 0.75, 0, 1.5, h)
          }

          const px = frac * w
          ctx.fillStyle = 'rgba(255,255,255,0.85)'
          ctx.fillRect(px - 0.75, 0, 1.5, h)
        }
      }
      raf = requestAnimationFrame(draw)
    }

    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [peaks, duration, getPosition, meta.color, loopA, loopB, isLoopActive])

  const handleSeek = (e: React.MouseEvent<HTMLCanvasElement>): void => {
    if (!peaks || duration <= 0) return
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = (e.clientX - rect.left) / rect.width
    onSeek(Math.max(0, Math.min(1, frac)) * duration)
  }

  return (
    <div
      className={`group flex items-center gap-3 rounded-xl px-3 py-2 transition-all duration-200 glass ${
        audible ? 'opacity-100' : 'opacity-45'
      }`}
    >
      <div className="flex items-center gap-2.5 w-44 shrink-0 min-w-0">
        <span
          className="flex items-center justify-center w-7 h-7 rounded-lg shrink-0"
          style={{ background: `${meta.color}22`, color: meta.color }}
        >
          {meta.icon}
        </span>
        <span className="text-sm font-medium capitalize truncate" title={meta.label}>
          {meta.label}
        </span>
      </div>

      <button
        onClick={onToggleMute}
        title="Mute"
        className={`no-drag w-6 h-6 rounded-md text-[11px] font-bold transition-colors ${
          muted
            ? 'bg-rose-400 text-black'
            : 'bg-white/5 text-white/50 hover:bg-white/10'
        }`}
      >
        M
      </button>
      <button
        onClick={onToggleSolo}
        title="Solo"
        className={`no-drag w-6 h-6 rounded-md text-[11px] font-bold transition-colors ${
          soloed
            ? 'bg-amber-300 text-black'
            : 'bg-white/5 text-white/50 hover:bg-white/10'
        }`}
      >
        S
      </button>

      <canvas
        ref={canvasRef}
        onClick={handleSeek}
        className="no-drag flex-1 h-14 rounded-lg cursor-pointer min-w-0"
      />

      <div
        className="no-drag flex items-center gap-2 shrink-0"
        onDoubleClick={() => onVolume(1)}
        title={`Volume: ${Math.round(volume * 100)}% (Double-click to reset to 100%)`}
      >
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={(e) => onVolume(parseFloat(e.target.value))}
          className="no-drag w-24 shrink-0"
          style={{
            background: `linear-gradient(to right, ${meta.color} ${volume * 100}%, rgba(255,255,255,0.14) ${volume * 100}%)`
          }}
        />
        <span className="w-9 text-right text-[11px] font-mono tabular-nums text-white/55 select-none">
          {Math.round(volume * 100)}%
        </span>
      </div>

      <PanKnob label={meta.label} color={meta.color} value={pan} onChange={onPan} />

      {onSubsplit && (
        <button
          onClick={onSubsplit}
          disabled={isSplitting}
          title={
            isSplitting
              ? splitMessage
                ? `${splitMessage} (${splitPct ?? 0}%)`
                : `Separating… ${splitPct ?? 0}%`
              : (subsplitLabel ?? 'Sub-split stem')
          }
          className={`no-drag h-6 rounded-md flex items-center justify-center gap-1.5 transition-all shrink-0 ${
            isSplitting
              ? 'px-2 bg-violet-500/20 text-violet-300 cursor-wait border border-violet-400/25'
              : 'w-6 bg-white/5 hover:bg-violet-500/20 text-white/45 hover:text-violet-300'
          }`}
        >
          {isSplitting ? (
            <>
              <span className="w-3 h-3 rounded-full border-2 border-violet-300/30 border-t-violet-300 animate-spin shrink-0" />
              <span className="text-[11px] font-mono font-semibold tabular-nums">
                {splitPct ?? 0}%
              </span>
            </>
          ) : (
            <SplitTrackIcon className="w-3.5 h-3.5" />
          )}
        </button>
      )}

      {onMergeParent && (
        <button
          onClick={onMergeParent}
          title={mergeLabel ?? 'Merge back to single stem'}
          className="no-drag w-6 h-6 rounded-md bg-white/5 hover:bg-amber-500/20 text-white/45 hover:text-amber-300 flex items-center justify-center transition-colors shrink-0"
        >
          <MergeTrackIcon className="w-3.5 h-3.5" />
        </button>
      )}

      {onExport && (
        <button
          onClick={onExport}
          title="Export stem"
          className="no-drag w-6 h-6 rounded-md bg-white/5 hover:bg-white/10 text-white/40 hover:text-white flex items-center justify-center transition-colors shrink-0"
        >
          <DownloadIcon className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  )
}

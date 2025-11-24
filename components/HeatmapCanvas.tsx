import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { CanvasTexture, DoubleSide } from 'three';
import { AccessPoint, Wall } from '../types';

interface HeatmapCanvasProps {
  aps: AccessPoint[];
  walls: Wall[];
  width: number;
  height: number;
  show: boolean;
}

type ColorScale = 'turbo' | 'viridis' | 'magma';

interface WallSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  attenuation: number;
}

const METERS_PER_PIXEL = 0.6;
const DEFAULT_FREQ_MHZ = 5200;
const ITU_DISTANCE_EXPONENT = 30; // Typical office environment exponent from P.1238

const palette: Record<ColorScale, (t: number) => [number, number, number]> = {
  turbo: (t) => {
    const r = Math.min(255, 255 * t);
    const g = Math.min(255, 255 * (1 - Math.abs(t - 0.5) * 2));
    const b = Math.min(255, 255 * (1 - t));
    return [r, g, b];
  },
  viridis: (t) => {
    const r = 68 + 187 * t;
    const g = 1 + 165 * t;
    const b = 84 + 101 * t;
    return [r, g, b];
  },
  magma: (t) => {
    const r = 32 + 223 * t;
    const g = 18 + 120 * t;
    const b = 72 + 80 * (1 - t);
    return [r, g, b];
  }
};

const segmentIntersects = (p1x: number, p1y: number, p2x: number, p2y: number, p3x: number, p3y: number, p4x: number, p4y: number) => {
  const d = (p2x - p1x) * (p4y - p3y) - (p2y - p1y) * (p4x - p3x);
  if (d === 0) return false;
  const u = ((p3x - p1x) * (p4y - p3y) - (p3y - p1y) * (p4x - p3x)) / d;
  const v = ((p3x - p1x) * (p2y - p1y) - (p3y - p1y) * (p2x - p1x)) / d;
  return u >= 0 && u <= 1 && v >= 0 && v <= 1;
};

const computeWallLoss = (walls: WallSegment[], ax: number, ay: number, px: number, py: number) => {
  let total = 0;
  for (const wall of walls) {
    if (segmentIntersects(ax, ay, px, py, wall.x1, wall.y1, wall.x2, wall.y2)) {
      total += wall.attenuation;
    }
  }
  return total;
};

const log10 = (value: number) => Math.log(value) / Math.LN10;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const HeatmapCanvas: React.FC<HeatmapCanvasProps> = ({ aps, walls, width, height, show }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const [colorScale, setColorScale] = useState<ColorScale>('turbo');
  const [minDbm, setMinDbm] = useState(-90);
  const [maxDbm, setMaxDbm] = useState(-40);
  const [coverageThreshold, setCoverageThreshold] = useState(-70);
  const [show3d, setShow3d] = useState(true);
  const [volumetric, setVolumetric] = useState(false);
  const [heatmapVersion, setHeatmapVersion] = useState(0);

  const wallSegments: WallSegment[] = useMemo(() => walls.map(wall => ({
    x1: wall.x1,
    y1: wall.y1,
    x2: wall.x2,
    y2: wall.y2,
    attenuation: wall.attenuation + (wall.type === 'Concrete' ? 4 : wall.type === 'Glass' ? 1 : 0)
  })), [walls]);

  useEffect(() => {
    if (!show) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const offCanvas = offscreenRef.current ?? document.createElement('canvas');
    offCanvas.width = width;
    offCanvas.height = height;
    offscreenRef.current = offCanvas;
    const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });
    if (!offCtx) return;

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    const offImage = offCtx.createImageData(width, height);
    const offData = offImage.data;

    const freqLog = 20 * log10(DEFAULT_FREQ_MHZ);
    const minVal = Math.min(minDbm, maxDbm - 1);
    const maxVal = Math.max(maxDbm, minDbm + 1);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let bestSignal = -Infinity;
        for (const ap of aps) {
          const dx = (x + 0.5 - ap.x) * METERS_PER_PIXEL;
          const dy = (y + 0.5 - ap.y) * METERS_PER_PIXEL;
          const distance = Math.max(Math.hypot(dx, dy), 0.5);
          const distanceKm = distance / 1000;

          const fspl = 20 * log10(distanceKm) + freqLog + 32.44;
          const wallLoss = computeWallLoss(wallSegments, ap.x, ap.y, x + 0.5, y + 0.5);
          const pathLoss = fspl + wallLoss;

          const ituLoss = 20 * log10(DEFAULT_FREQ_MHZ) + ITU_DISTANCE_EXPONENT * log10(distance) - 28;
          const effectiveLoss = wallLoss > 0 ? pathLoss : Math.min(pathLoss, ituLoss);

          const receivedPower = ap.power - effectiveLoss;
          if (receivedPower > bestSignal) {
            bestSignal = receivedPower;
          }
        }

        const clamped = clamp((bestSignal - minVal) / (maxVal - minVal), 0, 1);
        const [r, g, b] = palette[colorScale](clamped);
        const alpha = bestSignal < coverageThreshold ? 90 : 200;
        const idx = (y * width + x) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = alpha;

        offData[idx] = r;
        offData[idx + 1] = g;
        offData[idx + 2] = b;
        offData[idx + 3] = alpha;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    offCtx.putImageData(offImage, 0, 0);
    setHeatmapVersion(v => v + 1);
  }, [aps, wallSegments, width, height, show, colorScale, minDbm, maxDbm, coverageThreshold]);

  const heatmapTexture = useMemo(() => {
    if (!offscreenRef.current) return null;
    const texture = new CanvasTexture(offscreenRef.current);
    texture.needsUpdate = true;
    return texture;
  }, [heatmapVersion]);

  useEffect(() => {
    return () => {
      heatmapTexture?.dispose();
    };
  }, [heatmapTexture]);

  if (!show) return null;

  return (
    <>
      <div className="absolute top-4 left-4 z-30 bg-white/90 backdrop-blur rounded-lg shadow-lg border border-slate-200 p-3 w-64 text-xs text-slate-700 space-y-3">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-slate-800 text-sm">Signal Heatmap</span>
          <label className="flex items-center gap-1 text-[11px]">
            <input type="checkbox" checked={show3d} onChange={(e) => setShow3d(e.target.checked)} className="rounded text-blue-600" />
            <span>3D</span>
          </label>
        </div>
        <div className="space-y-1">
          <label className="text-[11px] font-semibold">Color Scale</label>
          <select
            value={colorScale}
            onChange={(e) => setColorScale(e.target.value as ColorScale)}
            className="w-full border border-slate-200 rounded px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="turbo">Turbo</option>
            <option value="viridis">Viridis</option>
            <option value="magma">Magma</option>
          </select>
        </div>
        <div className="space-y-1">
          <label className="text-[11px] font-semibold">Dynamic Range</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              value={minDbm}
              onChange={(e) => setMinDbm(Number(e.target.value))}
              className="w-20 border border-slate-200 rounded px-2 py-1 text-xs"
            />
            <span className="text-[11px]">to</span>
            <input
              type="number"
              value={maxDbm}
              onChange={(e) => setMaxDbm(Number(e.target.value))}
              className="w-20 border border-slate-200 rounded px-2 py-1 text-xs"
            />
          </div>
        </div>
        <div className="space-y-1">
          <label className="text-[11px] font-semibold">Coverage Threshold ({coverageThreshold} dBm)</label>
          <input
            type="range"
            min={-95}
            max={-40}
            value={coverageThreshold}
            onChange={(e) => setCoverageThreshold(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>
        <label className="flex items-center gap-2 text-[11px]">
          <input type="checkbox" checked={volumetric} onChange={(e) => setVolumetric(e.target.checked)} className="rounded text-blue-600" />
          Volumetric overlay
        </label>
      </div>

      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="absolute top-0 left-0 pointer-events-none z-0 opacity-80"
      />

      {show3d && (
        <div className="absolute bottom-4 right-4 w-80 h-64 bg-white/80 backdrop-blur rounded-lg shadow-lg border border-slate-200 overflow-hidden z-30">
          <Canvas camera={{ position: [width / 2, 180, height / 2], fov: 45 }} shadows>
            <ambientLight intensity={0.6} />
            <directionalLight position={[0, 150, 0]} intensity={0.8} />
            <group position={[0, 0, 0]}>
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[width / 2, 0, height / 2]} receiveShadow>
                <planeGeometry args={[width, height]} />
                <meshStandardMaterial color="#f8fafc" side={DoubleSide} />
              </mesh>
              {heatmapTexture && (
                <group>
                  <mesh rotation={[-Math.PI / 2, 0, 0]} position={[width / 2, volumetric ? 1 : 0.5, height / 2]}>
                    <planeGeometry args={[width, height]} />
                    <meshStandardMaterial
                      map={heatmapTexture}
                      transparent
                      opacity={volumetric ? 0.45 : 0.8}
                      depthWrite={false}
                      side={DoubleSide}
                    />
                  </mesh>
                  {volumetric && (
                    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[width / 2, 12, height / 2]}>
                      <planeGeometry args={[width, height]} />
                      <meshStandardMaterial
                        map={heatmapTexture}
                        transparent
                        opacity={0.2}
                        depthWrite={false}
                        side={DoubleSide}
                      />
                    </mesh>
                  )}
                </group>
              )}

              {wallSegments.map(wall => {
                const midX = (wall.x1 + wall.x2) / 2;
                const midY = (wall.y1 + wall.y2) / 2;
                const dx = wall.x2 - wall.x1;
                const dy = wall.y2 - wall.y1;
                const length = Math.hypot(dx, dy);
                const rotation = Math.atan2(dy, dx);
                return (
                  <mesh key={`${wall.x1}-${wall.y1}-${wall.x2}-${wall.y2}`} position={[midX, 20, midY]} rotation={[0, -rotation, 0]} castShadow receiveShadow>
                    <boxGeometry args={[length, 40, 6]} />
                    <meshStandardMaterial color="#0f172a" opacity={0.85} transparent />
                  </mesh>
                );
              })}

              {aps.map(ap => (
                <mesh key={ap.id} position={[ap.x, 12, ap.y]} castShadow>
                  <sphereGeometry args={[8, 24, 24]} />
                  <meshStandardMaterial color={ap.color} emissive={ap.color} emissiveIntensity={0.4} />
                </mesh>
              ))}
            </group>
            <OrbitControls enablePan enableZoom enableRotate />
          </Canvas>
        </div>
      )}
    </>
  );
};

export default HeatmapCanvas;

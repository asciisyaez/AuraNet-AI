import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, ThreeElements } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { CanvasTexture, DoubleSide } from 'three';
import { AccessPoint, Wall } from '../types';

// Extend JSX IntrinsicElements with Three.js types
declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace JSX {
    interface IntrinsicElements extends ThreeElements {}
  }
}

interface HeatmapCanvasProps {
  aps: AccessPoint[];
  walls: Wall[];
  width: number;
  height: number;
  show: boolean;
  metersPerPixel: number;
}

type ColorScale = 'turbo' | 'viridis' | 'magma';

interface WallSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  attenuation: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

type BandKey = '2.4GHz' | '5GHz' | '6GHz';

interface BandParameters {
  distanceExponent: number;
  frequencyMHz: number;
  additionalWallPenalty: number;
  ituExponent: number;
}

interface MultiWallModelConfig {
  metersPerPixel: number;
  fallbackDistanceMeters: number;
  bandParameters: Record<BandKey, BandParameters>;
}

const DEFAULT_CONFIG: MultiWallModelConfig = {
  metersPerPixel: 0.6,
  fallbackDistanceMeters: 1,
  bandParameters: {
    '2.4GHz': {
      distanceExponent: 3.1,
      frequencyMHz: 2400,
      additionalWallPenalty: 1.5,
      ituExponent: 28,
    },
    '5GHz': {
      distanceExponent: 3.3,
      frequencyMHz: 5200,
      additionalWallPenalty: 2.1,
      ituExponent: 30,
    },
    '6GHz': {
      distanceExponent: 3.5,
      frequencyMHz: 6000,
      additionalWallPenalty: 2.5,
      ituExponent: 31,
    },
  },
};

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

const log10 = (value: number) => Math.log(value) / Math.LN10;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const createHeatmapWorker = () => {
  const workerScript = `self.onmessage = (event) => {
    const { width, height, aps, walls, config } = event.data;
    const signals = new Float32Array(width * height);

    const log10 = (value) => Math.log(value) / Math.LN10;
    const segmentIntersects = (p1x, p1y, p2x, p2y, wall) => {
      if (p1x > wall.maxX && p2x > wall.maxX) return false;
      if (p1x < wall.minX && p2x < wall.minX) return false;
      if (p1y > wall.maxY && p2y > wall.maxY) return false;
      if (p1y < wall.minY && p2y < wall.minY) return false;

      const d = (p2x - p1x) * (wall.y2 - wall.y1) - (p2y - p1y) * (wall.x2 - wall.x1);
      if (d === 0) return false;
      const u = ((wall.x1 - p1x) * (wall.y2 - wall.y1) - (wall.y1 - p1y) * (wall.x2 - wall.x1)) / d;
      const v = ((wall.x1 - p1x) * (p2y - p1y) - (wall.y1 - p1y) * (p2x - p1x)) / d;
      return u >= 0 && u <= 1 && v >= 0 && v <= 1;
    };

    const computeWallCrossings = (ax, ay, px, py, wallList) => {
      let count = 0;
      let attenuation = 0;
      for (let i = 0; i < wallList.length; i++) {
        const wall = wallList[i];
        if (segmentIntersects(ax, ay, px, py, wall)) {
          count++;
          attenuation += wall.attenuation;
        }
      }
      return { count, attenuation };
    };

    const bandParams = config.bandParameters;
    const fsplLookup = {
      '2.4GHz': 32.44 + 20 * log10(bandParams['2.4GHz'].frequencyMHz) - 60,
      '5GHz': 32.44 + 20 * log10(bandParams['5GHz'].frequencyMHz) - 60,
      '6GHz': 32.44 + 20 * log10(bandParams['6GHz'].frequencyMHz) - 60,
    };
    const preparedAps = aps.map((ap) => ({
      x: ap.x,
      y: ap.y,
      power: ap.power,
      antennaGain: ap.antennaGain ?? 0,
      band: ap.band,
      params: bandParams[ap.band],
      fspl1m: fsplLookup[ap.band],
    }));

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let bestSignal = -Infinity;
        const px = x + 0.5;
        const py = y + 0.5;

        for (let i = 0; i < preparedAps.length; i++) {
          const ap = preparedAps[i];
          const params = ap.params;
          const dx = (px - ap.x) * config.metersPerPixel;
          const dy = (py - ap.y) * config.metersPerPixel;
          const distanceMeters = Math.max(Math.hypot(dx, dy), config.fallbackDistanceMeters);

          const crossings = computeWallCrossings(ap.x, ap.y, px, py, walls);
          const baseLoss = ap.fspl1m + 10 * params.distanceExponent * log10(distanceMeters);
          const multiWallPenalty = crossings.count > 1 ? (crossings.count - 1) * params.additionalWallPenalty : 0;
          const pathLoss = baseLoss + crossings.attenuation + multiWallPenalty;

          const ituLoss = 20 * log10(params.frequencyMHz) + params.ituExponent * log10(distanceMeters) - 28;
          const effectiveLoss = crossings.count > 0 ? pathLoss : Math.min(pathLoss, ituLoss);
          const receivedPower = ap.power + ap.antennaGain - effectiveLoss;

          if (receivedPower > bestSignal) {
            bestSignal = receivedPower;
          }
        }

        signals[y * width + x] = bestSignal;
      }
    }

    postMessage({ signals }, [signals.buffer]);
  };`;

  const blob = new Blob([workerScript], { type: 'application/javascript' });
  const worker = new Worker(URL.createObjectURL(blob));
  return worker;
};

const segmentIntersects = (p1x: number, p1y: number, p2x: number, p2y: number, wall: WallSegment) => {
  if (p1x > wall.maxX && p2x > wall.maxX) return false;
  if (p1x < wall.minX && p2x < wall.minX) return false;
  if (p1y > wall.maxY && p2y > wall.maxY) return false;
  if (p1y < wall.minY && p2y < wall.minY) return false;

  const d = (p2x - p1x) * (wall.y2 - wall.y1) - (p2y - p1y) * (wall.x2 - wall.x1);
  if (d === 0) return false;
  const u = ((wall.x1 - p1x) * (wall.y2 - wall.y1) - (wall.y1 - p1y) * (wall.x2 - wall.x1)) / d;
  const v = ((wall.x1 - p1x) * (p2y - p1y) - (wall.y1 - p1y) * (p2x - p1x)) / d;
  return u >= 0 && u <= 1 && v >= 0 && v <= 1;
};

const computeWallCrossings = (walls: WallSegment[], ax: number, ay: number, px: number, py: number) => {
  let count = 0;
  let attenuation = 0;
  for (const wall of walls) {
    if (segmentIntersects(ax, ay, px, py, wall)) {
      count++;
      attenuation += wall.attenuation;
    }
  }
  return { count, attenuation };
};

const HeatmapCanvas: React.FC<HeatmapCanvasProps> = ({ aps, walls, width, height, show, metersPerPixel }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const [colorScale, setColorScale] = useState<ColorScale>('turbo');
  const [minDbm, setMinDbm] = useState(-90);
  const [maxDbm, setMaxDbm] = useState(-40);
  const [coverageThreshold, setCoverageThreshold] = useState(-70);
  const [show3d, setShow3d] = useState(true);
  const [volumetric, setVolumetric] = useState(false);
  const [heatmapVersion, setHeatmapVersion] = useState(0);

  const wallSegments: WallSegment[] = useMemo(() => walls.map(wall => {
    const minX = Math.min(wall.x1, wall.x2);
    const maxX = Math.max(wall.x1, wall.x2);
    const minY = Math.min(wall.y1, wall.y2);
    const maxY = Math.max(wall.y1, wall.y2);
    return {
      x1: wall.x1,
      y1: wall.y1,
      x2: wall.x2,
      y2: wall.y2,
      attenuation: wall.attenuation + (wall.material === 'Concrete' ? 4 : wall.material === 'Glass' ? 1 : 0),
      minX,
      maxX,
      minY,
      maxY,
    };
  }), [walls]);

  const pixelScale = metersPerPixel || 0.6;

  useEffect(() => {
    if (!show) return;
    if (typeof Worker !== 'undefined' && !workerRef.current) {
      workerRef.current = createHeatmapWorker();
    }

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
    const minVal = Math.min(minDbm, maxDbm - 1);
    const maxVal = Math.max(maxDbm, minDbm + 1);

    const applyColors = (signals: Float32Array) => {
      for (let i = 0; i < signals.length; i++) {
        const bestSignal = signals[i];
        const clamped = clamp((bestSignal - minVal) / (maxVal - minVal), 0, 1);
        const [r, g, b] = palette[colorScale](clamped);
        const alpha = bestSignal < coverageThreshold ? 90 : 200;
        const idx = i * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = alpha;

        offData[idx] = r;
        offData[idx + 1] = g;
        offData[idx + 2] = b;
        offData[idx + 3] = alpha;
      }

      ctx.putImageData(imageData, 0, 0);
      offCtx.putImageData(offImage, 0, 0);
      setHeatmapVersion(v => v + 1);
    };

    const computeLocally = () => {
      const bandParams = DEFAULT_CONFIG.bandParameters;
      const fsplLookup = {
        '2.4GHz': 32.44 + 20 * log10(bandParams['2.4GHz'].frequencyMHz) - 60,
        '5GHz': 32.44 + 20 * log10(bandParams['5GHz'].frequencyMHz) - 60,
        '6GHz': 32.44 + 20 * log10(bandParams['6GHz'].frequencyMHz) - 60,
      };
      const preparedAps = aps.map((ap) => ({
        x: ap.x,
        y: ap.y,
        power: ap.power,
        antennaGain: ap.antennaGain ?? 0,
        params: bandParams[ap.band as BandKey],
        fspl1m: fsplLookup[ap.band as BandKey],
      }));

      const signals = new Float32Array(width * height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let bestSignal = -Infinity;
          const px = x + 0.5;
          const py = y + 0.5;
          for (const ap of preparedAps) {
            const params = ap.params;
            const dx = (px - ap.x) * DEFAULT_CONFIG.metersPerPixel;
            const dy = (py - ap.y) * DEFAULT_CONFIG.metersPerPixel;
            const distanceMeters = Math.max(Math.hypot(dx, dy), DEFAULT_CONFIG.fallbackDistanceMeters);
            const crossings = computeWallCrossings(wallSegments, ap.x, ap.y, px, py);
            const baseLoss = ap.fspl1m + 10 * params.distanceExponent * log10(distanceMeters);
            const multiWallPenalty = crossings.count > 1 ? (crossings.count - 1) * params.additionalWallPenalty : 0;
            const pathLoss = baseLoss + crossings.attenuation + multiWallPenalty;
            const ituLoss = 20 * log10(params.frequencyMHz) + params.ituExponent * log10(distanceMeters) - 28;
            const effectiveLoss = crossings.count > 0 ? pathLoss : Math.min(pathLoss, ituLoss);
            const receivedPower = ap.power + ap.antennaGain - effectiveLoss;
            if (receivedPower > bestSignal) {
              bestSignal = receivedPower;
            }
          }
          signals[y * width + x] = bestSignal;
        }
      }
      applyColors(signals);
    };

    if (workerRef.current) {
      const worker = workerRef.current;
      const computePromise = new Promise<Float32Array>((resolve) => {
        const listener = (event: MessageEvent) => {
          worker.removeEventListener('message', listener);
          resolve(new Float32Array(event.data.signals));
        };
        worker.addEventListener('message', listener);
      });

      worker.postMessage({
        width,
        height,
        aps,
        walls: wallSegments,
        config: DEFAULT_CONFIG,
      });

      computePromise.then(applyColors).catch(computeLocally);
    } else {
      computeLocally();
    }

    return undefined;
  }, [aps, wallSegments, width, height, show, colorScale, minDbm, maxDbm, coverageThreshold]);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
    };
  }, []);

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
        className="absolute top-0 left-0 pointer-events-none z-10 opacity-80"
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

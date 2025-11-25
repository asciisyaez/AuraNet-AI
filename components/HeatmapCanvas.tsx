import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AccessPoint, Wall } from '../types';

interface HeatmapCanvasProps {
  aps: AccessPoint[];
  walls: Wall[];
  width: number;
  height: number;
  show: boolean;
  metersPerPixel: number;
  // Configuration Props
  colorScale?: 'turbo' | 'viridis' | 'magma';
  minDbm?: number;
  maxDbm?: number;
  coverageThreshold?: number;
  onTextureReady?: (texture: any) => void; // Placeholder for future 3D integration
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

const HeatmapCanvas: React.FC<HeatmapCanvasProps> = ({
  aps,
  walls,
  width,
  height,
  show,
  metersPerPixel,
  colorScale = 'turbo',
  minDbm = -90,
  maxDbm = -40,
  coverageThreshold = -70
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);

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

  useEffect(() => {
    if (!show) return;
    if (typeof Worker !== 'undefined' && !workerRef.current) {
      workerRef.current = createHeatmapWorker();
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
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
      }

      ctx.putImageData(imageData, 0, 0);
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
        config: { ...DEFAULT_CONFIG, metersPerPixel: metersPerPixel || 0.6 },
      });

      computePromise.then(applyColors).catch(computeLocally);
    } else {
      computeLocally();
    }

    return undefined;
  }, [aps, wallSegments, width, height, show, colorScale, minDbm, maxDbm, coverageThreshold, metersPerPixel]);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  if (!show) return null;

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute top-0 left-0 pointer-events-none z-10 opacity-80"
    />
  );
};

export default HeatmapCanvas;

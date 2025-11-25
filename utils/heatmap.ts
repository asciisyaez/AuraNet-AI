import { AccessPoint, Wall } from '../types';

export const METERS_PER_PIXEL = 0.6;
export const DEFAULT_FREQ_MHZ = 5200;
export const ITU_DISTANCE_EXPONENT = 30;

export type WallSegment = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  attenuation: number;
};

export const segmentIntersects = (
  p1x: number,
  p1y: number,
  p2x: number,
  p2y: number,
  p3x: number,
  p3y: number,
  p4x: number,
  p4y: number,
) => {
  const d = (p2x - p1x) * (p4y - p3y) - (p2y - p1y) * (p4x - p3x);
  if (d === 0) return false;
  const u = ((p3x - p1x) * (p4y - p3y) - (p3y - p1y) * (p4x - p3x)) / d;
  const v = ((p3x - p1x) * (p2y - p1y) - (p3y - p1y) * (p2x - p1x)) / d;
  return u >= 0 && u <= 1 && v >= 0 && v <= 1;
};

export const computeWallLoss = (walls: WallSegment[], ax: number, ay: number, px: number, py: number) => {
  let total = 0;
  for (const wall of walls) {
    if (segmentIntersects(ax, ay, px, py, wall.x1, wall.y1, wall.x2, wall.y2)) {
      total += wall.attenuation;
    }
  }
  return total;
};

export const log10 = (value: number) => Math.log(value) / Math.LN10;

export const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

export const mapWallsToSegments = (walls: Wall[]): WallSegment[] =>
  walls.map(wall => ({
    x1: wall.x1,
    y1: wall.y1,
    x2: wall.x2,
    y2: wall.y2,
    attenuation: wall.attenuation + (wall.type === 'Concrete' ? 4 : wall.type === 'Glass' ? 1 : 0)
  }));

export const computePixelSignalStrength = (
  aps: AccessPoint[],
  wallSegments: WallSegment[],
  x: number,
  y: number,
  minDbm: number,
  maxDbm: number,
  coverageThreshold: number
) => {
  const freqLog = 20 * log10(DEFAULT_FREQ_MHZ);
  const minVal = Math.min(minDbm, maxDbm - 1);
  const maxVal = Math.max(maxDbm, minDbm + 1);

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
  const alpha = bestSignal < coverageThreshold ? 90 : 200;

  return { bestSignal, clamped, alpha };
};

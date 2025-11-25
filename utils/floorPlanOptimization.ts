import { AccessPoint, Wall } from '../types';

export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 600;
export const COVERAGE_TARGET_DBM = -65;

export const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(value, max));

export const segmentsIntersect = (
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
  dx: number,
  dy: number
) => {
  const det = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
  if (det === 0) return false;
  const lambda = ((dy - cy) * (dx - ax) + (cx - dx) * (dy - ay)) / det;
  const gamma = ((ay - cy) * (dx - ax) + (cx - ax) * (dy - ay)) / det;
  return lambda > 0 && lambda < 1 && gamma > 0 && gamma < 1;
};

export const calculateSignal = (ap: AccessPoint, x: number, y: number, walls: Wall[]) => {
  const horizontalDistance = Math.hypot(ap.x - x, ap.y - y);
  const distance = Math.hypot(horizontalDistance, ap.height);
  const pathLoss = 20 * Math.log10(distance + 1);
  const wallLoss = walls.reduce((loss, wall) => {
    const intersects = segmentsIntersect(ap.x, ap.y, x, y, wall.x1, wall.y1, wall.x2, wall.y2);
    return loss + (intersects ? wall.attenuation : 0);
  }, 0);
  return ap.power + ap.antennaGain - pathLoss - wallLoss;
};

export const evaluateCoverage = (aps: AccessPoint[], walls: Wall[], target: number = COVERAGE_TARGET_DBM) => {
  const gridSpacing = 60;
  const gridPoints: { x: number; y: number }[] = [];
  for (let x = gridSpacing; x < CANVAS_WIDTH; x += gridSpacing) {
    for (let y = gridSpacing; y < CANVAS_HEIGHT; y += gridSpacing) {
      gridPoints.push({ x, y });
    }
  }

  let covered = 0;
  let bestSignals: number[] = [];

  gridPoints.forEach(point => {
    const bestSignal = Math.max(
      ...aps.map(ap => calculateSignal(ap, point.x, point.y, walls))
    );
    bestSignals.push(bestSignal);
    if (bestSignal >= target) covered += 1;
  });

  const coveragePercent = (covered / gridPoints.length) * 100;
  const averageSignal = bestSignals.reduce((a, b) => a + b, 0) / bestSignals.length;
  const imbalancePenalty = aps.reduce((penalty, ap, idx) => {
    return aps.slice(idx + 1).reduce((inner, other) => {
      const distance = Math.hypot(ap.x - other.x, ap.y - other.y);
      return inner + (distance < 120 ? (120 - distance) * 0.5 : 0);
    }, penalty);
  }, 0);

  const score = (100 - coveragePercent) * 5 + (target - averageSignal) * 0.5 + imbalancePenalty;

  return { coveragePercent, averageSignal, score };
};

export const runSimulatedAnnealing = (aps: AccessPoint[], walls: Wall[], target: number) => {
  let current = aps.map(ap => ({ ...ap }));
  let best = current.map(ap => ({ ...ap }));
  let { score: bestScore } = evaluateCoverage(best, walls, target);
  let { score: currentScore } = evaluateCoverage(current, walls, target);

  const iterations = 250;
  for (let i = 0; i < iterations; i++) {
    const temperature = 1 - i / iterations;
    const candidate = current.map(ap => ({ ...ap }));
    const apIndex = Math.floor(Math.random() * candidate.length);
    const jitter = Math.max(10, 120 * temperature);
    const deltaX = (Math.random() - 0.5) * jitter;
    const deltaY = (Math.random() - 0.5) * jitter;

    candidate[apIndex].x = clamp(candidate[apIndex].x + deltaX, 40, CANVAS_WIDTH - 40);
    candidate[apIndex].y = clamp(candidate[apIndex].y + deltaY, 40, CANVAS_HEIGHT - 40);

    const { score: candidateScore } = evaluateCoverage(candidate, walls, target);
    const acceptance = Math.exp((currentScore - candidateScore) / Math.max(temperature, 0.01));
    if (candidateScore < currentScore || Math.random() < acceptance) {
      current = candidate;
      currentScore = candidateScore;
    }

    if (candidateScore < bestScore) {
      best = candidate.map(ap => ({ ...ap }));
      bestScore = candidateScore;
    }
  }

  const bestMetrics = evaluateCoverage(best, walls, target);
  return { bestAps: best, metrics: bestMetrics };
};

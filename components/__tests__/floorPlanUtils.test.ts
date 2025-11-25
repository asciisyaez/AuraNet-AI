import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';
import { AccessPoint, Wall } from '../../types';
import {
  CANVAS_HEIGHT,
  CANVAS_WIDTH,
  COVERAGE_TARGET_DBM,
  evaluateCoverage,
  runSimulatedAnnealing,
} from '../../utils/floorPlanOptimization';

const baseAps: AccessPoint[] = [
  {
    id: 'ap-1',
    x: 200,
    y: 200,
    model: 'Test',
    band: '5GHz',
    power: 24,
    channel: 36,
    height: 3,
    azimuth: 0,
    tilt: 0,
    antennaGain: 5,
    color: '#000',
  },
  {
    id: 'ap-2',
    x: 600,
    y: 400,
    model: 'Test',
    band: '5GHz',
    power: 24,
    channel: 36,
    height: 3,
    azimuth: 0,
    tilt: 0,
    antennaGain: 5,
    color: '#000',
  }
];

const diagonalWall: Wall = {
  id: 'wall-1',
  x1: 0,
  y1: 0,
  x2: CANVAS_WIDTH,
  y2: CANVAS_HEIGHT,
  material: 'Concrete',
  attenuation: 60,
  thickness: 10,
  height: 3,
  elevation: 0,
};

describe('evaluateCoverage', () => {
  it('returns lower coverage when walls attenuate signals', () => {
    const lowPowerAps: AccessPoint[] = [
      { ...baseAps[0], power: 10, antennaGain: 0 },
    ];
    const baseline = evaluateCoverage(lowPowerAps, [], -50);
    const walled = evaluateCoverage(lowPowerAps, [diagonalWall], -50);

    expect(walled.coveragePercent).toBeLessThan(baseline.coveragePercent);
    expect(walled.averageSignal).toBeLessThan(baseline.averageSignal);
  });
});

describe('runSimulatedAnnealing', () => {
  const walls: Wall[] = [];
  let mathRandomSpy: ReturnType<typeof vi.spyOn>;
  const values = [0.1, 0.9, 0.3, 0.7];
  let idx = 0;

  beforeEach(() => {
    idx = 0;
    mathRandomSpy = vi.spyOn(Math, 'random').mockImplementation(() => {
      const value = values[idx % values.length];
      idx += 1;
      return value;
    });
  });

  afterEach(() => {
    mathRandomSpy.mockRestore();
  });

  it('keeps optimized APs within bounds and returns metrics', () => {
    const { bestAps, metrics } = runSimulatedAnnealing(baseAps, walls, COVERAGE_TARGET_DBM);
    const recomputed = evaluateCoverage(bestAps, walls, COVERAGE_TARGET_DBM);

    bestAps.forEach(ap => {
      expect(ap.x).toBeGreaterThanOrEqual(40);
      expect(ap.x).toBeLessThanOrEqual(CANVAS_WIDTH - 40);
      expect(ap.y).toBeGreaterThanOrEqual(40);
      expect(ap.y).toBeLessThanOrEqual(CANVAS_HEIGHT - 40);
    });

    expect(metrics.coveragePercent).toBeGreaterThan(0);
    expect(Number.isFinite(metrics.score)).toBe(true);
    expect(metrics.score).toBeCloseTo(recomputed.score, 3);
  });
});

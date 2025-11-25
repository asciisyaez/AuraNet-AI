import { describe, expect, it } from 'vitest';
import { AccessPoint, Wall } from '../../types';
import {
  WallSegment,
  clamp,
  computePixelSignalStrength,
  computeWallLoss,
  mapWallsToSegments,
} from '../../utils/heatmap';

const aps: AccessPoint[] = [
  {
    id: 'ap-1',
    x: 10,
    y: 10,
    model: 'Test',
    band: '5GHz',
    power: 30,
    channel: 36,
    height: 2,
    azimuth: 0,
    tilt: 0,
    antennaGain: 5,
    color: '#000'
  }
];

const walls: Wall[] = [
  { id: 'w1', x1: 0, y1: 30, x2: 20, y2: 30, material: 'Concrete', attenuation: 60, thickness: 10, height: 3, elevation: 0 }
];

describe('heatmap utilities', () => {
  it('clamp keeps values within bounds', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-5, 0, 10)).toBe(0);
    expect(clamp(15, 0, 10)).toBe(10);
  });

  it('computeWallLoss adds attenuation for intersecting walls', () => {
    const segments: WallSegment[] = mapWallsToSegments(walls);
    const loss = computeWallLoss(segments, 10, 0, 10, 50);
    const noLoss = computeWallLoss(segments, 10, 0, 50, 0);

    expect(loss).toBeGreaterThan(0);
    expect(noLoss).toBe(0);
  });

  it('computePixelSignalStrength returns weaker signals behind walls', () => {
    const withWall = computePixelSignalStrength(aps, mapWallsToSegments(walls), 10, 80, -90, -40, -70);
    const withoutWall = computePixelSignalStrength(aps, [], 10, 80, -90, -40, -70);

    expect(withWall.bestSignal).toBeLessThan(withoutWall.bestSignal);
    expect(withWall.clamped).toBeLessThanOrEqual(1);
    expect(withWall.alpha).toBeDefined();
  });
});

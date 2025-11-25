import { Project, UserProfile, AccessPoint, Wall, GlobalSettings } from './types';

export const DEFAULT_GLOBAL_SETTINGS: GlobalSettings = {
  units: 'metric',
  defaultSignalProfiles: ['Data', 'Voice', 'Video'],
};

export const DEFAULT_FLOOR_PLAN = {
  opacity: 0.6,
  metersPerPixel: 0.6,
};

const baseSettings = (overrides: Partial<GlobalSettings> = {}): GlobalSettings => ({
  ...DEFAULT_GLOBAL_SETTINGS,
  ...overrides,
  defaultSignalProfiles: overrides.defaultSignalProfiles ?? DEFAULT_GLOBAL_SETTINGS.defaultSignalProfiles,
});

export const DEFAULT_PROJECTS: Project[] = [
  {
    id: '1',
    name: 'San Francisco HQ - Floor 1',
    status: 'Draft',
    optimizationStatus: 'Pending',
    lastModified: 'Oct 26, 2023',
    floorCount: 1,
    settings: baseSettings(),
    floorPlan: DEFAULT_FLOOR_PLAN,
  },
  {
    id: '2',
    name: 'Warehouse Expansion P1',
    status: 'Active',
    optimizationStatus: 'Optimized',
    lastModified: 'Oct 24, 2023',
    floorCount: 1,
    settings: baseSettings({ defaultSignalProfiles: ['Outdoor', 'IoT'] }),
    floorPlan: DEFAULT_FLOOR_PLAN,
  },
  {
    id: '3',
    name: 'Lobby Wi-Fi Upgrade',
    status: 'Archived',
    optimizationStatus: '-',
    lastModified: 'Sep 15, 2023',
    floorCount: 1,
    settings: baseSettings({ units: 'imperial' }),
    floorPlan: DEFAULT_FLOOR_PLAN,
  },
  {
    id: '4',
    name: 'Meeting Rooms Coverage',
    status: 'Active',
    optimizationStatus: 'Optimized',
    lastModified: 'Nov 01, 2023',
    floorCount: 1,
    settings: baseSettings({ defaultSignalProfiles: ['Voice', 'Collaboration'] }),
    floorPlan: DEFAULT_FLOOR_PLAN,
  },
];


export const ENV_TOOLS = [
  { id: 'brick', name: 'Brick', icon: 'Square', type: 'wall', material: 'Brick', attenuation: 14, thickness: 12, height: 3, elevation: 0, color: '#b45309' },
  { id: 'drywall', name: 'Drywall', icon: 'Square', type: 'wall', material: 'Drywall', attenuation: 3, thickness: 8, height: 2.8, elevation: 0, color: '#94a3b8' },
  { id: 'concrete', name: 'Concrete', icon: 'Square', type: 'wall', material: 'Concrete', attenuation: 12, thickness: 14, height: 3, elevation: 0, color: '#475569' },
  { id: 'glass', name: 'Glass', icon: 'Square', type: 'wall', material: 'Glass', attenuation: 2, thickness: 6, height: 3, elevation: 0, color: '#38bdf8' },
];

export const HARDWARE_TOOLS = [
  { id: 'ap-generic', name: 'Generic AP', icon: 'Wifi', type: 'ap' },
  { id: 'ap-high-density', name: 'High Density AP', icon: 'Router', type: 'ap' },
];

export const INITIAL_APS: AccessPoint[] = [];
export const INITIAL_WALLS: Wall[] = [];

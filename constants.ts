import { Project, UserProfile, AccessPoint, Wall, GlobalSettings } from './types';

export const DEFAULT_GLOBAL_SETTINGS: GlobalSettings = {
  units: 'metric',
  defaultSignalProfiles: ['Data', 'Voice', 'Video'],
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
  },
  {
    id: '2',
    name: 'Warehouse Expansion P1',
    status: 'Active',
    optimizationStatus: 'Optimized',
    lastModified: 'Oct 24, 2023',
    floorCount: 1,
    settings: baseSettings({ defaultSignalProfiles: ['Outdoor', 'IoT'] }),
  },
  {
    id: '3',
    name: 'Lobby Wi-Fi Upgrade',
    status: 'Archived',
    optimizationStatus: '-',
    lastModified: 'Sep 15, 2023',
    floorCount: 1,
    settings: baseSettings({ units: 'imperial' }),
  },
  {
    id: '4',
    name: 'Meeting Rooms Coverage',
    status: 'Active',
    optimizationStatus: 'Optimized',
    lastModified: 'Nov 01, 2023',
    floorCount: 1,
    settings: baseSettings({ defaultSignalProfiles: ['Voice', 'Collaboration'] }),
  },
];

export const MOCK_USERS: UserProfile[] = [
  { id: '1', name: 'Sarah Chen', email: 'sarah.c@techcorp.com', role: 'Network Engineer', status: 'Active' },
  { id: '2', name: 'Michael Ross', email: 'mross@techcorp.com', role: 'Viewer', status: 'Active' },
  { id: '3', name: 'James Wilson', email: 'j.wilson@techcorp.com', role: 'Admin', status: 'Active' },
];

export const INITIAL_APS: AccessPoint[] = [
  { id: 'AP-01', x: 200, y: 150, model: 'Wi-Fi 6E Omni', power: 20, channel: 'Auto', color: '#3b82f6' },
  { id: 'AP-02', x: 600, y: 150, model: 'Wi-Fi 6E Omni', power: 20, channel: 6, color: '#3b82f6' },
  { id: 'AP-03', x: 400, y: 400, model: 'High Density', power: 18, channel: 11, color: '#f59e0b' },
];

export const INITIAL_WALLS: Wall[] = [
  // Outer Box
  { id: 'w1', x1: 50, y1: 50, x2: 750, y2: 50, material: 'Concrete', attenuation: 12, thickness: 12, height: 3, elevation: 0 },
  { id: 'w2', x1: 750, y1: 50, x2: 750, y2: 550, material: 'Concrete', attenuation: 12, thickness: 12, height: 3, elevation: 0 },
  { id: 'w3', x1: 750, y1: 550, x2: 50, y2: 550, material: 'Concrete', attenuation: 12, thickness: 12, height: 3, elevation: 0 },
  { id: 'w4', x1: 50, y1: 550, x2: 50, y2: 50, material: 'Concrete', attenuation: 12, thickness: 12, height: 3, elevation: 0 },
  // Inner Rooms
  { id: 'w5', x1: 50, y1: 250, x2: 300, y2: 250, material: 'Drywall', attenuation: 3, thickness: 8, height: 2.8, elevation: 0 },
  { id: 'w6', x1: 300, y1: 50, x2: 300, y2: 250, material: 'Glass', attenuation: 2, thickness: 6, height: 2.8, elevation: 0 },
  { id: 'w7', x1: 500, y1: 350, x2: 750, y2: 350, material: 'Drywall', attenuation: 3, thickness: 8, height: 2.8, elevation: 0 },
];

export const HARDWARE_TOOLS = [
  { id: 'h1', name: 'Wi-Fi 6E Omni', icon: 'Router', type: 'ap' },
  { id: 'h2', name: 'Directional', icon: 'Wifi', type: 'ap' },
  { id: 'h3', name: 'Outdoor', icon: 'CloudRain', type: 'ap' },
];

export const ENV_TOOLS = [
  { id: 'brick', name: 'Brick', icon: 'Square', type: 'wall', material: 'Brick', attenuation: 14, thickness: 12, height: 3, elevation: 0, color: '#b45309' },
  { id: 'drywall', name: 'Drywall', icon: 'Square', type: 'wall', material: 'Drywall', attenuation: 3, thickness: 8, height: 2.8, elevation: 0, color: '#94a3b8' },
  { id: 'concrete', name: 'Concrete', icon: 'Square', type: 'wall', material: 'Concrete', attenuation: 12, thickness: 14, height: 3, elevation: 0, color: '#475569' },
  { id: 'glass', name: 'Glass', icon: 'Square', type: 'wall', material: 'Glass', attenuation: 2, thickness: 6, height: 3, elevation: 0, color: '#38bdf8' },
];

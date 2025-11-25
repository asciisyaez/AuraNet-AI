export interface GlobalSettings {
  units: 'metric' | 'imperial';
  defaultSignalProfiles: string[];
}

export interface Project {
  id: string;
  region: string;
  location: string;
  floor: string;
  name: string;
  status: 'Draft' | 'Active' | 'Archived';
  optimizationStatus: 'Pending' | 'Optimized' | '-';
  lastModified: string;
  floorCount: number;
  settings: GlobalSettings;
  aps?: AccessPoint[];
  walls?: Wall[];
  floorPlan?: FloorPlan;
}

export interface PersistedProjectData {
  projects: Project[];
  globalSettings: GlobalSettings;
}

export interface FloorPlan {
  imageDataUrl?: string;
  imageName?: string;
  width?: number;
  height?: number;
  opacity: number;
  metersPerPixel: number;
  reference?: ScaleReference;
}

export interface ScaleReference {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  pixelLength: number;
  distanceMeters: number;
}

export interface AccessPoint {
  id: string;
  x: number;
  y: number;
  model: string;
  band: '2.4GHz' | '5GHz' | '6GHz';
  power: number; // dBm
  channel: number | 'Auto';
  height: number; // meters
  azimuth: number; // degrees
  tilt: number; // degrees
  antennaGain: number; // dBi
  antennaPatternFile?: string;
  color: string;
}

export interface AccessPointModel {
  id: string;
  name: string;
  vendor: string;
  bands: ('2.4GHz' | '5GHz' | '6GHz')[];
  defaultPower: number;
  defaultHeight: number;
  antennaGain: number;
  defaultAzimuth?: number;
  defaultTilt?: number;
  patternFile?: string;
  notes?: string;
}

export interface Wall {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  material: 'Brick' | 'Concrete' | 'Drywall' | 'Glass';
  attenuation: number; // dB loss
  thickness: number; // px thickness for 2D view
  height: number; // meters, used for future 3D rendering
  elevation: number; // meters above ground
  metadata?: {
    pattern?: string;
    color?: string;
    [key: string]: any;
  };
}

export type UserRole =
  | 'Regional Admin'
  | 'Local Admin'
  | 'Regional Viewer'
  | 'Local Viewer';

export interface Region {
  id: string;
  name: string;
  status: 'Active' | 'Onboarding' | 'Maintenance';
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  region?: string;
  status: 'Active' | 'Inactive';
  lastLogin?: string;
  avatarColor?: string;
}

export interface AuthSession {
  userId: string;
  name: string;
  email: string;
  role: UserRole;
  region?: string;
  avatarUrl?: string;
  provider: 'google';
}

export interface RoamingDataPoint {
  time: string;
  signalStrength: number;
  apId: string;
}

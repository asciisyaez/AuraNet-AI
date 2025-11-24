export interface GlobalSettings {
  units: 'metric' | 'imperial';
  defaultSignalProfiles: string[];
}

export interface Project {
  id: string;
  name: string;
  status: 'Draft' | 'Active' | 'Archived';
  optimizationStatus: 'Pending' | 'Optimized' | '-';
  lastModified: string;
  floorCount: number;
  settings: GlobalSettings;
}

export interface PersistedProjectData {
  projects: Project[];
  globalSettings: GlobalSettings;
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
  };
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: 'Admin' | 'Network Engineer' | 'Viewer';
  status: 'Active' | 'Inactive';
}

export interface RoamingDataPoint {
  time: string;
  signalStrength: number;
  apId: string;
}

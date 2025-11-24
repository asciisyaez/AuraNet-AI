export interface Project {
  id: string;
  name: string;
  status: 'Draft' | 'Active' | 'Archived';
  optimizationStatus: 'Pending' | 'Optimized' | '-';
  lastModified: string;
  floorCount: number;
}

export interface AccessPoint {
  id: string;
  x: number;
  y: number;
  model: string;
  power: number; // dBm
  channel: number | 'Auto';
  color: string;
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

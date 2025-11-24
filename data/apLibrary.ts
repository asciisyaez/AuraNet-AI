import { AccessPointModel } from '../types';

export const AP_LIBRARY: AccessPointModel[] = [
  {
    id: 'ap-omni-6e',
    name: 'Wi-Fi 6E Omni',
    vendor: 'AuraNet',
    bands: ['2.4GHz', '5GHz', '6GHz'],
    defaultPower: 20,
    defaultHeight: 2.7,
    antennaGain: 5,
    patternFile: 'omni_90deg.ant',
    notes: 'Indoor omni with balanced gain and tri-band support.'
  },
  {
    id: 'ap-hd-panel',
    name: 'High Density Panel',
    vendor: 'AuraNet',
    bands: ['5GHz', '6GHz'],
    defaultPower: 18,
    defaultHeight: 2.5,
    antennaGain: 9,
    patternFile: 'panel_65deg.ant',
    defaultAzimuth: 0,
    defaultTilt: -5,
    notes: 'Directional 65° panel for auditoriums or atriums.'
  },
  {
    id: 'ap-outdoor-sector',
    name: 'Outdoor Sector',
    vendor: 'AuraNet',
    bands: ['5GHz'],
    defaultPower: 23,
    defaultHeight: 5,
    antennaGain: 13,
    patternFile: 'sector_90deg.ant',
    defaultAzimuth: 90,
    defaultTilt: -2,
    notes: 'Weather-rated sector antenna for exterior coverage.'
  }
];

export const ANTENNA_PATTERNS = [
  { label: 'Omni 90° (default)', file: 'omni_90deg.ant', gain: 5 },
  { label: 'Panel 65° tight beam', file: 'panel_65deg.ant', gain: 9 },
  { label: 'Sector 90° outdoor', file: 'sector_90deg.ant', gain: 13 }
];

export const CHANNEL_OPTIONS: (number | 'Auto')[] = [
  'Auto',
  1,
  6,
  11,
  36,
  40,
  44,
  48,
  149,
  153,
  157,
  161
];

import { readFileSync } from 'fs';
import { describe, expect, it, beforeEach } from 'vitest';
import { DEFAULT_GLOBAL_SETTINGS, DEFAULT_PROJECTS } from '../../constants';
import { PersistedProjectData } from '../../types';
import { __resetProjectStore, projectStoreApi } from '../projectStore';

const STORAGE_KEY = 'auranet-projects';

const loadFixture = (): PersistedProjectData => {
  const raw = readFileSync('tests/fixtures/persistedData.json', 'utf-8');
  return JSON.parse(raw) as PersistedProjectData;
};

describe('projectStore persistence', () => {
  beforeEach(() => {
    __resetProjectStore();
    window.localStorage.clear();
  });

  it('saves projects and settings into localStorage', () => {
    const updatedProject = { ...DEFAULT_PROJECTS[0], name: 'Updated' };
    projectStoreApi.setProjects([updatedProject]);

    const payload = projectStoreApi.saveToStorage();
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}');

    expect(payload.projects[0].name).toBe('Updated');
    expect(saved.projects[0].name).toBe('Updated');
    expect(saved.globalSettings).toEqual(DEFAULT_GLOBAL_SETTINGS);
  });

  it('loads defaults when storage is empty', () => {
    const payload = projectStoreApi.loadFromStorage();

    expect(payload.projects.length).toBe(DEFAULT_PROJECTS.length);
    expect(payload.globalSettings).toEqual(DEFAULT_GLOBAL_SETTINGS);
  });

  it('imports data from fixture JSON and persists selection', () => {
    const fixture = loadFixture();

    const result = projectStoreApi.importData(fixture);
    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}');

    expect(result.projects[0].id).toBe('fixture-1');
    expect(stored.projects[0].name).toBe('Fixture Project');
    expect(result.globalSettings.units).toBe('imperial');
  });

  it('loadFromStorage hydrates store from existing saved payload', () => {
    const fixture = loadFixture();
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(fixture));

    const payload = projectStoreApi.loadFromStorage();

    expect(payload.projects[0].id).toBe('fixture-1');
    expect(payload.globalSettings.defaultSignalProfiles).toEqual(['Data', 'Video']);
  });
});

describe('projectStore updates', () => {
  beforeEach(() => {
    __resetProjectStore();
    window.localStorage.clear();
  });

  it('addProject creates a new project and selects it', () => {
    projectStoreApi.addProject('New Project');
    const payload = projectStoreApi.saveToStorage();

    expect(payload.projects[0].name).toBe('New Project');
    expect(payload.projects[0].settings).toEqual(DEFAULT_GLOBAL_SETTINGS);
  });

  it('updateGlobalSettings merges defaults and updates projects', () => {
    const next = { units: 'imperial' as const, defaultSignalProfiles: ['Voice'] };

    projectStoreApi.updateGlobalSettings(next);
    const payload = projectStoreApi.saveToStorage();

    expect(payload.globalSettings.units).toBe('imperial');
    expect(payload.globalSettings.defaultSignalProfiles).toEqual(['Voice']);
    expect(payload.projects.every(p => p.settings.units === 'imperial')).toBe(true);
  });
});

import { useSyncExternalStore } from 'react';
import { DEFAULT_GLOBAL_SETTINGS, DEFAULT_PROJECTS } from '../constants';
import { GlobalSettings, PersistedProjectData, Project } from '../types';

type Listener = () => void;

interface ProjectState {
  projects: Project[];
  globalSettings: GlobalSettings;
  selectedProjectId?: string;
}

interface ProjectActions {
  addProject: (name?: string) => void;
  setProjects: (projects: Project[]) => void;
  updateProject: (projectId: string, updates: Partial<Project>) => void;
  updateGlobalSettings: (updates: Partial<GlobalSettings>) => void;
  saveToStorage: () => PersistedProjectData;
  loadFromStorage: () => PersistedProjectData;
  importData: (payload: PersistedProjectData) => PersistedProjectData;
}

export type ProjectStore = ProjectState & ProjectActions;

const STORAGE_KEY = 'auranet-projects';

const listeners = new Set<Listener>();

const notify = () => listeners.forEach((listener) => listener());

const getDefaultState = (): ProjectState => ({
  projects: DEFAULT_PROJECTS,
  globalSettings: DEFAULT_GLOBAL_SETTINGS,
  selectedProjectId: DEFAULT_PROJECTS[0]?.id,
});

let state: ProjectState = getDefaultState();

const generateId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2, 11);

const setState = (partial: Partial<ProjectState>) => {
  state = { ...state, ...partial };
  notify();
};

const persistPayload = (): PersistedProjectData => ({
  projects: state.projects,
  globalSettings: state.globalSettings,
});

const saveToStorage = (): PersistedProjectData => {
  const payload = persistPayload();
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }
  return payload;
};

const loadFromStorage = (): PersistedProjectData => {
  if (typeof window === 'undefined') {
    setState(getDefaultState());
    return persistPayload();
  }

  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved) {
    try {
      const parsed = JSON.parse(saved) as PersistedProjectData;
      setState({
        projects: parsed.projects ?? getDefaultState().projects,
        globalSettings: parsed.globalSettings ?? DEFAULT_GLOBAL_SETTINGS,
        selectedProjectId: parsed.projects?.[0]?.id,
      });
      return parsed;
    } catch (err) {
      console.error('Unable to read saved projects', err);
    }
  }

  setState(getDefaultState());
  return persistPayload();
};

const importData = (payload: PersistedProjectData): PersistedProjectData => {
  setState({
    projects: payload.projects ?? [],
    globalSettings: payload.globalSettings ?? DEFAULT_GLOBAL_SETTINGS,
    selectedProjectId: payload.projects?.[0]?.id,
  });
  saveToStorage();
  return persistPayload();
};

const addProject = (name?: string) => {
  const now = new Date();
  const formattedDate = now.toLocaleDateString('en-US', {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
  });

  const projectName = name || `New Project ${state.projects.length + 1}`;

  const newProject: Project = {
    id: generateId(),
    name: projectName,
    status: 'Draft',
    optimizationStatus: 'Pending',
    lastModified: formattedDate,
    floorCount: 1,
    settings: state.globalSettings,
    floorPlan: state.projects[0]?.floorPlan ?? { opacity: 0.6, metersPerPixel: 0.6 },
  };

  setState({
    projects: [newProject, ...state.projects],
    selectedProjectId: newProject.id,
  });
};

const setProjects = (projects: Project[]) => setState({ projects });

const updateProject = (projectId: string, updates: Partial<Project>) => {
  setState({
    projects: state.projects.map((project) =>
      project.id === projectId ? { ...project, ...updates } : project
    ),
  });
};

const updateGlobalSettings = (updates: Partial<GlobalSettings>) => {
  const merged = {
    ...state.globalSettings,
    ...updates,
    defaultSignalProfiles: updates.defaultSignalProfiles ?? state.globalSettings.defaultSignalProfiles,
  };

  setState({
    globalSettings: merged,
    projects: state.projects.map((project) => ({
      ...project,
      settings: {
        ...project.settings,
        ...merged,
        defaultSignalProfiles: merged.defaultSignalProfiles,
      },
    })),
  });
};

const subscribe = (listener: Listener) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const api: ProjectActions = {
  addProject,
  setProjects,
  updateProject,
  updateGlobalSettings,
  saveToStorage,
  loadFromStorage,
  importData,
};

const getSnapshot = () => ({ ...state, ...api });

export const useProjectStore = <T>(selector: (state: ProjectStore) => T): T =>
  useSyncExternalStore(subscribe, () => selector(getSnapshot()), () => selector(getSnapshot()));

export const projectStoreApi = api;

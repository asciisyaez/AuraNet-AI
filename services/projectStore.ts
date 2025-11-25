import { create } from 'zustand';
import { DEFAULT_GLOBAL_SETTINGS, DEFAULT_PROJECTS, DEFAULT_FLOOR_PLAN } from '../constants';
import { GlobalSettings, PersistedProjectData, Project } from '../types';

type ProjectState = {
  projects: Project[];
  globalSettings: GlobalSettings;
  selectedProjectId?: string;
};

type ProjectActions = {
  addProject: (name?: string) => void;
  setProjects: (projects: Project[]) => void;
  setSelectedProjectId: (projectId?: string) => void;
  updateProject: (projectId: string, updates: Partial<Project>) => void;
  updateGlobalSettings: (updates: Partial<GlobalSettings>) => void;
  saveToStorage: () => PersistedProjectData;
  loadFromStorage: () => PersistedProjectData;
  importData: (payload: PersistedProjectData) => PersistedProjectData;
};

export type ProjectStore = ProjectState & ProjectActions;

const STORAGE_KEY = 'auranet-projects';

const generateId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2, 11);

const getDefaultState = (): ProjectState => ({
  projects: DEFAULT_PROJECTS,
  globalSettings: DEFAULT_GLOBAL_SETTINGS,
  selectedProjectId: DEFAULT_PROJECTS[0]?.id,
});

const persistPayload = (state: ProjectState): PersistedProjectData => ({
  projects: state.projects,
  globalSettings: state.globalSettings,
});

export const useProjectStore = create<ProjectStore>((set, get) => ({
  ...getDefaultState(),

  addProject: (name?: string) => {
    const now = new Date();
    const formattedDate = now.toLocaleDateString('en-US', {
      month: 'short',
      day: '2-digit',
      year: 'numeric',
    });

    const projectName = name || `New Project ${get().projects.length + 1}`;

    const newProject: Project = {
      id: generateId(),
      name: projectName,
      status: 'Draft',
      optimizationStatus: 'Pending',
      lastModified: formattedDate,
      floorCount: 1,
      settings: get().globalSettings,
      floorPlan: { ...DEFAULT_FLOOR_PLAN },
      aps: [],
      walls: [],
    };

    set((state) => ({
      projects: [newProject, ...state.projects],
      selectedProjectId: newProject.id,
    }));
  },

  setProjects: (projects: Project[]) =>
    set({
      projects,
      selectedProjectId: projects[0]?.id,
    }),

  setSelectedProjectId: (projectId?: string) =>
    set((state) => ({
      selectedProjectId: projectId ?? state.projects[0]?.id,
    })),

  updateProject: (projectId: string, updates: Partial<Project>) =>
    set((state) => ({
      projects: state.projects.map((project) =>
        project.id === projectId ? { ...project, ...updates, lastModified: new Date().toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' }) } : project
      ),
    })),

  updateGlobalSettings: (updates: Partial<GlobalSettings>) => {
    const current = get();
    const merged = {
      ...current.globalSettings,
      ...updates,
      defaultSignalProfiles: updates.defaultSignalProfiles ?? current.globalSettings.defaultSignalProfiles,
    };

    set((state) => ({
      globalSettings: merged,
      projects: state.projects.map((project) => ({
        ...project,
        settings: {
          ...project.settings,
          ...merged,
          defaultSignalProfiles: merged.defaultSignalProfiles,
        },
      })),
    }));
  },

  saveToStorage: () => {
    const payload = persistPayload(get());
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    }
    return payload;
  },

  loadFromStorage: () => {
    if (typeof window === 'undefined') {
      const defaults = getDefaultState();
      set(defaults);
      return persistPayload(defaults);
    }

    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as PersistedProjectData;
        const nextProjects = parsed.projects ?? getDefaultState().projects;
        const nextSettings = parsed.globalSettings ?? DEFAULT_GLOBAL_SETTINGS;

        set({
          projects: nextProjects,
          globalSettings: nextSettings,
          selectedProjectId: nextProjects[0]?.id,
        });
        return { projects: nextProjects, globalSettings: nextSettings };
      } catch (err) {
        console.error('Unable to read saved projects', err);
      }
    }

    const defaults = getDefaultState();
    set(defaults);
    return persistPayload(defaults);
  },

  importData: (payload: PersistedProjectData) => {
    const nextProjects = payload.projects ?? [];
    const nextSettings = payload.globalSettings ?? DEFAULT_GLOBAL_SETTINGS;

    set({
      projects: nextProjects,
      globalSettings: nextSettings,
      selectedProjectId: nextProjects[0]?.id,
    });

    const persisted = persistPayload(get());
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    }
    return persisted;
  },
}));

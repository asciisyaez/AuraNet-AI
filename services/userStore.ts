import { create } from 'zustand';
import { DEFAULT_REGIONS, DEFAULT_USERS } from '../constants';
import { UserProfile, UserRole } from '../types';

const STORAGE_KEY = 'auranet-users';

const palette = ['#2563eb', '#7c3aed', '#0ea5e9', '#14b8a6', '#ef4444', '#eab308', '#16a34a'];

const generateId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2, 11);

const pickColor = (seed = '') => {
  const hash = seed.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return palette[hash % palette.length];
};

const getDefaultState = () => ({
  users: DEFAULT_USERS,
  regions: DEFAULT_REGIONS,
  filters: {
    search: '',
    role: 'All' as UserRole | 'All',
    region: 'All' as string,
  },
});

type UserState = ReturnType<typeof getDefaultState>;

type UserActions = {
  addUser: (
    payload: Omit<UserProfile, 'id' | 'status' | 'lastLogin' | 'avatarColor'> &
      Partial<Pick<UserProfile, 'status' | 'lastLogin' | 'avatarColor'>>
  ) => UserProfile;
  updateUser: (userId: string, updates: Partial<UserProfile>) => void;
  toggleStatus: (userId: string) => void;
  setFilters: (filters: Partial<UserState['filters']>) => void;
  resetFilters: () => void;
  ensureUser: (user: UserProfile) => UserProfile;
  loadFromStorage: () => void;
};

export type UserStore = UserState & UserActions;

const persist = (state: UserState) => {
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ users: state.users, filters: state.filters })
    );
  }
};

export const useUserStore = create<UserStore>((set, get) => ({
  ...getDefaultState(),

  addUser: (payload) => {
    const existing = get().users.find((user) => user.email === payload.email);
    if (existing) return existing;

    const newUser: UserProfile = {
      id: generateId(),
      status: payload.status ?? 'Active',
      lastLogin: payload.lastLogin ?? 'Invited · pending login',
      avatarColor: payload.avatarColor ?? pickColor(payload.email),
      ...payload,
    };

    set((state) => {
      const next = { ...state, users: [newUser, ...state.users] };
      persist(next);
      return next;
    });

    return newUser;
  },

  updateUser: (userId, updates) => {
    set((state) => {
      const nextUsers = state.users.map((user) =>
        user.id === userId ? { ...user, ...updates } : user
      );
      const next = { ...state, users: nextUsers };
      persist(next);
      return next;
    });
  },

  toggleStatus: (userId) => {
    set((state) => {
      const nextUsers = state.users.map((user) =>
        user.id === userId
          ? { ...user, status: user.status === 'Active' ? 'Inactive' : 'Active' }
          : user
      );
      const next = { ...state, users: nextUsers };
      persist(next);
      return next;
    });
  },

  setFilters: (filters) => set((state) => ({ filters: { ...state.filters, ...filters } })),

  resetFilters: () => set((state) => ({ filters: { ...state.filters, search: '', role: 'All', region: 'All' } })),

  ensureUser: (user) => {
    const existing = get().users.find((u) => u.email === user.email);
    if (existing) return existing;
    return get().addUser(user);
  },

  loadFromStorage: () => {
    if (typeof window === 'undefined') return;
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    try {
      const parsed = JSON.parse(saved) as Partial<UserState>;
      set((state) => ({
        ...state,
        users: parsed.users ?? state.users,
        filters: parsed.filters ?? state.filters,
      }));
    } catch (error) {
      console.warn('Unable to restore user preferences', error);
    }
  },
}));

import { create } from 'zustand';
import { DEFAULT_USERS } from '../constants';
import { AuthSession } from '../types';
import { useUserStore } from './userStore';

const AUTH_STORAGE_KEY = 'auranet-auth-session';

const safeId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2, 11);

const persistSession = (session?: AuthSession) => {
  if (typeof window === 'undefined') return;
  if (!session) {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
};

type AuthState = {
  session?: AuthSession;
  loading: boolean;
};

type AuthActions = {
  signInWithGoogle: (email?: string) => AuthSession;
  signOut: () => void;
  hydrate: () => void;
};

export type AuthStore = AuthState & AuthActions;

export const useAuthStore = create<AuthStore>((set, get) => ({
  session: undefined,
  loading: false,

  hydrate: () => {
    if (typeof window === 'undefined') return;
    const saved = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!saved) return;

    try {
      const parsed = JSON.parse(saved) as AuthSession;
      set({ session: parsed });
      useUserStore.getState().ensureUser({
        id: parsed.userId,
        name: parsed.name,
        email: parsed.email,
        role: parsed.role,
        region: parsed.region,
        status: 'Active',
        avatarColor: '#2563eb',
      });
    } catch (error) {
      console.warn('Unable to restore auth session', error);
    }
  },

  signInWithGoogle: (email) => {
    set({ loading: true });
    const userStore = useUserStore.getState();
    const roster = userStore.users.length ? userStore.users : DEFAULT_USERS;
    const normalized = email?.toLowerCase();
    const matched = normalized ? roster.find((user) => user.email.toLowerCase() === normalized) : roster[0];

    const selected = matched ?? {
      id: safeId(),
      name: normalized ?? 'Google User',
      email: normalized ?? 'google-user@auranet.ai',
      role: 'Regional Viewer',
      region: 'All Regions',
      status: 'Active',
      avatarColor: '#2563eb',
      lastLogin: 'Just now',
    };

    const ensured = userStore.ensureUser(selected);

    const session: AuthSession = {
      userId: ensured.id,
      name: ensured.name,
      email: ensured.email,
      role: ensured.role,
      region: ensured.region,
      provider: 'google',
      avatarUrl: undefined,
    };

    persistSession(session);
    set({ session, loading: false });
    return session;
  },

  signOut: () => {
    persistSession(undefined);
    set({ session: undefined });
  },
}));

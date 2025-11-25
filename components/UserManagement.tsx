import React, { useEffect, useMemo, useState } from 'react';
import { Shield, UserPlus, Users, Filter, Globe2, Mail, LockKeyhole, KeyRound, Eye, EyeOff } from 'lucide-react';
import { useAuthStore } from '../services/authStore';
import { useUserStore } from '../services/userStore';
import { UserRole } from '../types';

const roleDescriptions: Record<UserRole, string> = {
  'Regional Admin': 'Full administrative access across all regions.',
  'Local Admin': 'Full administrative access within their assigned region.',
  'Regional Viewer': 'Read-only visibility across all regions.',
  'Local Viewer': 'Read-only visibility within their assigned region.',
};

const roleBadgeColor: Record<UserRole, string> = {
  'Regional Admin': 'bg-blue-100 text-blue-700',
  'Local Admin': 'bg-emerald-100 text-emerald-700',
  'Regional Viewer': 'bg-amber-100 text-amber-700',
  'Local Viewer': 'bg-slate-200 text-slate-700',
};

const initials = (name: string) =>
  name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .toUpperCase();

const UserManagement: React.FC = () => {
  const { session, signInWithGoogle, signOut, hydrate } = useAuthStore();
  const {
    users,
    regions,
    filters,
    setFilters,
    resetFilters,
    toggleStatus,
    addUser,
    updateUser,
    loadFromStorage,
  } = useUserStore();

  const [formState, setFormState] = useState({
    name: '',
    email: '',
    role: 'Local Viewer' as UserRole,
    region: regions[0]?.name ?? 'North America',
  });

  useEffect(() => {
    loadFromStorage();
    hydrate();
  }, [hydrate, loadFromStorage]);

  const filteredUsers = useMemo(() => {
    return users.filter((user) => {
      const matchesSearch = `${user.name} ${user.email}`
        .toLowerCase()
        .includes(filters.search.toLowerCase());
      const matchesRole = filters.role === 'All' || user.role === filters.role;
      const matchesRegion =
        filters.region === 'All' ||
        user.region === filters.region ||
        user.region === 'All Regions';
      return matchesSearch && matchesRole && matchesRegion;
    });
  }, [filters.region, filters.role, filters.search, users]);

  const adminCount = users.filter((user) => user.role.includes('Admin')).length;
  const viewerCount = users.filter((user) => user.role.includes('Viewer')).length;
  const inactiveCount = users.filter((user) => user.status === 'Inactive').length;

  const handleAddUser = (event: React.FormEvent) => {
    event.preventDefault();
    if (!formState.name.trim() || !formState.email.trim()) return;

    addUser({
      name: formState.name.trim(),
      email: formState.email.trim().toLowerCase(),
      role: formState.role,
      region: formState.role.includes('Regional') ? 'All Regions' : formState.region,
    });

    setFormState((prev) => ({ ...prev, name: '', email: '' }));
  };

  return (
    <div className="p-8 bg-slate-50 h-full overflow-y-auto">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <p className="text-sm font-semibold text-blue-600 mb-1">Access Control</p>
          <h1 className="text-2xl font-bold text-slate-900">Admin & Viewer Management</h1>
          <p className="text-sm text-slate-600 mt-1">
            Configure Google sign-in, roles, and region-based permissions without touching floor plans.
          </p>
        </div>
        <div className="bg-white shadow-sm border border-slate-200 rounded-xl p-4 w-full lg:w-80">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Shield size={18} className="text-blue-600" />
              <span className="text-sm font-semibold text-slate-800">Google OAuth</span>
            </div>
            {session ? (
              <span className="text-xs px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">Connected</span>
            ) : (
              <span className="text-xs px-2 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-100">Signed out</span>
            )}
          </div>
          {session ? (
            <div className="space-y-3 text-sm text-slate-700">
              <div className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold"
                  style={{ background: '#2563eb' }}
                >
                  {initials(session.name)}
                </div>
                <div>
                  <div className="font-semibold text-slate-900">{session.name}</div>
                  <div className="text-xs text-slate-500">{session.email}</div>
                  <div className="text-xs text-blue-700 mt-1">{session.role}</div>
                </div>
              </div>
              <button
                onClick={() => signOut()}
                className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 text-sm font-medium text-slate-700 border border-slate-200 rounded-lg hover:bg-slate-50"
              >
                <EyeOff size={16} /> Disconnect Google
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-sm text-slate-600 leading-relaxed">
                Use Google sign-in to sync admin identity and enforce region-aware privileges.
              </p>
              <button
                onClick={() => signInWithGoogle()}
                className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg"
              >
                <KeyRound size={16} /> Sign in with Google
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="grid md:grid-cols-4 gap-4 mt-6">
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs text-slate-500">Total Accounts</div>
          <div className="text-2xl font-bold text-slate-900 mt-1">{users.length}</div>
          <div className="text-xs text-slate-500 mt-2">{adminCount} admins, {viewerCount} viewers</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs text-slate-500">Active</div>
          <div className="text-2xl font-bold text-emerald-600 mt-1">{users.length - inactiveCount}</div>
          <div className="text-xs text-slate-500 mt-2">{inactiveCount} suspended</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs text-slate-500">Regions</div>
          <div className="text-2xl font-bold text-blue-600 mt-1">{regions.length}</div>
          <div className="text-xs text-slate-500 mt-2">Regional coverage & health</div>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <div className="text-xs text-slate-500">Signed-in Context</div>
          <div className="text-2xl font-bold text-slate-900 mt-1">{session ? 'Linked' : 'Guest'}</div>
          <div className="text-xs text-slate-500 mt-2">Google OAuth for console access</div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6 mt-8">
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 p-4 border-b border-slate-200">
              <div className="flex items-center gap-2 text-slate-700 font-semibold">
                <Users size={18} />
                User Directory
              </div>
              <div className="flex flex-wrap gap-2">
                <div className="relative">
                  <input
                    className="pl-9 pr-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-100"
                    placeholder="Search by name or email"
                    value={filters.search}
                    onChange={(e) => setFilters({ search: e.target.value })}
                  />
                  <Mail size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                </div>
                <select
                  className="text-sm border border-slate-200 rounded-lg px-3 py-2 bg-white"
                  value={filters.role}
                  onChange={(e) => setFilters({ role: e.target.value as UserRole | 'All' })}
                >
                  <option value="All">All Roles</option>
                  <option value="Regional Admin">Regional Admin</option>
                  <option value="Local Admin">Local Admin</option>
                  <option value="Regional Viewer">Regional Viewer</option>
                  <option value="Local Viewer">Local Viewer</option>
                </select>
                <select
                  className="text-sm border border-slate-200 rounded-lg px-3 py-2 bg-white"
                  value={filters.region}
                  onChange={(e) => setFilters({ region: e.target.value })}
                >
                  <option value="All">All Regions</option>
                  <option value="All Regions">All Regions</option>
                  {regions.map((region) => (
                    <option key={region.id} value={region.name}>
                      {region.name}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => resetFilters()}
                  className="inline-flex items-center gap-2 px-3 py-2 text-xs font-semibold text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50"
                >
                  <Filter size={14} /> Reset
                </button>
              </div>
            </div>

            <div className="divide-y divide-slate-100">
              {filteredUsers.map((user) => (
                <div key={user.id} className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 p-4">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-11 h-11 rounded-full flex items-center justify-center text-white font-semibold shrink-0"
                      style={{ background: user.avatarColor ?? '#2563eb' }}
                    >
                      {initials(user.name)}
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-slate-900">{user.name}</div>
                      <div className="text-xs text-slate-500">{user.email}</div>
                      <div className="flex flex-wrap gap-2 mt-2 text-xs">
                        <span className={`px-2 py-1 rounded-full ${roleBadgeColor[user.role]}`}>{user.role}</span>
                        <span className="px-2 py-1 rounded-full bg-slate-100 text-slate-700 flex items-center gap-1">
                          <Globe2 size={12} /> {user.region ?? 'All Regions'}
                        </span>
                        <span
                          className={`px-2 py-1 rounded-full border text-xs ${
                            user.status === 'Active'
                              ? 'border-emerald-100 bg-emerald-50 text-emerald-700'
                              : 'border-amber-100 bg-amber-50 text-amber-700'
                          }`}
                        >
                          {user.status}
                        </span>
                      </div>
                      {user.lastLogin && (
                        <div className="text-xs text-slate-400 mt-1">Last login: {user.lastLogin}</div>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 items-center">
                    <select
                      value={user.role}
                      onChange={(e) =>
                        updateUser(user.id, {
                          role: e.target.value as UserRole,
                          region: e.target.value.includes('Regional') ? 'All Regions' : user.region,
                        })
                      }
                      className="text-xs border border-slate-200 rounded-lg px-3 py-2 bg-white"
                    >
                      <option value="Regional Admin">Regional Admin</option>
                      <option value="Local Admin">Local Admin</option>
                      <option value="Regional Viewer">Regional Viewer</option>
                      <option value="Local Viewer">Local Viewer</option>
                    </select>
                    <select
                      value={user.region ?? 'All Regions'}
                      disabled={user.role.includes('Regional')}
                      onChange={(e) => updateUser(user.id, { region: e.target.value })}
                      className={`text-xs border border-slate-200 rounded-lg px-3 py-2 bg-white ${
                        user.role.includes('Regional') ? 'opacity-60 cursor-not-allowed' : ''
                      }`}
                    >
                      <option value="All Regions">All Regions</option>
                      {regions.map((region) => (
                        <option key={region.id} value={region.name}>
                          {region.name}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={() => toggleStatus(user.id)}
                      className={`px-3 py-2 text-xs font-semibold rounded-lg border transition-colors ${
                        user.status === 'Active'
                          ? 'text-emerald-700 border-emerald-100 bg-emerald-50 hover:bg-emerald-100'
                          : 'text-amber-700 border-amber-100 bg-amber-50 hover:bg-amber-100'
                      }`}
                    >
                      {user.status === 'Active' ? 'Suspend' : 'Activate'}
                    </button>
                  </div>
                </div>
              ))}

              {filteredUsers.length === 0 && (
                <div className="p-6 text-center text-slate-500 text-sm">No users match the current filters.</div>
              )}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4">
            <div className="flex items-center gap-2 text-slate-800 font-semibold mb-4">
              <Shield size={18} /> Role Definitions
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              {(Object.keys(roleDescriptions) as UserRole[]).map((role) => (
                <div key={role} className="border border-slate-200 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-xs font-semibold px-2 py-1 rounded-full ${roleBadgeColor[role]}`}>{role}</span>
                    {role.includes('Admin') ? <LockKeyhole size={14} className="text-slate-500" /> : <Eye size={14} className="text-slate-500" />}
                  </div>
                  <p className="text-sm text-slate-600 leading-relaxed">{roleDescriptions[role]}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
            <div className="flex items-center gap-2 text-slate-800 font-semibold mb-3">
              <UserPlus size={18} /> Invite user
            </div>
            <form className="space-y-3" onSubmit={handleAddUser}>
              <div>
                <label className="text-xs text-slate-500">Full name</label>
                <input
                  type="text"
                  value={formState.name}
                  onChange={(e) => setFormState((prev) => ({ ...prev, name: e.target.value }))}
                  className="mt-1 w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-100"
                  placeholder="Jane Doe"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500">Work email</label>
                <input
                  type="email"
                  value={formState.email}
                  onChange={(e) => setFormState((prev) => ({ ...prev, email: e.target.value }))}
                  className="mt-1 w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-100"
                  placeholder="name@auranet.ai"
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-500">Role</label>
                  <select
                    value={formState.role}
                    onChange={(e) =>
                      setFormState((prev) => ({
                        ...prev,
                        role: e.target.value as UserRole,
                        region: e.target.value.includes('Regional') ? 'All Regions' : prev.region,
                      }))
                    }
                    className="mt-1 w-full px-3 py-2 border border-slate-200 rounded-lg text-sm bg-white"
                  >
                    <option value="Regional Admin">Regional Admin</option>
                    <option value="Local Admin">Local Admin</option>
                    <option value="Regional Viewer">Regional Viewer</option>
                    <option value="Local Viewer">Local Viewer</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-slate-500">Region</label>
                  <select
                    value={formState.region}
                    disabled={formState.role.includes('Regional')}
                    onChange={(e) => setFormState((prev) => ({ ...prev, region: e.target.value }))}
                    className={`mt-1 w-full px-3 py-2 border border-slate-200 rounded-lg text-sm bg-white ${
                      formState.role.includes('Regional') ? 'opacity-60 cursor-not-allowed' : ''
                    }`}
                  >
                    <option value="All Regions">All Regions</option>
                    {regions.map((region) => (
                      <option key={region.id} value={region.name}>
                        {region.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <button
                type="submit"
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg"
              >
                <UserPlus size={16} /> Send invite
              </button>
            </form>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 space-y-3">
            <div className="flex items-center gap-2 text-slate-800 font-semibold">
              <Globe2 size={18} /> Region health
            </div>
            <div className="space-y-2">
              {regions.map((region) => (
                <div key={region.id} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                    <span className="font-semibold text-slate-800">{region.name}</span>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded-full border ${
                      region.status === 'Active'
                        ? 'border-emerald-100 bg-emerald-50 text-emerald-700'
                        : region.status === 'Onboarding'
                          ? 'border-amber-100 bg-amber-50 text-amber-700'
                          : 'border-blue-100 bg-blue-50 text-blue-700'
                    }`}
                  >
                    {region.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserManagement;

import React, { useEffect, useRef, useState } from 'react';
import { Layout, Folder, Users, Settings, Bell, Search, Menu, LogOut, LayoutGrid, LogIn, KeyRound } from 'lucide-react';
import ProjectList from './components/ProjectList';
import FloorPlanEditor from './components/FloorPlanEditor';
import { useProjectStore } from './services/projectStore';
import UserManagement from './components/UserManagement';
import { useAuthStore } from './services/authStore';
import { useUserStore } from './services/userStore';

// Basic SVG Logo
const Logo = () => (
  <div className="flex items-center gap-2 text-blue-600 font-bold text-xl tracking-tight">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 10.93a10.24 10.24 0 0 1 20 0" />
        <path d="M6 14.65a5.12 5.12 0 0 1 12 0" />
        <line x1="12" y1="19" x2="12" y2="19.01" />
    </svg>
    <span>AuraNet Planner <span className="text-blue-400 font-normal">AI</span></span>
  </div>
);

type View = 'dashboard' | 'projects' | 'editor' | 'users' | 'settings';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<View>('projects');
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const projectCount = useProjectStore((state) => state.projects.length);
  const loadProjects = useProjectStore((state) => state.loadFromStorage);
  const loadUsers = useUserStore((state) => state.loadFromStorage);
  const { session, hydrate, signOut, signInWithGoogle } = useAuthStore();
  const hasHydrated = useRef(false);

  useEffect(() => {
    if (hasHydrated.current) return;
    hasHydrated.current = true;
    loadProjects();
    loadUsers();
    hydrate();
  }, [hydrate, loadProjects, loadUsers]);

  const NavItem = ({ view, icon: Icon, label, badge }: { view: View, icon: any, label: string, badge?: number }) => (
    <button
      onClick={() => setCurrentView(view)}
      className={`flex items-center justify-between w-full px-4 py-2.5 mb-1 rounded-md text-sm font-medium transition-colors ${currentView === view ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'}`}
    >
      <div className="flex items-center gap-3">
        <Icon size={18} />
        {isSidebarOpen && <span>{label}</span>}
      </div>
      {isSidebarOpen && badge !== undefined && <span className="bg-slate-100 text-slate-600 text-xs px-2 py-0.5 rounded-full">{badge}</span>}
    </button>
  );

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      {/* Sidebar */}
      <aside className={`${isSidebarOpen ? 'w-64' : 'w-20'} bg-white border-r border-slate-200 flex flex-col transition-all duration-300 z-20`}>
        <div className="h-16 flex items-center px-6 border-b border-slate-100">
           {isSidebarOpen ? <Logo /> : <div className="text-blue-600 font-bold">AN</div>}
        </div>

        <div className="flex-1 py-6 px-3 overflow-y-auto">
          <div className="mb-8">
            <h3 className={`px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 ${!isSidebarOpen ? 'hidden' : ''}`}>
              Main
            </h3>
            <NavItem view="dashboard" icon={LayoutGrid} label="Dashboard" />
            <NavItem view="projects" icon={Folder} label="Projects" badge={projectCount} />
            <NavItem view="editor" icon={Layout} label="Floor Plans" />
          </div>

          <div className="mb-8">
             <h3 className={`px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 ${!isSidebarOpen ? 'hidden' : ''}`}>
               Organization
             </h3>
             <NavItem view="users" icon={Users} label="Users" />
             <NavItem view="settings" icon={Settings} label="Settings" />
          </div>
        </div>

        <div className="p-4 border-t border-slate-100">
           <button
             onClick={() => (session ? signOut() : signInWithGoogle())}
             className={`flex items-center gap-3 w-full px-2 py-2 text-sm font-medium transition-colors ${
               session ? 'text-slate-600 hover:text-red-600' : 'text-blue-600 hover:text-blue-700'
             }`}
           >
              {session ? <LogOut size={18} /> : <LogIn size={18} />}
              {isSidebarOpen && <span>{session ? 'Sign Out' : 'Sign In with Google'}</span>}
           </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full relative">
        {/* Top Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 shrink-0">
           <div className="flex items-center gap-4">
              <button onClick={() => setSidebarOpen(!isSidebarOpen)} className="p-2 text-slate-500 hover:bg-slate-100 rounded-md">
                 <Menu size={20}/>
              </button>
              <h1 className="text-lg font-semibold text-slate-800">
                 {currentView === 'projects' ? 'Project Management' :
                  currentView === 'editor' ? 'Floor Plan Editor' :
                  currentView.charAt(0).toUpperCase() + currentView.slice(1)}
              </h1>
           </div>

           <div className="flex items-center gap-6">
              <div className="relative">
                 <Search size={18} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
                 <input type="text" placeholder="Search..." className="pl-10 pr-4 py-2 bg-slate-50 border-none rounded-full text-sm w-64 focus:ring-2 focus:ring-blue-100 outline-none transition-all" />
              </div>
              <button className="relative p-2 text-slate-500 hover:bg-slate-100 rounded-full">
                 <Bell size={20} />
                 <span className="absolute top-1 right-2 w-2 h-2 bg-red-500 rounded-full border border-white"></span>
              </button>
             <div className="flex items-center gap-3 pl-6 border-l border-slate-200">
                <div className="text-right hidden md:block">
                    <div className="text-sm font-medium text-slate-800">{session?.name ?? 'Guest'}</div>
                    <div className="text-xs text-slate-500">{session?.role ?? 'No Google session'}</div>
                </div>
                 <div className="w-10 h-10 rounded-full border-2 border-white shadow-sm bg-blue-600 text-white flex items-center justify-center font-semibold">
                   {session?.name ? session.name.split(' ').map((n) => n[0]).join('').toUpperCase() : 'AU'}
                 </div>
              </div>
              {!session && (
                <button
                  onClick={() => signInWithGoogle()}
                  className="hidden md:inline-flex items-center gap-2 px-3 py-2 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg"
                >
                  <KeyRound size={16} /> Connect Google
                </button>
              )}
             </div>
          </header>

        {/* Dynamic View Content */}
        <div className="flex-1 overflow-hidden relative">
           {currentView === 'projects' && <ProjectList />}
           {currentView === 'editor' && <FloorPlanEditor />}
           {currentView === 'dashboard' && (
             <div className="p-10 text-center text-slate-500 mt-20">
               <div className="text-6xl mb-4">📊</div>
               <h2 className="text-xl font-semibold">Dashboard Coming Soon</h2>
               <p>Navigate to Projects or Floor Plans to see implemented features.</p>
             </div>
           )}
            {currentView === 'users' && (
             <UserManagement />
           )}
        </div>
      </main>
    </div>
  );
};

export default App;

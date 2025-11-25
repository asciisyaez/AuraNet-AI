import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FileText, Edit, Trash, Plus, Save, Upload } from 'lucide-react';
import { useProjectStore } from '../services/projectStore';
import { PersistedProjectData, Project } from '../types';

const ProjectList: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'All' | Project['status']>('All');
  const [message, setMessage] = useState('');

  const { projects, globalSettings, selectedProjectId } = useProjectStore((state) => ({
    projects: state.projects,
    globalSettings: state.globalSettings,
    selectedProjectId: state.selectedProjectId,
  }));
  const addProject = useProjectStore((state) => state.addProject);
  const saveToStorage = useProjectStore((state) => state.saveToStorage);
  const loadFromStorage = useProjectStore((state) => state.loadFromStorage);
  const importData = useProjectStore((state) => state.importData);
  const updateGlobalSettings = useProjectStore((state) => state.updateGlobalSettings);
  const setSelectedProjectId = useProjectStore((state) => state.setSelectedProjectId);

  useEffect(() => {
    loadFromStorage();
  }, [loadFromStorage]);

  const filteredProjects = useMemo(() => {
    return projects.filter((project) => {
      const matchesSearch = project.name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === 'All' || project.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [projects, searchTerm, statusFilter]);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? projects[0],
    [projects, selectedProjectId]
  );

  const handleNewProject = () => {
    addProject();
    setMessage('New project created with current global settings.');
  };

  const handleSave = () => {
    const payload = saveToStorage();
    const dataStr = JSON.stringify(payload, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'auranet-projects.json';
    link.click();
    URL.revokeObjectURL(url);
    setMessage('Projects saved to browser storage and downloaded as JSON.');
  };

  const handleLoadFromLocal = () => {
    const payload = loadFromStorage();
    setMessage(`Loaded ${payload.projects.length} project(s) from browser storage.`);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(reader.result as string) as PersistedProjectData;
        importData(payload);
        setMessage(`Imported ${payload.projects.length} project(s) from JSON file.`);
      } catch (error) {
        console.error('Import failed', error);
        setMessage('Unable to import file. Please ensure it is valid JSON.');
      }
    };
    reader.readAsText(file);
  };

  const renderSignalProfiles = (profiles: string[]) => profiles.join(', ');

  return (
    <div className="p-8 max-w-7xl mx-auto h-full overflow-y-auto">
      <div className="flex justify-between items-end mb-8">
        <div>
           <h2 className="text-2xl font-bold text-slate-800 mb-2">San Francisco HQ - Floor 1 Projects</h2>
           <div className="flex gap-4">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value as 'All' | Project['status'])}
                className="bg-white border border-slate-200 text-slate-600 text-sm rounded-md px-3 py-1.5 outline-none focus:ring-2 focus:ring-blue-100"
              >
                <option value="All">Status: All</option>
                <option value="Active">Active</option>
                <option value="Draft">Draft</option>
                <option value="Archived">Archived</option>
              </select>
              <select
                value={selectedProject?.id ?? ''}
                onChange={(e) => setSelectedProjectId(e.target.value || undefined)}
                className="bg-white border border-slate-200 text-slate-600 text-sm rounded-md px-3 py-1.5 outline-none focus:ring-2 focus:ring-blue-100"
              >
                <option value="">Active project: Auto-select first</option>
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))}
              </select>
              <button
                onClick={handleLoadFromLocal}
                className="bg-white border border-slate-200 text-slate-600 text-sm rounded-md px-3 py-1.5 outline-none focus:ring-2 focus:ring-blue-100"
              >
                Sync from Browser
              </button>
           </div>
        </div>
        <div className="flex gap-3 items-center">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search projects..."
              className="bg-white border border-slate-200 rounded-md px-4 py-2 w-64 outline-none focus:border-blue-500 text-sm"
            />
            <button
              onClick={handleNewProject}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2"
            >
                <Plus size={16} /> New
            </button>
            <button
              onClick={handleSave}
              className="bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2"
            >
                <Save size={16} /> Save
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2"
            >
                <Upload size={16} /> Load
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json"
              className="hidden"
              onChange={handleFileChange}
            />
        </div>
      </div>

      {message && (
        <div className="mb-6 text-sm text-blue-700 bg-blue-50 border border-blue-100 rounded-md px-4 py-3">
          {message}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm lg:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-slate-700">Global Settings</h3>
            <span className="text-xs text-slate-500">Persisted with each save</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
            <label className="text-sm text-slate-600 font-medium">
              Units
              <select
                value={globalSettings.units}
                onChange={(e) => updateGlobalSettings({ units: e.target.value as 'metric' | 'imperial' })}
                className="mt-1 block w-full border border-slate-200 rounded-md px-3 py-2 text-sm text-slate-700 focus:ring-2 focus:ring-blue-100"
              >
                <option value="metric">Metric (meters)</option>
                <option value="imperial">Imperial (feet)</option>
              </select>
            </label>
            <label className="text-sm text-slate-600 font-medium">
              Default Signal Profiles
              <input
                value={globalSettings.defaultSignalProfiles.join(', ')}
                onChange={(e) =>
                  updateGlobalSettings({
                    defaultSignalProfiles: e.target.value
                      .split(',')
                      .map((profile) => profile.trim())
                      .filter(Boolean),
                  })
                }
                placeholder="Comma separated list"
                className="mt-1 block w-full border border-slate-200 rounded-md px-3 py-2 text-sm text-slate-700 focus:ring-2 focus:ring-blue-100"
              />
            </label>
          </div>
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">Data Export</h3>
          <p className="text-xs text-slate-500 mb-3">Save projects and settings to JSON for portability.</p>
          <button
            onClick={handleSave}
            className="w-full mb-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium flex items-center justify-center gap-2"
          >
            <Save size={16} /> Save & Download
          </button>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full mb-2 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 px-4 py-2 rounded-md text-sm font-medium flex items-center justify-center gap-2"
          >
            <Upload size={16} /> Load from File
          </button>
          <button
            onClick={handleLoadFromLocal}
            className="w-full bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 px-4 py-2 rounded-md text-sm font-medium"
          >
            Load from Browser
          </button>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-500 uppercase tracking-wider">
              <th className="px-6 py-4">Project Name</th>
              <th className="px-6 py-4">Status</th>
              <th className="px-6 py-4">AI Optimization</th>
              <th className="px-6 py-4">Units</th>
              <th className="px-6 py-4">Signal Profiles</th>
              <th className="px-6 py-4">Last Modified</th>
              <th className="px-6 py-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {filteredProjects.map((project) => (
              <tr
                key={project.id}
                onClick={() => setSelectedProjectId(project.id)}
                className={`hover:bg-slate-50 transition-colors group cursor-pointer ${
                  project.id === selectedProjectId ? 'bg-blue-50/70' : ''
                }`}
              >
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-blue-50 flex items-center justify-center text-blue-600">
                        <FileText size={16} />
                    </div>
                    <div>
                      <div className="font-medium text-slate-800">{project.name}</div>
                      <div className="text-xs text-slate-500">{project.floorCount} floor(s)</div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                    ${project.status === 'Active' ? 'bg-green-100 text-green-800' :
                      project.status === 'Draft' ? 'bg-slate-100 text-slate-800' :
                      'bg-yellow-100 text-yellow-800'}`}>
                    {project.status}
                  </span>
                </td>
                <td className="px-6 py-4">
                   <span className={`text-sm ${project.optimizationStatus === 'Optimized' ? 'text-green-600 font-medium' : 'text-slate-500'}`}>
                     {project.optimizationStatus}
                   </span>
                </td>
                <td className="px-6 py-4 text-sm text-slate-600">{project.settings.units === 'metric' ? 'Metric' : 'Imperial'}</td>
                <td className="px-6 py-4 text-sm text-slate-600">{renderSignalProfiles(project.settings.defaultSignalProfiles)}</td>
                <td className="px-6 py-4 text-sm text-slate-500">
                  {project.lastModified}
                </td>
                <td className="px-6 py-4 text-right">
                  <div className="flex items-center justify-end gap-2 text-slate-400">
                    <button className="p-1 hover:text-blue-600 hover:bg-blue-50 rounded"><Edit size={16}/></button>
                    <button className="p-1 hover:text-red-600 hover:bg-red-50 rounded"><Trash size={16}/></button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination Footer Mock */}
        <div className="px-6 py-4 border-t border-slate-200 flex items-center justify-between text-sm text-slate-500">
            <span>Showing {filteredProjects.length} of {projects.length} projects</span>
            <div className="flex gap-2">
                <button className="px-3 py-1 border border-slate-200 rounded hover:bg-slate-50 disabled:opacity-50" disabled>&lt;</button>
                <button className="px-3 py-1 border border-slate-200 rounded hover:bg-slate-50 disabled:opacity-50" disabled>&gt;</button>
            </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectList;

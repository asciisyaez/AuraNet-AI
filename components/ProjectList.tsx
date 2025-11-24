import React from 'react';
import { MOCK_PROJECTS } from '../constants';
import { FileText, MoreHorizontal, Edit, Trash, Plus } from 'lucide-react';

const ProjectList: React.FC = () => {
  return (
    <div className="p-8 max-w-7xl mx-auto h-full overflow-y-auto">
      <div className="flex justify-between items-end mb-8">
        <div>
           <h2 className="text-2xl font-bold text-slate-800 mb-2">San Francisco HQ - Floor 1 Projects</h2>
           <div className="flex gap-4">
              <select className="bg-white border border-slate-200 text-slate-600 text-sm rounded-md px-3 py-1.5 outline-none focus:ring-2 focus:ring-blue-100">
                <option>Status: All</option>
                <option>Active</option>
                <option>Draft</option>
              </select>
              <select className="bg-white border border-slate-200 text-slate-600 text-sm rounded-md px-3 py-1.5 outline-none focus:ring-2 focus:ring-blue-100">
                <option>Type: All</option>
              </select>
           </div>
        </div>
        <div className="flex gap-4">
            <input 
              type="text" 
              placeholder="Search projects..." 
              className="bg-white border border-slate-200 rounded-md px-4 py-2 w-64 outline-none focus:border-blue-500 text-sm"
            />
            <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2">
                <Plus size={16} /> New Project
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
              <th className="px-6 py-4">Last Modified</th>
              <th className="px-6 py-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {MOCK_PROJECTS.map((project) => (
              <tr key={project.id} className="hover:bg-slate-50 transition-colors group cursor-pointer">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-blue-50 flex items-center justify-center text-blue-600">
                        <FileText size={16} />
                    </div>
                    <span className="font-medium text-slate-800">{project.name}</span>
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
            <span>Showing 1-4 of 4 projects</span>
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
